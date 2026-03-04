import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Diabetes Analytics Dashboard",
    layout="wide"
)

# ================= STYLE CSS =================

st.markdown("""
<style>

section[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
    border-right: 1px solid rgba(128,128,128,0.2);
    padding-top: 25px;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 20px;
    color: var(--text-color);
}

.kpi-card {
    background: var(--secondary-background-color);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,0.15);
    transition: 0.3s;
    text-align: center;
}

.kpi-card:hover {
    transform: translateY(-6px);
    border: 1px solid rgba(0,123,255,0.4);
}

.kpi-title {
    font-size: 14px;
    opacity: 0.7;
}

.kpi-value {
    font-size: 30px;
    font-weight: 600;
    margin-top: 8px;
}

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================

@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    df["OutcomeLabel"] = df["Outcome"].map({0: "Non-Diabetic", 1: "Diabetic"})
    return df

df = load_data()

# ================= SIDEBAR =================

st.sidebar.markdown('<div class="sidebar-title">Filters</div>', unsafe_allow_html=True)

status_filter = st.sidebar.selectbox(
    "Diabetic Status",
    ["All", "Diabetic", "Non-Diabetic"]
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (int(df["Age"].min()), int(df["Age"].max()))
)

#  AJOUT DU BMI RANGE
bmi_range = st.sidebar.slider(
    "BMI Range",
    float(df["BMI"].min()),
    float(df["BMI"].max()),
    (float(df["BMI"].min()), float(df["BMI"].max()))
)

# ================= FILTERING =================

filtered_df = df[
    (df["Age"] >= age_range[0]) &
    (df["Age"] <= age_range[1]) &
    (df["BMI"] >= bmi_range[0]) &
    (df["BMI"] <= bmi_range[1])
]

if status_filter != "All":
    filtered_df = filtered_df[filtered_df["OutcomeLabel"] == status_filter]

# ================= KPI CALCULATIONS =================

total_patients = len(filtered_df)
diabetic_count = len(filtered_df[filtered_df["Outcome"] == 1])
non_diabetic_count = len(filtered_df[filtered_df["Outcome"] == 0])
diabetes_rate = round((diabetic_count / total_patients) * 100, 2) if total_patients > 0 else 0
avg_glucose = round(filtered_df["Glucose"].mean(), 2) if total_patients > 0 else 0

# ================= TITLE =================

st.title("Diabetes Analytics Dashboard")

# ================= KPI SECTION =================

col1, col2, col3, col4 = st.columns(4)

kpi_data = [
    ("Total Patients", total_patients),
    ("Diabetic Patients", diabetic_count),
    ("Diabetes Rate (%)", diabetes_rate),
    ("Average Glucose", avg_glucose)
]

for col, (title, value) in zip([col1, col2, col3, col4], kpi_data):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ================= HISTOGRAM =================

fig1 = px.histogram(
    filtered_df,
    x="Glucose",
    nbins=30,
    title="Glucose Distribution",
    color_discrete_sequence=["#3A7BD5"]
)

fig1.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig1, use_container_width=True)

# ================= BOX PLOT =================

fig2 = px.box(
    filtered_df,
    x="OutcomeLabel",
    y="BMI",
    title="BMI by Diabetic Status",
    color="OutcomeLabel",
    color_discrete_sequence=["#27AE60", "#E74C3C"]
)

fig2.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False
)

st.plotly_chart(fig2, use_container_width=True)

# ================= HEATMAP =================

st.subheader("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include="number")
corr = numeric_df.corr()

fig3 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto"
)

fig3.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig3, use_container_width=True)
