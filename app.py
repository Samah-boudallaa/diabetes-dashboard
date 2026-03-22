import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Diabetes Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= STYLE =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #0a0f1e; }
.block-container { background: transparent; padding: 2rem 3rem; max-width: 1600px; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,200,255,0.15);
}
[data-testid="stSidebar"] * { color: #e0f0ff !important; }

/* HEADER */
.dashboard-header {
    background: linear-gradient(135deg, rgba(0,180,255,0.08) 0%, rgba(100,0,255,0.08) 100%);
    border: 1px solid rgba(0,200,255,0.2);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
}
.dashboard-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.5px; }
.dashboard-title span { color: #00c8ff; }
.dashboard-subtitle { color: #7ecfff; font-size: 0.95rem; margin-top: 6px; }
.badge {
    display: inline-block;
    background: rgba(0,200,255,0.15);
    border: 1px solid rgba(0,200,255,0.4);
    color: #00c8ff;
    font-size: 0.72rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 1px; text-transform: uppercase;
    margin-right: 6px; margin-top: 10px;
}

/* KPI CARDS */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 22px 20px;
    text-align: center; position: relative; overflow: hidden;
    height: 110px; display: flex; flex-direction: column; justify-content: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
.kpi-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}
.kpi-card.blue::after   { background: linear-gradient(90deg, #00c8ff, #0080ff); }
.kpi-card.red::after    { background: linear-gradient(90deg, #ff4d6d, #c9184a); }
.kpi-card.orange::after { background: linear-gradient(90deg, #ff9500, #ff6000); }
.kpi-card.green::after  { background: linear-gradient(90deg, #06d6a0, #1b9aaa); }
.kpi-card.purple::after { background: linear-gradient(90deg, #a855f7, #7c3aed); }
.kpi-label { font-size: 0.70rem; font-weight: 600; color: #8aa8c8; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1; font-family: 'JetBrains Mono', monospace; }
.kpi-delta { font-size: 0.72rem; color: #6b8caa; margin-top: 5px; }

/* SECTION CARD (Sankey, Feature Importance, Guide Sante) */
.section-card {
    background: linear-gradient(135deg, rgba(13,27,58,0.95) 0%, rgba(8,18,40,0.95) 100%);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 18px; padding: 24px;
    margin-bottom: 20px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    box-sizing: border-box;
}
.section-card:hover { border-color: rgba(0,200,255,0.32); box-shadow: 0 4px 20px rgba(0,200,255,0.07); }

/* CARD-HEADER : titre+desc seulement, sans barre coloree */
.card-header {
    background: linear-gradient(135deg, rgba(13,27,58,0.95) 0%, rgba(8,18,40,0.95) 100%);
    border: 1px solid rgba(0,200,255,0.12);
    border-radius: 18px; padding: 18px 22px;
    margin-bottom: 10px;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
    box-sizing: border-box;
}
.card-header:hover { border-color: rgba(0,200,255,0.32); box-shadow: 0 4px 20px rgba(0,200,255,0.07); }

.section-title { font-size: 1rem; font-weight: 600; color: #ffffff !important; margin-bottom: 4px; }
.section-desc  { font-size: 0.80rem; color: #5a7a9a; margin-bottom: 0; line-height: 1.5; }

/* INSIGHT BOXES */
.insight-box {
    background: rgba(0,200,255,0.06); border-left: 3px solid #00c8ff;
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin-top: 10px;
}
.insight-box p { color: #a8d8f0 !important; font-size: 0.82rem; margin: 0; line-height: 1.5; }
.insight-title { color: #00c8ff !important; font-size: 0.73rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.insight-box.warning { background: rgba(255,149,0,0.06); border-left-color: #ff9500; }
.insight-box.warning p { color: #ffd580 !important; }
.insight-box.warning .insight-title { color: #ff9500 !important; }
.insight-box.success { background: rgba(6,214,160,0.06); border-left-color: #06d6a0; }
.insight-box.success p { color: #a8f0dc !important; }
.insight-box.success .insight-title { color: #06d6a0 !important; }
.insight-box.danger { background: rgba(255,77,109,0.06); border-left-color: #ff4d6d; }
.insight-box.danger p { color: #ffb3c1 !important; }
.insight-box.danger .insight-title { color: #ff4d6d !important; }

/* DIVIDER */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,200,255,0.2), transparent);
    margin: 20px 0;
}

/* RESULT CARDS — style KPI sans barre coloree en haut, hover translateY seulement */
.result-high {
    background: linear-gradient(135deg, rgba(255,77,109,0.12), rgba(200,24,74,0.08));
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 16px; padding: 28px 20px;
    text-align: center; color: #ff4d6d;
    font-weight: 700; font-size: 1rem;
    transition: transform 0.2s ease;
    cursor: default;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 6px;
}
.result-high:hover { transform: translateY(-3px); }

.result-low {
    background: linear-gradient(135deg, rgba(6,214,160,0.12), rgba(27,154,170,0.08));
    border: 1px solid rgba(6,214,160,0.35);
    border-radius: 16px; padding: 28px 20px;
    text-align: center; color: #06d6a0;
    font-weight: 700; font-size: 1rem;
    transition: transform 0.2s ease;
    cursor: default;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 6px;
}
.result-low:hover { transform: translateY(-3px); }

.metric-badge {
    display: inline-block;
    background: rgba(168,85,247,0.15); border: 1px solid rgba(168,85,247,0.3);
    color: #c084fc; font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; padding: 4px 10px; border-radius: 6px; margin: 3px;
}

/* REFERENCE TABLE — classes CSS sans inline styles */
.ref-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.82rem; margin-top: 10px; background: transparent;
}
.ref-table th {
    color: #7ecfff !important; font-weight: 600;
    text-transform: uppercase; font-size: 0.70rem; letter-spacing: 1px;
    padding: 10px 12px; border-bottom: 1px solid rgba(0,200,255,0.25);
    text-align: left; background: rgba(0,200,255,0.05);
}
.ref-table td {
    color: #c0d8f0 !important; padding: 9px 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: middle;
}
.ref-table tr { transition: background 0.15s ease; }
.ref-table tbody tr:hover td { background: rgba(0,200,255,0.06) !important; }
.tag-normal  { display:inline-block; background: rgba(6,214,160,0.18);  color: #06d6a0 !important; padding: 3px 10px; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }
.tag-warning { display:inline-block; background: rgba(255,149,0,0.18);   color: #ff9500 !important; padding: 3px 10px; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }
.tag-danger  { display:inline-block; background: rgba(255,77,109,0.18);  color: #ff4d6d !important; padding: 3px 10px; border-radius: 10px; font-size: 0.72rem; font-weight: 600; }
.td-mono { font-family: 'JetBrains Mono', monospace !important; color: #ffffff !important; font-weight: 600 !important; padding: 9px 12px; }
.td-name { color: #e0f0ff !important; font-weight: 600 !important; padding: 9px 12px; }

/* STREAMLIT OVERRIDES */
h1, h2, h3, h4, h5, h6, p, span, label, div { color: #d0e8ff; }
.stMetric { background: transparent; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'JetBrains Mono', monospace; }
.stNumberInput label, .stSlider label, .stSelectbox label {
    color: #7ecfff !important; font-size: 0.8rem !important; font-weight: 600 !important;
}
.stNumberInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    color: #ffffff !important; border-radius: 8px !important;
}
.sidebar-title {
    font-size: 1rem; font-weight: 700; color: #00c8ff !important;
    letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 16px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,200,255,0.2);
}
.sidebar-section {
    font-size: 0.7rem; font-weight: 700; color: #3a6080 !important;
    text-transform: uppercase; letter-spacing: 1.5px; margin: 14px 0 5px 0;
}
</style>
""", unsafe_allow_html=True)


# ================= LOAD & CLEAN DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "diabetes.csv"))
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)
    df.drop_duplicates(inplace=True)
    df["OutcomeLabel"] = df["Outcome"].map({0: "Non-Diabétique", 1: "Diabétique"})
    df["AgeGroup"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 90],
                            labels=["20-30", "30-40", "40-50", "50-60", "60+"])
    df["GlucoseLevel"] = pd.cut(df["Glucose"], bins=[0, 100, 126, 300],
                                labels=["Normal (<100)", "Pre-diabete (100-126)", "Eleve (>126)"])
    df["BMILevel"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                            labels=["Insuffisant", "Normal", "Surpoids", "Obese"])
    df["RiskScore"] = (
        (df["Glucose"] / df["Glucose"].max()) * 0.35 +
        (df["BMI"] / df["BMI"].max()) * 0.20 +
        (df["Age"] / df["Age"].max()) * 0.15 +
        (df["Insulin"] / df["Insulin"].max()) * 0.15 +
        (df["DiabetesPedigreeFunction"] / df["DiabetesPedigreeFunction"].max()) * 0.15
    ).round(3)
    return df


@st.cache_resource
def train_model(df):
    features = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                   "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]
    target = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, acc, report, features.columns.tolist()


df = load_data()
model, accuracy, clf_report, feature_cols = train_model(df)

# ================= PLOT THEME =================
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#a8c8e8", family="Space Grotesk"),
    title_font=dict(color="#ffffff"),
    legend=dict(bgcolor="rgba(10,20,40,0.8)", bordercolor="rgba(0,200,255,0.2)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)")
)
COLOR_MAP = {"Non-Diabétique": "#00c8ff", "Diabétique": "#ff4d6d"}

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="sidebar-title">Filtres Medicaux</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Statut</div>', unsafe_allow_html=True)
    status_filter = st.selectbox("Statut", ["Tous", "Diabétique", "Non-Diabétique"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Age et IMC</div>', unsafe_allow_html=True)
    age_range = st.slider("Age", int(df.Age.min()), int(df.Age.max()),
                          (int(df.Age.min()), int(df.Age.max())))
    bmi_range = st.slider("IMC (kg/m2)", float(df.BMI.min()), float(df.BMI.max()),
                          (float(df.BMI.min()), float(df.BMI.max())), step=0.5)

    st.markdown('<div class="sidebar-section">Marqueurs Sanguins</div>', unsafe_allow_html=True)
    glucose_range = st.slider("Glucose (mg/dL)", int(df.Glucose.min()), int(df.Glucose.max()),
                              (int(df.Glucose.min()), int(df.Glucose.max())))
    bp_range = st.slider("Pression Arterielle", int(df.BloodPressure.min()), int(df.BloodPressure.max()),
                         (int(df.BloodPressure.min()), int(df.BloodPressure.max())))
    insulin_range = st.slider("Insuline (uU/mL)", int(df.Insulin.min()), int(df.Insulin.max()),
                              (int(df.Insulin.min()), int(df.Insulin.max())))

    st.markdown('<div class="sidebar-section">Autres</div>', unsafe_allow_html=True)
    preg_range = st.slider("Grossesses", int(df.Pregnancies.min()), int(df.Pregnancies.max()),
                           (int(df.Pregnancies.min()), int(df.Pregnancies.max())))

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.72rem; color:#3a6080; line-height:1.8;'>
    Dataset : Pima Indians Diabetes<br>
    Source : Kaggle / UCI<br>
    Modele : Random Forest (n=100)<br>
    Accuracy : <span style='color:#06d6a0; font-weight:700;'>{accuracy*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

# ================= FILTER DATA =================
filtered = df[
    (df.Age >= age_range[0]) & (df.Age <= age_range[1]) &
    (df.BMI >= bmi_range[0]) & (df.BMI <= bmi_range[1]) &
    (df.Glucose >= glucose_range[0]) & (df.Glucose <= glucose_range[1]) &
    (df.BloodPressure >= bp_range[0]) & (df.BloodPressure <= bp_range[1]) &
    (df.Insulin >= insulin_range[0]) & (df.Insulin <= insulin_range[1]) &
    (df.Pregnancies >= preg_range[0]) & (df.Pregnancies <= preg_range[1])
]
if status_filter != "Tous":
    filtered = filtered[filtered["OutcomeLabel"] == status_filter]

total       = len(filtered)
diabetic    = len(filtered[filtered.Outcome == 1])
rate        = round(diabetic / total * 100, 1) if total > 0 else 0
avg_glucose = round(filtered.Glucose.mean(), 1) if total > 0 else 0
avg_bmi     = round(filtered.BMI.mean(), 1) if total > 0 else 0

# ================= HEADER =================
st.markdown(f"""
<div class="dashboard-header">
    <div class="dashboard-title">AI Diabetes <span>Intelligence</span> Dashboard</div>
    <div class="dashboard-subtitle">
        Analyse predictive avancee des facteurs de risque diabetiques — Dataset Pima Indians (768 patientes)
    </div>
    <div style="margin-top:12px;">
        <span class="badge">Machine Learning</span>
        <span class="badge">DataViz</span>
        <span class="badge">Sante Publique</span>
        <span class="badge" style="background:rgba(6,214,160,0.15);border-color:rgba(6,214,160,0.4);color:#06d6a0;">
            Accuracy {accuracy*100:.1f}%
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= KPIs =================
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("blue",   "Total Patientes",  total,       "dataset filtre"),
    ("red",    "Cas Diabetiques",  diabetic,    f"{rate}% du total"),
    ("orange", "Taux Diabete",     f"{rate}%",  "prevalence filtree"),
    ("green",  "Glucose Moyen",    avg_glucose, "mg/dL"),
    ("purple", "IMC Moyen",        avg_bmi,     "kg/m2"),
]
for col, (color, label, val, delta) in zip([c1, c2, c3, c4, c5], kpis):
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ================= ROW 1 : Glucose + IMC =================
col_a, col_b = st.columns([3, 2])

with col_a:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">Distribution du Glucose</div>
        <div class="section-desc">
            Repartition du taux de glucose plasmatique selon le statut diabetique.
            Le seuil clinique de 126 mg/dL marque le diagnostic de diabete.
        </div>
    </div>
    """, unsafe_allow_html=True)
    fig = px.histogram(filtered, x="Glucose", color="OutcomeLabel", nbins=35,
                       color_discrete_map=COLOR_MAP, barmode="overlay", opacity=0.75)
    fig.add_vline(x=126, line_dash="dash", line_color="#ff9500", line_width=2,
                  annotation_text="Seuil diabete (126)", annotation_font_color="#ff9500",
                  annotation_bgcolor="rgba(0,0,0,0.7)")
    fig.add_vline(x=100, line_dash="dot", line_color="#06d6a0", line_width=1,
                  annotation_text="Pre-diabete (100)", annotation_font_color="#06d6a0",
                  annotation_bgcolor="rgba(0,0,0,0.7)")
    fig.update_layout(**PLOT_THEME, height=320, showlegend=True,
                      title="", xaxis_title="Glucose (mg/dL)", yaxis_title="Nombre de patientes")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">Insight Cle</div>
        <p>Les patientes diabetiques ont un glucose significativement plus eleve,
        avec un pic autour de 140-180 mg/dL. Le glucose est le predicteur le plus fort de ce dataset.</p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">IMC par Statut Diabetique</div>
        <div class="section-desc">
            Distribution de l'IMC entre les deux groupes. L'obesite (IMC > 30) est un facteur de risque majeur.
        </div>
    </div>
    """, unsafe_allow_html=True)
    fig = px.violin(filtered, x="OutcomeLabel", y="BMI", color="OutcomeLabel",
                    box=True, points="outliers", color_discrete_map=COLOR_MAP)
    fig.add_hline(y=30, line_dash="dash", line_color="#ff9500", line_width=1.5,
                  annotation_text="Obesite (30)", annotation_font_color="#ff9500")
    fig.add_hline(y=25, line_dash="dot", line_color="#7ecfff", line_width=1,
                  annotation_text="Surpoids (25)", annotation_font_color="#7ecfff")
    fig.update_layout(**PLOT_THEME, height=320, showlegend=False,
                      title="", xaxis_title="", yaxis_title="IMC (kg/m2)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box warning">
        <div class="insight-title">Observation</div>
        <p>La mediane de l'IMC chez les diabetiques (~34) depasse le seuil d'obesite,
        confirmant le lien fort entre surpoids et diabete de type 2.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)

# ================= ROW 2 : Scatter + Heatmap =================
col_c, col_d = st.columns([2, 3])

with col_c:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">Age vs Glucose</div>
        <div class="section-desc">Chaque point est une patiente. La taille du point encode l'IMC.</div>
    </div>
    """, unsafe_allow_html=True)
    fig = px.scatter(filtered, x="Age", y="Glucose", color="OutcomeLabel",
                     size="BMI", size_max=18, opacity=0.7,
                     color_discrete_map=COLOR_MAP,
                     hover_data=["BloodPressure", "Insulin"])
    fig.add_hline(y=126, line_dash="dash", line_color="#ff9500", line_width=1.5)
    fig.update_layout(**PLOT_THEME, height=340, showlegend=True,
                      title="", xaxis_title="Age (annees)", yaxis_title="Glucose (mg/dL)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box">
        <div class="insight-title">Tendance</div>
        <p>Le risque augmente avec l'age. Les cas diabetiques se concentrent
        au-dessus du seuil de 126 mg/dL et chez les patientes de plus de 35 ans.</p>
    </div>
    """, unsafe_allow_html=True)

with col_d:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">Matrice de Correlation</div>
        <div class="section-desc">
            Relations lineaires entre toutes les variables cliniques.
            Les valeurs proches de +1 indiquent une forte correlation positive avec le diabete.
        </div>
    </div>
    """, unsafe_allow_html=True)
    corr = filtered.select_dtypes(include="number").drop(columns=["Outcome", "RiskScore"], errors="ignore").corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_traces(textfont=dict(color="white", size=10))
    theme_no_font = {k: v for k, v in PLOT_THEME.items() if k != "font"}
    fig.update_layout(**theme_no_font, height=340,
                      font=dict(color="#a8c8e8", family="Space Grotesk"),
                      title="",
                      coloraxis_colorbar=dict(
                          tickfont=dict(color="#a8c8e8"),
                          title=dict(text="r", font=dict(color="#a8c8e8"))))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box success">
        <div class="insight-title">Correlations Principales</div>
        <p>Glucose -> Outcome (r~0.47) est la correlation la plus forte.
        IMC et Age montrent egalement des correlations significatives avec le diagnostic.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)

# ================= ROW 3 : Barplot + Pie + Line =================
col_e, col_f, col_g = st.columns(3)

with col_e:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">Diabete par Groupe d'Age</div>
        <div class="section-desc">Nombre de cas diabetiques et non-diabetiques par tranche d'age.</div>
    </div>
    """, unsafe_allow_html=True)
    age_counts = filtered.groupby(["AgeGroup", "OutcomeLabel"], observed=True).size().reset_index(name="count")
    fig = px.bar(age_counts, x="AgeGroup", y="count", color="OutcomeLabel",
                 barmode="group", color_discrete_map=COLOR_MAP)
    fig.update_layout(**PLOT_THEME, height=300, showlegend=True,
                      title="", xaxis_title="Groupe d'age", yaxis_title="Patientes", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

with col_f:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">Repartition Globale</div>
        <div class="section-desc">Proportion de cas diabetiques dans la selection actuelle.</div>
    </div>
    """, unsafe_allow_html=True)
    pie_data = filtered["OutcomeLabel"].value_counts().reset_index()
    pie_data.columns = ["Statut", "count"]
    fig = px.pie(pie_data, names="Statut", values="count",
                 color="Statut", color_discrete_map=COLOR_MAP, hole=0.55)
    fig.update_traces(textfont_color="white", textfont_size=12)
    fig.update_layout(**PLOT_THEME, height=300, showlegend=True, title="", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

with col_g:
    st.markdown("""
    <div class="card-header">
        <div class="section-title">Score de Risque par Age</div>
        <div class="section-desc">Score composite moyen (glucose, IMC, age, insuline, genetique) par groupe d'age.</div>
    </div>
    """, unsafe_allow_html=True)
    risk_line = filtered.groupby(["AgeGroup", "OutcomeLabel"], observed=True)["RiskScore"].mean().reset_index()
    fig = px.line(risk_line, x="AgeGroup", y="RiskScore", color="OutcomeLabel",
                  markers=True, color_discrete_map=COLOR_MAP)
    fig.update_traces(line_width=2.5, marker_size=8)
    fig.update_layout(**PLOT_THEME, height=300, showlegend=True,
                      title="", xaxis_title="Groupe d'age", yaxis_title="Score de risque", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)

# ================= SANKEY =================
st.markdown("""
<div class="section-card">
    <div class="section-title">Diagramme de Flux (Sankey) — Parcours de Risque</div>
    <div class="section-desc">
        Comment lire ce diagramme ? Il montre le parcours des patientes depuis leur niveau de glucose
        vers leur categorie d'IMC, puis vers leur diagnostic final.
        Plus une bande est epaisse, plus le nombre de patientes concernees est eleve.
        Objectif : identifier quels profils combinees (glucose eleve + obesite) menent le plus souvent au diabete.
    </div>
""", unsafe_allow_html=True)

sankey_df   = filtered.dropna(subset=["GlucoseLevel", "BMILevel"])
nodes_glucose = sankey_df["GlucoseLevel"].cat.categories.tolist()
nodes_bmi   = sankey_df["BMILevel"].cat.categories.tolist()
nodes_outcome = ["Non-Diabetique", "Diabetique"]
all_nodes   = nodes_glucose + nodes_bmi + nodes_outcome
node_idx    = {n: i for i, n in enumerate(all_nodes)}

source, target, value, link_color = [], [], [], []
for g in nodes_glucose:
    for b in nodes_bmi:
        cnt = len(sankey_df[(sankey_df["GlucoseLevel"] == g) & (sankey_df["BMILevel"] == b)])
        if cnt > 0:
            source.append(node_idx[g]); target.append(node_idx[b])
            value.append(cnt); link_color.append("rgba(168,85,247,0.3)")

for b in nodes_bmi:
    for o in ["Non-Diabetique", "Diabetique"]:
        label_col = "Non-Diabétique" if o == "Non-Diabetique" else "Diabétique"
        cnt = len(sankey_df[(sankey_df["BMILevel"] == b) & (sankey_df["OutcomeLabel"] == label_col)])
        if cnt > 0:
            source.append(node_idx[b]); target.append(node_idx[o])
            value.append(cnt)
            link_color.append("rgba(255,77,109,0.3)" if o == "Diabetique" else "rgba(0,200,255,0.3)")

node_colors = ["#a855f7"] * len(nodes_glucose) + ["#0080ff"] * len(nodes_bmi) + ["#00c8ff", "#ff4d6d"]

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(pad=20, thickness=22, label=all_nodes, color=node_colors,
              line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
    link=dict(source=source, target=target, value=value, color=link_color)
)])
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=12, family="Space Grotesk"),
    height=380
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight-box">
    <div class="insight-title">Comment interpreter les resultats</div>
    <p>Le flux le plus epais entre "Eleve (>126)" vers "Obese" vers "Diabetique" confirme que
    la combinaison glucose eleve + obesite est le chemin dominant. A l'inverse, les patientes
    avec un glucose normal restent majoritairement non-diabetiques quel que soit leur IMC.</p>
</div>
</div>
""", unsafe_allow_html=True)

# ================= PREDICTION IA =================
st.markdown(f"""
<div class="section-card">
    <div class="section-title">Prediction IA — Risque Diabetique Individuel</div>
    <div class="section-desc">
        Modele Random Forest entraine sur 80% du dataset (614 patientes).
        Entrez vos mesures cliniques pour obtenir une estimation de votre risque.
        Accuracy du modele : <span style="color:#06d6a0; font-weight:700;">{accuracy*100:.1f}%</span>
    </div>
""", unsafe_allow_html=True)

col_form, col_result = st.columns([3, 2])

with col_form:
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1: preg_val    = st.number_input("Grossesses",       0,    20,   1)
    with r1c2: glucose_val = st.number_input("Glucose (mg/dL)", 44,   200, 120)
    with r1c3: bp_val      = st.number_input("Pression Art.",   24,   122,  70)
    with r1c4: skin_val    = st.number_input("Epaisseur Peau",   7,    99,  20)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1: insulin_val = st.number_input("Insuline (uU/mL)", 14,  846,  80)
    with r2c2: bmi_val     = st.number_input("IMC (kg/m2)",    18.2, 67.1, 25.0, step=0.1)
    with r2c3: dpf_val     = st.number_input("Pedigree",       0.08, 2.42, 0.47, step=0.01)
    with r2c4: age_val     = st.number_input("Age",              21,   81,  30)

with col_result:
    input_data = [[preg_val, glucose_val, bp_val, skin_val, insulin_val, bmi_val, dpf_val, age_val]]
    pred       = model.predict(input_data)[0]
    prob       = model.predict_proba(input_data)[0][1]
    prob_pct   = round(prob * 100, 1)

    if pred == 1:
        st.markdown(f"""
        <div class="result-high">
            RISQUE ELEVE DETECTE<br>
            <span style="font-size:2.6rem; font-family:'JetBrains Mono',monospace;">{prob_pct}%</span><br>
            <span style="font-size:0.8rem; opacity:0.75;">probabilite de diabete</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            RISQUE FAIBLE<br>
            <span style="font-size:2.6rem; font-family:'JetBrains Mono',monospace;">{prob_pct}%</span><br>
            <span style="font-size:0.8rem; opacity:0.75;">probabilite de diabete</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:12px;">
        <div class="metric-badge">Accuracy : {accuracy*100:.1f}%</div>
        <div class="metric-badge">Precision : {clf_report['1']['precision']*100:.1f}%</div>
        <div class="metric-badge">Rappel : {clf_report['1']['recall']*100:.1f}%</div>
        <div class="metric-badge">F1-Score : {clf_report['1']['f1-score']*100:.1f}%</div>
    </div>
    <div style="font-size:0.72rem; color:#3a6080; margin-top:10px;">
        Cet outil est purement educatif et ne remplace pas un avis medical professionnel.
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================= GUIDE SANTE PERSONNELLE =================
st.markdown("""
<div class="section-card">
    <div class="section-title">Guide Sante — Comprendre vos Mesures</div>
    <div class="section-desc">
        Ce tableau de reference vous permet de situer chaque indicateur medical dans une echelle clinique reconnue.
        Si plusieurs de vos valeurs se trouvent dans la zone Eleve, consultez un professionnel de sante.
    </div>
""", unsafe_allow_html=True)

def get_status(val, normal_max, warning_max, unit):
    if val <= normal_max:
        return '<span class="tag-normal">Normal</span>', "success"
    elif val <= warning_max:
        return '<span class="tag-warning">Vigilance</span>', "warning"
    else:
        return '<span class="tag-danger">Eleve</span>', "danger"

indicators = [
    ("Glucose (mg/dL)",      glucose_val, 99,   125,  "mg/dL",  "< 100",       "100 - 125",  "> 126"),
    ("Pression Art. (mmHg)", bp_val,      79,   89,   "mmHg",   "< 80",        "80 - 89",    "> 90"),
    ("IMC (kg/m2)",          bmi_val,     24.9, 29.9, "kg/m2",  "18.5 - 24.9", "25 - 29.9",  "> 30"),
    ("Insuline (uU/mL)",     insulin_val, 25,   166,  "uU/mL",  "2 - 25",      "25 - 166",   "> 166"),
]

rows_html = ""
alerts = []
for name, val, n_max, w_max, unit, n_label, w_label, d_label in indicators:
    tag, level = get_status(val, n_max, w_max, unit)
    if level == "danger":
        alerts.append(name)
    rows_html += f"""
    <tr>
        <td class="td-name">{name}</td>
        <td><span class="tag-normal">{n_label}</span></td>
        <td><span class="tag-warning">{w_label}</span></td>
        <td><span class="tag-danger">{d_label}</span></td>
        <td class="td-mono">{val}</td>
        <td>{tag}</td>
    </tr>"""

table_html = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: transparent; font-family: 'Space Grotesk', sans-serif; }}
table {{
    width: 100%; border-collapse: collapse;
    font-size: 0.82rem; background: transparent;
}}
th {{
    color: #7ecfff; font-weight: 600; text-transform: uppercase;
    font-size: 0.70rem; letter-spacing: 1px;
    padding: 10px 14px; border-bottom: 1px solid rgba(0,200,255,0.25);
    text-align: left; background: rgba(0,200,255,0.07);
}}
td {{
    color: #c0d8f0; padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    vertical-align: middle;
}}
tr:hover td {{ background: rgba(0,200,255,0.06); }}
.td-mono {{ font-family: 'JetBrains Mono', monospace; color: #ffffff; font-weight: 600; }}
.td-name {{ color: #e0f0ff; font-weight: 600; }}
.tag-normal  {{ display:inline-block; background:rgba(6,214,160,0.18);  color:#06d6a0; padding:3px 10px; border-radius:10px; font-size:0.72rem; font-weight:600; }}
.tag-warning {{ display:inline-block; background:rgba(255,149,0,0.18);  color:#ff9500; padding:3px 10px; border-radius:10px; font-size:0.72rem; font-weight:600; }}
.tag-danger  {{ display:inline-block; background:rgba(255,77,109,0.18); color:#ff4d6d; padding:3px 10px; border-radius:10px; font-size:0.72rem; font-weight:600; }}
</style>
<table>
<thead>
<tr>
    <th>Indicateur</th>
    <th>Zone Normale</th>
    <th>Zone de Vigilance</th>
    <th>Zone a Risque</th>
    <th>Votre Valeur</th>
    <th>Statut</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
"""
components.html(table_html, height=220, scrolling=False)

if pred == 1 or len(alerts) >= 2:
    st.markdown(f"""
    <div class="insight-box danger" style="margin-top:16px;">
        <div class="insight-title">Recommandation Medicale</div>
        <p>Votre profil presente {len(alerts)} indicateur(s) en zone a risque
        ({', '.join(alerts) if alerts else 'combinaison de facteurs'}) et une probabilite de diabete de {prob_pct}%.
        Nous vous recommandons de consulter un medecin ou endocrinologue pour un bilan glycemique complet (HbA1c, HGPO).
        Un diagnostic precoce permet d'agir efficacement.</p>
    </div>""", unsafe_allow_html=True)
elif len(alerts) == 1:
    st.markdown(f"""
    <div class="insight-box warning" style="margin-top:16px;">
        <div class="insight-title">Point de Vigilance</div>
        <p>Votre indicateur {alerts[0]} se situe en zone de vigilance.
        Maintenez une alimentation equilibree, une activite physique reguliere et un bilan annuel chez votre medecin.</p>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="insight-box success" style="margin-top:16px;">
        <div class="insight-title">Profil Favorable</div>
        <p>Tous vos indicateurs sont dans les zones normales. Continuez un mode de vie sain :
        alimentation equilibree, activite physique reguliere et controles medicaux preventifs annuels.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ================= FEATURE IMPORTANCE =================
st.markdown("""
<div class="section-card">
    <div class="section-title">Importance des Variables — Modele IA</div>
    <div class="section-desc">
        Le Random Forest calcule la contribution de chaque variable dans la prise de decision.
        Une importance elevee signifie que la variable est tres discriminante pour distinguer les deux groupes.
    </div>
""", unsafe_allow_html=True)

importance_df = pd.DataFrame({
    "Variable": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

labels_fr = {
    "Pregnancies": "Grossesses", "Glucose": "Glucose",
    "BloodPressure": "Pression Arterielle", "SkinThickness": "Epaisseur Peau",
    "Insulin": "Insuline", "BMI": "IMC",
    "DiabetesPedigreeFunction": "Pedigree Diabetique", "Age": "Age"
}
importance_df["Variable_FR"] = importance_df["Variable"].map(labels_fr)
importance_df["Color"] = importance_df["Importance"].apply(
    lambda x: "#ff4d6d" if x > 0.15 else ("#ff9500" if x > 0.08 else "#00c8ff")
)

fig = go.Figure(go.Bar(
    x=importance_df["Importance"],
    y=importance_df["Variable_FR"],
    orientation="h",
    marker_color=importance_df["Color"],
    text=[f"{v*100:.1f}%" for v in importance_df["Importance"]],
    textposition="outside",
    textfont=dict(color="white", size=11)
))
theme_imp = {k: v for k, v in PLOT_THEME.items() if k not in ("xaxis", "yaxis")}
fig.update_layout(**theme_imp, height=320, title="",
                  xaxis=dict(title="Importance relative", gridcolor="rgba(255,255,255,0.05)",
                             linecolor="rgba(255,255,255,0.1)", tickformat=".0%"),
                  yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)",
                             linecolor="rgba(255,255,255,0.1)"))
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div class="insight-box success">
    <div class="insight-title">Variables Cles identifiees par l'IA</div>
    <p>Le Glucose domine largement (>20%), suivi du Pedigree Diabetique et de l'IMC.
    Cela valide l'approche clinique : la glycemie reste le biomarqueur principal de diagnostic.</p>
</div>
</div>""", unsafe_allow_html=True)

# ================= DATA TABLE =================
with st.expander("Voir le dataset filtre"):
    display_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "AgeGroup", "OutcomeLabel", "RiskScore"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "Pregnancies": "Grossesses", "BloodPressure": "Pression Art.",
            "SkinThickness": "Epaisseur Peau", "AgeGroup": "Groupe Age",
            "OutcomeLabel": "Statut", "RiskScore": "Score Risque"
        }),
        use_container_width=True
    )
    st.caption(f"Affichage de {len(filtered)} patientes sur {len(df)} — apres application des filtres.")

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center; padding:30px 0 10px; border-top:1px solid rgba(255,255,255,0.05); margin-top:30px;">
    <div style="font-size:0.75rem; color:#2a4060; letter-spacing:1px;">
        AI DIABETES INTELLIGENCE DASHBOARD  |  Pima Indians Dataset  |
        Streamlit + Plotly + Scikit-learn  |  Mini-Projet DataViz 2025-2026
    </div>
</div>
""", unsafe_allow_html=True)
