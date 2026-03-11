import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
page_title="AI Diabetes Intelligence Dashboard",
layout="wide"
)

# ================= STYLE =================

st.markdown("""
<style>

/* BACKGROUND */

.stApp{
background-image:url("https://images.unsplash.com/photo-1576091160550-2173dba999ef");
background-size:cover;
background-position:center;
background-attachment:fixed;
}

/* OVERLAY GLOBAL */

.block-container{
background:rgba(10,10,10,0.45);
padding:30px;
border-radius:18px;
}

/* KPI CARDS */

.kpi-card{
background:rgba(255,255,255,0.12);
backdrop-filter:blur(10px);
padding:22px;
border-radius:16px;
text-align:center;
color:white;
box-shadow:0 8px 30px rgba(0,0,0,0.6);
transition:0.3s;
border:2px solid rgba(255,255,255,0.35);
}

.kpi-card:hover{
transform:scale(1.07);
box-shadow:0 10px 40px rgba(0,0,0,0.8);
}

/* SECTION GRAPHES */

.section{
background:rgba(0,0,0,0.65);
padding:25px;
border-radius:18px;
margin-top:25px;
margin-bottom:25px;
border:1px solid rgba(255,255,255,0.15);
}

/* TEXT */

h1,h2,h3,h4,h5,p,span{
color:white !important;
}

</style>
""",unsafe_allow_html=True)

# ================= LOAD DATA =================

@st.cache_data
def load_data():

    df=pd.read_csv("diabetes.csv")

    df["OutcomeLabel"]=df["Outcome"].map({
        0:"Non-Diabetic",
        1:"Diabetic"
    })

    df["Gender"]=np.random.choice(["Male","Female"],len(df))

    return df

df=load_data()

# ================= SIDEBAR =================

st.sidebar.title("Medical Filters")

status_filter=st.sidebar.selectbox(
"Diabetes Status",
["All","Diabetic","Non-Diabetic"]
)

gender_filter=st.sidebar.selectbox(
"Gender",
["All","Male","Female"]
)

age_range=st.sidebar.slider(
"Age",
int(df.Age.min()),
int(df.Age.max()),
(int(df.Age.min()),int(df.Age.max()))
)

bmi_range=st.sidebar.slider(
"BMI",
float(df.BMI.min()),
float(df.BMI.max()),
(float(df.BMI.min()),float(df.BMI.max()))
)

glucose_range=st.sidebar.slider(
"Glucose",
int(df.Glucose.min()),
int(df.Glucose.max()),
(int(df.Glucose.min()),int(df.Glucose.max()))
)

bp_range=st.sidebar.slider(
"Blood Pressure",
int(df.BloodPressure.min()),
int(df.BloodPressure.max()),
(int(df.BloodPressure.min()),int(df.BloodPressure.max()))
)

# ================= FILTER DATA =================

filtered=df[
(df.Age>=age_range[0]) &
(df.Age<=age_range[1]) &
(df.BMI>=bmi_range[0]) &
(df.BMI<=bmi_range[1]) &
(df.Glucose>=glucose_range[0]) &
(df.Glucose<=glucose_range[1]) &
(df.BloodPressure>=bp_range[0]) &
(df.BloodPressure<=bp_range[1])
]

if status_filter!="All":
    filtered=filtered[filtered["OutcomeLabel"]==status_filter]

if gender_filter!="All":
    filtered=filtered[filtered["Gender"]==gender_filter]

# ================= KPI =================

total=len(filtered)
diabetic=len(filtered[filtered.Outcome==1])
rate=round(diabetic/total*100,2) if total>0 else 0
avg_glucose=round(filtered.Glucose.mean(),2)
avg_bmi=round(filtered.BMI.mean(),2)

# ================= TITLE =================

st.title("AI Diabetes Intelligence Dashboard")

st.write("""
Advanced medical analytics platform exploring relationships between
health indicators and diabetes risk using interactive visualization
and machine learning.
""")

# ================= KPI DISPLAY =================

c1,c2,c3,c4,c5=st.columns(5)

kpis=[
("Total Patients",total),
("Diabetic Patients",diabetic),
("Diabetes Rate %",rate),
("Average Glucose",avg_glucose),
("Average BMI",avg_bmi)
]

for col,(t,v) in zip([c1,c2,c3,c4,c5],kpis):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
        <div style="font-size:14px;opacity:0.8">{t}</div>
        <div style="font-size:34px;font-weight:700">{v}</div>
        </div>
        """,unsafe_allow_html=True)

# ================= HISTOGRAM =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("Glucose Distribution")

fig=px.histogram(filtered,x="Glucose",color="OutcomeLabel",nbins=30)

fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig,use_container_width=True)

st.write("Higher glucose levels significantly increase diabetes risk.")

st.markdown('</div>',unsafe_allow_html=True)

# ================= BOXPLOT =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("BMI Distribution by Diabetes")

fig=px.box(filtered,x="OutcomeLabel",y="BMI",color="OutcomeLabel")

fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig,use_container_width=True)

st.write("Diabetic patients generally show higher BMI values.")

st.markdown('</div>',unsafe_allow_html=True)

# ================= SCATTER =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("Age vs Glucose Relationship")

fig=px.scatter(
filtered,
x="Age",
y="Glucose",
color="OutcomeLabel",
size="BMI"
)

fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig,use_container_width=True)

st.write("Age combined with glucose level highlights risk groups.")

st.markdown('</div>',unsafe_allow_html=True)

# ================= HEATMAP =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("Medical Correlation Matrix")

corr=filtered.select_dtypes(include="number").corr()

fig=px.imshow(
corr,
text_auto=True,
color_continuous_scale="RdBu"
)

fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
font_color="white"
)

st.plotly_chart(fig,use_container_width=True)

st.write("The heatmap highlights relationships between medical variables.")

st.markdown('</div>',unsafe_allow_html=True)

# ================= SANKEY =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("Medical Flow Analysis (Sankey Diagram)")

filtered["GlucoseLevel"]=pd.cut(filtered["Glucose"],
bins=[0,100,140,300],
labels=["Normal","Elevated","High"])

filtered["BMILevel"]=pd.cut(filtered["BMI"],
bins=[0,25,30,70],
labels=["Normal","Overweight","Obese"])

source=[]
target=[]
value=[]

levels=list(filtered["GlucoseLevel"].unique())+list(filtered["BMILevel"].unique())

level_map={k:i for i,k in enumerate(levels)}

for g in filtered["GlucoseLevel"].unique():
    for b in filtered["BMILevel"].unique():
        count=len(filtered[(filtered["GlucoseLevel"]==g)&(filtered["BMILevel"]==b)])
        if count>0:
            source.append(level_map[g])
            target.append(level_map[b])
            value.append(count)

fig=go.Figure(data=[go.Sankey(
node=dict(label=levels,pad=15,thickness=20),
link=dict(source=source,target=target,value=value)
)])

fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
font_color="white"
)

st.plotly_chart(fig,use_container_width=True)

st.write("""
The Sankey diagram illustrates how **glucose levels influence BMI categories**.
It helps identify **risk pathways leading to diabetes**.
""")

st.markdown('</div>',unsafe_allow_html=True)

# ================= MACHINE LEARNING =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("AI Diabetes Risk Prediction")

features=df.drop(columns=["Outcome","OutcomeLabel","Gender"])
target=df["Outcome"]

model=RandomForestClassifier()
model.fit(features,target)

preg=st.number_input("Pregnancies",0,20,1)
glucose=st.number_input("Glucose",0,200,120)
bp=st.number_input("Blood Pressure",0,140,70)
skin=st.number_input("Skin Thickness",0,100,20)
insulin=st.number_input("Insulin",0,900,80)
bmi=st.number_input("BMI",0.0,70.0,25.0)
dpf=st.number_input("Diabetes Pedigree Function",0.0,3.0,0.5)
age=st.number_input("Age",10,100,30)

prediction=model.predict([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])[0]
prob=model.predict_proba([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])[0][1]

st.write("Predicted Risk Probability:",round(prob*100,2),"%")

if prediction==1:
    st.error("High Diabetes Risk")
else:
    st.success("Low Diabetes Risk")

st.markdown('</div>',unsafe_allow_html=True)

# ================= FEATURE IMPORTANCE =================

st.markdown('<div class="section">',unsafe_allow_html=True)

st.subheader("AI Feature Importance")

importance=pd.DataFrame({
"feature":features.columns,
"importance":model.feature_importances_
}).sort_values("importance",ascending=False)

fig=px.bar(
importance,
x="importance",
y="feature",
orientation="h"
)

fig.update_layout(
paper_bgcolor="rgba(0,0,0,0)",
plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig,use_container_width=True)

st.write("This graph explains which medical factors influence the AI model the most.")

st.markdown('</div>',unsafe_allow_html=True)

# ================= DATA TABLE =================

st.subheader("Filtered Dataset")

st.dataframe(filtered,use_container_width=True)
