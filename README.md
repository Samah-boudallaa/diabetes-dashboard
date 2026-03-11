# AI Diabetes Intelligence Dashboard

## Project Overview

The AI Diabetes Intelligence Dashboard is an interactive data analytics and machine learning application developed using Streamlit, Plotly, and Scikit-learn.

The objective of this project is to explore medical indicators related to diabetes and provide an intelligent visualization platform capable of both data analysis and predictive modeling.

The dashboard enables users to explore the dataset interactively, analyze relationships between health indicators, and estimate diabetes risk using a machine learning model.

---

# Dataset Information

Source: Kaggle  
Dataset: Pima Indians Diabetes Dataset

The dataset contains medical diagnostic measurements for female patients used to predict diabetes.

## Dataset Features

- Pregnancies: Number of pregnancies
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure
- SkinThickness: Triceps skin fold thickness
- Insulin: Serum insulin level
- BMI: Body Mass Index
- DiabetesPedigreeFunction: Genetic influence indicator
- Age: Patient age
- Outcome: 0 (Non-Diabetic), 1 (Diabetic)

---

# Project Objectives

The main objectives of this project are:

- Perform exploratory data analysis (EDA) on diabetes medical data
- Build an interactive analytics dashboard
- Identify key medical indicators influencing diabetes risk
- Integrate machine learning for diabetes prediction
- Provide a clean and intuitive interface for healthcare data exploration

---

# Key Features

## Interactive Medical Filters

The dashboard allows dynamic filtering of the dataset using multiple parameters:

- Diabetes status (Diabetic / Non-Diabetic)
- Gender simulation
- Age range
- BMI range
- Glucose range
- Blood pressure range

These filters allow users to explore specific patient groups and identify hidden patterns.

---

# KPI Indicators

The dashboard displays dynamic KPI cards including:

- Total number of patients
- Number of diabetic patients
- Diabetes prevalence rate
- Average glucose level
- Average BMI

All indicators update automatically based on selected filters.

---

# Data Visualization

The dashboard includes several interactive visualizations to explore medical patterns.

## Glucose Distribution

Histogram showing the distribution of glucose levels across diabetic and non-diabetic patients.

## BMI Distribution by Diabetes Status

Boxplot highlighting BMI differences between diabetic and non-diabetic individuals.

## Age vs Glucose Relationship

Scatter plot exploring how age and glucose levels interact and influence diabetes risk.

## Correlation Heatmap

Correlation matrix showing relationships between medical variables.

---

# Advanced Analytics

## Sankey Flow Analysis

The Sankey diagram visualizes how glucose levels transition into BMI categories, helping identify possible medical risk pathways.

This visualization provides an intuitive way to understand relationships between health indicators.

---

# Machine Learning Model

The application integrates a Random Forest Classifier to estimate diabetes risk.

Users can input medical indicators including:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The model predicts:

- Diabetes risk probability
- Risk classification (High Risk or Low Risk)

---

# Model Interpretability

The dashboard includes a Feature Importance visualization that highlights which medical variables have the strongest influence on the machine learning model.

Typical important features include:

- Glucose
- BMI
- Age
- Diabetes Pedigree Function

---

# Technologies Used

This project was developed using the following technologies:

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn

---

# How to Run the Application

## Install dependencies
pip install -r requirements.txt

## Run the Streamlit application
streamlit run app.py

The dashboard will open automatically in your browser.

---

# Key Insights

Some important insights obtained from the analysis include:

- Glucose level shows a strong correlation with diabetes outcome.
- Higher BMI values are associated with increased diabetes risk.
- Diabetes prevalence increases with age.
- Combining multiple medical indicators improves prediction performance.

---

# Future Improvements

Possible future enhancements include:

- Deep learning based prediction models
- Patient risk clustering
- Multi-page analytics dashboards
- Real-time healthcare monitoring
- Deployment on Streamlit Cloud

---

# Author

This project was developed as a data analytics and machine learning practice project focusing on interactive visualization and predictive modeling.

