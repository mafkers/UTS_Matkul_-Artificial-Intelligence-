import streamlit as st
import pandas as pd
import numpy as np
import joblib

import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Diabetes Predictor", page_icon="🏥")

@st.cache_resource
def load_model():
    if os.path.exists('diabetes_model.pkl'):
        return joblib.load('diabetes_model.pkl')
    else:
        df = pd.read_csv('diabetes.csv')
        df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'diabetes_model.pkl')
        return model

model = load_model()

st.title("Diabetes Risk Prediction")

st.sidebar.header("Input Patient Data")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

st.subheader("Patient Data Preview")
st.write(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
