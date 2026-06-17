import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Diabetes Predictor", page_icon="🏥", layout="wide")

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

st.title("🏥 Diabetes Risk Prediction System")
st.markdown("Aplikasi sederhana untuk memprediksi risiko penyakit diabetes berdasarkan rekam medis pasien.")
st.markdown("---")

st.sidebar.header("📝 Input Data Pasien")
pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

st.subheader("🔍 Ringkasan Data Pasien")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pregnancies", pregnancies)
col2.metric("Glucose", glucose)
col3.metric("Blood Pressure", blood_pressure)
col4.metric("Skin Thickness", skin_thickness)

col5, col6, col7, col8 = st.columns(4)
col5.metric("Insulin", insulin)
col6.metric("BMI", bmi)
col7.metric("DPF", dpf)
col8.metric("Age", age)

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

st.markdown("---")
st.subheader("⚙️ Hasil Analisis")

if st.button("Lakukan Prediksi Risiko", type="primary", use_container_width=True):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("🚨 **Peringatan:** Pasien memiliki risiko TINGGI terkena diabetes. Disarankan untuk segera melakukan pemeriksaan medis lebih lanjut.")
    else:
        st.success("✅ **Aman:** Pasien memiliki risiko RENDAH terkena diabetes. Tetap pertahankan gaya hidup sehat.")
