import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

st.set_page_config(page_title="Resiko Diabetes", layout="wide")

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

st.title("Resiko Diabetes")
st.markdown("---")

st.sidebar.header("Data Pasien")
pregnancies = st.sidebar.number_input("Kehamilan", min_value=0, max_value=20, value=0)
glucose = st.sidebar.number_input("Kadar Glukosa", min_value=0, max_value=200, value=120)
blood_pressure = st.sidebar.number_input("Tekanan Darah", min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.sidebar.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input("Fungsi Silsilah Diabetes", min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input("Usia", min_value=1, max_value=120, value=30)

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

st.subheader("Hasil Analisis")

if st.button("Lakukan Prediksi Risiko", type="primary", use_container_width=True):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("Peringatan: Pasien memiliki risiko TINGGI terkena diabetes. Disarankan untuk segera melakukan pemeriksaan medis lebih lanjut.")
    else:
        st.success("Aman: Pasien memiliki risiko RENDAH terkena diabetes. Tetap pertahankan gaya hidup sehat.")
