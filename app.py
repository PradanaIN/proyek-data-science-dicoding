import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load model dan scaler
model = joblib.load('model/rf_model.pkl')
scaler = joblib.load('model/scaler.pkl')

with open('model/columns.json', 'r') as f:
    columns = json.load(f)

# Mapping label
label_map = {0: 'Dropout', 1: 'Masih Kuliah (Enrolled)', 2: 'Lulus (Graduate)'}

# Fungsi prediksi
def prediksi_status(input_data):
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)

    for col in columns:
        if col not in df_input:
            df_input[col] = 0
    df_input = df_input[columns]

    df_scaled = scaler.transform(df_input)
    pred = model.predict(df_scaled)
    proba = model.predict_proba(df_scaled)[0]

    label_pred = label_map[int(pred[0])]
    proba_dict = {label_map[i]: f"{p:.2%}" for i, p in enumerate(proba)}

    return label_pred, proba_dict

# ===== Streamlit UI =====

st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="centered")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Masukkan informasi akademik dan pribadi mahasiswa untuk memprediksi apakah mereka akan **Dropout**, **Masih Kuliah (Enrolled)**, atau **Lulus (Graduate)**.")

# Form input
with st.form("form_mahasiswa"):
    st.subheader("📝 Data Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        admission_grade = st.slider("Nilai Masuk Perguruan Tinggi", 95.0, 200.0, 150.0)
        cu_1st_approved = st.number_input("Mata Kuliah 1 - Lulus", min_value=0, max_value=10, value=5)
        cu_1st_grade = st.number_input("Rata-rata Nilai Semester 1", min_value=0.0, max_value=20.0, value=13.0)
        cu_1st_eval = st.number_input("Jumlah Evaluasi Semester 1", min_value=0, max_value=15, value=6)

    with col2:
        cu_2nd_approved = st.number_input("Mata Kuliah 2 - Lulus", min_value=0, max_value=10, value=5)
        cu_2nd_grade = st.number_input("Rata-rata Nilai Semester 2", min_value=0.0, max_value=20.0, value=14.0)
        cu_2nd_eval = st.number_input("Jumlah Evaluasi Semester 2", min_value=0, max_value=15, value=5)
        course = st.selectbox("Program Studi", ['Nursing', 'Informatics', 'Management', 'Social Service', 'Psychology'])

    prev_qual_grade = st.slider("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 145.0)
    fathers_occ = st.selectbox("Pekerjaan Ayah", ['technical', 'services', 'student', 'unemployed', 'others'])

    submitted = st.form_submit_button("🔍 Prediksi Status")

# Prediksi
if submitted:
    input_data = {
        'Admission_grade': admission_grade,
        'Curricular_units_1st_sem_approved': cu_1st_approved,
        'Curricular_units_1st_sem_grade': cu_1st_grade,
        'Curricular_units_1st_sem_evaluations': cu_1st_eval,
        'Curricular_units_2nd_sem_approved': cu_2nd_approved,
        'Curricular_units_2nd_sem_grade': cu_2nd_grade,
        'Curricular_units_2nd_sem_evaluations': cu_2nd_eval,
        'Previous_qualification_grade': prev_qual_grade,
        'Course': course,
        'Fathers_occupation': fathers_occ
    }

    prediksi, probabilitas = prediksi_status(input_data)

    st.success(f"📌 Prediksi Status Mahasiswa: **{prediksi}**")
    st.markdown("### 📊 Probabilitas Tiap Status:")
    for k, v in probabilitas.items():
        st.markdown(f"- **{k}**: {v}")
