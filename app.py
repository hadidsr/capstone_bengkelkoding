import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model dan preprocessing
model = joblib.load("obesity_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # nama kolom training

st.title("Prediksi Tingkat Obesitas")
st.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas:")

# Form input pengguna
with st.form("form_obesitas"):
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Usia", min_value=10, max_value=100, value=25)
    height = st.number_input("Tinggi Badan (meter)", min_value=1.0, max_value=2.5, step=0.01)
    weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, step=0.5)
    family = st.selectbox("Riwayat Keluarga Overweight", ["yes", "no"])
    favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori?", ["yes", "no"])
    fcvc = st.slider("Frekuensi Konsumsi Sayuran (0-3)", 0, 3)
    ncp = st.slider("Jumlah Makan Besar per Hari", 1, 5)
    caec = st.selectbox("Kebiasaan Camilan", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Merokok?", ["yes", "no"])
    ch2o = st.slider("Jumlah Air Minum (Liter/Hari)", 0.0, 5.0, step=0.1)
    scc = st.selectbox("Memantau Kalori?", ["yes", "no"])
    faf = st.slider("Aktivitas Fisik Mingguan (jam)", 0.0, 7.0, step=0.5)
    tue = st.slider("Waktu Layar per Hari (jam)", 0, 24)
    calc = st.selectbox("Frekuensi Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportasi Utama", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submitted = st.form_submit_button("Prediksi")

# Saat tombol ditekan
if submitted:
    # Buat DataFrame input
    input_data = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }
    input_df = pd.DataFrame([input_data])

    # Encode kolom kategorikal
    for col in encoders:
        try:
            input_df[col] = encoders[col].transform(input_df[col])
        except ValueError:
            st.error(f"Nilai '{input_df[col][0]}' di kolom '{col}' tidak dikenali. Coba ubah input.")
            st.stop()

    # Urutkan kolom sesuai training
    input_df = input_df[feature_columns]

    # Normalisasi
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    label = target_encoder.inverse_transform([prediction])[0]

    # Tampilkan hasil
    st.success(f"Tingkat Obesitas: **{label}**")
