import streamlit as st
import joblib
import numpy as np

model = joblib.load('model_covid_dt.pkl')

st.set_page_config(page_title="Deteksi COVID-19", layout="centered")

st.title("🩺 Sistem Deteksi Gejala COVID-19")
st.write("Masukkan kondisi gejala berikut:")

def konversi(x):
    return 1 if x == "Ya" else 0

demam = st.selectbox("Demam", ["Ya", "Tidak"])
batuk = st.selectbox("Batuk", ["Ya", "Tidak"])
sesak = st.selectbox("Sesak Nafas", ["Ya", "Tidak"])
penciuman = st.selectbox("Kehilangan Penciuman", ["Ya", "Tidak"])

if st.button("Diagnosa"):
    data = np.array([[
        konversi(demam),
        konversi(batuk),
        konversi(sesak),
        konversi(penciuman)
    ]])

    hasil = model.predict(data)

    if hasil[0] == 1:
        st.error("⚠️ Hasil: POSITIF COVID-19")
    else:
        st.success("✅ Hasil: NEGATIF COVID-19")