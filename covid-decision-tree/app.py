import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(
    page_title="Deteksi COVID-19",
    layout="centered"
)

st.title("🩺 Sistem Diagnosis COVID-19")
st.write("Metode Decision Tree (Entropy)")

# ===============================
# Dataset
# ===============================
data = {
    "Demam": ["Ya","Ya","Ya","Tidak","Tidak","Ya","Tidak","Tidak"],
    "Batuk": ["Ya","Ya","Tidak","Ya","Tidak","Ya","Ya","Tidak"],
    "Sesak_Napas": ["Ya","Tidak","Tidak","Tidak","Tidak","Ya","Ya","Ya"],
    "Hilang_Penciuman": ["Ya","Ya","Tidak","Tidak","Tidak","Tidak","Ya","Tidak"],
    "Status_COVID": ["Positif","Positif","Negatif","Negatif","Negatif","Positif","Positif","Negatif"]
}

df = pd.DataFrame(data)

# ===============================
# Encoding
# ===============================
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop("Status_COVID", axis=1)
y = df["Status_COVID"]

# ===============================
# TRAIN MODEL (WAJIB DI LUAR BUTTON)
# ===============================
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)
model.fit(X, y)

# ===============================
# Input User
# ===============================
st.subheader("Masukkan Gejala Pasien")

demam = st.selectbox("Demam", ["Ya", "Tidak"])
batuk = st.selectbox("Batuk", ["Ya", "Tidak"])
sesak = st.selectbox("Sesak Napas", ["Ya", "Tidak"])
penciuman = st.selectbox("Kehilangan Penciuman", ["Ya", "Tidak"])

input_data = np.array([[
    1 if demam == "Ya" else 0,
    1 if batuk == "Ya" else 0,
    1 if sesak == "Ya" else 0,
    1 if penciuman == "Ya" else 0
]])

# ===============================
# Prediksi
# ===============================
if st.button("🔍 Diagnosa"):
    hasil = model.predict(input_data)

    if hasil[0] == 1:
        st.error("⚠️ HASIL: POSITIF COVID-19")
    else:
        st.success("✅ HASIL: NEGATIF COVID-19")
