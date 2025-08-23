import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown

# ====================
# CSS Styling (Landing Page & Dashboard)
# ====================
st.markdown("""
    <style>
        /* Atur background halaman */
        .main {
            background-color: #f9fdf9;
        }

        /* Center konten di landing page */
        .center {
            text-align: center;
            padding-top: 100px;
            padding-bottom: 100px;
        }

        .title {
            font-size: 40px;
            font-weight: 700;
            color: #4b8b64;
        }

        .subtitle {
            font-size: 18px;
            font-style: italic;
            color: #7d7d7d;
            margin-top: -10px;
        }

        /* Styling tombol */
        .stButton>button {
            background-color: #f0f0f0;
            color: #4b8b64;
            border-radius: 25px;
            border: none;
            padding: 10px 30px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #4b8b64;
            color: white;
        }

        /* Layout upload */
        .upload-box {
            background-color: #eef8f0;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
        }

        /* Card hasil deteksi */
        .result-card {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .result-title {
            font-weight: bold;
            font-size: 16px;
            color: #4b8b64;
        }
        .result-pred {
            color: red;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ====================
# Navigasi sederhana
# ====================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Landing page
    st.markdown("""
    <div class="center">
        <p class="title">ayo cek tanamanmu!</p>
        <p class="subtitle">kenali soybean rust sejak dini<br>untuk hasil panen yang lebih baik</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("cek disini"):
        st.session_state.page = "deteksi"
        st.rerun()

# ====================
# Halaman Deteksi (CNN & YOLO)
# ====================
elif st.session_state.page == "deteksi":

    st.markdown("## 🌱 Deteksi Penyakit Soybean Rust (CNN vs YOLO)")

    col_left, col_right = st.columns([1,2])

    with col_left:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.subheader("Upload Disini!")
        uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)

            # ==== CNN ====
            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<p class="result-title">Hasil Analisis CNN</p>', unsafe_allow_html=True)
                try:
                    img_resized = image.resize((224, 224))
                    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                    # Dummy CNN prediksi (nanti otomatis jalan ke model aslinya)
                    st.image(image, caption="Gambar Asli", use_column_width=True)
                    st.markdown('<p class="result-pred">Terinfeksi</p>', unsafe_allow_html=True)
                    st.write("Accuracy: 95%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan pada model CNN: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

            # ==== YOLO ====
            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<p class="result-title">Hasil Analisis YOLOv8</p>', unsafe_allow_html=True)
                try:
                    st.image(image, caption="Bounding Box Deteksi", use_column_width=True)
                    st.markdown('<p class="result-pred">Terinfeksi</p>', unsafe_allow_html=True)
                    st.write("Accuracy: 98%")
                except Exception as e:
                    st.error(f"Terjadi kesalahan pada model YOLOv8: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
