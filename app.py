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
        /* Atur font global */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }

        /* Atur background halaman */
        .main {
            background-color: #f9fdf9;
            color: #555555; /* default teks abu-abu sama dengan landing */
        }
        
        /* Background untuk landing page dengan gambar daun */
        .landing-page {
            background-image: url("https://placehold.co/1000x1000/F2FBF1/F2FBF1?text=."), 
                              url("https://placehold.co/1000x1000/F2FBF1/F2FBF1?text=.");
            background-size: 400px, 400px;
            background-repeat: no-repeat, no-repeat;
            background-position: top -50px left -100px, bottom -50px right -100px;
            background-color: #F2FBF1;
        }
        
        /* Styling untuk logo dan navigasi di header */
        .header-section {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 40px;
            background-color: white;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .logo {
            display: flex;
            align-items: center;
            font-weight: 700;
            color: #4b8b64;
        }
        
        .logo-text {
            margin-left: 10px;
            font-size: 20px;
        }
        
        .nav-links {
            display: flex;
            gap: 20px;
            font-size: 14px;
        }
        
        .nav-links a {
            color: #555555;
            text-decoration: none;
            transition: color 0.3s;
        }
        
        .nav-links a:hover {
            color: #4b8b64;
        }

        /* Center konten di landing page */
        .center {
            text-align: center;
            padding-top: 150px;
            padding-bottom: 150px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .title {
            font-size: 48px;
            font-weight: 700;
            color: #4b8b64; /* hijau daun */
        }

        .subtitle {
            font-size: 20px;
            font-style: italic;
            color: #555555; /* abu lembut */
            margin-top: -10px;
        }

        /* Styling tombol */
        .stButton>button {
            background-color: white;
            color: #4b8b64;
            border-radius: 25px;
            border: 1px solid #4b8b64;
            padding: 10px 30px;
            font-weight: 600;
            cursor: pointer;
            transition: 0.3s;
            position: center;
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            background-color: #4b8b64;
            color: white;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
        }

        /* Layout deteksi */
        .detection-container {
            display: flex;
            justify-content: center;
            gap: 40px;
            padding: 40px;
            background-color: #f9fdf9;
        }

        /* Kolom kiri untuk upload */
        .upload-column {
            background-color: #eef8f0;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            color: #555555;
            flex: 1;
        }

        /* Styling tombol browse/upload */
        .st-emotion-cache-1c0y53j, .st-emotion-cache-1c0y53j > div > button {
            background-color: white !important;
            color: #4b8b64 !important;
            border: 1px solid #4b8b64 !important;
            border-radius: 10px !important;
            padding: 8px 20px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
            transition: 0.3s !important;
        }
        .st-emotion-cache-1c0y53j > div > button:hover {
            background-color: #4b8b64 !important;
            color: white !important;
        }

        /* Card hasil deteksi */
        .result-card {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            color: #555555; /* teks abu */
            min-height: 250px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        
        .result-card h5 {
            font-weight: bold;
            font-size: 18px;
            color: #4b8b64; /* judul card tetap hijau */
            margin-bottom: 10px;
        }
        
        .result-pred {
            color: red;
            font-weight: 600;
            font-size: 1.2em;
        }

        .image-placeholder {
            background-color: #F2FBF1;
            border: 2px dashed #4b8b64;
            border-radius: 10px;
            height: 150px;
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #4b8b64;
            font-style: italic;
            font-size: 12px;
            margin-bottom: 10px;
        }
        
        /* Kolom untuk hasil deteksi */
        .result-column {
            display: flex;
            flex-direction: column;
            gap: 20px;
            flex: 2;
        }

    </style>
""", unsafe_allow_html=True)


# ====================
# Header Navigasi
# ====================
def show_header():
    st.markdown("""
        <div class="header-section">
            <div class="logo">
                🌱 <span class="logo-text">SoybeanCare</span>
            </div>
            <div class="nav-links">
                <a href="#" onclick="window.location.reload();">Dashboard</a>
                <a href="#" onclick="window.location.reload();">Deteksi Penyakit</a>
                <a href="#" onclick="window.location.reload();">Evaluasi</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ====================
# Navigasi sederhana
# ====================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Landing page
    st.markdown('<div class="landing-page">', unsafe_allow_html=True)
    st.markdown("""
    <div class="center">
        <p class="title">ayo cek tanamanmu!</p>
        <p class="subtitle">kenali soybean rust sejak dini<br>untuk hasil panen yang lebih baik</p>
        <div style="text-align: center;">
            <button class="stButton">
                <a href="#" onclick="parent.postMessage({target: 'streamlit_rerun', data: {page: 'deteksi'}}, '*'); return false;">cek disini</a>
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Menangani klik tombol di landing page
    if st.button("cek disini", key="home_button"):
        st.session_state.page = "deteksi"
        st.rerun()

# ====================
# Halaman Deteksi (CNN & YOLO)
# ====================
elif st.session_state.page == "deteksi":
    show_header()
    st.markdown("## 🌱 Deteksi Penyakit Soybean Rust (CNN vs YOLO)", unsafe_allow_html=True)
    st.markdown("""<div class="detection-container">""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown('<div class="upload-column">', unsafe_allow_html=True)
        st.subheader("Upload Disini!")
        st.markdown('<p>kamu bisa upload disini biar kita bantu deteksi</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="result-column">', unsafe_allow_html=True)
        
        # Row 1
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<h5>Gambar Asli</h5>', unsafe_allow_html=True)
            if uploaded_file is None:
                st.markdown('<div class="image-placeholder">Gambar akan muncul disini</div>', unsafe_allow_html=True)
            else:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<h5>Hasil Deteksi Grad-Cam CNN</h5>', unsafe_allow_html=True)
            if uploaded_file is not None:
                # Dummy CNN prediksi
                st.markdown('<p class="result-pred">Terinfeksi</p>', unsafe_allow_html=True)
                st.write("Accuracy: 95%")
            else:
                st.markdown('<div class="image-placeholder">Hasil deteksi akan muncul disini</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<h5>Gambar Asli</h5>', unsafe_allow_html=True)
            if uploaded_file is None:
                st.markdown('<div class="image-placeholder">Gambar akan muncul disini</div>', unsafe_allow_html=True)
            else:
                st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<h5>Hasil Deteksi Bounding Box</h5>', unsafe_allow_html=True)
            if uploaded_file is not None:
                # Dummy YOLO prediksi
                st.markdown('<p class="result-pred">Terinfeksi</p>', unsafe_allow_html=True)
                st.write("Accuracy: 98%")
            else:
                st.markdown('<div class="image-placeholder">Hasil deteksi akan muncul disini</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
