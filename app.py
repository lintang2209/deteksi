import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown

# ====================
# CSS Styling baru
# ====================
st.markdown("""
    <style>
        /* Background halaman */
        .main {
            background-color: #F5F9F6;
        }

        /* Daun dekorasi */
        body::before {
            content: "";
            position: absolute;
            top: -30px;
            left: -50px;
            width: 220px;
            height: 220px;
            background: url("https://i.ibb.co/Lh2W1tV/leaf-top.png") no-repeat;
            background-size: contain;
            z-index: -1;
        }
        body::after {
            content: "";
            position: absolute;
            bottom: -30px;
            right: -50px;
            width: 220px;
            height: 220px;
            background: url("https://i.ibb.co/Z2ShYDC/leaf-bottom.png") no-repeat;
            background-size: contain;
            z-index: -1;
        }

        /* Landing Page */
        .center {
            text-align: center;
            padding-top: 120px;
        }
        .title {
            font-size: 38px;
            font-weight: 700;
            color: #2E7D32;
        }
        .subtitle {
            font-size: 17px;
            font-style: italic;
            color: #4E6E58;
            margin-top: -5px;
        }

        /* Input uploader */
        .uploadedFile {
            color: #2E3A59 !important;  /* teks uploader jadi gelap */
            font-weight: 500 !important;
        }
        .stFileUploader label {
            color: #2E3A59 !important;  /* label file uploader lebih gelap */
            font-weight: 600 !important;
        }

        /* Button */
        .stButton>button {
            background-color: #66BB6A;
            color: white;
            border-radius: 12px;
            padding: 10px 25px;
            font-weight: 600;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #388E3C;
            color: white;
        }

        /* Card hasil deteksi */
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        .card h3 {
            color: #2E7D32;
            font-weight: 600;
        }
        .detected {
            color: red;
            font-weight: 600;
            margin-top: 10px;
        }
        .accuracy {
            font-size: 14px;
            color: #2E3A59; /* ganti dari abu ke gelap */
        }
    </style>
""", unsafe_allow_html=True

# ====================
# Navigasi sederhana
# ====================
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    # Landing page
    st.markdown("""
    <div class="center">
        <p class="title">Ayo Cek Tanamanmu!</p>
        <p class="subtitle">Kenali soybean rust sejak dini<br>untuk hasil panen yang lebih baik 🌱</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🌿 Cek Disini"):
        st.session_state.page = "deteksi"
        st.rerun()

# ====================
# Halaman Deteksi (CNN & YOLO)
# ====================
elif st.session_state.page == "deteksi":

    st.markdown("<h2 style='color:#2E7D32;'>Perbandingan Deteksi Penyakit Soybean Rust (CNN vs YOLO)</h2>", unsafe_allow_html=True)
    st.write("Unggah satu gambar daun kedelai untuk melihat hasil deteksi dari kedua model secara bersamaan.")

    @st.cache_resource
    def load_cnn_model():
        GOOGLE_DRIVE_FILE_ID = "1sZegfJRnGu2tr00qtinTAeZeLaQnllrO"
        MODEL_PATH = "models/cnn.h5"
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        if not os.path.exists(MODEL_PATH):
            st.info("Mengunduh model dari Google Drive...")
            try:
                gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model dari Google Drive: {e}")
                return None
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model CNN: {e}")
            return None

    @st.cache_resource
    def load_yolo_model():
        MODEL_PATH = "models/best.pt"
        if not os.path.exists(MODEL_PATH):
            st.error(f"File model YOLOv8 tidak ditemukan: {MODEL_PATH}")
            return None
        try:
            model = YOLO(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Gagal memuat model YOLOv8: {e}")
            return None

    # Muat model
    cnn_model = load_cnn_model()
    yolo_model = load_yolo_model()

    if cnn_model is None or yolo_model is None:
        st.stop()

    uploaded_file = st.file_uploader("📤 Pilih gambar daun...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📸 Gambar yang diunggah", use_column_width=True)
        st.write("---")

        col1, col2 = st.columns(2)

        # ==== CNN ====
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Hasil Analisis CNN")
            try:
                img_resized = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                prediction = cnn_model.predict(img_array)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                
                class_names = ["Sehat", "Soybean Rust"]
                predicted_class_name = class_names[class_id]

                st.write(f"### Prediksi: **{predicted_class_name}**")
                st.markdown(f"<div class='accuracy'>Confidence: {confidence:.2f}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan pada model CNN: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        # ==== YOLOv8 ====
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Hasil Analisis YOLOv8")
            try:
                results = yolo_model(image)
                results_img = results[0].plot()
                st.image(results_img, caption="📊 Hasil Deteksi YOLOv8", use_column_width=True)

                if len(results[0].boxes) > 0:
                    st.markdown("<div class='detected'>Terdeteksi Soybean Rust</div>", unsafe_allow_html=True)
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        st.markdown(f"<div class='accuracy'>Confidence: {conf:.2f}</div>", unsafe_allow_html=True)
                else:
                    st.write("✅ Tidak ditemukan penyakit Soybean Rust.")
            except Exception as e:
                st.error(f"Terjadi kesalahan pada model YOLOv8: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
