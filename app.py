import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from ultralytics import YOLO
import gdown

# ====================
# CSS Styling
# ====================
st.markdown("""
    <style>
        body {
            background-color: #f8faf8;
        }
        .upload-box {
            background-color: #f0f7f2;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .card h4 {
            color: #4b8b64;
            margin-bottom: 10px;
        }
        .detected {
            color: red;
            font-weight: bold;
        }
        .healthy {
            color: green;
            font-weight: bold;
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
    <div style="text-align:center; padding-top:120px;">
        <p style="font-size:36px; font-weight:700; color:#4b8b64;">ayo cek tanamanmu!</p>
        <p style="font-size:16px; font-style:italic; color:#7d7d7d;">
            kenali soybean rust sejak dini<br>untuk hasil panen yang lebih baik
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("cek disini"):
        st.session_state.page = "deteksi"
        st.rerun()


# ====================
# Halaman Deteksi
# ====================
elif st.session_state.page == "deteksi":

    st.markdown("<h2 style='color:#4b8b64;'>Deteksi Penyakit Daun Soybean 🌱</h2>", unsafe_allow_html=True)
    st.write("Unggah gambar daun kedelai untuk melihat hasil deteksi dari model CNN dan YOLO.")

    # Fungsi muat model CNN
    @st.cache_resource
    def load_cnn_model():
        GOOGLE_DRIVE_FILE_ID = "1JeSvrid8Zw2xurG-pciDrw6EdI2qXuAd"
        MODEL_PATH = "models/cnn.h5"
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        if not os.path.exists(MODEL_PATH):
            st.info("Mengunduh model CNN dari Google Drive...")
            try:
                gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}', MODEL_PATH, quiet=False)
                st.success("Model berhasil diunduh!")
            except Exception as e:
                st.error(f"Gagal mengunduh model CNN: {e}")
                return None
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model CNN: {e}")
            return None

    # Fungsi muat model YOLO
    @st.cache_resource
    def load_yolo_model():
        MODEL_PATH = "models/best.pt"
        if not os.path.exists(MODEL_PATH):
            st.error("File model YOLOv8 tidak ditemukan.")
            return None
        try:
            return YOLO(MODEL_PATH)
        except Exception as e:
            st.error(f"Gagal memuat model YOLOv8: {e}")
            return None

    cnn_model = load_cnn_model()
    yolo_model = load_yolo_model()
    if cnn_model is None or yolo_model is None:
        st.stop()

    # Layout 2 kolom
    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown("<div class='upload-box'><h3>Upload Disini!</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Pilih gambar daun...", type=["jpg","png","jpeg"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            # Card CNN
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Hasil Deteksi Grad-CAM CNN")
            st.image(image, caption="Gambar Asli", use_column_width=True)

            try:
                img_resized = image.resize((224, 224))
                img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
                prediction = cnn_model.predict(img_array)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                class_names = ["Daun Sehat", "Soybean Rust"]
                predicted_class = class_names[class_id]

                if predicted_class == "Soybean Rust":
                    st.markdown(f"<p class='detected'>Terinfeksi</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='healthy'>Sehat</p>", unsafe_allow_html=True)
                st.write(f"Accuracy: {confidence*100:.2f}%")
            except Exception as e:
                st.error(f"Error CNN: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Card YOLO
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Hasil Deteksi Bounding Box YOLO")
            try:
                results = yolo_model(image)
                results_img = results[0].plot()
                st.image(results_img, caption="Deteksi YOLOv8", use_column_width=True)

                if len(results[0].boxes) > 0:
                    st.markdown(f"<p class='detected'>Terinfeksi</p>", unsafe_allow_html=True)
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        st.write(f"Confidence: {conf*100:.2f}%")
                else:
                    st.markdown(f"<p class='healthy'>Sehat</p>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error YOLO: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
