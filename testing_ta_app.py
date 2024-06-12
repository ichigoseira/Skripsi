import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Memuat model
model = tf.keras.models.load_model('model-percobaan3-MobileNetV2 (3).h5')
# Fungsi untuk memprediksi kelas gambar
def predict_image(image, model):
    # Pastikan gambar dalam mode RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Ubah ukuran gambar sesuai dengan input model Anda
    size = (224, 224)  # Sesuaikan ukuran input model
    image = ImageOps.fit(image, size, Image.LANCZOS)  # Ganti ANTIALIAS dengan LANCZOS
    
    # Normalisasi gambar
    image_array = np.asarray(image)
    image_array = image_array / 255.0

    # Tambahkan batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    # Prediksi gambar
    prediction = model.predict(image_array)
    return np.argmax(prediction), prediction

# Judul aplikasi
st.title("Testing : Deteksi Gerakan Tangan Bahasa Isyarat SIBI")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Gambar.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Prediksi gambar
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # 24 gerakan tangan bahasa isyarat
    label, prediction = predict_image(image, model)
    st.write(f"Prediction: {class_names[label]}")
    
    # Membuat grafik batang dengan nama kelas
    prediction_dict = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
    st.bar_chart(prediction_dict)