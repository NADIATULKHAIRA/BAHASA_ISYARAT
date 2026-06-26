# **Sistem Pendeteksian Bahasa Isyarat BISINDO**

## **Deskripsi**

Sistem Pendeteksian Bahasa Isyarat BISINDO merupakan aplikasi berbasis Computer Vision dan Machine Learning yang dirancang untuk mengenali bahasa isyarat Indonesia secara real-time melalui kamera/webcam.
Sistem ini bekerja dengan mendeteksi tangan pengguna menggunakan MediaPipe Hands untuk mengekstraksi 21 titik landmark tangan. Data landmark tersebut kemudian diproses dan digunakan sebagai masukan ke model Multi-Layer Perceptron (MLP) untuk melakukan klasifikasi isyarat BISINDO secara langsung.
Hasil pengenalan bahasa isyarat ditampilkan secara real-time melalui antarmuka aplikasi, sehingga sistem ini dapat digunakan sebagai media pembelajaran dan pengenalan bahasa isyarat Indonesia berbasis teknologi.

----------------------------------------------------------------------------------------------------
## **Fitur**
- Deteksi tangan secara real-time menggunakan MediaPipe Hands.
- Ekstraksi 21 landmark tangan dengan koordinat x dan y pada setiap tangan.
- Normalisasi landmark tangan berdasarkan titik wrist sebagai titik acuan.
- Pembentukan 84 fitur dari maksimal dua tangan (2 × 42 fitur).
- Klasifikasi gesture bahasa isyarat BISINDO menggunakan Multi-Layer Perceptron (MLP).
- Sistem backend berbasis Flask untuk pemrosesan dan pengiriman data.
- Prediksi gesture bahasa isyarat secara real-time melalui webcam dan ditampilkan pada antarmuka web
------------------------------------------------------------------------------------------------------
## **Teknologi yang Digunakan** 
### Bahasa Pemrograman 
- Python 
### Machine Learning
- Multi-Layer Perceptron (MLP) untuk klasifikasi huruf BISINDO
### Computer Vision
- MediaPipe Hands untuk deteksi dan ekstraksi landmark tangan
- OpenCV untuk pengolahan citra dan akses webcam
### Framework & Library
- Flask sebagai backend dan antarmuka web
- NumPy untuk pengolahan data numerik
- Joblib untuk menyimpan dan memuat model machine learning
---------------------------------------------------------------------------------------------------------

## Struktur Proyek
```
BHS_ISYARAT/
├── data/
│   ├── Huruf/
│   ├── Angka/
│   └── Kalimat/
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── realtime.py
├── train_mlp.py
└── model.pkl
```
-----------------------------------------------------------------------------------------------------------
## 👩‍💻**Pengembang**
**Nadiatul Khaira**<br>
Mahasiswa Teknik Informatika<br>
Jurusan Teknologi Informasi dan Komputer<br>
Politeknik Negeri Lhokseumawe

