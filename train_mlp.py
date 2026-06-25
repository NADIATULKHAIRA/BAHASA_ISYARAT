import os
import cv2
import numpy as np
import mediapipe as mp
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler    
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ==========================
# MediaPipe
# ==========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

dataset_path = "data"

X = []
y = []

print("Memulai membaca dataset...")

for category in ["Huruf", "Angka", "Kalimat"]:

    category_path = os.path.join(dataset_path, category)

    if not os.path.exists(category_path):
        continue

    for label in os.listdir(category_path):

        folder = os.path.join(category_path, label)

        if not os.path.isdir(folder):
            continue

        print("Proses:", label)

        for file in os.listdir(folder):

            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Mendeteksi titik tangan pada gambar
            result = hands.process(rgb)

            data = []

            # ==========================
            # AMBIL MAKS 2 TANGAN
            # ==========================

            if result.multi_hand_landmarks:

                hands_data = result.multi_hand_landmarks[:2]

                for hand in hands_data:

                    temp = []

                    # Mengambil 21 titik landmark tangan
                    for lm in hand.landmark:
                        temp.extend([lm.x, lm.y])
                        
                    # Melakukan normalisasi posisi tangan
                    base_x = temp[0]
                    base_y = temp[1]

                    
                    for i in range(0, len(temp), 2):
                        temp[i] -= base_x
                        temp[i + 1] -= base_y

                    data.extend(temp)

            # ==========================
            #  Membuat jumlah fitur menjadi 84
            # ==========================

            while len(data) < 84:
                data.append(0)

            data = data[:84]

            X.append(data)
            y.append(label)

print("\nJumlah data:", len(X))

if len(X) == 0:
    print("Dataset kosong")
    exit()

X = np.array(X)
y = np.array(y)

# Membagi dataset:
# 80% untuk training dan 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(128, 64),  # Jumlah neuron pada hidden layer
        max_iter=1500,
        random_state=42
    )
)

print("\nTraining dimulai...")

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print(f"Akurasi: {acc*100:.2f}%")

joblib.dump(model, "model.pkl")

print("\nModel berhasil disimpan")