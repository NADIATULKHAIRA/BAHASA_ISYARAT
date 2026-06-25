from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

latest_pred = "-"
latest_conf = 0


def process_frame(frame):
    global latest_pred, latest_conf

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    data = []
    hand_list = []

    # ==========================
    # CEK ADA TANGAN ATAU TIDAK
    # ==========================
    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            xs = [lm.x for lm in hand.landmark]
            avg_x = sum(xs) / len(xs)
            hand_list.append((avg_x, hand))

        hand_list.sort(key=lambda x: x[0])

        for _, hand in hand_list[:2]:

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            temp = []

            for lm in hand.landmark:
                temp.extend([lm.x, lm.y])

            base_x = temp[0]
            base_y = temp[1]

            for i in range(0, len(temp), 2):
                temp[i] -= base_x
                temp[i + 1] -= base_y

            data.extend(temp)

        # ==========================
        # PREDIKSI HANYA JIKA ADA TANGAN
        # ==========================
        while len(data) < 84:
            data.append(0)

        data = data[:84]
        data = np.array(data).reshape(1, -1)

        pred = model.predict(data)[0]
        conf = np.max(model.predict_proba(data)) * 100

        latest_pred = str(pred)
        latest_conf = round(conf, 2)

    else:
        # ==========================
        # TIDAK ADA TANGAN
        # ==========================
        latest_pred = "-"
        latest_conf = 0

    return frame


def generate():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def prediction():
    return jsonify({
        "result": latest_pred,
        "confidence": latest_conf
    })


if __name__ == "__main__":
    app.run(debug=True)