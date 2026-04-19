from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1Z-pxIwlP1Bf0MUoKmkDSrE-gSUbIvMQi"
MODEL_PATH = "mosquito_model.h5"

model = None


# ✅ FIXED DOWNLOAD FUNCTION (Google Drive compatible)
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")

        session = requests.Session()
        response = session.get(MODEL_URL, stream=True)

        # Handle Google Drive large file confirmation
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': MODEL_URL.split("id=")[-1], 'confirm': value}
                response = session.get(MODEL_URL, params=params, stream=True)
                break

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

        print("✅ Model downloaded correctly")


# ✅ LOAD MODEL WHEN NEEDED
def load_model():
    global model
    if model is None:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")


# ✅ HOME ROUTE
@app.route('/')
def home():
    return "Mosquito API is running successfully 🚀"


# ✅ PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model

        if model is None:
            load_model()

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        raw_conf = float(pred[0][0])

        if raw_conf > 0.5:
            prediction = "Aedes"
            final_conf = raw_conf
        else:
            prediction = "Culex"
            final_conf = 1 - raw_conf

        confidence = round(final_conf * 100, 2)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ FOR LOCAL RUN ONLY
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
