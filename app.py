from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown   # ✅ IMPORTANT

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?id=1Z-pxIwlP1Bf0MUoKmkDSrE-gSUbIvMQi"
MODEL_PATH = "mosquito_model.h5"

model = None


# ✅ SIMPLE & CORRECT DOWNLOAD
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded")


# ✅ LOAD MODEL
def load_model():
    global model
    if model is None:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded")


@app.route('/')
def home():
    return "Mosquito API is running successfully 🚀"


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
