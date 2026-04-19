from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub   # 🔥 IMPORTANT
import os
import requests

app = Flask(__name__)

MODEL_URL = "https://github.com/Sivaneshm03/mosquito-webapp/releases/download/v1/swin_model_final.keras"
MODEL_PATH = "model.keras"

model = None


# 🔥 DOWNLOAD MODEL
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")

        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print("✅ Model downloaded")


# 🔥 LOAD MODEL (FIXED FOR SWIN + TF HUB)
def load_model():
    global model

    if model is None:
        download_model()

        # ✅ Check file size (avoid corrupted file)
        if os.path.getsize(MODEL_PATH) < 10_000_000:
            raise Exception("❌ Model file corrupted or not downloaded properly")

        print("📦 Loading model...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'KerasLayer': hub.KerasLayer},  # 🔥 FIX
            compile=False                                  # 🔥 FIX
        )

        print("✅ Model loaded successfully")


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
            confidence = raw_conf
        else:
            prediction = "Culex"
            confidence = 1 - raw_conf

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
