from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?id=1Z-pxIwlP1Bf0MUoKmkDSrE-gSUbIvMQi"
MODEL_PATH = "mosquito_model.h5"

model = None


# ✅ ROBUST DOWNLOAD (WITH VERIFICATION)
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # 🔥 CHECK FILE SIZE (IMPORTANT)
    file_size = os.path.getsize(MODEL_PATH)

    if file_size < 5 * 1024 * 1024:   # < 5MB → corrupted
        print("❌ Model file corrupted. Re-downloading...")
        os.remove(MODEL_PATH)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    print("✅ Model ready")


# ✅ LOAD MODEL SAFELY
def load_model():
    global model
    if model is None:
        download_model()
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded")
        except Exception:
            print("❌ Error loading model. Re-downloading...")
            os.remove(MODEL_PATH)
            download_model()
            model = tf.keras.models.load_model(MODEL_PATH)


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

        # ✅ IMAGE PREPROCESS
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # ✅ PREDICTION
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
