from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests

app = Flask(__name__)

# 🔗 YOUR MODEL DOWNLOAD LINK (PUT YOUR LINK HERE)
MODEL_URL = "https://your-model-link.com/mosquito_model.h5"
MODEL_PATH = "mosquito_model.h5"

model = None

# ✅ Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded")

# ✅ Load model safely
def load_model_safe():
    global model
    try:
        if model is None:
            download_model()
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Model loading failed:", e)

# ✅ ROOT ROUTE
@app.route('/')
def home():
    return "Mosquito API is running successfully 🚀"

# ✅ PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    try:
        global model

        # 🔥 Ensure model is loaded
        if model is None:
            load_model_safe()

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # 🔍 Check file
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # 🔍 Process image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # 🔥 MODEL PREDICTION
        pred = model.predict(img)
        raw_conf = float(pred[0][0])

        # 🔍 Classification logic
        if raw_conf > 0.5:
            prediction = "Aedes"
            final_conf = raw_conf
        else:
            prediction = "Culex"
            final_conf = 1 - raw_conf

        confidence = round(final_conf * 100, 2)

        # ✅ RESULT DATA
        if prediction == "Aedes":
            data = {
                "prediction": prediction,
                "confidence": confidence,
                "species": "Aedes aegypti",
                "family": "Culicidae",
                "biting_time": "Daytime",
                "habitat": "Clean stagnant water",
                "appearance": "Black with white stripes",
                "disease": "Dengue, Zika, Chikungunya"
            }
        else:
            data = {
                "prediction": prediction,
                "confidence": confidence,
                "species": "Culex",
                "family": "Culicidae",
                "biting_time": "Night",
                "habitat": "Dirty stagnant water",
                "appearance": "Brownish color",
                "disease": "Filariasis, West Nile Virus"
            }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ Render will use gunicorn, no need for app.run
