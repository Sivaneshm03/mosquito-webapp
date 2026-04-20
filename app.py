from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import os
import requests

app = Flask(__name__)

MODEL_URL = "https://github.com/Sivaneshm03/mosquito-webapp/releases/download/v1/swin_model_final.h5"
MODEL_PATH = "model.h5"

CLASS_NAMES = ["Aedes", "Culex"]

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model...")
        headers = {"Accept": "application/octet-stream"}
        with requests.get(MODEL_URL, stream=True, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists, skipping download")

def load_model():
    global model
    download_model()

    file_size = os.path.getsize(MODEL_PATH)
    print(f"📦 Model file size: {file_size / (1024 * 1024):.2f} MB")

    if file_size < 10_000_000:
        os.remove(MODEL_PATH)
        raise Exception("❌ Model file too small — corrupted. Deleted, please restart.")

    print("📦 Loading model...")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"KerasLayer": hub.KerasLayer}
    )
    print("✅ Model loaded successfully!")
    return model

model = load_model()

@app.route("/")
def home():
    return "🚀 Mosquito Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Use key='file' in form-data"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename. Please upload a valid image."}), 400

        try:
            img = Image.open(file).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image file. Please upload a JPEG or PNG."}), 400

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        pred = model.predict(img_array)
        print(f"🔍 Raw prediction output: {pred}")

        if pred.shape[-1] == 1:
            raw_conf = float(pred[0][0])
            if raw_conf > 0.5:
                prediction = "Aedes"
                confidence = raw_conf
            else:
                prediction = "Culex"
                confidence = 1 - raw_conf
        else:
            class_index = int(np.argmax(pred[0]))
            confidence = float(pred[0][class_index])
            prediction = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else f"Class_{class_index}"

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "raw_output": pred.tolist()
        })

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__
