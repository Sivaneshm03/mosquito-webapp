from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import os
import requests
import threading

app = Flask(__name__)

MODEL_URL = "https://huggingface.co/Sivaneshm03/mosquito-classifier/resolve/main/swin_model_final.h5"
MODEL_PATH = "model.h5"
CLASS_NAMES = ["Aedes", "Culex"]
model = None
model_ready = False
model_error = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        headers = {"Accept": "application/octet-stream"}
        with requests.get(MODEL_URL, stream=True, headers=headers, allow_redirects=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        print("Model downloaded")
    else:
        print("Model exists")

def load_model_background():
    global model, model_ready, model_error
    try:
        download_model()
        size = os.path.getsize(MODEL_PATH)
        if size < 10000000:
            os.remove(MODEL_PATH)
            raise Exception("Model file corrupted or incomplete")
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"KerasLayer": hub.KerasLayer}
        )
        model_ready = True
        print("Model loaded successfully")
    except Exception as e:
        model_error = str(e)
        print(f"Model loading failed: {e}")

# Start loading in background AFTER Flask starts
threading.Thread(target=load_model_background, daemon=True).start()

@app.route("/")
def home():
    if model_ready:
        return "Mosquito API running — Model ready ✅"
    elif model_error:
        return f"Mosquito API running — Model error ❌: {model_error}"
    else:
        return "Mosquito API running — Model loading... ⏳"

@app.route("/predict", methods=["POST"])
def predict():
    if not model_ready:
        if model_error:
            return jsonify({"error": f"Model failed to load: {model_error}"}), 503
        return jsonify({"error": "Model is still loading, please try again in a moment"}), 503

    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        try:
            img = Image.open(file).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid image"}), 400

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        pred = model.predict(img_array)

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
            prediction = CLASS_NAMES[class_index] if class_index < len(CLASS_NAMES) else "Unknown"

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "raw_output": pred.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
