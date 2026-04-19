from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

app = Flask(__name__)

# =========================
# DOWNLOAD MODEL FROM DRIVE
# =========================
url = "https://drive.google.com/uc?id=1Z-pxIwlP1Bf0MUoKmkDSrE-gSUbIvMQi"
model_path = "model.h5"

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(url, model_path, quiet=False)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(model_path)

classes = ["Aedes", "Culex"]

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "API Running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file)

    processed = preprocess(image)
    prediction = model.predict(processed)

    confidence = float(np.max(prediction))
    label = classes[np.argmax(prediction)]

    if confidence < 0.7:
        label = "Unknown"

    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2)
    })

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)