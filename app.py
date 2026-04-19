from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# ✅ Load trained model
model = tf.keras.models.load_model("mosquito_model.h5")

# ✅ ROOT ROUTE
@app.route('/')
def home():
    return "Mosquito API is running successfully 🚀"

# ✅ PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    try:
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

        # 🔍 Classification logic + correct confidence
        if raw_conf > 0.5:
            prediction = "Aedes"
            final_conf = raw_conf
        else:
            prediction = "Culex"
            final_conf = 1 - raw_conf

        confidence = round(final_conf * 100, 2)  # ✅ percentage out of 100

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


# ✅ RUN APP (Render compatible)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
