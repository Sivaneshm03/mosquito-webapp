from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

# ✅ ROOT ROUTE (VERY IMPORTANT)
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

        # 🔍 Read image
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # 🔥 TEMP PREDICTION (replace with model later)
        prediction = "Aedes"
        confidence = "95%"

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


# ✅ RUN APP
if __name__ == '__main__':
    app.run(debug=True)
