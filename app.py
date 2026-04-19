from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = "Aedes"  # (later connect your model)

    if prediction == "Aedes":
        data = {
            "species": "Aedes",
            "family": "Culicidae",
            "biting_time": "Daytime",
            "habitat": "Clean stagnant water",
            "appearance": "Black with white stripes",
            "disease": "Dengue, Zika, Chikungunya"
        }
    else:
        data = {
            "species": "Culex",
            "family": "Culicidae",
            "biting_time": "Night",
            "habitat": "Dirty stagnant water",
            "appearance": "Brownish color",
            "disease": "Filariasis, West Nile Virus"
        }

    return jsonify(data)
