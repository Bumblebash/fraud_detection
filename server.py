from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("fraud_detection_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([np.array(data)])
    return jsonify({"fraudulent": bool(prediction[0])})

def home():
    return "Model is running!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
