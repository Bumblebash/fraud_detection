from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Re-save the model properly
joblib.dump(model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Serve HTML UI
@app.route("/")
def home():
    return render_template("index.html")

# API endpoint for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
