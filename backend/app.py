from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained models
diabetes_model = joblib.load("../models/diabetes_model.pkl")
symptom_model = joblib.load("../models/symptom_model.pkl")
headache_model = joblib.load("../models/headache_model.pkl")

@app.route("/")
def home():
    return "AI Healthcare System Running"

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    data = request.json["features"]
    prediction = diabetes_model.predict([data])[0]

    result = "High Risk of Diabetes" if prediction == 1 else "Low Risk"
    return jsonify({"result": result})

@app.route("/predict_symptom", methods=["POST"])
def predict_symptom():
    data = request.json["features"]
    prediction = symptom_model.predict([data])[0]

    return jsonify({"disease": prediction})

@app.route("/predict_headache", methods=["POST"])
def predict_headache():
    data = request.json["features"]
    prediction = headache_model.predict([data])[0]

    return jsonify({"headache_type": prediction})

if __name__ == "__main__":
    app.run(debug=True)