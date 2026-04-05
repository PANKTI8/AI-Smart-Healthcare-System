from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load trained models

diabetes_model = joblib.load("../models/diabetes_model.pkl")
symptom_model = joblib.load("../models/symptom_model.pkl")
headache_model = joblib.load("../models/headache_model.pkl")

HISTORY_FILE = "history.json"

def save_history(data):
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except:
        history = []

    history.append(data)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

@app.route("/")
def home():
    return "AI Healthcare System Running"

@app.route("/predict_diabetes", methods=["POST"])
def predict_diabetes():
    data = request.json["features"]
    prediction = diabetes_model.predict([data])[0]
    
    result = "High Risk of Diabetes" if prediction == 1 else "Low Risk"
    
    save_history({
        "type": "diabetes",
        "input": data,
        "result": result,
        "time": str(datetime.now())
    })
    
    return jsonify({"result": result})


@app.route("/predict_headache", methods=["POST"])
def predict_headache():
    data = request.json["features"]
    prediction = headache_model.predict([data])[0]
    
    save_history({
        "type": "headache",
        "input": data,
        "result": str(prediction),
        "time": str(datetime.now())
    })
    
    return jsonify({"headache_type": str(prediction)})

import os

UPLOAD_FOLDER = "videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    video = request.files['video']   # get video
    filename = video.filename        # get filename
    
    video.save("videos/" + filename) # save video

    save_history({
        "type": "video_feedback",
        "file": filename,
        "time": str(datetime.now())
    })

    return jsonify({"message": "Video uploaded successfully"})
    
    app.run(host="0.0.0.0", port=10000)