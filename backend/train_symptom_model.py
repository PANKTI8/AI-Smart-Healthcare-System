import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("../datasets/symptoms.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "../models/symptom_model.pkl")

print("Symptom model trained!")