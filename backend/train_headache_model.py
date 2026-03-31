import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("../datasets/headache.csv")

X = data.drop("HeadacheType", axis=1)
y = data["HeadacheType"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "../models/headache_model.pkl")

print("Headache model trained!")