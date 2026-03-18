from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load the model once when the server starts
model = joblib.load("model.pkl")

class PredictRequest(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "Champion model inference API is running."}

@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.features])
    return {"prediction": int(prediction[0])}
