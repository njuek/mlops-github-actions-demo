from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Load the model once when the server starts
model_path = os.getenv("MODEL_PATH", "outputs/model.pkl")
model = joblib.load(model_path)

class PredictRequest(BaseModel):
    features: list

@app.get("/")
def root():
    return {"message": "Champion model inference API is running."}

@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model.predict([request.features])
    return {"prediction": int(prediction[0])}
