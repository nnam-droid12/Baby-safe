from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load

api = FastAPI(title="Pre Term API")


class Patient(BaseModel):
    Year: int
    Race: int
    Total_Birth: int
    Events: int
    percent: float
    upper: float
    lower: float

model = load("model.pkl")


@api.get("/")
def index():
    return {"message": "Get predictions using /predict"}

@api.get("/predict")
def predict(patient: Patient):
    data = np.array(patient.dict().values()).reshape(-1,1)
    pred = model.predict(data)
    return {"prediction": str(pred)}