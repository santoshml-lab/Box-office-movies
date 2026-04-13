
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

model = pickle.load(open("model_v2.pkl", "rb"))

class InputData(BaseModel):
    title: str
    domestic_lifetime_gross: float
    domestic_percentage: float
    foreign_lifetime_gross: float
    foreign_percentage: float
    year: int

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": float(pred[0])}
