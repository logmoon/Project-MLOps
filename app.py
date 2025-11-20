import numpy as np
from model_pipeline import (
    load_model,
)
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class PredictionFeatures(BaseModel):
    ph: float
    hardness: float
    solids: float
    chloramines: float
    sulfate: float
    conductivity: float
    organic_carbon: float
    trihalomethanes: float
    turbidity: float

# Cache the model
model, scaler = load_model() # From default path

@app.post('/predict')
def predict(input: PredictionFeatures):
    try:
        ratio = input.hardness / (input.solids + 1)
        features = [
            input.ph,
            input.hardness,
            input.solids,
            input.chloramines,
            input.sulfate,
            input.conductivity,
            input.organic_carbon,
            input.trihalomethanes,
            input.turbidity,
            ratio
        ]

        # Convert to 2D array
        X = np.array([features])

        # Scale
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)
        return { "message": "Success", "prediction": int(prediction[0]) }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
