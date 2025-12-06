from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import numpy as np
from model_pipeline import (
    prepare_data, 
    split_and_scale_data, 
    train_and_log, 
    save_model, 
    load_model
)

app = FastAPI()
templates = Jinja2Templates(directory=".")

# Cache the model and scaler, load them at startup
model, scaler = load_model()

# Base features list (without engineered features)
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page with prediction form"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": FEATURES,
        "values": {},
        "prediction": None,
        "message": None
    })

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ph: float = Form(...),
    Hardness: float = Form(...),
    Solids: float = Form(...),
    Chloramines: float = Form(...),
    Sulfate: float = Form(...),
    Conductivity: float = Form(...),
    Organic_carbon: float = Form(...),
    Trihalomethanes: float = Form(...),
    Turbidity: float = Form(...)
):
    """
    Make prediction on user-provided water quality parameters
    
    Applies the same feature engineering as training pipeline
    """
    values = {
        "ph": ph,
        "Hardness": Hardness,
        "Solids": Solids,
        "Chloramines": Chloramines,
        "Sulfate": Sulfate,
        "Conductivity": Conductivity,
        "Organic_carbon": Organic_carbon,
        "Trihalomethanes": Trihalomethanes,
        "Turbidity": Turbidity,
    }
    
    try:
        # Create feature vector in correct order
        features = [values[f] for f in FEATURES]
        
        # Add engineered feature: hardness_solids_ratio
        # This must match what was done during training
        hardness_solids_ratio = Hardness / (Solids + 1)
        features.append(hardness_solids_ratio)
        
        # Scale features using the loaded scaler
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction_raw = model.predict(features_scaled)[0]
        prediction = "✅ Potable" if prediction_raw == 1 else "❌ Not Potable"
        
    except Exception as e:
        prediction = f"⚠️ Error: {str(e)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": FEATURES,
        "values": values,
        "prediction": prediction,
        "message": None
    })

@app.post("/retrain", response_class=HTMLResponse)
async def retrain(
    request: Request,
    file: UploadFile = File(...),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    missing_strategy: str = Form("median"),
    engineer_features: bool = Form(True),
    outlier_method: str = Form("IQR"),
    outlier_threshold: float = Form(1.5),
    var_smoothing: float = Form(1e-9)
):
    """
    Retrain the model with uploaded dataset and specified hyperparameters
    
    Supports configurable preprocessing and model hyperparameters
    """
    temp_path = f"temp_{file.filename}"
    
    try:
        # Save uploaded file temporarily
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Prepare data with specified preprocessing parameters
        df, engineered_features = prepare_data(
            temp_path,
            missing_value_strategy=missing_strategy,
            engineer_features=engineer_features,
            outlier_method=outlier_method if outlier_method != "none" else None,
            outlier_threshold=outlier_threshold
        )
        
        # Split and scale data
        X_train, X_test, y_train, y_test, new_scaler = split_and_scale_data(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Train model with MLflow logging
        new_model, run_id = train_and_log(
            X_train, y_train, X_test, y_test, new_scaler,
            dataset_path=file.filename,
            test_size=test_size,
            random_state=random_state,
            experiment_name="water_potability",
            engineered_features=engineered_features,
            outlier_method=outlier_method if outlier_method != "none" else None,
            outlier_threshold=outlier_threshold,
            missing_value_strategy=missing_strategy,
            var_smoothing=var_smoothing,
            run_source="api"
        )
        
        # Save the new model
        save_model(new_model, new_scaler)

        # Update global model and scaler
        global model, scaler
        model = new_model
        scaler = new_scaler

        message = f"✅ Model retrained successfully from {file.filename}! MLflow run: {run_id}"
        
    except Exception as e:
        message = f"❌ Retrain failed: {str(e)}"
        
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": FEATURES,
        "values": {},
        "prediction": None,
        "message": message
    })

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/model-info")
async def model_info():
    """Return information about the currently loaded model"""
    try:
        return {
            "model_type": type(model).__name__,
            "scaler_type": type(scaler).__name__,
            "n_features": scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else "unknown",
            "feature_names": FEATURES + ["hardness_solids_ratio"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))