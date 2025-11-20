from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from model_pipeline import prepare_data, split_and_scale_data, train_model, save_model, load_model

app = FastAPI()
templates = Jinja2Templates(directory=".")

# Cache the model and scaler, load em at startup
model, scaler = load_model()

# Features list
FEATURES = ["ph","hardness","solids","chloramines","sulfate","conductivity","organic_carbon","trihalomethanes","turbidity"]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
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
    hardness: float = Form(...),
    solids: float = Form(...),
    chloramines: float = Form(...),
    sulfate: float = Form(...),
    conductivity: float = Form(...),
    organic_carbon: float = Form(...),
    trihalomethanes: float = Form(...),
    turbidity: float = Form(...)
):
    values = {
        "ph": ph,
        "hardness": hardness,
        "solids": solids,
        "chloramines": chloramines,
        "sulfate": sulfate,
        "conductivity": conductivity,
        "organic_carbon": organic_carbon,
        "trihalomethanes": trihalomethanes,
        "turbidity": turbidity,
    }
    try:
        features = [values[f] for f in FEATURES]
        features.append(hardness / (solids + 1)) # Add in `hardness_solids_ratio`
        prediction_raw = model.predict([features])[0]
        prediction = "Potable" if prediction_raw == 1 else "Not Potable"
    except Exception as e:
        prediction = f"Error: {str(e)}"

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
    random_state: int = Form(42)
):
    temp_path = f"temp_{file.filename}"
    try:
        contents = file.file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        df = prepare_data(temp_path)
        X_train, X_test, y_train, y_test, new_scaler = split_and_scale_data(df, test_size, random_state)
        new_model = train_model(X_train, y_train)
        save_model(new_model, new_scaler)

        global model, scaler
        model = new_model
        scaler = new_scaler

        message = f"Model retrained and saved successfully from {file.filename}!"
    except Exception as e:
        message = f"Retrain failed: {str(e)}"
    finally:
        file.file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "features": FEATURES,
        "values": {},
        "prediction": None,
        "message": message
    })

