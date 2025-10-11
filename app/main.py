from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from app.schemas import TextRequest

app = FastAPI(title="Wildfire Risk System")

# Load model and preprocessor once
pipeline = PredictionPipeline()

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def build_features(data: TextRequest) -> pd.DataFrame:
    """Return DataFrame with all features including computed ones, excluding datetime"""
    features = {
        # Original fields
        "latitude": data.latitude,
        "longitude": data.longitude,
        "pr": data.pr,
        "rmax": data.rmax,
        "rmin": data.rmin,
        "sph": data.sph,
        "srad": data.srad,
        "tmmn": data.tmmn,
        "tmmx": data.tmmx,
        "vs": data.vs,
        "bi": data.bi,
        "fm100": data.fm100,
        "fm1000": data.fm1000,
        "erc": data.erc,
        "etr": data.etr,
        "pet": data.pet,
        "vpd": data.vpd,
        # Computed fields
        "year": data.year,
        "month": data.month,
        "day": data.day,
        "dayofweek": data.dayofweek,
        "quarter": data.quarter,
        "dayofyear": data.dayofyear,
        "weekofyear": data.weekofyear,
        "is_weekend": data.is_weekend,
        "trange": data.trange,
        "rrange": data.rrange,
        "fm_ratio": data.fm_ratio,
        "pet_minus_etr": data.pet_minus_etr,
        "trange_srad": data.trange_srad,
        "vpd_tmmx": data.vpd_tmmx,
        "fm_wind": data.fm_wind,
        "pr_rmax_ratio": data.pr_rmax_ratio,
        "fm_diff": data.fm_diff
    }
    return pd.DataFrame([features])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the HTML form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_wildfire(data: TextRequest):
    """Predict wildfire risk from input data"""
    try:
        logging.info(f"Received prediction request: {data.dict()}")
        df = build_features(data)
        pred = pipeline.predict(df)[0]
        label = "🔥 High Wildfire Risk" if pred == 1 else "🌿 Low Wildfire Risk"
        return {"prediction": label, "numeric_prediction": int(pred)}
    except CustomException as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")
