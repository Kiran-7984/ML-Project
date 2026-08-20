from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI(title="IRI Deterioration Prediction API")

pipeline = PredictionPipeline()

DATA = pd.read_csv("data/processed/iri_deterioration.csv")
MIN_AGE = float(DATA["PAVEMENT_AGE"].min())
MAX_AGE = float(DATA["PAVEMENT_AGE"].max())

class PredictionInput(BaseModel):
    initial_iri: float
    aadtt_all_trucks_trend: int
    total_ann_precip: float
    mean_ann_temp_avg: float
    freeze_index_yr: int
    freeze_thaw_yr: int
    sn: float


@app.get("/")
def home():
    return {"message": "IRI Deterioration Prediction API is running"}


@app.post("/predict/{pavement_age}")
def predict(
    pavement_age: float = Path(
        ...,
        description="Pavement age in years",
        ge=MIN_AGE,
        le=MAX_AGE
    ),
    input_data: PredictionInput = None
):

    if pavement_age < MIN_AGE or pavement_age > MAX_AGE:
        raise HTTPException(
            status_code=400,
            detail=f"pavement_age must be between {MIN_AGE:.1f} "
                   f"and {MAX_AGE:.1f} years (training data range). "
                   f"Prediction outside this range is unreliable."
        )

    record = pd.DataFrame([{
        "INITIAL_IRI": input_data.initial_iri,
        "AADTT_ALL_TRUCKS_TREND": input_data.aadtt_all_trucks_trend,
        "TOTAL_ANN_PRECIP": input_data.total_ann_precip,
        "MEAN_ANN_TEMP_AVG": input_data.mean_ann_temp_avg,
        "FREEZE_INDEX_YR": input_data.freeze_index_yr,
        "FREEZE_THAW_YR": input_data.freeze_thaw_yr,
        "PAVEMENT_AGE": pavement_age,
        "SN": input_data.sn
    }])

    prediction = pipeline.predict(record)

    return {
        "pavement_age": pavement_age,
        "predicted_mri": round(float(prediction[0]), 4)
    }