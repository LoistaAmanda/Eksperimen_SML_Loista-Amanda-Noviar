from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time

from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ======================
# LOAD MODEL
# ======================
model = joblib.load("model.pkl")  

app = FastAPI(title="IMDB Sentiment Inference")

# ======================
# PROMETHEUS METRICS
# ======================
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)

PREDICTION_COUNT = Counter(
    "prediction_total",
    "Total predictions by class",
    ["label"]
)

# ======================
# SCHEMA
# ======================
class TextInput(BaseModel):
    text: str

# ======================
# ENDPOINTS
# ======================
@app.post("/predict")
def predict(data: TextInput):
    start = time.time()
    REQUEST_COUNT.inc()

    pred = model.predict([data.text])[0]
    PREDICTION_COUNT.labels(label=str(pred)).inc()

    latency = time.time() - start
    REQUEST_LATENCY.observe(latency)

    return {
        "prediction": int(pred),
        "latency": latency
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
