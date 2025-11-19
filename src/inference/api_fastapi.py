"""
LedgerX – FastAPI Inference Deployment
======================================

Run with:
    uvicorn src.inference.api_fastapi:app --reload --port 8000

Provides:
    POST /predict
"""

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .inference_service import predict_invoice

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ledgerx_fastapi")

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="LedgerX Invoice Quality & Failure Prediction API",
    description="AI-based Invoice Quality Screening and Failure Risk Detection",
    version="1.0.0",
)

# -------------------------------------------------------------------
# INPUT SCHEMA
# -------------------------------------------------------------------
class InvoiceFeatures(BaseModel):
    blur_score: float
    contrast_score: float
    ocr_confidence: float
    file_size_kb: float

    vendor_name: str
    vendor_freq: float

    total_amount: float
    invoice_number: str
    invoice_date: str
    currency: str

# -------------------------------------------------------------------
# ENDPOINT
# -------------------------------------------------------------------
@app.post("/predict")
def predict(features: InvoiceFeatures):
    logger.info("===============================================")
    logger.info("[API] Incoming /predict request")
    logger.info("===============================================")

    try:
        result = predict_invoice(features.dict())
        return {"status": "ok", "result": result}

    except Exception as e:
        logger.exception("[API] Error during inference")
        raise HTTPException(status_code=400, detail=str(e))
