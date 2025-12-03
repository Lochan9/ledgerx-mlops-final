"""
LedgerX FastAPI Application - Cloud Run Optimized
Lightweight inference API without Airflow
"""

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import io
from PIL import Image
import pytesseract

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LedgerX Invoice Intelligence API",
    description="Production ML API for invoice quality and failure prediction",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load models
MODELS_DIR = Path("/app/models")
quality_model = joblib.load(MODELS_DIR / "quality_model.pkl")
failure_model = joblib.load(MODELS_DIR / "failure_model.pkl")

logger.info("✅ Models loaded successfully")

# Mock user database (replace with Cloud SQL)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin"
    }
}

# ============================================================================
# AUTHENTICATION
# ============================================================================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "LedgerX Invoice Intelligence API",
        "version": "2.0.0",
        "status": "operational",
        "models": {
            "quality": "CatBoost v10 (F1: 77.1%, AUC: 0.826)",
            "failure": "CatBoost v10 (F1: 70.9%, AUC: 0.790)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    """
    Upload and process invoice image
    Returns OCR extraction + ML predictions
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # OCR with Tesseract (lightweight)
        text = pytesseract.image_to_string(image)
        
        # Extract invoice data (simplified)
        invoice_data = extract_invoice_fields(text)
        
        # Engineer features for ML
        features = engineer_features(invoice_data)
        
        # Predict quality
        quality_pred = quality_model.predict([features])[0]
        quality_prob = quality_model.predict_proba([features])[0]
        
        # Predict failure
        failure_pred = failure_model.predict([features])[0]
        failure_prob = failure_model.predict_proba([features])[0]
        
        return {
            "status": "success",
            "filename": file.filename,
            "extracted_data": invoice_data,
            "quality": {
                "prediction": "good" if quality_pred == 0 else "bad",
                "probability": float(quality_prob[0]),
                "quality": "good" if quality_pred == 0 else "bad"
            },
            "failure": {
                "prediction": "safe" if failure_pred == 0 else "risk",
                "probability": float(failure_prob[1]),
                "risk": "low" if failure_pred == 0 else "high"
            },
            "ocr_method": "tesseract",
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/stats")
async def get_stats(current_user: str = Depends(get_current_user)):
    """Get system statistics"""
    return {
        "models": {
            "quality_f1": 0.771,
            "failure_f1": 0.709
        },
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_invoice_fields(ocr_text: str) -> Dict:
    """Extract fields from OCR text (simplified)"""
    import re
    
    # Simple regex extraction
    invoice_num = re.search(r'INV[- ]?(\d+)', ocr_text)
    amount = re.search(r'\$?(\d+\.?\d*)', ocr_text)
    
    return {
        "invoice_number": invoice_num.group(0) if invoice_num else "N/A",
        "vendor_name": "Unknown",
        "total_amount": float(amount.group(1)) if amount else 0.0,
        "currency": "USD",
        "invoice_date": datetime.now().strftime("%Y-%m-%d"),
        "vendor_freq": 0,
        "blur_score": 55.0,
        "contrast_score": 35.0,
        "ocr_confidence": 0.85
    }

def engineer_features(invoice_data: Dict) -> list:
    """Engineer 21 features for quality model"""
    # This should match your training features
    # For now, returning dummy values
    return [
        invoice_data.get('blur_score', 55.0),
        invoice_data.get('contrast_score', 35.0),
        invoice_data.get('ocr_confidence', 0.85),
        1,  # num_pages_fake
        55.0 * 0.85,  # blur_ocr_interaction
        55.0 / 35.0,  # blur_contrast_ratio
        0.85 * 0.55,  # ocr_blur_product
        55.0 ** 2,  # blur_squared
        0.85 ** 2,  # ocr_squared
        35.0 ** 2,  # contrast_squared
        0.75,  # overall_image_quality
        0,  # is_critical_low_blur
        0,  # is_low_blur
        0,  # is_excellent_blur
        0,  # is_low_ocr
        1,  # is_medium_ocr
        0,  # is_high_ocr
        0,  # is_low_contrast
        0,  # is_multipage
        0,  # is_high_risk_ocr
        0   # is_multipage_low_quality
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)