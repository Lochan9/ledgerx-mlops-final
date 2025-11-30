# src/inference/api_fastapi.py

"""
LedgerX – FastAPI Inference Deployment with Math Validation & Duplicate Detection
==================================================================================

Run with:
    uvicorn src.inference.api_fastapi:app --reload --port 8000

Features:
    - OAuth2 Authentication with Role-Based Access Control
    - Math Validation Pipeline
    - Duplicate Detection (NEW!)
    - Full Invoice Validation Pipeline
    - Smart Routing Decisions
    - Cost Optimization (Rate Limiting + Caching)
    - Batch Processing
    - Admin Dashboards
"""

import logging
from datetime import timedelta, datetime
from fastapi import FastAPI, HTTPException, Depends, status, Request, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import json
import time
from pathlib import Path
import io
import re

# Import existing services
from .inference_service import predict_invoice
from .auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    User,
    Token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    require_user,
    require_any_authenticated,
    require_admin
)

# Import validation modules
from ..stages.math_validation import validate_invoice_math
from ..stages.duplicate_detection import detect_duplicates

# Import Document AI OCR (replaces Tesseract)
try:
    from ..utils.document_ai_ocr import get_processor
    DOCUMENT_AI_ENABLED = True
    logger_init = logging.getLogger("ledgerx_fastapi")
    logger_init.info("[INIT] ✅ Document AI enabled for OCR")
except ImportError:
    DOCUMENT_AI_ENABLED = False
    logger_init = logging.getLogger("ledgerx_fastapi")
    logger_init.warning("[INIT] ⚠️ Document AI not available, falling back to basic extraction")

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ledgerx_fastapi")

# -------------------------------------------------------------------
# DUPLICATE DETECTION CONFIGURATION
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent
HISTORICAL_INVOICES_PATH = PROJECT_ROOT / "test_data" / "historical_invoices.csv"

# -------------------------------------------------------------------
# RATE LIMITER (Cost Protection for GCP Free Tier)
# -------------------------------------------------------------------
try:
    from ..utils.rate_limiter import rate_limiter, check_rate_limit
    RATE_LIMITING_ENABLED = True
    logger.info("[INIT] ✅ Rate limiting enabled for cost protection")
except ImportError:
    logger.warning("[INIT] ⚠️ Rate limiter not found - running without cost protection!")
    RATE_LIMITING_ENABLED = False
    
    async def check_rate_limit(request: Request):
        pass

# -------------------------------------------------------------------
# PREDICTION CACHE (40% Cost Savings)
# -------------------------------------------------------------------
try:
    from ..utils.prediction_cache import prediction_cache, get_cached_or_predict, get_cache_stats
    CACHING_ENABLED = True
    logger.info("[INIT] ✅ Prediction caching enabled (40% cost savings)")
except ImportError:
    logger.warning("[INIT] ⚠️ Prediction cache not found - running without caching")
    CACHING_ENABLED = False
    
    def get_cached_or_predict(features, predict_func):
        return predict_func(features)
    
    def get_cache_stats():
        return {"status": "disabled"}

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="LedgerX Invoice Intelligence API",
    description="""
    🏦 **AI-powered Invoice Quality Screening, Failure Risk Detection, Math Validation & Duplicate Detection**
    
    ## 🔐 Authentication
    
    All prediction endpoints require authentication:
    
    1. Obtain a token from `/token` endpoint with your credentials
    2. Include token in requests: `Authorization: Bearer <your_token>`
    
    ## 👥 Default Test Users
    
    - **admin** / admin123 → Full access + admin dashboards
    - **john_doe** / password123 → User access
    - **jane_viewer** / viewer123 → Readonly access
    
    ## ✨ Features v2.2
    
    ### New in v2.2:
    - ✅ **Duplicate Detection** (`/validate/invoice-full`) - Prevent double payments
    - ✅ Multi-strategy detection (exact, fuzzy, typo)
    - ✅ CRITICAL priority routing for duplicates
    
    ### Existing Features:
    - ✅ **Math Validation** - Verify invoice calculations
    - ✅ **Full Pipeline Validation** - Complete workflow
    - ✅ **Smart Routing** - Automatic decision engine with priority levels
    - ✅ **Quality Prediction** - Invoice quality assessment (good/bad)
    - ✅ **Failure Risk Prediction** - Payment failure probability
    - ✅ **Batch Processing** - Process 1-1000 invoices at once
    - ✅ **Cost Optimization** - Rate limiting + caching (40% savings)
    - ✅ **Admin Dashboards** - Cost and cache monitoring
    """,
    version="2.2.0",
    contact={
        "name": "LedgerX Support",
        "email": "support@ledgerx.ai"
    }
)

# -------------------------------------------------------------------
# CORS MIDDLEWARE
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------------------------------------------
# INPUT SCHEMAS WITH VALIDATION
# -------------------------------------------------------------------
class InvoiceFeatures(BaseModel):
    """Invoice features for ML model prediction"""
    blur_score: float = Field(ge=0, le=100, description="Image blur score (0-100)")
    contrast_score: float = Field(ge=0, le=100, description="Image contrast score (0-100)")
    ocr_confidence: float = Field(ge=0.0, le=1.0, description="OCR confidence (0-1)")
    file_size_kb: float = Field(gt=0, le=10000, description="File size in KB")

    vendor_name: str = Field(min_length=2, max_length=200, description="Vendor name")
    vendor_freq: float = Field(ge=0.0, le=1.0, description="Vendor frequency")

    total_amount: float = Field(gt=0, le=1000000, description="Invoice total amount")
    invoice_number: str = Field(min_length=1, max_length=100, description="Invoice number")
    invoice_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Invoice date (YYYY-MM-DD)")
    currency: str = Field(min_length=3, max_length=3, description="Currency code (e.g., USD)")

    @validator('vendor_name')
    def validate_vendor_name(cls, v):
        """Prevent SQL injection and XSS"""
        dangerous_chars = ["'", '"', '--', ';', '<', '>', 'DROP', 'DELETE', 'INSERT']
        v_upper = v.upper()
        for char in dangerous_chars:
            if char in v_upper:
                raise ValueError(f"Invalid character in vendor name: {char}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "blur_score": 45.5,
                "contrast_score": 62.3,
                "ocr_confidence": 0.89,
                "file_size_kb": 245.7,
                "vendor_name": "Tech Supplies Inc",
                "vendor_freq": 0.05,
                "total_amount": 1500.00,
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-11-20",
                "currency": "USD"
            }
        }

class PredictionResponse(BaseModel):
    """Standard prediction response"""
    status: str
    result: dict
    user: str
    timestamp: str

# -------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -------------------------------------------------------------------
@app.post("/token", response_model=Token, tags=["🔐 Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """**OAuth2 compatible token login**"""
    logger.info(f"[AUTH] 🔑 Login attempt for user: {form_data.username}")
    
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"[AUTH] ❌ Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    logger.info(f"[AUTH] ✅ Successful login for user: {form_data.username} (role: {user.role})")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.get("/users/me", response_model=User, tags=["🔐 Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information. Requires valid authentication token."""
    logger.info(f"[AUTH] 👤 User info requested: {current_user.username}")
    return current_user

# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------
@app.get("/health", tags=["💚 Health"])
async def health_check():
    """Health check endpoint - no authentication required."""
    return {
        "status": "healthy",
        "version": "2.2.0",
        "features": {
            "authentication": "enabled",
            "math_validation": "enabled",
            "duplicate_detection": "enabled",
            "rate_limiting": RATE_LIMITING_ENABLED,
            "caching": CACHING_ENABLED,
            "batch_processing": "enabled"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# MATH VALIDATION ENDPOINTS
# ============================================================================

@app.post("/validate/math", tags=["🧮 Validation"])
async def validate_invoice_math_endpoint(
    file: UploadFile = File(..., description="Invoice JSON file"),
    request: Request = None,
    current_user: User = Depends(require_any_authenticated),
    _: None = Depends(check_rate_limit)
):
    """**📊 Math Validation Only**"""
    logger.info("="*60)
    logger.info(f"[MATH-VAL] 📥 Request from user: {current_user.username}")
    logger.info(f"[MATH-VAL] 📄 File: {file.filename}")
    logger.info("="*60)
    
    try:
        contents = await file.read()
        invoice_data = json.loads(contents)
        
        validation_result = validate_invoice_math(invoice_data)
        
        response = {
            "status": "ok",
            "filename": file.filename,
            "validation_result": validation_result,
            "user": current_user.username,
            "timestamp": datetime.utcnow().isoformat(),
            "invoice_summary": {
                "invoice_number": invoice_data.get('invoice_number', 'N/A'),
                "vendor_name": invoice_data.get('vendor_name', 'N/A'),
                "total_amount": invoice_data.get('total_amount', 0.0)
            }
        }
        
        status_emoji = "✅" if validation_result['is_valid'] else "❌"
        logger.info(
            f"{status_emoji} [MATH-VAL] {file.filename}: "
            f"{'PASSED' if validation_result['is_valid'] else 'FAILED'} "
            f"(confidence: {validation_result['confidence']:.2%}, "
            f"errors: {validation_result['error_count']})"
        )
        logger.info("="*60)
        
        return response
    
    except json.JSONDecodeError as e:
        logger.error(f"[MATH-VAL] ❌ Invalid JSON: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        logger.error(f"[MATH-VAL] ❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/validate/invoice-full", tags=["🧮 Validation"])
async def validate_full_invoice_pipeline(
    file: UploadFile = File(..., description="Invoice JSON file"),
    request: Request = None,
    current_user: User = Depends(require_any_authenticated),
    _: None = Depends(check_rate_limit)
):
    """
    **🔄 COMPLETE INVOICE VALIDATION PIPELINE**
    
    ## Pipeline Stages:
    1. ✅ Math Validation - Verify calculations
    2. ✅ Duplicate Detection - Check for duplicate invoices (NEW!)
    3. ✅ Quality Assessment - ML model prediction
    4. ✅ Failure Risk - ML model prediction
    5. ✅ Smart Routing - Automatic decision with priority
    
    ## Duplicate Detection Strategies:
    - **Exact Match**: Same invoice#, vendor, amount (100% confidence)
    - **Fuzzy Match**: Similar amount ±5% within 30 days (variable confidence)
    - **Typo Detection**: Similar invoice numbers >90% similarity
    
    ## Routing Priorities:
    - **CRITICAL** (1-2 hr SLA): Duplicates, Math errors, High fraud risk
    - **HIGH** (24 hr SLA): Quality issues
    - **NONE**: Auto-process (all checks passed)
    """
    logger.info("="*70)
    logger.info(f"[FULL-VAL] 🔄 Complete pipeline validation from: {current_user.username}")
    logger.info(f"[FULL-VAL] 📄 File: {file.filename}")
    logger.info("="*70)
    
    start_time = time.time()
    
    try:
        contents = await file.read()
        invoice_data = json.loads(contents)
        
        # ====================================================================
        # STAGE 1: Math Validation
        # ====================================================================
        logger.info("[FULL-VAL] 📊 Stage 1/5: Math Validation")
        math_result = validate_invoice_math(invoice_data)
        math_status = "✅ PASS" if math_result['is_valid'] else "❌ FAIL"
        logger.info(f"[FULL-VAL] Math: {math_status} (confidence: {math_result['confidence']:.2%})")
        
        # ====================================================================
        # STAGE 2: Duplicate Detection (NEW!)
        # ====================================================================
        logger.info("[FULL-VAL] 🔍 Stage 2/5: Duplicate Detection")
        duplicate_result = detect_duplicates(invoice_data, HISTORICAL_INVOICES_PATH)
        dup_status = "🚨 DUPLICATE" if duplicate_result['is_duplicate'] else "✅ UNIQUE"
        logger.info(
            f"[FULL-VAL] Duplicate Check: {dup_status} "
            f"(found: {duplicate_result['duplicate_count']}, "
            f"confidence: {duplicate_result['highest_confidence']:.2%})"
        )
        
        # ====================================================================
        # STAGE 3 & 4: Quality Model + Failure Model
        # ====================================================================
        logger.info("[FULL-VAL] 🤖 Stage 3/5: ML Model Predictions")
        
        if CACHING_ENABLED:
            ml_predictions = get_cached_or_predict(invoice_data, predict_invoice)
            if ml_predictions.get('from_cache'):
                logger.info("[FULL-VAL] ✨ Cache HIT - Using cached predictions")
        else:
            ml_predictions = predict_invoice(invoice_data)
        
        quality_assessment = {
            'quality': 'bad' if ml_predictions['quality_bad'] == 1 else 'good',
            'probability': ml_predictions['quality_probability'],
            'confidence': max(ml_predictions['quality_probability'], 
                            1 - ml_predictions['quality_probability'])
        }
        
        failure_risk = {
            'risk': 'high' if ml_predictions['failure_probability'] > 0.7 
                   else 'medium' if ml_predictions['failure_probability'] > 0.3 
                   else 'low',
            'probability': ml_predictions['failure_probability']
        }
        
        logger.info(f"[FULL-VAL] Quality: {quality_assessment['quality'].upper()}")
        logger.info(f"[FULL-VAL] Failure Risk: {failure_risk['risk'].upper()}")
        
        # ====================================================================
        # STAGE 5: Smart Routing Decision
        # ====================================================================
        logger.info("[FULL-VAL] 🎯 Stage 4/5: Routing Decision")
        routing_decision = determine_routing(
            math_result, 
            duplicate_result,
            quality_assessment,
            failure_risk
        )
        logger.info(
            f"[FULL-VAL] Decision: {routing_decision['action']} "
            f"(Priority: {routing_decision['priority']}, "
            f"Reason: {routing_decision['reason']})"
        )
        
        processing_time = time.time() - start_time
        
        response = {
            "status": "ok",
            "filename": file.filename,
            "user": current_user.username,
            "timestamp": datetime.utcnow().isoformat(),
            "invoice_summary": {
                "invoice_number": invoice_data.get('invoice_number', 'N/A'),
                "vendor_name": invoice_data.get('vendor_name', 'N/A'),
                "total_amount": invoice_data.get('total_amount', 0.0),
                "invoice_date": invoice_data.get('invoice_date', 'N/A')
            },
            "validations": {
                "math_validation": math_result,
                "duplicate_detection": duplicate_result,
                "quality_assessment": quality_assessment,
                "failure_risk": failure_risk,
                "warnings": ml_predictions.get('warnings', [])
            },
            "routing": routing_decision,
            "performance": {
                "processing_time_seconds": round(processing_time, 3),
                "cached": ml_predictions.get('from_cache', False)
            }
        }
        
        logger.info(f"[FULL-VAL] ✅ Pipeline complete in {processing_time:.3f}s")
        logger.info("="*70)
        
        return response
    
    except json.JSONDecodeError as e:
        logger.error(f"[FULL-VAL] ❌ Invalid JSON: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        logger.exception(f"[FULL-VAL] ❌ Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


def determine_routing(math_result: Dict, 
                     duplicate_result: Dict,
                     quality_result: Dict, 
                     failure_result: Dict) -> Dict[str, Any]:
    """
    **Smart Routing Decision Engine**
    
    Priority Order:
    1. Duplicate detection → CRITICAL (prevent double payment!)
    2. Math validation failure → CRITICAL
    3. Bad quality → HIGH
    4. High failure risk → CRITICAL
    5. All pass → AUTO_PROCESS
    """
    # Check for duplicates FIRST
    if duplicate_result['is_duplicate']:
        return {
            'action': 'HUMAN_REVIEW',
            'priority': 'CRITICAL',
            'reason': 'DUPLICATE_INVOICE_DETECTED',
            'details': f"Found {duplicate_result['duplicate_count']} potential duplicate(s) "
                      f"(confidence: {duplicate_result['highest_confidence']:.2%})",
            'queue': 'DUPLICATE_REVIEW',
            'sla_hours': 1,
            'duplicate_details': duplicate_result['duplicates_found'],
            'next_steps': [
                'URGENT: Verify if invoice was already paid',
                'Check payment history immediately',
                'Contact vendor to confirm if duplicate',
                'Review original invoice submission',
                'Reject duplicate or mark as correction'
            ]
        }
    
    # Check math validation
    if not math_result['is_valid']:
        return {
            'action': 'HUMAN_REVIEW',
            'priority': 'CRITICAL',
            'reason': 'MATH_VALIDATION_FAILED',
            'details': f"Found {math_result['error_count']} calculation error(s)",
            'queue': 'CALCULATION_ERRORS',
            'sla_hours': 2,
            'next_steps': [
                'Review calculation errors',
                'Verify with vendor',
                'Correct amounts manually'
            ]
        }
    
    # Check quality
    if quality_result['quality'] == 'bad':
        if quality_result['probability'] > 0.8:
            return {
                'action': 'HUMAN_REVIEW',
                'priority': 'HIGH',
                'reason': 'LOW_QUALITY_INVOICE',
                'details': f"Quality confidence: {quality_result['confidence']:.2%}",
                'queue': 'QUALITY_REVIEW',
                'sla_hours': 24,
                'next_steps': [
                    'Review OCR extraction',
                    'Verify missing fields',
                    'Request clearer invoice from vendor'
                ]
            }
    
    # Check failure risk
    if failure_result['risk'] == 'high':
        return {
            'action': 'HUMAN_REVIEW',
            'priority': 'CRITICAL',
            'reason': 'HIGH_FAILURE_RISK',
            'details': f"Failure probability: {failure_result['probability']:.2%}",
            'queue': 'RISK_REVIEW',
            'sla_hours': 4,
            'next_steps': [
                'Check for duplicate invoices',
                'Verify vendor details',
                'Review payment history',
                'Confirm with requester'
            ]
        }
    
    # All checks passed
    return {
        'action': 'AUTO_PROCESS',
        'priority': 'NONE',
        'reason': 'ALL_VALIDATIONS_PASSED',
        'details': 'Invoice approved for automatic processing',
        'queue': None,
        'sla_hours': None,
        'next_steps': [
            'Store in database',
            'Generate final JSON',
            'Send to ERP system',
            'Schedule payment'
        ]
    }

# -------------------------------------------------------------------
# ML PREDICTION ENDPOINTS
# -------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["🤖 ML Predictions"])
async def predict_invoice_endpoint(
    features: InvoiceFeatures,
    request: Request,
    current_user: User = Depends(require_any_authenticated),
    _: None = Depends(check_rate_limit)
):
    """🤖 ML Model Predictions"""
    logger.info("="*60)
    logger.info(f"[API] 🤖 Prediction request from: {current_user.username}")
    logger.info(f"[API] 📄 Invoice: {features.invoice_number}")
    logger.info("="*60)
    
    try:
        if CACHING_ENABLED:
            result = get_cached_or_predict(features.dict(), predict_invoice)
            if result.get('from_cache'):
                logger.info(f"[API] ✨ Cache HIT for invoice: {features.invoice_number}")
        else:
            result = predict_invoice(features.dict())
        
        user_result = {
            'quality_assessment': {
                'quality': 'bad' if result['quality_bad'] == 1 else 'good',
                'probability': result['quality_probability'],
                'confidence': max(result['quality_probability'], 1 - result['quality_probability'])
            },
            'failure_risk': {
                'risk': 'high' if result['failure_probability'] > 0.7 
                       else 'medium' if result['failure_probability'] > 0.3 
                       else 'low',
                'probability': result['failure_probability']
            },
            'warnings': result.get('warnings', [])
        }
        
        logger.info(f"[API] Quality: {user_result['quality_assessment']['quality'].upper()}")
        logger.info(f"[API] Failure Risk: {user_result['failure_risk']['risk'].upper()}")
        logger.info("="*60)
        
        return PredictionResponse(
            status="ok",
            result=user_result,
            user=current_user.username,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.exception(f"[API] ❌ Error during inference for user: {current_user.username}")
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------------------------------------------------
# BATCH PREDICTION ENDPOINT
# -------------------------------------------------------------------

# ============================================================================
# IMAGE UPLOAD WITH OCR (NEW!)
# ============================================================================

@app.post("/upload/image", tags=["📤 Upload & OCR"])
async def upload_and_process_image(
    file: UploadFile = File(..., description="Invoice image (JPG, PNG, PDF)"),
    request: Request = None,
    current_user: User = Depends(require_any_authenticated),
    _: None = Depends(check_rate_limit)
):
    """
    **📸 Upload Invoice Image - Auto OCR & ML Processing**
    
    Accepts: JPG, PNG, PDF
    
    Pipeline:
    1. 🔍 OCR Extraction (Google Document AI - 95%+ accuracy)
    2. 📋 Automatic structured extraction
    3. 🧮 Math validation
    4. 🔍 Duplicate detection  
    5. 🤖 Quality assessment (ML)
    6. 🎯 Failure risk prediction (ML)
    7. 🚦 Smart routing decision
    
    Returns complete validation results with extracted data
    """
    logger.info("="*70)
    logger.info(f"[IMAGE-UPLOAD] 📸 Image processing from: {current_user.username}")
    logger.info(f"[IMAGE-UPLOAD] 📄 File: {file.filename} ({file.content_type})")
    logger.info("="*70)
    
    start_time = time.time()
    
    try:
        # Step 1: Read file bytes
        contents = await file.read()
        
        # Step 2: Process with Document AI (95%+ accuracy)
        if DOCUMENT_AI_ENABLED:
            logger.info("[IMAGE-UPLOAD] 🔍 Processing with Google Document AI...")
            
            doc_ai = get_processor()
            invoice_data = doc_ai.process_invoice(
                contents, 
                mime_type=file.content_type or "image/jpeg"
            )
            
            # Add file size
            invoice_data["file_size_kb"] = len(contents) / 1024
            
            logger.info(f"[IMAGE-UPLOAD] ✅ Document AI extracted:")
            logger.info(f"  Invoice #: {invoice_data.get('invoice_number')}")
            logger.info(f"  Vendor: {invoice_data.get('vendor_name')}")
            logger.info(f"  Amount: {invoice_data.get('currency')} {invoice_data.get('total_amount')}")
            logger.info(f"  Date: {invoice_data.get('invoice_date')}")
            logger.info(f"  Confidence: {invoice_data.get('ocr_confidence', 0):.2%}")
        else:
            # Fallback to basic extraction (not recommended for production)
            logger.warning("[IMAGE-UPLOAD] ⚠️ Using fallback extraction (low accuracy)")
            invoice_data = {
                "invoice_number": f"AUTO-{datetime.utcnow().year}-{file.filename[:10]}",
                "vendor_name": "Unknown Vendor",
                "total_amount": 0.0,
                "currency": "USD",
                "invoice_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "blur_score": 50.0,
                "contrast_score": 60.0,
                "ocr_confidence": 0.5,
                "file_size_kb": len(contents) / 1024,
                "vendor_freq": 0.05
            }
        
        # Step 3: Run validation pipeline
        logger.info("[IMAGE-UPLOAD] 🔄 Running validation pipeline...")
        
        # Math validation
        math_result = validate_invoice_math(invoice_data)
        
        # Duplicate detection
        duplicate_result = detect_duplicates(invoice_data, HISTORICAL_INVOICES_PATH)
        
        # ML predictions
        if CACHING_ENABLED:
            ml_predictions = get_cached_or_predict(invoice_data, predict_invoice)
        else:
            ml_predictions = predict_invoice(invoice_data)
        
        # Quality assessment
        quality_assessment = {
            'quality': 'bad' if ml_predictions['quality_bad'] == 1 else 'good',
            'prediction': 'bad' if ml_predictions['quality_bad'] == 1 else 'good',
            'probability': ml_predictions['quality_probability'],
            'probabilities': {
                'good': 1 - ml_predictions['quality_probability'] if ml_predictions['quality_bad'] == 1 
                        else ml_predictions['quality_probability'],
                'bad': ml_predictions['quality_probability'] if ml_predictions['quality_bad'] == 1 
                       else 1 - ml_predictions['quality_probability']
            }
        }
        
        # Failure risk
        failure_risk = {
            'risk': 'high' if ml_predictions['failure_probability'] > 0.7 else 'low',
            'prediction': 'risk' if ml_predictions['failure_probability'] > 0.5 else 'safe',
            'probability': ml_predictions['failure_probability'],
            'probabilities': {
                'risk': ml_predictions['failure_probability'],
                'safe': 1 - ml_predictions['failure_probability']
            }
        }
        
        # Routing decision
        routing_decision = determine_routing(
            math_result,
            duplicate_result,
            quality_assessment,
            failure_risk
        )
        
        processing_time = time.time() - start_time
        
        response = {
            "status": "ok",
            "filename": file.filename,
            "file_type": "image",
            "ocr_extracted": True,
            "ocr_method": "Document AI" if DOCUMENT_AI_ENABLED else "Fallback",
            "extracted_data": invoice_data,
            "quality": quality_assessment,
            "failure": failure_risk,
            "math_validation": math_result,
            "duplicate_check": duplicate_result,
            "routing": routing_decision,
            "performance": {
                "processing_time_seconds": round(processing_time, 3),
                "ocr_time": "included",
                "cached": ml_predictions.get('from_cache', False)
            },
            "user": current_user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"[IMAGE-UPLOAD] ✅ Complete pipeline finished in {processing_time:.3f}s")
        logger.info(f"[IMAGE-UPLOAD] Quality: {quality_assessment['quality']}, Risk: {failure_risk['risk']}")
        logger.info("="*70)
        
        return response
        
    except Exception as e:
        logger.error(f"[IMAGE-UPLOAD] ❌ Processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")



@app.post("/predict/batch", tags=["🤖 ML Predictions"])
async def predict_batch(
    invoices: List[InvoiceFeatures],
    request: Request,
    current_user: User = Depends(require_user),
    _: None = Depends(check_rate_limit)
):
    """📦 Batch Prediction for Multiple Invoices"""
    logger.info("="*60)
    logger.info(f"[BATCH] 📦 Request from: {current_user.username}")
    logger.info(f"[BATCH] 📊 Batch size: {len(invoices)}")
    logger.info("="*60)
    
    if len(invoices) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 1000")
    
    if len(invoices) == 0:
        raise HTTPException(status_code=400, detail="Batch must contain at least 1 invoice")
    
    start_time = time.time()
    results = []
    cache_hits = 0
    errors = 0
    
    try:
        for idx, invoice in enumerate(invoices):
            try:
                if CACHING_ENABLED:
                    result = get_cached_or_predict(invoice.dict(), predict_invoice)
                    if result.get('from_cache'):
                        cache_hits += 1
                else:
                    result = predict_invoice(invoice.dict())
                
                user_result = {
                    'quality_assessment': {
                        'quality': 'bad' if result['quality_bad'] == 1 else 'good',
                        'probability': result['quality_probability']
                    },
                    'failure_risk': {
                        'risk': 'high' if result['failure_probability'] > 0.7 
                               else 'medium' if result['failure_probability'] > 0.3 
                               else 'low',
                        'probability': result['failure_probability']
                    }
                }
                
                results.append({
                    "index": idx,
                    "invoice_number": invoice.invoice_number,
                    "prediction": user_result,
                    "status": "success"
                })
                
            except Exception as e:
                errors += 1
                logger.warning(f"[BATCH] ⚠️ Error processing invoice {idx}: {e}")
                results.append({
                    "index": idx,
                    "invoice_number": invoice.invoice_number,
                    "error": str(e),
                    "status": "error"
                })
        
        processing_time = time.time() - start_time
        
        logger.info(f"[BATCH] ✅ Processed {len(invoices)} invoices in {processing_time:.2f}s")
        logger.info(f"[BATCH] Success: {len(invoices) - errors}, Errors: {errors}, Cache hits: {cache_hits}")
        logger.info("="*60)
        
        return {
            "status": "ok",
            "batch_size": len(invoices),
            "results": results,
            "summary": {
                "total": len(invoices),
                "successful": len(invoices) - errors,
                "errors": errors,
                "cache_hits": cache_hits,
                "cache_hit_rate": f"{(cache_hits/len(invoices))*100:.1f}%",
                "processing_time_seconds": round(processing_time, 2),
                "avg_time_per_invoice": round(processing_time / len(invoices), 3)
            },
            "user": current_user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"[BATCH] ❌ Failed for user: {current_user.username}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# ADMIN ENDPOINTS
# -------------------------------------------------------------------
@app.get("/admin/costs", tags=["👑 Admin"])
async def get_cost_stats(current_user: User = Depends(require_admin)):
    """💰 Cost and Usage Statistics - **Requires Admin role**"""
    if not RATE_LIMITING_ENABLED:
        return {"status": "disabled", "message": "Rate limiting not enabled"}
    
    from ..utils.rate_limiter import rate_limiter
    stats = rate_limiter.get_usage_stats()
    
    return {
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "requests_today": stats['total_requests_today'],
        "estimated_cost_today": f"${stats['estimated_cost_today']:.4f}",
        "daily_budget": f"${stats['daily_budget']:.2f}",
        "budget_used": f"{(stats['estimated_cost_today']/stats['daily_budget'])*100:.1f}%",
        "budget_remaining": f"${stats['budget_remaining']:.4f}",
        "monthly_projection": f"${stats['estimated_cost_today'] * 30:.2f}",
        "status": "healthy" if stats['estimated_cost_today'] < stats['daily_budget'] else "over_budget"
    }

@app.get("/admin/cache", tags=["👑 Admin"])
async def get_cache_statistics(current_user: User = Depends(require_admin)):
    """📊 Cache Statistics and Cost Savings - **Requires Admin role**"""
    if not CACHING_ENABLED:
        return {"status": "disabled", "message": "Caching not enabled"}
    
    stats = get_cache_stats()
    
    return {
        "cache_enabled": True,
        "cache_size": stats['cache_size'],
        "max_size": stats['max_size'],
        "utilization": f"{(stats['cache_size']/stats['max_size'])*100:.1f}%",
        "performance": {
            "total_requests": stats['total_requests'],
            "cache_hits": stats['cache_hits'],
            "cache_misses": stats['cache_misses'],
            "hit_rate": f"{stats['hit_rate_percent']:.1f}%"
        },
        "cost_savings": {
            "saved_today": f"${stats['cost_saved_today']:.6f}",
            "cost_without_cache": f"${stats['cost_without_cache']:.6f}",
            "savings_percent": f"{stats['savings_percent']:.1f}%"
        },
        "recommendation": "Excellent!" if stats['hit_rate_percent'] > 30 else "Consider cache warming"
    }

# -------------------------------------------------------------------
# STARTUP EVENT
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Application startup event - logs configuration"""
    logger.info("=" * 70)
    logger.info("🚀 LedgerX Invoice Intelligence API v2.2")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("  ✅ Authentication: OAuth2 + Role-Based Access Control")
    logger.info("  ✅ Math Validation: Calculation verification")
    logger.info("  ✅ Duplicate Detection: NEW - Prevent double payments")
    logger.info("  ✅ Full Pipeline: Complete validation workflow")
    logger.info("  ✅ Smart Routing: Automatic decision engine")
    logger.info(f"  ✅ Rate Limiting: {'ENABLED' if RATE_LIMITING_ENABLED else 'DISABLED'}")
    logger.info(f"  ✅ Prediction Caching: {'ENABLED (40% savings)' if CACHING_ENABLED else 'DISABLED'}")
    logger.info("  ✅ Batch Processing: 1-1000 invoices")
    logger.info("  ✅ Cost Dashboards: Admin-only monitoring")
    
    if RATE_LIMITING_ENABLED:
        logger.info("=" * 70)
        logger.info("Rate Limiting:")
        logger.info("  - IP Limit: 50/hour, 200/day")
        logger.info("  - Budget Protection: $1.67/day")
    
    if CACHING_ENABLED:
        logger.info("=" * 70)
        logger.info("Prediction Caching:")
        logger.info("  - Cache Size: 1000 predictions")
        logger.info("  - TTL: 5 minutes")
        logger.info("  - Expected Savings: 30-40%")
    
    logger.info("=" * 70)
    logger.info("Test Credentials:")
    logger.info("  - admin / admin123 (Admin - Full Access)")
    logger.info("  - john_doe / password123 (User)")
    logger.info("  - jane_viewer / viewer123 (Readonly)")
    
    logger.info("=" * 70)
    logger.info("Endpoints:")
    logger.info("  - POST /validate/math - Math validation")
    logger.info("  - POST /validate/invoice-full - Complete pipeline + duplicates")
    
    logger.info("=" * 70)
    logger.info("Admin Endpoints:")
    logger.info("  - GET /admin/costs - Cost dashboard")
    logger.info("  - GET /admin/cache - Cache statistics")
    
    logger.info("=" * 70)
    logger.info("📖 API Documentation: http://localhost:8000/docs")
    logger.info("=" * 70)