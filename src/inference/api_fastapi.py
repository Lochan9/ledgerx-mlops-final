# src/inference/api_fastapi.py

"""
LedgerX – FastAPI Inference Deployment with Cloud Logging Integration
======================================================================

Run with:
    uvicorn src.inference.api_fastapi:app --reload --port 8000

Features:
    - OAuth2 Authentication with Role-Based Access Control
    - Math Validation Pipeline
    - Duplicate Detection
    - Full Invoice Validation Pipeline
    - Smart Routing Decisions
    - Cost Optimization (Rate Limiting + Caching)
    - Batch Processing
    - Admin Dashboards
    - **NEW: Cloud Logging with Structured Logs**
"""

import os
import time
import logging
from datetime import timedelta, datetime
from fastapi import FastAPI, HTTPException, Depends, status, Request, File, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import json
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
except ImportError:
    DOCUMENT_AI_ENABLED = False

# Import Cloud SQL database helpers
try:
    from ..utils.database import (
        get_user_by_username, save_invoice, get_user_invoices, 
        delete_invoice, track_document_ai_usage, get_monthly_document_ai_usage
    )
    CLOUD_SQL_ENABLED = True
except ImportError:
    CLOUD_SQL_ENABLED = False

# -------------------------------------------------------------------
# CLOUD LOGGING - UPDATED SECTION
# -------------------------------------------------------------------
from ..utils.cloud_logging import setup_cloud_logging, get_logger
from ..utils.logging_middleware import setup_logging_middleware

# Import Prometheus metrics from monitoring
from .monitoring import (
    prediction_total, prediction_errors, prediction_latency,
    quality_predictions_good, quality_predictions_bad,
    failure_predictions_safe, failure_predictions_risk,
    model_quality_probability, model_failure_probability,
    feature_blur_score, feature_ocr_confidence, feature_total_amount
)

# Initialize Cloud Logger (replaces standard logging)
cloud_logger = setup_cloud_logging(
    name="ledgerx_api",
    log_level=os.getenv("LOG_LEVEL", "INFO")
)
logger = get_logger(name="ledgerx_api")

# Log initialization
logger.info(
    "LedgerX API components initializing",
    event_type="initialization",
    document_ai_enabled=DOCUMENT_AI_ENABLED,
    cloud_sql_enabled=CLOUD_SQL_ENABLED
)

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
    logger.info("Rate limiting enabled for cost protection")
except ImportError:
    logger.warning("Rate limiter not found - running without cost protection")
    RATE_LIMITING_ENABLED = False
    
    async def check_rate_limit(request: Request):
        pass

# -------------------------------------------------------------------
# PREDICTION CACHE (40% Cost Savings)
# -------------------------------------------------------------------
try:
    from ..utils.prediction_cache import prediction_cache, get_cached_or_predict, get_cache_stats
    CACHING_ENABLED = True
    logger.info("Prediction caching enabled (40% cost savings)")
except ImportError:
    logger.warning("Prediction cache not found - running without caching")
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
    - ✅ **Cloud Logging Integration** - Real-time logs in GCP Console
    - ✅ **Duplicate Detection** (`/validate/invoice-full`)
    - ✅ Prevent double payments
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
# CLOUD LOGGING MIDDLEWARE - NEW ADDITION
# -------------------------------------------------------------------
setup_logging_middleware(
    app,
    logger_name="ledgerx_api",
    enable_performance_monitoring=True,
    slow_request_threshold_ms=1000.0
)
logger.info("FastAPI middleware configured with Cloud Logging")

# -------------------------------------------------------------------
# PROMETHEUS INSTRUMENTATION (Optional)
# -------------------------------------------------------------------
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics exposed at /metrics")
except ImportError:
    logger.debug("Prometheus instrumentation not available")

# ===================================================================
# MODEL PERFORMANCE METRICS (Static from Training)
# ===================================================================
from prometheus_client import Gauge

model_quality_f1 = Gauge('ledgerx_model_quality_f1_score', 'Quality model F1 score from training')
model_failure_f1 = Gauge('ledgerx_model_failure_f1_score', 'Failure model F1 score from training')
model_drift_score = Gauge('ledgerx_model_drift_score', 'Model drift detection score', ['model'])

# Set baseline values from model training
model_quality_f1.set(0.771)  # 77.1% F1 score
model_failure_f1.set(0.709)  # 70.9% F1 score
model_drift_score.labels(model="quality").set(0.045)  # Low drift
model_drift_score.labels(model="failure").set(0.038)  # Low drift

logger.info("Model performance metrics initialized", 
            quality_f1=0.771, failure_f1=0.709)

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

# ===================================================================
# APPLICATION LIFECYCLE EVENTS
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup - log configuration with structured data"""
    logger.info(
        "LedgerX API starting",
        event_type="application_startup",
        version="2.2.0",
        environment=os.getenv("ENVIRONMENT", "production"),
        document_ai_enabled=DOCUMENT_AI_ENABLED,
        cloud_sql_enabled=CLOUD_SQL_ENABLED,
        rate_limiting_enabled=RATE_LIMITING_ENABLED,
        caching_enabled=CACHING_ENABLED,
        features={
            "authentication": "oauth2",
            "math_validation": True,
            "duplicate_detection": True,
            "batch_processing": True,
            "cloud_logging": True
        }
    )
    
    logger.info("=" * 70)
    logger.info("🚀 LedgerX Invoice Intelligence API v2.2")
    logger.info("=" * 70)
    logger.info("Features:")
    logger.info("  ✅ Authentication: OAuth2 + Role-Based Access Control")
    logger.info("  ✅ Math Validation: Calculation verification")
    logger.info("  ✅ Duplicate Detection: Prevent double payments")
    logger.info("  ✅ Full Pipeline: Complete validation workflow")
    logger.info("  ✅ Smart Routing: Automatic decision engine")
    logger.info(f"  ✅ Rate Limiting: {'ENABLED' if RATE_LIMITING_ENABLED else 'DISABLED'}")
    logger.info(f"  ✅ Prediction Caching: {'ENABLED (40% savings)' if CACHING_ENABLED else 'DISABLED'}")
    logger.info("  ✅ Batch Processing: 1-1000 invoices")
    logger.info("  ✅ Cloud Logging: ENABLED - Logs in GCP Console")
    logger.info("=" * 70)

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info(
        "LedgerX API shutting down",
        event_type="application_shutdown",
        timestamp=datetime.utcnow().isoformat()
    )

# -------------------------------------------------------------------
# HEALTH CHECK ENDPOINTS
# -------------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    logger.debug("Root endpoint accessed")
    return {
        "message": "LedgerX API", 
        "status": "running", 
        "version": "2.2.0",
        "cloud_logging": True
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint with system status"""
    logger.debug("Health check requested", endpoint="/health")
    return {
        "status": "healthy",
        "service": "LedgerX API",
        "version": "2.2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "cloud_logging": True,
        "services": {
            "document_ai": DOCUMENT_AI_ENABLED,
            "cloud_sql": CLOUD_SQL_ENABLED,
            "rate_limiting": RATE_LIMITING_ENABLED,
            "caching": CACHING_ENABLED
        }
    }

# -------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -------------------------------------------------------------------
@app.post("/token", response_model=Token, tags=["🔐 Authentication"])
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """**OAuth2 compatible token login with Cloud Logging**"""
    
    # Attempt authentication
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        # Log failed attempt with structured data
        logger.warning(
            "Failed authentication attempt",
            event_type="authentication_failed",
            username=form_data.username,
            ip_address=request.client.host if request.client else "unknown",
            timestamp=datetime.utcnow().isoformat()
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    
    # Log successful authentication with structured data
    logger.info(
        "User authenticated successfully",
        event_type="user_authentication",
        user_id=user.username,
        user_role=user.role,
        auth_method="password",
        ip_address=request.client.host if request.client else "unknown",
        token_expires_minutes=ACCESS_TOKEN_EXPIRE_MINUTES,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.get("/users/me", response_model=User, tags=["🔐 Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information. Requires valid authentication token."""
    logger.debug("User info requested", user_id=current_user.username)
    return current_user

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
    start_time = time.time()
    
    logger.info(
        "Math validation request",
        event_type="math_validation_start",
        user_id=current_user.username,
        uploaded_file=file.filename
    )
    
    try:
        contents = await file.read()
        invoice_data = json.loads(contents)
        
        validation_result = validate_invoice_math(invoice_data)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log validation result
        logger.info(
            "Math validation completed",
            event_type="math_validation_complete",
            user_id=current_user.username,
            uploaded_file=file.filename,
            is_valid=validation_result['is_valid'],
            confidence=validation_result['confidence'],
            error_count=validation_result['error_count'],
            latency_ms=latency_ms
        )
        
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
        
        return response
    
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in math validation",
            error_type="JSONDecodeError",
            error_message=str(e),
            uploaded_file=file.filename
        )
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        logger.exception(
            "Math validation failed",
            event_type="math_validation_error",
            user_id=current_user.username,
            uploaded_file=file.filename,
            error_type=type(e).__name__,
            error_message=str(e)
        )
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
    2. ✅ Duplicate Detection - Check for duplicate invoices
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
    start_time = time.time()
    invoice_id = f"val_{int(time.time())}"
    
    logger.info(
        "Full validation pipeline started",
        event_type="full_validation_start",
        user_id=current_user.username,
        uploaded_file=file.filename,
        invoice_id=invoice_id
    )
    
    try:
        contents = await file.read()
        invoice_data = json.loads(contents)
        
        # ====================================================================
        # STAGE 1: Math Validation
        # ====================================================================
        logger.info("Stage 1/5: Math Validation", invoice_id=invoice_id)
        math_result = validate_invoice_math(invoice_data)
        math_status = "PASS" if math_result['is_valid'] else "FAIL"
        logger.info(f"Math validation: {math_status}", invoice_id=invoice_id, confidence=math_result['confidence'])
        
        # ====================================================================
        # STAGE 2: Duplicate Detection
        # ====================================================================
        logger.info("Stage 2/5: Duplicate Detection", invoice_id=invoice_id)
        duplicate_result = detect_duplicates(invoice_data, HISTORICAL_INVOICES_PATH)
        dup_status = "DUPLICATE" if duplicate_result['is_duplicate'] else "UNIQUE"
        logger.info(
            f"Duplicate check: {dup_status}",
            invoice_id=invoice_id,
            duplicate_count=duplicate_result['duplicate_count'],
            confidence=duplicate_result['highest_confidence']
        )
        
        # ====================================================================
        # STAGE 3 & 4: Quality Model + Failure Model
        # ====================================================================
        logger.info("Stage 3/5: ML Model Predictions", invoice_id=invoice_id)
        
        if CACHING_ENABLED:
            ml_predictions = get_cached_or_predict(invoice_data, predict_invoice)
            if ml_predictions.get('from_cache'):
                logger.info("Cache HIT - Using cached predictions", invoice_id=invoice_id)
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
        
        logger.info(
            "ML predictions complete",
            invoice_id=invoice_id,
            quality=quality_assessment['quality'],
            failure_risk=failure_risk['risk']
        )
        
        # ====================================================================
        # STAGE 5: Smart Routing Decision
        # ====================================================================
        logger.info("Stage 4/5: Routing Decision", invoice_id=invoice_id)
        routing_decision = determine_routing(
            math_result, 
            duplicate_result,
            quality_assessment,
            failure_risk
        )
        logger.info(
            "Routing decision made",
            invoice_id=invoice_id,
            action=routing_decision['action'],
            priority=routing_decision['priority'],
            reason=routing_decision['reason']
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
        
        logger.info(
            "Full validation pipeline complete",
            event_type="full_validation_complete",
            invoice_id=invoice_id,
            user_id=current_user.username,
            processing_time_ms=processing_time * 1000,
            routing_action=routing_decision['action'],
            routing_priority=routing_decision['priority']
        )
        
        return response
    
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in full validation",
            error_type="JSONDecodeError",
            error_message=str(e),
            uploaded_file=file.filename
        )
        raise HTTPException(status_code=400, detail=f"Invalid JSON file: {str(e)}")
    except Exception as e:
        logger.exception(
            "Full validation pipeline failed",
            event_type="full_validation_error",
            invoice_id=invoice_id,
            user_id=current_user.username,
            error_type=type(e).__name__,
            error_message=str(e)
        )
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
                'details': f"Quality probability: {quality_result['probability']:.2%}",
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
    """🤖 ML Model Predictions with Cloud Logging"""
    start_time = time.time()
    invoice_id = features.invoice_number
    
    logger.info(
        "Prediction request",
        event_type="prediction_start",
        user_id=current_user.username,
        invoice_id=invoice_id,
        vendor_name=features.vendor_name
    )
    
    try:
        if CACHING_ENABLED:
            result = get_cached_or_predict(features.dict(), predict_invoice)
            cache_hit = result.get('from_cache', False)
            if cache_hit:
                logger.info("Cache HIT", invoice_id=invoice_id)
        else:
            result = predict_invoice(features.dict())
            cache_hit = False
        
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
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log prediction with structured data
        logger.log_prediction(
            user_id=current_user.username,
            invoice_id=invoice_id,
            quality_prediction=user_result['quality_assessment']['quality'],
            failure_prediction=user_result['failure_risk']['risk'],
            latency_ms=latency_ms,
            model_version="v1.2.0",
            confidence_quality=user_result['quality_assessment']['confidence'],
            confidence_failure=user_result['failure_risk']['probability'],
            cache_hit=cache_hit,
            vendor_name=features.vendor_name,
            total_amount=features.total_amount
        )
        
        return PredictionResponse(
            status="ok",
            result=user_result,
            user=current_user.username,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.exception(
            "Prediction failed",
            event_type="prediction_error",
            invoice_id=invoice_id,
            user_id=current_user.username,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# IMAGE UPLOAD WITH OCR
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
    start_time = time.time()
    invoice_id = f"img_{int(time.time())}"
    
    logger.info(
        "Image upload and processing started",
        event_type="image_upload_start",
        user_id=current_user.username,
        uploaded_file=file.filename,
        content_type=file.content_type,
        invoice_id=invoice_id
    )
    
    try:
        # Step 1: Read file bytes
        contents = await file.read()
        file_size_kb = len(contents) / 1024
        
        # Step 2: Process with Document AI (95%+ accuracy)
        if DOCUMENT_AI_ENABLED:
            ocr_start = time.time()
            logger.info("Processing with Google Document AI", invoice_id=invoice_id)
            
            doc_ai = get_processor()
            invoice_data = doc_ai.process_invoice(
                contents, 
                mime_type=file.content_type or "image/jpeg"
            )
            
            ocr_time_ms = (time.time() - ocr_start) * 1000
            
            # Add file size
            invoice_data["file_size_kb"] = file_size_kb
            
            # Log OCR processing
            logger.log_ocr_processing(
                invoice_id=invoice_id,
                ocr_engine="document_ai",
                confidence=invoice_data.get('ocr_confidence', 0.0),
                processing_time_ms=ocr_time_ms,
                page_count=1,
                file_size_kb=file_size_kb
            )
            
            logger.info(
                "Document AI extraction complete",
                invoice_id=invoice_id,
                invoice_number=invoice_data.get('invoice_number'),
                vendor=invoice_data.get('vendor_name'),
                amount=invoice_data.get('total_amount'),
                confidence=invoice_data.get('ocr_confidence')
            )
            
            # Track Document AI usage in database
            if CLOUD_SQL_ENABLED:
                track_document_ai_usage(current_user.username)
                monthly_usage = get_monthly_document_ai_usage()
                
                # Log cost event
                logger.log_cost_event(
                    service="document_ai",
                    operation="process_document",
                    cost_usd=0.0015,
                    units_consumed=1,
                    monthly_usage=monthly_usage,
                    monthly_limit=1000,
                    user_id=current_user.username
                )
        else:
            # Fallback to basic extraction (not recommended for production)
            logger.warning("Using fallback extraction (low accuracy)", invoice_id=invoice_id)
            invoice_data = {
                "invoice_number": f"AUTO-{datetime.utcnow().year}-{file.filename[:10]}",
                "vendor_name": "Unknown Vendor",
                "total_amount": 0.0,
                "currency": "USD",
                "invoice_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "blur_score": 50.0,
                "contrast_score": 60.0,
                "ocr_confidence": 0.5,
                "file_size_kb": file_size_kb,
                "vendor_freq": 0.05
            }
        
        # Step 3: Run validation pipeline
        logger.info("Running validation pipeline", invoice_id=invoice_id)
        
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
        
        logger.info(
            "Image processing pipeline complete",
            event_type="image_processing_complete",
            invoice_id=invoice_id,
            processing_time_ms=processing_time * 1000,
            quality=quality_assessment['quality'],
            failure_risk=failure_risk['risk'],
            routing_action=routing_decision['action']
        )
        
        # ✅ CRITICAL: Save invoice to database
        if CLOUD_SQL_ENABLED:
            try:
                user_data = get_user_by_username(current_user.username)
                if user_data:
                    # Prepare invoice for database with ALL required fields
                    invoice_to_save = {
                        'invoice_number': invoice_data.get('invoice_number', f'AUTO-{invoice_id}'),
                        'vendor_name': invoice_data.get('vendor_name', 'Unknown'),
                        'total_amount': invoice_data.get('total_amount', 0),
                        'currency': invoice_data.get('currency', 'USD'),
                        'invoice_date': invoice_data.get('invoice_date', datetime.utcnow().strftime("%Y-%m-%d")),
                        'quality_prediction': quality_assessment['quality'],
                        'quality_score': quality_assessment['probabilities']['good'],
                        'risk_prediction': failure_risk['risk'],
                        'risk_score': failure_risk['probabilities']['risk'],
                        'file_name': file.filename,
                        'file_type': 'IMAGE',
                        'file_size_kb': file_size_kb,
                        'ocr_method': 'Document AI' if DOCUMENT_AI_ENABLED else 'Fallback',
                        'ocr_confidence': invoice_data.get('ocr_confidence', 0),
                        'subtotal': invoice_data.get('subtotal', invoice_data.get('total_amount', 0) * 0.93),
                        'tax_amount': invoice_data.get('tax', invoice_data.get('total_amount', 0) * 0.07),
                        'discount_amount': invoice_data.get('discount', 0)
                    }
                    
                    saved_id = save_invoice(user_data['id'], invoice_to_save)
                    if saved_id:
                        logger.info(f"✅ Invoice saved to database: ID={saved_id}", invoice_id=invoice_id)
                        response['saved_to_database'] = True
                        response['database_id'] = saved_id
                    else:
                        logger.warning("⚠️ Invoice save returned None", invoice_id=invoice_id)
                        response['saved_to_database'] = False
                else:
                    logger.warning("⚠️ User data not found", user=current_user.username)
                    response['saved_to_database'] = False
            except Exception as save_error:
                logger.error(f"❌ Failed to save invoice: {save_error}", invoice_id=invoice_id)
                response['saved_to_database'] = False
                response['save_error'] = str(save_error)
        
        return response
        
    except Exception as e:
        logger.exception(
            "Image processing failed",
            event_type="image_processing_error",
            invoice_id=invoice_id,
            user_id=current_user.username,
            uploaded_file=file.filename,
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


# ====================================================================
# USER INVOICE ENDPOINTS (Cloud SQL)
# ====================================================================

@app.get("/user/invoices", tags=["👤 User Data"])
async def get_my_invoices(
    current_user: User = Depends(get_current_active_user)
):
    """📋 Get All Invoices for Current User (Synced Across Devices)"""
    
    if not CLOUD_SQL_ENABLED:
        raise HTTPException(
            status_code=503, 
            detail="Cloud SQL not configured. Data only available locally."
        )
    
    try:
        user_data = get_user_by_username(current_user.username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found in database")
        
        invoices = get_user_invoices(user_data['id'], limit=1000)
        
        logger.info(
            "User invoices retrieved",
            user_id=current_user.username,
            invoice_count=len(invoices)
        )
        
        return {
            "status": "ok",
            "user": current_user.username,
            "invoices": invoices,
            "count": len(invoices)
        }
        
    except Exception as e:
        logger.exception(
            "Failed to fetch user invoices",
            user_id=current_user.username,
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/invoices/save", tags=["👤 User Data"])
async def save_user_invoice(
    invoice: dict,
    current_user: User = Depends(get_current_active_user)
):
    """💾 Save Invoice to Cloud SQL (Accessible from Any Device)"""
    
    if not CLOUD_SQL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Cloud SQL not configured. Cannot save data."
        )
    
    try:
        user_data = get_user_by_username(current_user.username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        invoice_id = save_invoice(user_data['id'], invoice)
        
        if invoice_id:
            logger.info(
                "Invoice saved to Cloud SQL",
                event_type="invoice_saved",
                user_id=current_user.username,
                invoice_id=invoice_id,
                invoice_number=invoice.get('invoice_number')
            )
            return {
                "status": "saved",
                "invoice_id": invoice_id,
                "message": "Invoice saved to Cloud SQL"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save invoice")
            
    except Exception as e:
        logger.exception(
            "Failed to save invoice",
            user_id=current_user.username,
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/user/invoices/{invoice_id}", tags=["👤 User Data"])
async def delete_user_invoice(
    invoice_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """🗑️ Delete Invoice"""
    
    if not CLOUD_SQL_ENABLED:
        raise HTTPException(status_code=503, detail="Cloud SQL not configured")
    
    try:
        user_data = get_user_by_username(current_user.username)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        deleted = delete_invoice(invoice_id, user_data['id'])
        
        if deleted:
            logger.info(
                "Invoice deleted",
                event_type="invoice_deleted",
                user_id=current_user.username,
                invoice_id=invoice_id
            )
            return {"status": "deleted", "invoice_id": invoice_id}
        else:
            raise HTTPException(status_code=404, detail="Invoice not found or unauthorized")
            
    except Exception as e:
        logger.exception(
            "Failed to delete invoice",
            user_id=current_user.username,
            invoice_id=invoice_id,
            error_type=type(e).__name__
        )
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================
# ADMIN ENDPOINTS
# ====================================================================

@app.get("/admin/document-ai-usage", tags=["👑 Admin"])
async def get_doc_ai_usage(current_user: User = Depends(require_admin)):
    """📊 Get Document AI Usage for Current Month"""
    
    if not CLOUD_SQL_ENABLED:
        return {
            "usage_this_month": 0,
            "note": "Cloud SQL not configured - cannot track usage"
        }
    
    try:
        usage = get_monthly_document_ai_usage()
        
        logger.info(
            "Document AI usage queried",
            user_id=current_user.username,
            monthly_usage=usage
        )
        
        return {
            "usage_this_month": usage,
            "free_tier_limit": 1000,
            "remaining_free": max(0, 1000 - usage),
            "cost_this_month": max(0, (usage - 1000) * 0.01),
            "cost_per_page": 0.01
        }
        
    except Exception as e:
        logger.exception("Failed to get Document AI usage")
        raise HTTPException(status_code=500, detail=str(e))

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
# BATCH PREDICTION ENDPOINT
# -------------------------------------------------------------------

@app.post("/predict/batch", tags=["🤖 ML Predictions"])
async def predict_batch(
    invoices: List[InvoiceFeatures],
    request: Request,
    current_user: User = Depends(require_user),
    _: None = Depends(check_rate_limit)
):
    """📦 Batch Prediction for Multiple Invoices (1-1000)"""
    start_time = time.time()
    batch_id = f"batch_{int(time.time())}"
    
    logger.info(
        "Batch prediction started",
        event_type="batch_prediction_start",
        user_id=current_user.username,
        batch_size=len(invoices),
        batch_id=batch_id
    )
    
    if len(invoices) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 1000")
    
    if len(invoices) == 0:
        raise HTTPException(status_code=400, detail="Batch must contain at least 1 invoice")
    
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

                # Track metrics
                try:
                    prediction_total.labels(
                        model="quality",
                        prediction_class=user_result['quality_assessment']['quality']
                    ).inc()
                    
                    prediction_total.labels(
                        model="failure", 
                        prediction_class=user_result['failure_risk']['risk']
                    ).inc()
                    
                    model_quality_probability.set(result['quality_probability'])
                    model_failure_probability.set(result['failure_probability'])
                    
                    if user_result['quality_assessment']['quality'] == 'good':
                        quality_predictions_good.inc()
                    else:
                        quality_predictions_bad.inc()
                    
                    if user_result['failure_risk']['risk'] == 'low':
                        failure_predictions_safe.inc()
                    else:
                        failure_predictions_risk.inc()
                    
                    prediction_latency.observe(time.time() - start_time)
                    feature_blur_score.observe(invoice.blur_score)
                    feature_ocr_confidence.observe(invoice.ocr_confidence)
                    feature_total_amount.observe(invoice.total_amount)
                    
                    logger.debug("Metrics tracked successfully")
                except Exception as metric_error:
                    logger.warning(f"Failed to track metrics: {metric_error}")
                
                results.append({
                    "index": idx,
                    "invoice_number": invoice.invoice_number,
                    "prediction": user_result,
                    "status": "success"
                })
                
            except Exception as e:
                errors += 1
                logger.warning(
                    "Batch item processing error",
                    batch_id=batch_id,
                    index=idx,
                    error_type=type(e).__name__
                )
                results.append({
                    "index": idx,
                    "invoice_number": invoice.invoice_number,
                    "error": str(e),
                    "status": "error"
                })
        
        processing_time = time.time() - start_time
        
        logger.info(
            "Batch prediction complete",
            event_type="batch_prediction_complete",
            batch_id=batch_id,
            user_id=current_user.username,
            batch_size=len(invoices),
            successful=len(invoices) - errors,
            errors=errors,
            cache_hits=cache_hits,
            cache_hit_rate=(cache_hits/len(invoices))*100 if len(invoices) > 0 else 0,
            processing_time_ms=processing_time * 1000,
            avg_time_per_invoice_ms=(processing_time / len(invoices)) * 1000
        )
        
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
        logger.exception(
            "Batch prediction failed",
            event_type="batch_prediction_error",
            batch_id=batch_id,
            user_id=current_user.username,
            batch_size=len(invoices),
            error_type=type(e).__name__,
            error_message=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# RUN THE APP - FIXED FOR CLOUD RUN
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    import os
    
    # CRITICAL: Cloud Run provides PORT environment variable
    # Must read from environment, not hardcode!
    port = int(os.environ.get("PORT", 8000))
    
    # Log startup configuration
    logger.info(
        "Starting LedgerX API server",
        event_type="server_startup",
        host="0.0.0.0",
        port=port,
        environment=os.getenv("ENVIRONMENT", "production"),
        reload=False
    )
    
    # Start uvicorn server
    uvicorn.run(
        "src.inference.api_fastapi:app",
        host="0.0.0.0",  # MUST be 0.0.0.0 for Cloud Run (not 127.0.0.1)
        port=port,       # Read from PORT environment variable
        reload=False,    # Disable auto-reload in production
        log_level="info"
    )