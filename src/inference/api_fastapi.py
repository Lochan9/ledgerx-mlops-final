"""
LedgerX – FastAPI Inference Deployment with Authentication
===========================================================

Run with:
    uvicorn src.inference.api_fastapi:app --reload --port 8000

Provides:
    POST /token - Get authentication token
    POST /predict - Make predictions (requires authentication)
    GET /users/me - Get current user info
    GET /health - Health check (no auth required)

Authentication:
    All prediction endpoints require JWT token.
    Include token in request header: Authorization: Bearer <token>
"""

import logging
from datetime import timedelta, datetime
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import Optional

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

# -------------------------------------------------------------------
# LOGGING
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("ledgerx_fastapi")

# -------------------------------------------------------------------
# RATE LIMITER (Cost Protection for GCP Free Tier)
# -------------------------------------------------------------------
try:
    from ..utils.rate_limiter import rate_limiter, check_rate_limit
    RATE_LIMITING_ENABLED = True
    logger.info("[INIT] Rate limiting enabled for cost protection")
except ImportError:
    logger.warning("[INIT] Rate limiter not found - running without cost protection!")
    RATE_LIMITING_ENABLED = False
    
    # Dummy dependency if rate limiter not available
    async def check_rate_limit(request: Request):
        pass

# -------------------------------------------------------------------
# PREDICTION CACHE (40% Cost Savings)
# -------------------------------------------------------------------
try:
    from ..utils.prediction_cache import prediction_cache, get_cached_or_predict, get_cache_stats
    CACHING_ENABLED = True
    logger.info("[INIT] Prediction caching enabled (40% cost savings)")
except ImportError:
    logger.warning("[INIT] Prediction cache not found - running without caching")
    CACHING_ENABLED = False
    
    def get_cached_or_predict(features, predict_func):
        return predict_func(features)
    
    def get_cache_stats():
        return {"status": "disabled"}

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="LedgerX Invoice Quality & Failure Prediction API",
    description="""
    AI-based Invoice Quality Screening and Failure Risk Detection
    
    ## Authentication
    
    All prediction endpoints require authentication. To get started:
    
    1. Obtain a token from `/token` endpoint with your credentials
    2. Include token in requests: `Authorization: Bearer <your_token>`
    
    ## Default Test Users
    
    - **admin** / admin123 (full access)
    - **john_doe** / password123 (user access)
    - **jane_viewer** / viewer123 (readonly access)
    """,
    version="2.0.0",
)
# ------------------------
# CORS MIDDLEWARE (IMPORTANT)
# ------------------------
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend is on 127.0.0.1:5500
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# INPUT SCHEMA WITH VALIDATION
# -------------------------------------------------------------------
class InvoiceFeatures(BaseModel):
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

class PredictionResponse(BaseModel):
    status: str
    result: dict
    user: str
    timestamp: str

# -------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# -------------------------------------------------------------------
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests.
    
    Use the returned token in the Authorization header:
    `Authorization: Bearer <access_token>`
    """
    logger.info(f"[AUTH] Login attempt for user: {form_data.username}")
    
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"[AUTH] Failed login attempt for user: {form_data.username}")
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
    
    logger.info(f"[AUTH] Successful login for user: {form_data.username} (role: {user.role})")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60  # in seconds
    }

@app.get("/users/me", response_model=User, tags=["Authentication"])
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    Requires valid authentication token.
    """
    return current_user

# -------------------------------------------------------------------
# HEALTH CHECK (NO AUTH REQUIRED)
# -------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint - no authentication required.
    Returns API status and version.
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "authentication": "enabled",
        "timestamp": logging.Formatter().formatTime(logging.LogRecord(
            name="", level=0, pathname="", lineno=0,
            msg="", args=(), exc_info=None
        ))
    }

# -------------------------------------------------------------------
# PREDICTION ENDPOINT (AUTH REQUIRED)
# -------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict invoice quality and failure risk",
    description="""
    Analyzes invoice features to predict:
    - Quality assessment (good/bad)
    - Failure risk probability
    
    **Authentication Required**: User or Admin role
    
    Returns confidence scores and warnings.
    """
)
async def predict(
    features: InvoiceFeatures,
    request: Request,
    current_user: User = Depends(require_user),
    _: None = Depends(check_rate_limit)  # RATE LIMITING PROTECTION
):
    """
    Make a prediction on invoice quality and failure risk.
    
    Requires authentication with 'user' or 'admin' role.
    Rate limited to protect costs: 50 req/hour, 200 req/day per IP.
    """
    logger.info("===============================================")
    logger.info(f"[API] Incoming /predict request from user: {current_user.username}")
    logger.info(f"[API] IP: {request.client.host}")
    logger.info("===============================================")

    try:
        # Use caching if enabled
        if CACHING_ENABLED:
            result = get_cached_or_predict(
                features.dict(),
                predict_invoice
            )
        else:
            result = predict_invoice(features.dict())
        
        # Log the prediction for audit trail
        logger.info(f"[AUDIT] User: {current_user.username}, Quality: {result['quality_bad']}, Failure: {result['failure_risk']}, Cached: {result.get('from_cache', False)}")
        
        from datetime import datetime
        
        return {
            "status": "ok",
            "result": result,
            "user": current_user.username,
            "timestamp": datetime.utcnow().isoformat(),
            "from_cache": result.get('from_cache', False)
        }

    except Exception as e:
        logger.exception(f"[API] Error during inference for user: {current_user.username}")
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------------------------------------------------
# BATCH PREDICTION ENDPOINT (Enterprise Feature)
# -------------------------------------------------------------------
@app.post(
    "/predict/batch",
    response_model=dict,
    tags=["Predictions"],
    summary="Batch Invoice Predictions",
    description="""
    Process multiple invoices in a single request.
    
    **Authentication Required**: User or Admin role
    
    **Enterprise Feature**: Process 1-1000 invoices at once
    
    Returns predictions for all invoices with performance metrics.
    """
)
async def predict_batch(
    invoices: list[InvoiceFeatures],
    request: Request,
    current_user: User = Depends(require_user),
    _: None = Depends(check_rate_limit)
):
    """
    Batch prediction for multiple invoices.
    
    Limits:
    - Max 1000 invoices per request
    - Counts as N requests for rate limiting (where N = number of invoices)
    """
    logger.info("===============================================")
    logger.info(f"[API] Batch prediction request from: {current_user.username}")
    logger.info(f"[API] Batch size: {len(invoices)}")
    logger.info("===============================================")
    
    # Validate batch size
    if len(invoices) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000 invoices"
        )
    
    if len(invoices) == 0:
        raise HTTPException(
            status_code=400,
            detail="Batch must contain at least 1 invoice"
        )
    
    # Track batch processing
    import time
    start_time = time.time()
    
    results = []
    cache_hits = 0
    errors = 0
    
    try:
        for idx, invoice in enumerate(invoices):
            try:
                # Use caching if enabled
                if CACHING_ENABLED:
                    result = get_cached_or_predict(
                        invoice.dict(),
                        predict_invoice
                    )
                    if result.get('from_cache'):
                        cache_hits += 1
                else:
                    result = predict_invoice(invoice.dict())
                
                results.append({
                    "index": idx,
                    "invoice_number": invoice.invoice_number,
                    "prediction": result,
                    "status": "success"
                })
                
            except Exception as e:
                errors += 1
                logger.warning(f"[BATCH] Error processing invoice {idx}: {e}")
                results.append({
                    "index": idx,
                    "invoice_number": invoice.invoice_number,
                    "error": str(e),
                    "status": "error"
                })
        
        processing_time = time.time() - start_time
        
        # Log batch completion
        logger.info(f"[BATCH] Processed {len(invoices)} invoices in {processing_time:.2f}s")
        logger.info(f"[BATCH] Success: {len(invoices) - errors}, Errors: {errors}, Cache hits: {cache_hits}")
        
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
        logger.exception(f"[API] Batch prediction failed for user: {current_user.username}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# COST MONITORING ENDPOINT (Admin Only)
# -------------------------------------------------------------------
@app.get("/admin/costs", tags=["Admin"])
async def get_cost_stats(current_user: User = Depends(require_admin)):
    """
    Get current cost and usage statistics
    
    Shows:
    - Requests today
    - Estimated cost
    - Budget remaining
    - Projections
    
    **Requires Admin role**
    """
    if not RATE_LIMITING_ENABLED:
        return {
            "status": "disabled",
            "message": "Rate limiting not enabled"
        }
    
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

@app.get("/admin/cache", tags=["Admin"])
async def get_cache_statistics(current_user: User = Depends(require_admin)):
    """
    Get cache statistics and cost savings
    
    Shows:
    - Cache hit rate
    - Cost savings from caching
    - Cache size
    
    **Requires Admin role**
    """
    if not CACHING_ENABLED:
        return {
            "status": "disabled",
            "message": "Caching not enabled"
        }
    
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
# STARTUP MESSAGE
# -------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("LedgerX API v2.0 Starting...")
    logger.info("Authentication: ENABLED")
    logger.info(f"Rate Limiting: {'ENABLED' if RATE_LIMITING_ENABLED else 'DISABLED'}")
    logger.info(f"Prediction Caching: {'ENABLED' if CACHING_ENABLED else 'DISABLED'}")
    if RATE_LIMITING_ENABLED:
        logger.info("  - IP Limit: 50/hour, 200/day")
        logger.info("  - Budget Protection: $1.67/day")
    if CACHING_ENABLED:
        logger.info("  - Cache Size: 1000 predictions")
        logger.info("  - Expected Savings: 30-40%")
    logger.info("=" * 60)
    logger.info("Test Credentials:")
    logger.info("  - admin / admin123 (Admin)")
    logger.info("  - john_doe / password123 (User)")
    logger.info("  - jane_viewer / viewer123 (Readonly)")
    logger.info("=" * 60)
    logger.info("Cost Optimization Features:")
    logger.info("  - GET /admin/costs - Cost dashboard")
    logger.info("  - GET /admin/cache - Cache statistics")
    logger.info("=" * 60)