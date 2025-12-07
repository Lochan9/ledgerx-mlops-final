"""
LedgerX - FastAPI Cloud Logging Middleware
===========================================

Automatic request/response logging middleware that integrates
with Google Cloud Logging for comprehensive API observability.
"""

import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .cloud_logging import get_logger


class CloudLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic request/response logging to Cloud Logging
    
    Captures:
    - Request method, path, headers
    - Response status code, latency
    - User information (if authenticated)
    - Errors and exceptions
    """
    
    def __init__(
        self,
        app: ASGIApp,
        logger_name: str = "ledgerx_api",
        log_request_body: bool = False,
        log_response_body: bool = False
    ):
        """
        Initialize middleware
        
        Args:
            app: FastAPI application
            logger_name: Logger name for Cloud Logging
            log_request_body: Whether to log request bodies (be careful with PII)
            log_response_body: Whether to log response bodies
        """
        super().__init__(app)
        self.logger = get_logger(name=logger_name)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details"""
        # Start timer
        start_time = time.time()
        
        # Extract request details
        request_id = request.headers.get("X-Request-ID", f"req_{int(start_time * 1000)}")
        user_agent = request.headers.get("User-Agent", "unknown")
        client_ip = self._get_client_ip(request)
        
        # Extract user info if available
        user_id = None
        user_email = None
        try:
            if hasattr(request.state, "user"):
                user_id = getattr(request.state.user, "username", None)
                user_email = getattr(request.state.user, "email", None)
        except:
            pass
        
        # Log request start
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            event_type="request_start",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=str(request.query_params),
            user_id=user_id,
            user_email=user_email,
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log successful request
            self.logger.log_api_request(
                method=request.method,
                endpoint=request.url.path,
                user_id=user_id,
                status_code=response.status_code,
                latency_ms=latency_ms,
                request_id=request_id,
                client_ip=client_ip,
                user_agent=user_agent
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Log error
            self.logger.error(
                f"Request failed: {request.method} {request.url.path}",
                event_type="request_error",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                user_id=user_id,
                error_type=type(e).__name__,
                error_message=str(e),
                latency_ms=latency_ms,
                client_ip=client_ip
            )
            
            # Re-raise exception
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request (handles proxies)"""
        # Check for forwarded headers (Cloud Run, load balancers)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        if request.client:
            return request.client.host
        
        return "unknown"


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """
    Additional middleware for performance monitoring
    Logs slow requests and performance metrics
    """
    
    def __init__(
        self,
        app: ASGIApp,
        slow_request_threshold_ms: float = 1000.0
    ):
        """
        Initialize performance middleware
        
        Args:
            app: FastAPI application
            slow_request_threshold_ms: Threshold for slow request warning (milliseconds)
        """
        super().__init__(app)
        self.logger = get_logger(name="ledgerx_performance")
        self.slow_threshold_ms = slow_request_threshold_ms
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log slow requests
            if latency_ms > self.slow_threshold_ms:
                self.logger.warning(
                    f"Slow request detected: {request.method} {request.url.path}",
                    event_type="slow_request",
                    method=request.method,
                    path=request.url.path,
                    latency_ms=latency_ms,
                    threshold_ms=self.slow_threshold_ms
                )
            
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            raise


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured error logging
    Captures all exceptions with full context
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = get_logger(name="ledgerx_errors")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Catch and log errors"""
        try:
            return await call_next(request)
        except Exception as e:
            # Log exception with full context
            self.logger.exception(
                f"Unhandled exception: {type(e).__name__}",
                event_type="unhandled_exception",
                exception_type=type(e).__name__,
                exception_message=str(e),
                method=request.method,
                path=request.url.path,
                query_params=str(request.query_params)
            )
            raise


def setup_logging_middleware(
    app,
    logger_name: str = "ledgerx_api",
    enable_performance_monitoring: bool = True,
    slow_request_threshold_ms: float = 1000.0
):
    """
    Setup all logging middleware for FastAPI app
    
    Args:
        app: FastAPI application
        logger_name: Logger name for Cloud Logging
        enable_performance_monitoring: Enable performance monitoring middleware
        slow_request_threshold_ms: Threshold for slow request warnings
    
    Example:
        from fastapi import FastAPI
        from src.utils.logging_middleware import setup_logging_middleware
        
        app = FastAPI()
        setup_logging_middleware(app)
    """
    # Add error logging first (innermost)
    app.add_middleware(ErrorLoggingMiddleware)
    
    # Add performance monitoring
    if enable_performance_monitoring:
        app.add_middleware(
            PerformanceLoggingMiddleware,
            slow_request_threshold_ms=slow_request_threshold_ms
        )
    
    # Add request/response logging (outermost)
    app.add_middleware(
        CloudLoggingMiddleware,
        logger_name=logger_name
    )


if __name__ == "__main__":
    """Test middleware setup"""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    
    app = FastAPI(title="Test App")
    
    # Setup logging middleware
    setup_logging_middleware(app)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "Test endpoint"}
    
    @app.get("/slow")
    async def slow_endpoint():
        import asyncio
        await asyncio.sleep(2)  # Simulate slow request
        return {"message": "Slow endpoint"}
    
    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")
    
    print("Test FastAPI app with logging middleware created")
    print("Run with: uvicorn test_middleware:app --reload")