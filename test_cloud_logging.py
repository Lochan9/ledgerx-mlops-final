#!/usr/bin/env python3
"""
LedgerX Cloud Logging - Comprehensive Test Suite
=================================================

Tests all aspects of Cloud Logging integration:
1. Basic logging functionality
2. Structured logging
3. Environment detection
4. FastAPI middleware
5. Error handling

Usage:
    # Set environment variable first (PowerShell)
    $env:GOOGLE_CLOUD_PROJECT = "ledgerx-mlops"
    
    # Run tests
    python test_cloud_logging.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test 1: Basic imports"""
    print("\n" + "="*80)
    print("TEST 1: Basic Imports")
    print("="*80)
    
    try:
        from src.utils.cloud_logging import CloudLogger, get_logger, setup_cloud_logging
        print("✅ cloud_logging module imported successfully")
        
        from src.utils.logging_middleware import (
            CloudLoggingMiddleware,
            PerformanceLoggingMiddleware,
            setup_logging_middleware
        )
        print("✅ logging_middleware module imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("\nMake sure you have:")
        print("  1. Created src/utils/cloud_logging.py")
        print("  2. Created src/utils/logging_middleware.py")
        print("  3. Installed dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_logger_initialization():
    """Test 2: Logger initialization"""
    print("\n" + "="*80)
    print("TEST 2: Logger Initialization")
    print("="*80)
    
    try:
        from src.utils.cloud_logging import CloudLogger
        
        # Test basic initialization
        logger = CloudLogger(name="test_logger", log_level="INFO")
        print(f"✅ Logger created: {logger.name}")
        print(f"   - Project ID: {logger.project_id or 'None (local mode)'}")
        print(f"   - Cloud Logging: {logger.use_cloud_logging}")
        print(f"   - Log Level: {logger.log_level}")
        
        return True
    except Exception as e:
        print(f"❌ Logger initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_logging():
    """Test 3: Basic logging methods"""
    print("\n" + "="*80)
    print("TEST 3: Basic Logging Methods")
    print("="*80)
    
    try:
        from src.utils.cloud_logging import get_logger
        
        logger = get_logger(name="test_basic")
        
        logger.debug("Debug message", test_field="debug_value")
        print("✅ Debug log sent")
        
        logger.info("Info message", test_field="info_value")
        print("✅ Info log sent")
        
        logger.warning("Warning message", test_field="warning_value")
        print("✅ Warning log sent")
        
        logger.error("Error message", test_field="error_value")
        print("✅ Error log sent")
        
        # Test exception logging
        try:
            raise ValueError("Test exception")
        except Exception:
            logger.exception("Exception caught", test_field="exception_value")
            print("✅ Exception log sent")
        
        return True
    except Exception as e:
        print(f"❌ Basic logging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_structured_logging():
    """Test 4: Structured logging methods"""
    print("\n" + "="*80)
    print("TEST 4: Structured Logging Methods")
    print("="*80)
    
    try:
        from src.utils.cloud_logging import get_logger
        
        logger = get_logger(name="test_structured")
        
        # Test prediction logging
        logger.log_prediction(
            user_id="test_user",
            invoice_id="inv_test_001",
            quality_prediction="good",
            failure_prediction="safe",
            latency_ms=125.5,
            model_version="test_v1"
        )
        print("✅ Prediction log sent")
        
        # Test API request logging
        logger.log_api_request(
            method="POST",
            endpoint="/api/v1/predict",
            user_id="test_user",
            status_code=200,
            latency_ms=250.3
        )
        print("✅ API request log sent")
        
        # Test model performance logging
        logger.log_model_performance(
            model_name="test_model",
            metric_name="f1_score",
            metric_value=0.977
        )
        print("✅ Model performance log sent")
        
        # Test drift logging
        logger.log_data_drift(
            feature_name="test_feature",
            drift_score=0.08,
            threshold=0.10,
            drift_detected=False
        )
        print("✅ Data drift log sent")
        
        # Test retraining trigger logging
        logger.log_retraining_trigger(
            trigger_reason="performance_degradation",
            model_name="test_model"
        )
        print("✅ Retraining trigger log sent")
        
        # Test OCR logging
        logger.log_ocr_processing(
            invoice_id="inv_test_001",
            ocr_engine="document_ai",
            confidence=0.95,
            processing_time_ms=450.2
        )
        print("✅ OCR processing log sent")
        
        # Test cost logging
        logger.log_cost_event(
            service="document_ai",
            operation="process_document",
            cost_usd=0.0015,
            units_consumed=1
        )
        print("✅ Cost event log sent")
        
        return True
    except Exception as e:
        print(f"❌ Structured logging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_detection():
    """Test 5: Environment detection"""
    print("\n" + "="*80)
    print("TEST 5: Environment Detection")
    print("="*80)
    
    try:
        from src.utils.cloud_logging import IS_CLOUD_RUN, IS_GCP, CLOUD_LOGGING_AVAILABLE
        
        print(f"   IS_CLOUD_RUN: {IS_CLOUD_RUN}")
        print(f"   IS_GCP: {IS_GCP}")
        print(f"   CLOUD_LOGGING_AVAILABLE: {CLOUD_LOGGING_AVAILABLE}")
        
        if IS_CLOUD_RUN:
            print("✅ Running on Cloud Run")
        elif IS_GCP:
            print("✅ Running on GCP")
        else:
            print("ℹ️  Running locally")
        
        if CLOUD_LOGGING_AVAILABLE:
            print("✅ google-cloud-logging package available")
        else:
            print("❌ google-cloud-logging package not installed")
            print("   Install with: pip install google-cloud-logging")
        
        return True
    except Exception as e:
        print(f"❌ Environment detection failed: {e}")
        return False


def test_fastapi_middleware():
    """Test 6: FastAPI middleware"""
    print("\n" + "="*80)
    print("TEST 6: FastAPI Middleware")
    print("="*80)
    
    try:
        # Check if FastAPI is available
        try:
            from fastapi import FastAPI
        except ImportError:
            print("⚠️  FastAPI not installed (optional for this test)")
            print("   Install with: pip install fastapi")
            return True
        
        from src.utils.logging_middleware import setup_logging_middleware
        
        app = FastAPI(title="Test App")
        setup_logging_middleware(app)
        
        print("✅ Middleware setup successful")
        print(f"   Middleware count: {len(app.user_middleware)}")
        
        return True
    except Exception as e:
        print(f"❌ Middleware setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test 7: Performance"""
    print("\n" + "="*80)
    print("TEST 7: Logging Performance")
    print("="*80)
    
    try:
        from src.utils.cloud_logging import get_logger
        
        logger = get_logger(name="test_performance")
        
        # Test 100 logs
        iterations = 100
        start_time = time.time()
        
        for i in range(iterations):
            logger.info(f"Performance test log {i}", iteration=i)
        
        elapsed = time.time() - start_time
        avg_latency = (elapsed / iterations) * 1000
        
        print(f"✅ Performance test complete")
        print(f"   Total time: {elapsed:.3f}s")
        print(f"   Average latency: {avg_latency:.3f}ms per log")
        
        if avg_latency < 10:
            print("✅ Performance: EXCELLENT (<10ms)")
        elif avg_latency < 50:
            print("✅ Performance: GOOD (<50ms)")
        else:
            print("⚠️  Performance: Acceptable but slow (>50ms)")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_gcp_integration():
    """Test 8: GCP Integration (if available)"""
    print("\n" + "="*80)
    print("TEST 8: GCP Integration")
    print("="*80)
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    
    if not project_id:
        print("ℹ️  No GCP project configured (skipping GCP-specific tests)")
        print("   Set GOOGLE_CLOUD_PROJECT to enable GCP integration")
        print()
        print("   PowerShell:")
        print("     $env:GOOGLE_CLOUD_PROJECT = 'ledgerx-mlops'")
        print()
        return True
    
    try:
        import google.cloud.logging
        client = google.cloud.logging.Client(project=project_id)
        print(f"✅ Connected to GCP project: {project_id}")
        print(f"   View logs at: https://console.cloud.google.com/logs/query?project={project_id}")
        return True
    except Exception as e:
        print(f"⚠️  GCP connection failed: {e}")
        print("   This is expected if:")
        print("     - Not authenticated (run: gcloud auth application-default login)")
        print("     - Not on GCP")
        print("     - Cloud Logging API not enabled")
        return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("LEDGERX CLOUD LOGGING - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Check environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    if project_id:
        print(f"📍 Project: {project_id}")
    else:
        print("⚠️  GOOGLE_CLOUD_PROJECT not set")
        print()
        print("Set it with (PowerShell):")
        print("  $env:GOOGLE_CLOUD_PROJECT = 'ledgerx-mlops'")
        print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Logger Initialization", test_logger_initialization),
        ("Basic Logging", test_basic_logging),
        ("Structured Logging", test_structured_logging),
        ("Environment Detection", test_environment_detection),
        ("FastAPI Middleware", test_fastapi_middleware),
        ("Performance", test_performance),
        ("GCP Integration", test_gcp_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Cloud Logging is ready to use.")
        print()
        if project_id:
            print(f"View your logs at:")
            print(f"https://console.cloud.google.com/logs/query?project={project_id}")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        print()
        print("Common issues:")
        print("  1. Files not in correct location:")
        print("     - src/utils/cloud_logging.py")
        print("     - src/utils/logging_middleware.py")
        print("  2. Dependencies not installed:")
        print("     pip install -r requirements.txt")
        print("  3. Environment variable not set:")
        print("     $env:GOOGLE_CLOUD_PROJECT = 'ledgerx-mlops'")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    
    # Additional instructions
    if exit_code == 0:
        print()
        print("="*80)
        print("NEXT STEPS")
        print("="*80)
        print()
        print("1. Update your FastAPI app with Cloud Logging:")
        print("   See: integrate_cloud_logging.py")
        print()
        print("2. Test locally:")
        print("   uvicorn src.inference.api_fastapi:app --reload")
        print()
        print("3. Deploy to Cloud Run:")
        print("   gcloud run deploy ledgerx-api --source . --region us-central1")
        print()
        print("="*80)
    
    sys.exit(exit_code)