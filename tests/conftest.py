import pytest
import os
from pathlib import Path

@pytest.fixture
def data_dir():
    """Fixture for data directory"""
    return Path("data/processed")

@pytest.fixture
def models_dir():
    """Fixture for models directory"""
    return Path("models")

@pytest.fixture
def sample_invoice():
    """Fixture for sample invoice data"""
    return {
        'blur_score': 45.2,
        'contrast_score': 28.5,
        'ocr_confidence': 0.87,
        'file_size_kb': 245.3,
        'vendor_name': 'Test Corp',
        'vendor_freq': 0.03,
        'total_amount': 1250.00,
        'currency': 'USD',
        'has_tax': True,
        'has_discount': False,
        'line_items_count': 5
    }