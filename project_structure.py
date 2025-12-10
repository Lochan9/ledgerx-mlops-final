import os
import sys
from pathlib import Path

# ANSI Colors
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RED = '\033[91m'
END = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}{text}{END}")
    print(f"{BLUE}{'='*60}{END}\n")

def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return True

def create_file(path, content=""):
    """Create file with optional content"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(path).exists():
        with open(path, 'w') as f:
            f.write(content)
        return True
    return False

print_header("🚀 LEDGERX PROJECT STRUCTURE CREATOR")

# ============================================================================
# STEP 1: Create Directories
# ============================================================================

print(f"{YELLOW}[STEP 1/4]{END} Creating directories...")

directories = [
    'src',
    'src/utils',
    'src/inference',
    'src/training',
    'tests',
    'config',
    'logs',
    'data/raw',
    'data/processed',
    'data/cache',
    'models',
    '.github/workflows',
    'scripts',
]

for dir_path in directories:
    create_directory(dir_path)
    print(f"  ✅ {dir_path}/")

# ============================================================================
# STEP 2: Create __init__.py Files
# ============================================================================

print(f"\n{YELLOW}[STEP 2/4]{END} Creating __init__.py files...")

init_files = {
    'src/__init__.py': '''"""LedgerX - Enterprise Invoice Intelligence Platform"""

__version__ = "1.0.0"
__author__ = "LedgerX Team"
__description__ = "AI-powered invoice processing with MLOps"
''',
    'src/utils/__init__.py': '''"""Utilities module"""
''',
    'src/inference/__init__.py': '''"""Inference module"""
''',
    'src/training/__init__.py': '''"""Training module"""
''',
    'tests/__init__.py': '''"""Test suite for LedgerX"""
''',
}

for file_path, content in init_files.items():
    if create_file(file_path, content):
        print(f"  ✅ {file_path}")

# ============================================================================
# STEP 3: Create Configuration Files
# ============================================================================

print(f"\n{YELLOW}[STEP 3/4]{END} Creating configuration files...")

config_yaml = '''app:
  name: "LedgerX"
  version: "1.0.0"
  environment: "development"
  debug: true

api:
  host: "0.0.0.0"
  port: 8000
  reload: true
  workers: 1
  timeout: 300

database:
  host: "localhost"
  port: 5432
  name: "ledgerx"
  user: "postgres"
  pool_size: 10

models:
  quality_model: "models/quality_catboost.cbm"
  failure_model: "models/failure_catboost.cbm"
  encoders: "models/encoders.pkl"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  cache_dir: "data/cache"

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "ledgerx-training"

security:
  jwt_secret: "your-secret-key-change-this"
  jwt_algorithm: "HS256"
  access_token_expire_minutes: 30
'''

env_example = '''ENVIRONMENT=development
DEBUG=true
PORT=8000

GCP_PROJECT_ID=ledgerx-mlops
GCP_REGION=us-central1

DB_HOST=localhost
DB_PORT=5432
DB_NAME=ledgerx
DB_USER=postgres
DB_PASSWORD=postgres

JWT_SECRET_KEY=your-secret-key-change-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

MLFLOW_TRACKING_URI=http://localhost:5000
'''

gitignore = '''__pycache__/
*.pyc
*.pyo
.pytest_cache/
.coverage
dist/
build/
*.egg-info/
.venv/
venv/
env/
.git/
.env
*.log
.DS_Store
models/*.cbm
data/raw/*
data/cache/*
mlruns/
'''

pytest_ini = '''[pytest]
testpaths = tests
python_files = test_*.py
addopts = -v --tb=short
'''

config_files = {
    'config/config.yaml': config_yaml,
    'config/.env.example': env_example,
    '.gitignore': gitignore,
    'pytest.ini': pytest_ini,
}

for file_path, content in config_files.items():
    if create_file(file_path, content):
        print(f"  ✅ {file_path}")

# ============================================================================
# STEP 4: Create .gitkeep Files
# ============================================================================

print(f"\n{YELLOW}[STEP 4/4]{END} Creating .gitkeep files...")

gitkeep_dirs = [
    'data/raw/.gitkeep',
    'data/cache/.gitkeep',
    'logs/.gitkeep',
    'models/.gitkeep',
]

for file_path in gitkeep_dirs:
    if create_file(file_path):
        print(f"  ✅ {file_path}")

# ============================================================================
# Summary
# ============================================================================

print_header("✅ PROJECT STRUCTURE CREATED SUCCESSFULLY!")

print(f"{GREEN}Summary:{END}")
print(f"  ✅ {len(directories)} directories created")
print(f"  ✅ {len(init_files)} __init__.py files created")
print(f"  ✅ {len(config_files)} configuration files created")
print(f"  ✅ {len(gitkeep_dirs)} .gitkeep files created")

print(f"\n{YELLOW}Next Steps:{END}")
print(f"  1. Copy main source files from artifacts:")
print(f"     - src/data_pipeline.py (from ledgerx_data_pipeline_fixed)")
print(f"     - src/train_models.py (from ledgerx_training_pipeline_fixed)")
print(f"     - src/api_inference.py (from ledgerx_api_inference)")
print(f"     - src/utils/logging_config.py")
print(f"     - src/utils/config.py")
print(f"     - src/inference/predictor.py")
print(f"     - tests/test_*.py files")
print(f"")
print(f"  2. Copy Docker files:")
print(f"     - Dockerfile (from ledgerx_docker_files)")
print(f"     - Dockerfile.cloudrun")
print(f"     - docker-compose.yml")
print(f"     - .dockerignore")
print(f"")
print(f"  3. Copy other files:")
print(f"     - setup.py")
print(f"     - README.md")
print(f"     - requirements*.txt")
print(f"     - Makefile")
print(f"     - .github/workflows/*.yml")
print(f"")
print(f"  4. Run setup:")
print(f"     python verify_installation.py")
print(f"     python src/data_pipeline.py")
print(f"     python src/train_models.py")

print(f"\n{GREEN}You're ready to proceed with copying the source files!{END}\n")