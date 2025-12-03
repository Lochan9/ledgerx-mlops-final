"""
Run Monitoring Check - Clean Runner Script
Place in project root: run_monitoring_check.py
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run
if __name__ == "__main__":
    from src.monitoring.auto_retrain_trigger import main
    main()