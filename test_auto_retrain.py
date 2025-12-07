"""
Wrapper script to test auto-retrain trigger
Fixes the relative import issue
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Now import and run
from monitoring.auto_retrain_trigger import main
import logging

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    main()
