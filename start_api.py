"""
LedgerX - Start API Server
===========================

Simple script to start the FastAPI server for the LedgerX platform.

Usage:
    python start_api.py
    
Then open website/index.html in your browser.
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 Starting LedgerX API Server")
    print("=" * 60)
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent
    api_file = project_root / "src" / "inference" / "api_fastapi.py"
    
    if not api_file.exists():
        print("❌ Error: api_fastapi.py not found!")
        print(f"   Expected at: {api_file}")
        sys.exit(1)
    
    print(f"✓ Found API file: {api_file}")
    print()
    print("Starting server on http://localhost:8000")
    print()
    print("📝 Test credentials:")
    print("   - admin / admin123 (full access)")
    print("   - john_doe / password123 (user access)")
    print()
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print()
    print("🌐 Open website/index.html in your browser to access the dashboard")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.inference.api_fastapi:app",
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\nMake sure you have installed all requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()