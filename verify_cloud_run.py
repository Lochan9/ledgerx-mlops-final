"""
LedgerX Cloud Run Integration Verification
Tests production deployment on GCP
"""
import requests
from datetime import datetime

BACKEND_URL = "https://ledgerx-api-671429123152.us-central1.run.app"
FRONTEND_URL = "https://storage.googleapis.com/ledgerx-dashboard-671429123152/index_v3.html"

print("""
╔══════════════════════════════════════════════════════════╗
║     LedgerX Cloud Run Deployment Verification            ║
╚══════════════════════════════════════════════════════════╝
""")

print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🌐 API: {BACKEND_URL}")
print()

# Test 1: Backend Health
print("1️⃣ Testing Cloud Run Health...")
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=10)
    health = response.json()
    print(f"   ✅ Service: {health['service']} v{health['version']}")
    print(f"   ✅ Cloud Logging: {health['cloud_logging']}")
    print(f"   ✅ Document AI: {health['services']['document_ai']}")
    print(f"   ✅ Cloud SQL: {health['services']['cloud_sql']}")
    print(f"   ✅ Rate Limiting: {health['services']['rate_limiting']}")
    print(f"   ✅ Caching: {health['services']['caching']}")
except Exception as e:
    print(f"   ❌ Backend Error: {e}")
    exit(1)

# Test 2: Database Authentication
print("\n2️⃣ Testing Database Authentication...")
try:
    response = requests.post(
        f"{BACKEND_URL}/token",
        data={"username": "admin", "password": "admin123"},
        timeout=10
    )
    
    print(f"   Status Code: {response.status_code}")
    print(f"   Response: {response.text[:100]}")
    
    if response.status_code == 200:
        token_data = response.json()
        token = token_data['access_token']
        print(f"   ✅ Authentication: SUCCESS")
        print(f"   ✅ JWT Token: {token[:30]}...")
        
        # Test authenticated endpoint
        headers = {"Authorization": f"Bearer {token}"}
        user_response = requests.get(f"{BACKEND_URL}/users/me", headers=headers, timeout=10)
        
        if user_response.status_code == 200:
            user = user_response.json()
            print(f"   ✅ User Info: {user['username']} ({user['role']})")
        else:
            print(f"   ❌ User endpoint failed: {user_response.status_code}")
            
    else:
        print(f"   ❌ Authentication FAILED")
        print(f"   Response: {response.json()}")
        
except Exception as e:
    print(f"   ❌ Auth Error: {e}")

# Test 3: CORS
print("\n3️⃣ Testing CORS Configuration...")
try:
    response = requests.options(
        f"{BACKEND_URL}/upload/image",
        headers={
            "Origin": "https://storage.googleapis.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"
        },
        timeout=5
    )
    
    if 'access-control-allow-origin' in response.headers:
        print(f"   ✅ CORS: Configured")
        print(f"   ✅ Allowed Origin: {response.headers.get('access-control-allow-origin')}")
    else:
        print(f"   ❌ CORS: Not configured")
        
except Exception as e:
    print(f"   ❌ CORS Error: {e}")

# Test 4: Check Revision
print("\n4️⃣ Checking Active Revision...")
import subprocess
result = subprocess.run(
    ["gcloud", "run", "services", "describe", "ledgerx-api", 
     "--region=us-central1", "--format=value(status.latestReadyRevisionName)"],
    capture_output=True, text=True
)
revision = result.stdout.strip()
print(f"   ✅ Active Revision: {revision}")

# Final Summary
print("\n" + "="*60)
print("📊 CLOUD RUN DEPLOYMENT STATUS")
print("="*60)

if response.status_code == 200:
    print("✅ PRODUCTION READY")
    print(f"✅ API: {BACKEND_URL}")
    print(f"✅ Website: {FRONTEND_URL}")
    print(f"✅ API Docs: {BACKEND_URL}/docs")
else:
    print("⚠️ PARTIAL DEPLOYMENT")
    print("✅ Health check: Working")
    print("❌ Authentication: Issue detected")
    print("\nCheck logs:")
    print(f"   gcloud logging read 'resource.labels.revision_name={revision}' --limit=20")

print("="*60)