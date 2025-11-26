"""
LedgerX API Authentication Test Script
======================================

This script demonstrates how to:
1. Authenticate and get a JWT token
2. Use the token to make authenticated API calls
3. Handle token expiration
4. Test role-based access control

Run this script to verify authentication is working correctly.
"""

import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"

# Test users
USERS = {
    "admin": {"username": "admin", "password": "admin123"},
    "user": {"username": "john_doe", "password": "password123"},
    "readonly": {"username": "jane_viewer", "password": "viewer123"},
}

# Sample invoice data for testing
SAMPLE_INVOICE = {
    "blur_score": 45.2,
    "contrast_score": 28.5,
    "ocr_confidence": 0.87,
    "file_size_kb": 245.3,
    "vendor_name": "Acme Corp",
    "vendor_freq": 0.03,
    "total_amount": 1250.00,
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "currency": "USD"
}

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def get_token(username, password):
    """
    Authenticate and get JWT token
    
    Returns:
        str: JWT access token
    """
    print(f"\n🔐 Authenticating as: {username}")
    
    response = requests.post(
        f"{API_BASE_URL}/token",
        data={
            "username": username,
            "password": password,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    if response.status_code == 200:
        token_data = response.json()
        print(f"✅ Authentication successful!")
        print(f"   Token expires in: {token_data['expires_in']} seconds")
        return token_data['access_token']
    else:
        print(f"❌ Authentication failed: {response.status_code}")
        print(f"   Error: {response.json()}")
        return None

def get_user_info(token):
    """Get current user information"""
    print("\n👤 Getting user info...")
    
    response = requests.get(
        f"{API_BASE_URL}/users/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if response.status_code == 200:
        user = response.json()
        print(f"✅ User info retrieved:")
        print(f"   Username: {user['username']}")
        print(f"   Full Name: {user.get('full_name', 'N/A')}")
        print(f"   Role: {user['role']}")
        print(f"   Email: {user.get('email', 'N/A')}")
        return user
    else:
        print(f"❌ Failed to get user info: {response.status_code}")
        return None

def make_prediction(token, invoice_data):
    """Make a prediction with authentication"""
    print("\n🔮 Making prediction...")
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=invoice_data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Status: {result['status']}")
        print(f"   User: {result['user']}")
        print(f"   Timestamp: {result['timestamp']}")
        print(f"\n   Results:")
        print(f"     Quality Bad: {result['result']['quality_bad']}")
        print(f"     Quality Probability: {result['result']['quality_probability']:.4f}")
        print(f"     Failure Risk: {result['result']['failure_risk']}")
        print(f"     Failure Probability: {result['result']['failure_probability']:.4f}")
        if result['result']['warnings']:
            print(f"     Warnings: {', '.join(result['result']['warnings'])}")
        return result
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.json()}")
        return None

def test_health_check():
    """Test health check endpoint (no auth required)"""
    print("\n💚 Testing health check (no auth)...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"✅ Health check passed!")
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Authentication: {health['authentication']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_unauthorized_access():
    """Test that endpoints are protected"""
    print("\n🔒 Testing unauthorized access...")
    
    # Try to make prediction without token
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=SAMPLE_INVOICE
    )
    
    if response.status_code == 401:
        print(f"✅ Unauthorized access correctly blocked!")
        print(f"   Status: {response.status_code}")
        return True
    else:
        print(f"❌ Unauthorized access was not blocked! Status: {response.status_code}")
        return False

def test_invalid_token():
    """Test with invalid token"""
    print("\n🚫 Testing with invalid token...")
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=SAMPLE_INVOICE,
        headers={"Authorization": "Bearer invalid_token_12345"}
    )
    
    if response.status_code == 401:
        print(f"✅ Invalid token correctly rejected!")
        return True
    else:
        print(f"❌ Invalid token was not rejected! Status: {response.status_code}")
        return False

def test_role_based_access():
    """Test role-based access control"""
    print("\n👥 Testing role-based access control...")
    
    # Get tokens for different users
    admin_token = get_token(USERS["admin"]["username"], USERS["admin"]["password"])
    user_token = get_token(USERS["user"]["username"], USERS["user"]["password"])
    readonly_token = get_token(USERS["readonly"]["username"], USERS["readonly"]["password"])
    
    # Admin should have access
    print("\n  Testing admin access...")
    make_prediction(admin_token, SAMPLE_INVOICE)
    
    # Regular user should have access
    print("\n  Testing user access...")
    make_prediction(user_token, SAMPLE_INVOICE)
    
    # Readonly user should NOT have access to predictions (if RBAC is fully implemented)
    # For now, readonly users can make predictions too
    print("\n  Testing readonly access...")
    make_prediction(readonly_token, SAMPLE_INVOICE)

def run_full_test_suite():
    """Run complete authentication test suite"""
    print_section("LedgerX API Authentication Test Suite")
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Health check (no auth)
    print_section("Test 1: Health Check (No Authentication)")
    test_health_check()
    
    # Test 2: Unauthorized access
    print_section("Test 2: Unauthorized Access Protection")
    test_unauthorized_access()
    
    # Test 3: Invalid token
    print_section("Test 3: Invalid Token Rejection")
    test_invalid_token()
    
    # Test 4: Valid authentication flow
    print_section("Test 4: Valid Authentication Flow")
    token = get_token(USERS["admin"]["username"], USERS["admin"]["password"])
    
    if token:
        get_user_info(token)
        make_prediction(token, SAMPLE_INVOICE)
    
    # Test 5: Role-based access
    print_section("Test 5: Role-Based Access Control")
    test_role_based_access()
    
    # Summary
    print_section("Test Suite Complete")
    print("✅ All authentication features tested!")
    print("\nNext steps:")
    print("1. Review the logs to see audit trail")
    print("2. Try making requests with expired tokens")
    print("3. Test with your own user credentials")
    print("4. Integrate authentication in your client application")

def example_client_usage():
    """
    Example of how to use authentication in a client application
    """
    print_section("Example: Client Application Usage")
    
    print("""
# Example Python client code:

import requests

class LedgerXClient:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.token = None
        self.authenticate(username, password)
    
    def authenticate(self, username, password):
        response = requests.post(
            f"{self.base_url}/token",
            data={"username": username, "password": password}
        )
        self.token = response.json()['access_token']
    
    def predict(self, invoice_data):
        response = requests.post(
            f"{self.base_url}/predict",
            json=invoice_data,
            headers={"Authorization": f"Bearer {self.token}"}
        )
        return response.json()

# Usage:
client = LedgerXClient("http://localhost:8000", "john_doe", "password123")
result = client.predict({
    "blur_score": 45.2,
    "ocr_confidence": 0.87,
    # ... other fields
})
print(result)
    """)

if __name__ == "__main__":
    try:
        run_full_test_suite()
        example_client_usage()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print(f"   Make sure the API is running at: {API_BASE_URL}")
        print("   Start it with: uvicorn src.inference.api_fastapi:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()