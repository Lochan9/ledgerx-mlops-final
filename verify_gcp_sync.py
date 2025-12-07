"""
LedgerX GCP Sync Verification
Checks if everything is up-to-date with Google Cloud Platform
"""
import subprocess
import json
from datetime import datetime

print("""
╔══════════════════════════════════════════════════════════╗
║         LedgerX GCP Sync Verification                    ║
╚══════════════════════════════════════════════════════════╝
""")

print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def run_command(cmd, description):
    """Run a command and return output"""
    print(f"🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        return "Command timed out", 1
    except Exception as e:
        return str(e), 1

# 1. Git Status
print("\n1️⃣ GIT STATUS")
print("="*60)
output, code = run_command("git status --short", "Checking git status")
if output.strip():
    print("📝 Uncommitted changes:")
    print(output)
else:
    print("✅ Git working directory clean")

# 2. Git Remote
output, code = run_command("git remote -v", "Checking git remote")
if "github.com" in output or "gitlab.com" in output:
    print("✅ Git remote configured")
    print(output)
else:
    print("⚠️ No git remote found")

# 3. DVC Status
print("\n2️⃣ DVC STATUS")
print("="*60)
output, code = run_command("dvc status", "Checking DVC status")
if "Data and pipelines are up to date" in output or not output.strip():
    print("✅ DVC data and pipelines up to date")
else:
    print("📝 DVC changes detected:")
    print(output)

# 4. DVC Remote
output, code = run_command("dvc remote list", "Checking DVC remote")
if "gs://" in output:
    print("✅ DVC remote configured (Google Cloud Storage)")
    print(output)
else:
    print("⚠️ DVC remote not configured")

# 5. GCP Project
print("\n3️⃣ GCP PROJECT")
print("="*60)
output, code = run_command("gcloud config get-value project", "Checking active GCP project")
if "ledgerx-mlops" in output:
    print("✅ Active GCP project: ledgerx-mlops")
else:
    print(f"⚠️ Current project: {output.strip()}")

# 6. Cloud SQL Instance
print("\n4️⃣ CLOUD SQL")
print("="*60)
output, code = run_command(
    "gcloud sql instances describe ledgerx-db --format=json",
    "Checking Cloud SQL instance"
)
try:
    if code == 0:
        data = json.loads(output)
        print(f"✅ Instance: {data.get('name')}")
        print(f"   State: {data.get('state')}")
        print(f"   Region: {data.get('region')}")
        print(f"   IP: {data.get('ipAddresses', [{}])[0].get('ipAddress')}")
except:
    print("⚠️ Could not fetch Cloud SQL details")

# 7. Cloud Storage (DVC Remote)
print("\n5️⃣ CLOUD STORAGE (DVC)")
print("="*60)
output, code = run_command(
    "gsutil ls gs://ledgerx-dvc-storage/",
    "Checking DVC storage bucket"
)
if code == 0:
    print("✅ DVC storage bucket exists")
    # Count files
    output2, _ = run_command(
        "gsutil ls -r gs://ledgerx-dvc-storage/ | wc -l",
        "Counting files"
    )
    print(f"   Files in bucket: {output2.strip()}")
else:
    print("⚠️ Could not access DVC storage bucket")

# 8. Cloud Run Services
print("\n6️⃣ CLOUD RUN")
print("="*60)
output, code = run_command(
    "gcloud run services list --region=us-central1 --format=json",
    "Checking Cloud Run services"
)
try:
    if code == 0:
        services = json.loads(output)
        if services:
            for svc in services:
                print(f"✅ Service: {svc.get('metadata', {}).get('name')}")
                print(f"   URL: {svc.get('status', {}).get('url')}")
                print(f"   Region: {svc.get('metadata', {}).get('labels', {}).get('cloud.googleapis.com/location')}")
        else:
            print("📝 No Cloud Run services deployed")
except:
    print("⚠️ Could not fetch Cloud Run details")

# 9. Document AI
print("\n7️⃣ DOCUMENT AI")
print("="*60)
output, code = run_command(
    "gcloud services list --enabled --filter='documentai.googleapis.com'",
    "Checking Document AI API"
)
if "documentai.googleapis.com" in output:
    print("✅ Document AI API enabled")
else:
    print("⚠️ Document AI API not enabled")

# 10. Cloud Logging
print("\n8️⃣ CLOUD LOGGING")
print("="*60)
output, code = run_command(
    "gcloud logging logs list --limit=5",
    "Checking recent logs"
)
if "ledgerx" in output.lower():
    print("✅ LedgerX logs found in Cloud Logging")
else:
    print("📝 Checking for application logs...")

# Summary
print("\n" + "="*60)
print("📊 SYNC SUMMARY")
print("="*60)

print("""
To push everything to GCP:

1. Git Sync:
   git add .
   git commit -m "Update: Backend-Frontend integration verified"
   git push origin main

2. DVC Sync:
   dvc push

3. Deploy to Cloud Run:
   gcloud run deploy ledgerx-api --source . --region us-central1

4. Check deployment:
   gcloud run services describe ledgerx-api --region us-central1
""")

print("="*60)