from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path
import time

# === STEP 1: AUTHENTICATE (with offline refresh support) ===
gauth = GoogleAuth(settings_file="settings.yaml")
gauth.LoadCredentialsFile("drive_creds.json")

if gauth.credentials is None:
    gauth.LocalWebserverAuth()  # opens browser (first time)
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()

gauth.SaveCredentialsFile("drive_creds.json")
drive = GoogleDrive(gauth)

# === STEP 2: PATH CONFIGURATION ===
FOLDER_ID = "1nDGBfjE3BSEPobsMewLJi8lGknsKTgfT"  # your Google Drive folder ID
LOCAL_PATH = Path("data/raw/FATURA")

# === STEP 3: UPLOAD WITH SKIP LOGIC ===
uploaded = 0
skipped = 0
errors = 0

print(f"\n🚀 Starting upload from {LOCAL_PATH} → Drive folder {FOLDER_ID}\n")

for file_path in LOCAL_PATH.rglob("*"):
    if not file_path.is_file():
        continue

    try:
        # Check if file already exists in Drive folder
        query = (
            f"'{FOLDER_ID}' in parents and title='{file_path.name}' and trashed=false"
        )
        existing_files = drive.ListFile({"q": query}).GetList()
        if existing_files:
            print(f"⏭️  Skipping (already uploaded): {file_path.name}")
            skipped += 1
            continue

        # Upload file
        f = drive.CreateFile({"parents": [{"id": FOLDER_ID}], "title": file_path.name})
        f.SetContentFile(str(file_path))
        f.Upload()
        print(f"✅ Uploaded: {file_path.name}")
        uploaded += 1

        # small delay to avoid hitting Drive API rate limits
        time.sleep(0.3)

    except Exception as e:
        print(f"❌ Error uploading {file_path.name}: {e}")
        errors += 1

print(
    f"\n🎯 Upload summary: {uploaded} uploaded, {skipped} skipped, {errors} errors."
)
