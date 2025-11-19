from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Step 1: Local OAuth flow
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # launches browser for login

# Step 2: Save credentials to a local file
gauth.SaveCredentialsFile("drive_creds.json")

print("✅ Authentication complete. Token saved as drive_creds.json.")
