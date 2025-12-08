# check_gcp_sync_simple.ps1
# Simple GCP Sync Status Check

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "GCP SYNC & DEPLOYMENT STATUS CHECK" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

$PROJECT = "ledgerx-mlops"
$BUCKET = "ledgerx-dashboard-671429123152"
$WEBSITE_URL = "https://storage.googleapis.com/$BUCKET/index.html"

# 1. Check Cloud Run Service
Write-Host "[1/5] Checking Cloud Run Service..." -ForegroundColor Yellow
gcloud run services describe ledgerx-api --region=us-central1 --format="table(status.url,status.latestReadyRevisionName,status.traffic)"
Write-Host ""

# 2. Check Cloud Storage Bucket
Write-Host "[2/5] Checking Cloud Storage Bucket Files..." -ForegroundColor Yellow
Write-Host "Files in gs://$BUCKET/:" -ForegroundColor Cyan
gsutil ls gs://$BUCKET/
Write-Host ""

# 3. Check Specific Website Files
Write-Host "[3/5] Checking Website Files Status..." -ForegroundColor Yellow
$files = @("index.html", "app.js", "styles.css")
foreach ($file in $files) {
    Write-Host "Checking $file..." -ForegroundColor Gray
    gsutil stat gs://$BUCKET/$file 2>&1 | Select-String -Pattern "Updated|Size"
}
Write-Host ""

# 4. Test Website Accessibility
Write-Host "[4/5] Testing Website..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri $WEBSITE_URL -UseBasicParsing -TimeoutSec 10
    Write-Host "  Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "  Size: $($response.Content.Length) bytes" -ForegroundColor Green
    
    # Check if it contains API URL
    if ($response.Content -like "*ledgerx-api*") {
        Write-Host "  API URL: Found in HTML" -ForegroundColor Green
    } else {
        Write-Host "  API URL: NOT found" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ERROR: Cannot access website" -ForegroundColor Red
}
Write-Host ""

# 5. Compare Local vs Remote
Write-Host "[5/5] Comparing Local vs Remote..." -ForegroundColor Yellow
if (Test-Path "website/index.html") {
    $localSize = (Get-Item "website/index.html").Length
    Write-Host "  Local index.html: $localSize bytes" -ForegroundColor Gray
    
    try {
        $remoteSize = (Invoke-WebRequest -Uri $WEBSITE_URL -UseBasicParsing).Content.Length
        Write-Host "  Remote index.html: $remoteSize bytes" -ForegroundColor Gray
        
        if ($localSize -eq $remoteSize) {
            Write-Host "  Status: FILES MATCH" -ForegroundColor Green
        } else {
            Write-Host "  Status: FILES DIFFER - Need to upload!" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  Could not compare" -ForegroundColor Red
    }
} else {
    Write-Host "  Local website/index.html not found" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "QUICK ACTIONS" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "To upload website:" -ForegroundColor Yellow
Write-Host "  gsutil -m cp website/* gs://$BUCKET/" -ForegroundColor White
Write-Host ""

Write-Host "To clear browser cache:" -ForegroundColor Yellow
Write-Host "  gsutil setmeta -h Cache-Control:no-cache gs://$BUCKET/index.html" -ForegroundColor White
Write-Host ""

Write-Host "To view website:" -ForegroundColor Yellow
Write-Host "  Start-Process '$WEBSITE_URL'" -ForegroundColor White
Write-Host ""

Write-Host "To check DVC sync:" -ForegroundColor Yellow
Write-Host "  dvc status -c" -ForegroundColor White
Write-Host "  dvc push" -ForegroundColor White
Write-Host ""