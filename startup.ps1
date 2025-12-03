# startup.ps1
Write-Host "Starting LedgerX Production System..." -ForegroundColor Cyan
Write-Host ""

$PROJECT = "ledgerx-mlops"
$INSTANCE = "ledgerx-postgres"
$API = "https://ledgerx-api-671429123152.us-central1.run.app"

Write-Host "Step 1: Starting Cloud SQL..." -ForegroundColor Yellow
gcloud sql instances patch $INSTANCE --activation-policy=ALWAYS --project=$PROJECT

Write-Host ""
Write-Host "Waiting for Cloud SQL to start (90 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 90

$status = gcloud sql instances describe $INSTANCE --project=$PROJECT --format="value(state)"
Write-Host "Cloud SQL Status: $status" -ForegroundColor Cyan

Write-Host ""
Write-Host "Step 2: Testing API health..." -ForegroundColor Yellow
$health = Invoke-WebRequest -Uri "$API/health" -UseBasicParsing -TimeoutSec 30
Write-Host "API Status: $($health.StatusCode)" -ForegroundColor Green

Write-Host ""
Write-Host "Step 3: Testing authentication..." -ForegroundColor Yellow
$body = "username=admin&password=admin123"
$auth = Invoke-WebRequest -Uri "$API/token" -Method POST -ContentType "application/x-www-form-urlencoded" -Body $body -UseBasicParsing -TimeoutSec 30
Write-Host "Auth Status: $($auth.StatusCode)" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "STARTUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Website: https://storage.googleapis.com/ledgerx-mlops-website/index.html" -ForegroundColor Cyan
Write-Host "Login: admin / admin123" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to test!" -ForegroundColor Green