# shutdown.ps1
Write-Host "Stopping LedgerX to save costs..." -ForegroundColor Cyan
Write-Host ""

$PROJECT = "ledgerx-mlops"
$INSTANCE = "ledgerx-postgres"

Write-Host "This will stop Cloud SQL (saves ~`$0.50/day)" -ForegroundColor Yellow
Write-Host "Continue? (y/n): " -NoNewline
$confirm = Read-Host

if ($confirm -ne "y") {
    Write-Host "Cancelled" -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "Stopping Cloud SQL..." -ForegroundColor Yellow
gcloud sql instances patch $INSTANCE --activation-policy=NEVER --project=$PROJECT

Write-Host ""
Start-Sleep -Seconds 10

$status = gcloud sql instances describe $INSTANCE --project=$PROJECT --format="value(state)"
Write-Host "Cloud SQL Status: $status" -ForegroundColor Cyan

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SHUTDOWN COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Savings: ~`$0.50/day" -ForegroundColor Green
Write-Host ""
Write-Host "To restart: .\startup.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Goodnight!" -ForegroundColor Cyan