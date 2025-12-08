# test_invoice_save.ps1
# Test if invoices are being saved to Cloud SQL

Write-Host "Testing Invoice Save Functionality..." -ForegroundColor Cyan
Write-Host ""

$API = "https://ledgerx-api-zyz6umftna-uc.a.run.app"

# 1. Login
Write-Host "[1/4] Logging in..." -ForegroundColor Yellow
$body = "username=admin&password=admin123"
$auth = Invoke-RestMethod -Uri "$API/token" -Method POST -ContentType "application/x-www-form-urlencoded" -Body $body
$token = $auth.access_token
Write-Host "  Token: $($token.Substring(0,20))..." -ForegroundColor Green
Write-Host ""

# 2. Check current invoices
Write-Host "[2/4] Checking current invoices..." -ForegroundColor Yellow
$headers = @{ "Authorization" = "Bearer $token" }
$before = Invoke-RestMethod -Uri "$API/user/invoices" -Headers $headers
Write-Host "  Current count: $($before.total)" -ForegroundColor Green
Write-Host ""

# 3. Make a test prediction (this might save to DB)
Write-Host "[3/4] Making test prediction..." -ForegroundColor Yellow
$testData = @{
    blur_score = 45.2
    contrast_score = 28.5
    ocr_confidence = 0.87
    file_size_kb = 245.3
    vendor_name = "Test Vendor $(Get-Date -Format 'HHmmss')"
    vendor_freq = 0.03
    total_amount = 1250.00
    invoice_number = "TEST-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    invoice_date = "2024-12-07"
    currency = "USD"
} | ConvertTo-Json

try {
    $prediction = Invoke-RestMethod -Uri "$API/predict" -Method POST -Headers $headers -Body $testData -ContentType "application/json"
    Write-Host "  Quality: $($prediction.quality.prediction)" -ForegroundColor Green
    Write-Host "  Failure: $($prediction.failure.prediction)" -ForegroundColor Green
} catch {
    Write-Host "  /predict endpoint not available (old revision)" -ForegroundColor Yellow
}
Write-Host ""

# 4. Check if count increased
Write-Host "[4/4] Checking if invoice was saved..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
$after = Invoke-RestMethod -Uri "$API/user/invoices" -Headers $headers
Write-Host "  New count: $($after.total)" -ForegroundColor Green
Write-Host ""

if ($after.total -gt $before.total) {
    Write-Host "SUCCESS: Invoice was saved to database!" -ForegroundColor Green
} else {
    Write-Host "ISSUE: Invoice was NOT saved to database" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible reasons:" -ForegroundColor Yellow
    Write-Host "  1. /upload/image endpoint exists but doesn't save to DB" -ForegroundColor Gray
    Write-Host "  2. Old revision (00016-cs9) has different behavior" -ForegroundColor Gray
    Write-Host "  3. Database connection issue" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Current invoices in database:" -ForegroundColor Cyan
if ($after.invoices -and $after.invoices.Count -gt 0) {
    $after.invoices | Select-Object -First 5 | Format-Table invoice_number, vendor_name, total_amount, quality_prediction, created_at
} else {
    Write-Host "  (none)" -ForegroundColor Gray
}