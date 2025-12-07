param([int]$DurationMinutes = 2, [int]$RequestsPerSecond = 3)
Write-Host "Generating demo traffic..." -ForegroundColor Cyan

$endTime = (Get-Date).AddMinutes($DurationMinutes)
$count = 0

# Auth
$authBody = "username=admin&password=admin123&grant_type=password"
try {
    $auth = Invoke-RestMethod -Uri "http://localhost:8000/token" -Method Post -Body $authBody -ContentType "application/x-www-form-urlencoded"
    $token = $auth.access_token
    Write-Host "Authenticated!" -ForegroundColor Green
} catch {
    Write-Host "Auth failed!" -ForegroundColor Red
    exit
}

# Traffic
while ((Get-Date) -lt $endTime) {
    $invoice = @{
        vendor_name = "Demo Vendor"
        invoice_number = "INV-" + (Get-Random -Min 1000 -Max 9999)
        total_amount = (Get-Random -Min 100 -Max 5000)
        blur_score = [math]::Round((Get-Random -Min 0 -Max 100) / 100, 2)
        ocr_confidence = [math]::Round((Get-Random -Min 70 -Max 100) / 100, 2)
    } | ConvertTo-Json
    
    try {
        $headers = @{Authorization = "Bearer $token"; "Content-Type" = "application/json"}
        Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Headers $headers -Body $invoice | Out-Null
        $count++
        if ($count % 10 -eq 0) { Write-Host "Sent $count requests" -ForegroundColor Green }
    } catch {}
    
    Start-Sleep -Milliseconds (1000 / $RequestsPerSecond)
}
Write-Host "Done! $count requests sent. Check Grafana!" -ForegroundColor Cyan
