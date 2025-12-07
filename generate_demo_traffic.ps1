# generate_demo_traffic.ps1
# Generate realistic traffic for LedgerX Grafana dashboard demonstration

param(
    [int]$DurationMinutes = 5,
    [int]$RequestsPerSecond = 2,
    [string]$ApiUrl = "http://localhost:8000",
    [string]$Username = "demo_user",
    [string]$Password = "demo123"
)

Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     LedgerX Dashboard - Demo Traffic Generator             ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚙️  Configuration:" -ForegroundColor Yellow
Write-Host "   Duration: $DurationMinutes minutes" -ForegroundColor White
Write-Host "   Target RPS: $RequestsPerSecond" -ForegroundColor White
Write-Host "   API URL: $ApiUrl" -ForegroundColor White
Write-Host ""

$endTime = (Get-Date).AddMinutes($DurationMinutes)
$requestCount = 0
$successCount = 0
$failureCount = 0

# Sample vendor names for variety
$vendors = @(
    "Tech Solutions Inc",
    "Office Supplies Co",
    "Cloud Services LLC",
    "Digital Marketing Pro",
    "Consulting Group",
    "Equipment Rentals Ltd",
    "Software Licensing",
    "Professional Services"
)

# Function to generate random invoice
function New-RandomInvoice {
    return @{
        vendor_name = Get-Random -InputObject $vendors
        invoice_number = "INV-$(Get-Date -Format 'yyyyMMdd')-$(Get-Random -Minimum 1000 -Maximum 9999)"
        invoice_date = (Get-Date).AddDays(-(Get-Random -Minimum 1 -Maximum 30)).ToString("yyyy-MM-dd")
        total_amount = [math]::Round((Get-Random -Minimum 100.0 -Maximum 5000.0), 2)
        currency = "USD"
        blur_score = [math]::Round((Get-Random -Minimum 0.0 -Maximum 1.0), 3)
        ocr_confidence = [math]::Round((Get-Random -Minimum 0.70 -Maximum 1.0), 3)
        has_watermark = $(Get-Random -InputObject @($true, $false))
        has_seal = $(Get-Random -InputObject @($true, $false))
        item_count = $(Get-Random -Minimum 1 -Maximum 25)
        has_discount = $(Get-Random -InputObject @($true, $false))
        has_tax = $true
        payment_terms = $(Get-Random -InputObject @("NET30", "NET45", "NET60", "Due on Receipt"))
    } | ConvertTo-Json
}

# Authenticate
Write-Host "🔐 Authenticating..." -ForegroundColor Yellow
try {
    $authBody = "username=$Username&password=$Password&grant_type=password"
    $authResponse = Invoke-RestMethod -Uri "$ApiUrl/token" `
        -Method Post `
        -Body $authBody `
        -ContentType "application/x-www-form-urlencoded"
    
    $token = $authResponse.access_token
    Write-Host "✅ Authenticated successfully as $Username" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "❌ Authentication failed!" -ForegroundColor Red
    Write-Host "   Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Tips:" -ForegroundColor Yellow
    Write-Host "   1. Make sure the API is running: uvicorn src.inference.api_fastapi:app --port 8000" -ForegroundColor White
    Write-Host "   2. Verify user exists in database" -ForegroundColor White
    Write-Host "   3. Check username/password are correct" -ForegroundColor White
    exit 1
}

# Generate traffic
Write-Host "📊 Generating prediction requests..." -ForegroundColor Yellow
Write-Host "   Press Ctrl+C to stop early" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date
$lastProgressUpdate = Get-Date

while ((Get-Date) -lt $endTime) {
    try {
        # Create random invoice
        $invoiceData = New-RandomInvoice
        
        # Make prediction request
        $headers = @{
            "Authorization" = "Bearer $token"
            "Content-Type" = "application/json"
        }
        
        $response = Invoke-RestMethod -Uri "$ApiUrl/predict" `
            -Method Post `
            -Headers $headers `
            -Body $invoiceData `
            -ErrorAction Stop
        
        $requestCount++
        $successCount++
        
        # Update progress every 5 seconds
        $now = Get-Date
        if (($now - $lastProgressUpdate).TotalSeconds -ge 5) {
            $elapsed = ($now - $startTime).TotalSeconds
            $currentRPS = if ($elapsed -gt 0) { [math]::Round($requestCount / $elapsed, 2) } else { 0 }
            $remaining = ($endTime - $now).TotalSeconds
            $progress = [math]::Round((1 - ($remaining / ($DurationMinutes * 60))) * 100, 1)
            
            Write-Host "   ✓ Requests: $requestCount | RPS: $currentRPS | Success: $successCount | Failed: $failureCount | Progress: $progress%" -ForegroundColor Green
            $lastProgressUpdate = $now
        }
        
        # Calculate delay to achieve target RPS
        $delayMs = [math]::Max(50, (1000 / $RequestsPerSecond))
        Start-Sleep -Milliseconds $delayMs
        
    } catch {
        $failureCount++
        if ($requestCount % 10 -eq 0) {
            Write-Host "   ⚠ Request $requestCount failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        Start-Sleep -Milliseconds 500
    }
}

# Final summary
$totalTime = ((Get-Date) - $startTime).TotalSeconds
$actualRPS = if ($totalTime -gt 0) { [math]::Round($requestCount / $totalTime, 2) } else { 0 }
$successRate = if ($requestCount -gt 0) { [math]::Round(($successCount / $requestCount) * 100, 1) } else { 0 }

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    DEMO COMPLETE                             ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "📈 Statistics:" -ForegroundColor Yellow
Write-Host "   Total Requests:  $requestCount" -ForegroundColor White
Write-Host "   Successful:      $successCount" -ForegroundColor Green
Write-Host "   Failed:          $failureCount" -ForegroundColor $(if ($failureCount -gt 0) { "Red" } else { "White" })
Write-Host "   Success Rate:    $successRate%" -ForegroundColor $(if ($successRate -ge 95) { "Green" } else { "Yellow" })
Write-Host "   Duration:        $([math]::Round($totalTime, 1)) seconds" -ForegroundColor White
Write-Host "   Actual RPS:      $actualRPS" -ForegroundColor White
Write-Host ""
Write-Host "📊 View Results:" -ForegroundColor Yellow
Write-Host "   Grafana:    http://localhost:3000" -ForegroundColor Cyan
Write-Host "   Prometheus: http://localhost:9090" -ForegroundColor Cyan
Write-Host "   API Metrics: http://localhost:8000/metrics" -ForegroundColor Cyan
Write-Host ""
Write-Host "✨ Your dashboard should now show live metrics!" -ForegroundColor Green
Write-Host ""