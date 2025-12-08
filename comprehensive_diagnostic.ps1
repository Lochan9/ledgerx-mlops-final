# comprehensive_diagnostic.ps1
# Complete end-to-end diagnostic for LedgerX deployment

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "LEDGERX COMPLETE END-TO-END DIAGNOSTIC" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

$PROJECT = "ledgerx-mlops"
$SERVICE = "ledgerx-api"
$REGION = "us-central1"
$API = "https://ledgerx-api-671429123152.us-central1.run.app"

# ============================================================================
# 1. CLOUD RUN SERVICE STATUS
# ============================================================================
Write-Host "[1/10] Cloud Run Service Status" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud run services describe $SERVICE --region=$REGION --format="table(metadata.name,status.url,status.latestReadyRevisionName,status.traffic)"
Write-Host ""

# ============================================================================
# 2. CURRENT REVISION DETAILS
# ============================================================================
Write-Host "[2/10] Current Revision Configuration" -ForegroundColor Yellow
Write-Host "-" * 80
$currentRev = gcloud run services describe $SERVICE --region=$REGION --format="value(status.latestReadyRevisionName)"
Write-Host "Current Revision: $currentRev" -ForegroundColor Cyan
Write-Host ""

# Get environment variables
Write-Host "Environment Variables:" -ForegroundColor Cyan
gcloud run revisions describe $currentRev --region=$REGION --format="table(spec.containers[0].env)" 2>$null
Write-Host ""

# Get Cloud SQL instances
Write-Host "Cloud SQL Instances:" -ForegroundColor Cyan
gcloud run revisions describe $currentRev --region=$REGION --format="value(metadata.annotations.'run.googleapis.com/cloudsql-instances')"
Write-Host ""

# ============================================================================
# 3. CLOUD SQL INSTANCE STATUS
# ============================================================================
Write-Host "[3/10] Cloud SQL Instance Status" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud sql instances describe ledgerx-postgres --format="table(name,state,databaseVersion,connectionName)"
Write-Host ""

# ============================================================================
# 4. DATABASE EXISTENCE
# ============================================================================
Write-Host "[4/10] Databases in Instance" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud sql databases list --instance=ledgerx-postgres
Write-Host ""

# ============================================================================
# 5. USERS IN DATABASE
# ============================================================================
Write-Host "[5/10] Database Users" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud sql users list --instance=ledgerx-postgres
Write-Host ""

# ============================================================================
# 6. API HEALTH CHECK
# ============================================================================
Write-Host "[6/10] API Health Check" -ForegroundColor Yellow
Write-Host "-" * 80
try {
    $health = Invoke-RestMethod -Uri "$API/health"
    Write-Host "✅ API is healthy" -ForegroundColor Green
    $health | ConvertTo-Json -Depth 3
} catch {
    Write-Host "❌ API health check failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Gray
}
Write-Host ""

# ============================================================================
# 7. STARTUP LOGS (Last 50 lines)
# ============================================================================
Write-Host "[7/10] Service Startup Logs" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud logging read "resource.labels.revision_name=$currentRev" --limit=50 --format="table(timestamp,severity,textPayload)" | Select-String -Pattern "Starting|Connected|Loaded|Cloud SQL|database|Error|Warning" | Select-Object -First 20
Write-Host ""

# ============================================================================
# 8. AUTHENTICATION ATTEMPT LOGS
# ============================================================================
Write-Host "[8/10] Recent Authentication Attempts" -ForegroundColor Yellow
Write-Host "-" * 80
Write-Host "Attempting login to trigger logs..." -ForegroundColor Gray
try {
    $body = "username=admin&password=admin123"
    Invoke-RestMethod -Uri "$API/token" -Method POST -ContentType "application/x-www-form-urlencoded" -Body $body | Out-Null
} catch {
    Write-Host "Login failed (expected - checking logs)" -ForegroundColor Gray
}

Start-Sleep -Seconds 3

Write-Host "Authentication logs:" -ForegroundColor Cyan
gcloud logging read "resource.labels.revision_name=$currentRev AND (textPayload:admin OR textPayload:authentication OR textPayload:password OR textPayload:user)" --limit=10 --format="table(timestamp,textPayload)"
Write-Host ""

# ============================================================================
# 9. DATABASE CONNECTION LOGS
# ============================================================================
Write-Host "[9/10] Database Connection Logs" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud logging read "resource.labels.revision_name=$currentRev AND (textPayload:database OR textPayload:SQL OR textPayload:Connected OR textPayload:Connection)" --limit=15 --format="table(timestamp,severity,textPayload)"
Write-Host ""

# ============================================================================
# 10. ERROR LOGS
# ============================================================================
Write-Host "[10/10] Recent Errors" -ForegroundColor Yellow
Write-Host "-" * 80
gcloud logging read "resource.labels.revision_name=$currentRev AND severity>=ERROR" --limit=10 --format="table(timestamp,textPayload)"
Write-Host ""

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DIAGNOSTIC SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "Key Information:" -ForegroundColor Yellow
Write-Host "  API URL: $API" -ForegroundColor White
Write-Host "  Current Revision: $currentRev" -ForegroundColor White
Write-Host "  Cloud SQL Instance: ledgerx-mlops:us-central1:ledgerx-postgres" -ForegroundColor White
Write-Host "  Database: ledgerx" -ForegroundColor White
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review the logs above for specific errors" -ForegroundColor White
Write-Host "  2. Check if Cloud SQL connection annotation is set" -ForegroundColor White
Write-Host "  3. Verify environment variables are correct" -ForegroundColor White
Write-Host "  4. Confirm database 'ledgerx' exists and has users table" -ForegroundColor White
Write-Host ""

Write-Host "To check database tables manually:" -ForegroundColor Yellow
Write-Host "  gcloud sql connect ledgerx-postgres --user=postgres --database=ledgerx" -ForegroundColor Gray
Write-Host "  Then run: \dt" -ForegroundColor Gray
Write-Host ""