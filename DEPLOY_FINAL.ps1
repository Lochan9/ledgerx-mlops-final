# DEPLOY_FINAL.ps1
# Complete deployment script - Run this after updating files

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "LEDGERX FINAL DEPLOYMENT" -ForegroundColor Cyan  
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Verify Docker is running
Write-Host "[1/5] Checking Docker..." -ForegroundColor Yellow
docker info 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker Desktop is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop first" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Docker is running" -ForegroundColor Green
Write-Host ""

# Build
Write-Host "[2/5] Building Docker image..." -ForegroundColor Yellow
docker build -f Dockerfile.cloudrun -t gcr.io/ledgerx-mlops/ledgerx-api:final-working .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Build complete" -ForegroundColor Green
Write-Host ""

# Push
Write-Host "[3/5] Pushing to registry..." -ForegroundColor Yellow
docker push gcr.io/ledgerx-mlops/ledgerx-api:final-working
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Pushed to GCR" -ForegroundColor Green
Write-Host ""

# Deploy
Write-Host "[4/5] Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy ledgerx-api `
  --image=gcr.io/ledgerx-mlops/ledgerx-api:final-working `
  --region=us-central1 `
  --allow-unauthenticated `
  --port=8000 `
  --cpu=2 `
  --memory=2Gi `
  --add-cloudsql-instances=ledgerx-mlops:us-central1:ledgerx-postgres `
  --set-env-vars="ENVIRONMENT=production,DB_NAME=ledgerx,DB_USER=postgres,DB_HOST=/cloudsql/ledgerx-mlops:us-central1:ledgerx-postgres,DB_PORT=5432" `
  --set-secrets="DB_PASSWORD=db-password:latest,OPENAI_API_KEY=openai-api-key:latest"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Deployed successfully" -ForegroundColor Green
Write-Host ""

# Test
Write-Host "[5/5] Testing deployment..." -ForegroundColor Yellow
$API = "https://ledgerx-api-671429123152.us-central1.run.app"

Start-Sleep -Seconds 10

# Health
try {
    $health = Invoke-RestMethod -Uri "$API/health"
    Write-Host "✅ Health: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "❌ Health check failed" -ForegroundColor Red
}

# Login
try {
    $body = "username=admin&password=admin123"
    $auth = Invoke-RestMethod -Uri "$API/token" -Method POST -ContentType "application/x-www-form-urlencoded" -Body $body
    Write-Host "✅ Login: SUCCESS" -ForegroundColor Green
    
    # Quick endpoint test
    $headers = @{ "Authorization" = "Bearer $($auth.access_token)" }
    $user = Invoke-RestMethod -Uri "$API/users/me" -Headers $headers
    Write-Host "✅ User endpoint: $($user.username)" -ForegroundColor Green
    
} catch {
    Write-Host "❌ Login failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "API URL: $API" -ForegroundColor White
Write-Host "Website: https://storage.googleapis.com/ledgerx-dashboard-671429123152/index_v3.html" -ForegroundColor White
Write-Host ""