# Deploy Lightweight LedgerX API to Cloud Run
# PowerShell script

Write-Host "🚀 LedgerX Cloud Run Deployment (Lightweight API)" -ForegroundColor Cyan
Write-Host "=" * 60

# Configuration
$PROJECT_ID = "ledgerx-mlops"
$SERVICE_NAME = "ledgerx-api"
$REGION = "us-central1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME" + ":v2-lightweight"

Write-Host "`n📋 Configuration:" -ForegroundColor Yellow
Write-Host "  Project: $PROJECT_ID"
Write-Host "  Service: $SERVICE_NAME"
Write-Host "  Region: $REGION"
Write-Host "  Image: $IMAGE_NAME"

# Step 1: Build lightweight Docker image
Write-Host "`n[1/3] Building lightweight Docker image..." -ForegroundColor Green
docker build -f Dockerfile.api -t $IMAGE_NAME .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Build complete!" -ForegroundColor Green

# Step 2: Push to Container Registry
Write-Host "`n[2/3] Pushing to Google Container Registry..." -ForegroundColor Green
docker push $IMAGE_NAME

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Push complete!" -ForegroundColor Green

# Step 3: Deploy to Cloud Run
Write-Host "`n[3/3] Deploying to Cloud Run..." -ForegroundColor Green
gcloud run deploy $SERVICE_NAME `
    --image=$IMAGE_NAME `
    --platform=managed `
    --region=$REGION `
    --allow-unauthenticated `
    --port=8080 `
    --cpu=1 `
    --memory=512Mi `
    --timeout=60 `
    --max-instances=10 `
    --min-instances=0 `
    --startup-cpu-boost `
    --set-env-vars="ENVIRONMENT=production"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✅ Deployment successful!" -ForegroundColor Green
Write-Host "=" * 60

# Get service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
Write-Host "`n🌐 Service URL: $SERVICE_URL" -ForegroundColor Cyan

# Test health endpoint
Write-Host "`n🧪 Testing health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$SERVICE_URL/health" -Method Get
    Write-Host "✅ Health check passed!" -ForegroundColor Green
    Write-Host "   Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor Gray
} catch {
    Write-Host "⚠️ Health check failed (service may still be starting)" -ForegroundColor Yellow
}

Write-Host "`n🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "API is ready at: $SERVICE_URL" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Update website/index.html with new URL"
Write-Host "  2. Test with: curl $SERVICE_URL"
Write-Host "  3. Upload invoice through dashboard"