# deploy_with_docker.ps1
# LedgerX Docker Deployment Script

Write-Host "LedgerX Deployment - Build Locally Method" -ForegroundColor Cyan
Write-Host "=" * 70
Write-Host ""

$PROJECT_ID = "ledgerx-mlops"
$SERVICE_NAME = "ledgerx-api"
$REGION = "us-central1"
$IMAGE_TAG = "gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Project: $PROJECT_ID" -ForegroundColor Gray
Write-Host "  Service: $SERVICE_NAME" -ForegroundColor Gray
Write-Host "  Region: $REGION" -ForegroundColor Gray
Write-Host "  Image: $IMAGE_TAG" -ForegroundColor Gray
Write-Host ""

# Step 1: Check prerequisites
Write-Host "[Step 1/5] Checking prerequisites..." -ForegroundColor Green

# Check Docker
try {
    docker version | Out-Null
    Write-Host "  Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  Docker is not running!" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check Dockerfile
if (-not (Test-Path "Dockerfile.cloudrun")) {
    Write-Host "  Dockerfile.cloudrun not found!" -ForegroundColor Red
    exit 1
}
Write-Host "  Dockerfile.cloudrun found" -ForegroundColor Green

# Check required files
$requiredFiles = @("src", "models", "requirements_docker.txt")
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "  Required path not found: $file" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  All required files present" -ForegroundColor Green
Write-Host ""

# Step 2: Configure Docker for GCR
Write-Host "[Step 2/5] Configuring Docker for Google Container Registry..." -ForegroundColor Green
gcloud auth configure-docker gcr.io --quiet
Write-Host "  Docker configured" -ForegroundColor Green
Write-Host ""

# Step 3: Build Docker image locally
Write-Host "[Step 3/5] Building Docker image locally..." -ForegroundColor Green
Write-Host "  This may take 5-10 minutes on first build..." -ForegroundColor Yellow
Write-Host ""

$buildStart = Get-Date
docker build -f Dockerfile.cloudrun -t $IMAGE_TAG .

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Docker build failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Common fixes:" -ForegroundColor Yellow
    Write-Host "    1. Check Dockerfile.cloudrun syntax" -ForegroundColor Gray
    Write-Host "    2. Ensure requirements_docker.txt exists" -ForegroundColor Gray
    Write-Host "    3. Check Docker has enough memory (4GB)" -ForegroundColor Gray
    exit 1
}

$buildEnd = Get-Date
$buildTime = ($buildEnd - $buildStart).TotalSeconds
Write-Host ""
Write-Host "  Build complete in $buildTime seconds" -ForegroundColor Green
Write-Host ""

# Step 4: Push to Container Registry
Write-Host "[Step 4/5] Pushing image to Google Container Registry..." -ForegroundColor Green
Write-Host "  This may take 2-5 minutes..." -ForegroundColor Yellow
Write-Host ""

docker push $IMAGE_TAG

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Push failed!" -ForegroundColor Red
    Write-Host "  Check your GCP authentication: gcloud auth login" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "  Image pushed successfully" -ForegroundColor Green
Write-Host ""

# Step 5: Deploy to Cloud Run
Write-Host "[Step 5/5] Deploying to Cloud Run..." -ForegroundColor Green
Write-Host ""

gcloud run deploy $SERVICE_NAME `
    --image=$IMAGE_TAG `
    --project=$PROJECT_ID `
    --region=$REGION `
    --platform=managed `
    --allow-unauthenticated `
    --port=8000 `
    --cpu=2 `
    --memory=2Gi `
    --timeout=300 `
    --min-instances=0 `
    --max-instances=10 `
    --set-env-vars="ENVIRONMENT=production" `
    --quiet

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=" * 70
    Write-Host "DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "=" * 70
    Write-Host ""
    
    # Get service URL
    $SERVICE_URL = gcloud run services describe $SERVICE_NAME `
        --project=$PROJECT_ID `
        --region=$REGION `
        --format="value(status.url)"
    
    Write-Host "Your API is live at:" -ForegroundColor Cyan
    Write-Host "   $SERVICE_URL" -ForegroundColor White
    Write-Host ""
    
    # Test health
    Write-Host "Running health check..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    
    try {
        $health = Invoke-RestMethod -Uri "$SERVICE_URL/health" -TimeoutSec 30
        Write-Host "   Service is healthy!" -ForegroundColor Green
        Write-Host "   Status: $($health.status)" -ForegroundColor Gray
    } catch {
        Write-Host "   Service is starting (try again in 30 seconds)" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "Quick Start:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Test authentication:" -ForegroundColor White
    Write-Host "    `$body = 'username=admin&password=admin123'" -ForegroundColor Gray
    Write-Host "    `$token = (Invoke-RestMethod -Uri '$SERVICE_URL/token' -Method POST -ContentType 'application/x-www-form-urlencoded' -Body `$body).access_token" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  View API docs:" -ForegroundColor White
    Write-Host "    Start-Process '$SERVICE_URL/docs'" -ForegroundColor Gray
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "Deployment failed!" -ForegroundColor Red
    Write-Host "Check logs: gcloud logging read 'resource.type=cloud_run_revision' --limit=50" -ForegroundColor Yellow
    exit 1
}