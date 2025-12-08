# VERIFY_FIXES.ps1
# Check if all fixes are applied before building

Write-Host "Verifying all fixes are in place..." -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check 1: database.py fix (no connection pooling)
Write-Host "[1/4] Checking database.py fix..." -ForegroundColor Yellow
$dbContent = Get-Content src/utils/database.py -Raw
if ($dbContent -match "Fresh connection every time" -or $dbContent -notmatch "_connection = None") {
    Write-Host "  ✅ database.py: Connection pooling removed" -ForegroundColor Green
} else {
    Write-Host "  ❌ database.py: Still using connection pooling (BUG)" -ForegroundColor Red
    Write-Host "     Download: database.py and replace src/utils/database.py" -ForegroundColor Yellow
    $allGood = $false
}

# Check 2: inference_service.py using .cbm
Write-Host "[2/4] Checking inference_service.py..." -ForegroundColor Yellow
$infContent = Get-Content src/inference/inference_service.py -Raw
if ($infContent -match "quality_catboost.cbm" -and $infContent -match "CatBoostClassifier") {
    Write-Host "  ✅ inference_service.py: Using .cbm models" -ForegroundColor Green
} else {
    Write-Host "  ❌ inference_service.py: Not using .cbm files" -ForegroundColor Red
    Write-Host "     Download: inference_service.py and replace src/inference/inference_service.py" -ForegroundColor Yellow
    $allGood = $false
}

# Check 3: All 37 features implemented
if ($infContent -match "engineer_all_features" -and $infContent -match "37 features") {
    Write-Host "  ✅ inference_service.py: All 37 features implemented" -ForegroundColor Green
} else {
    Write-Host "  ⚠️ inference_service.py: May not have all 37 features" -ForegroundColor Yellow
}

# Check 4: requirements_docker.txt has sentence-transformers
Write-Host "[3/4] Checking requirements..." -ForegroundColor Yellow
$reqContent = Get-Content requirements_docker.txt -Raw
if ($reqContent -match "sentence-transformers") {
    Write-Host "  ⚠️ sentence-transformers present (20 min build)" -ForegroundColor Yellow
    Write-Host "     Remove this line for 5 min build" -ForegroundColor Gray
} else {
    Write-Host "  ✅ No heavy dependencies (fast build)" -ForegroundColor Green
}

# Check 5: Models exist
Write-Host "[4/4] Checking model files..." -ForegroundColor Yellow
if ((Test-Path "models/quality_catboost.cbm") -and (Test-Path "models/failure_catboost.cbm")) {
    Write-Host "  ✅ CatBoost .cbm models present" -ForegroundColor Green
} else {
    Write-Host "  ❌ .cbm model files missing!" -ForegroundColor Red
    $allGood = $false
}

Write-Host ""
Write-Host "=" * 60
if ($allGood) {
    Write-Host "✅ ALL FIXES APPLIED - READY TO BUILD!" -ForegroundColor Green
    Write-Host "=" * 60
    Write-Host ""
    Write-Host "Run: .\DEPLOY_FINAL.ps1" -ForegroundColor Cyan
} else {
    Write-Host "❌ FIXES MISSING - Download required files first" -ForegroundColor Red
    Write-Host "=" * 60
    Write-Host ""
    Write-Host "Download these files from outputs:" -ForegroundColor Yellow
    Write-Host "  1. database.py → src/utils/" -ForegroundColor White
    Write-Host "  2. inference_service.py → src/inference/" -ForegroundColor White
    Write-Host "  3. requirements_docker.txt → root" -ForegroundColor White
}