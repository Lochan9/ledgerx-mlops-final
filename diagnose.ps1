# diagnose.ps1 - LedgerX Diagnostic for Windows PowerShell

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "🔍 LedgerX Diagnostic Report" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

$PROJECT_ID = "vedgrap-mlops"
$REGION = "us-central1"
$SERVICE_NAME = "ledgerx-api"

# Check gcloud auth
Write-Host "1. AUTHENTICATION CHECK" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------------------------"
try {
    $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if ($account) {
        Write-Host "✅ Authenticated as: $account" -ForegroundColor Green
    } else {
        Write-Host "❌ Not authenticated" -ForegroundColor Red
        Write-Host "   Run: gcloud auth login" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ gcloud not found or error occurred" -ForegroundColor Red
}
Write-Host ""

# Check project
Write-Host "2. PROJECT CONFIGURATION" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------------------------"
try {
    $currentProject = gcloud config get-value project 2>$null
    if ($currentProject -eq $PROJECT_ID) {
        Write-Host "✅ Project: $PROJECT_ID" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Current project: $currentProject" -ForegroundColor Yellow
        Write-Host "   Expected: $PROJECT_ID" -ForegroundColor Yellow
        Write-Host "   Run: gcloud config set project $PROJECT_ID" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Could not get project" -ForegroundColor Red
}
Write-Host ""

# Check service status
Write-Host "3. CLOUD RUN SERVICE STATUS" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------------------------"
try {
    $serviceExists = gcloud run services describe $SERVICE_NAME `
        --region $REGION `
        --project $PROJECT_ID `
        --format='value(metadata.name)' 2>$null

    if ($serviceExists) {
        Write-Host "✅ Service exists: $SERVICE_NAME" -ForegroundColor Green
        
        # Get service URL
        $serviceUrl = gcloud run services describe $SERVICE_NAME `
            --region $REGION `
            --project $PROJECT_ID `
            --format='value(status.url)'
        Write-Host "   URL: $serviceUrl" -ForegroundColor Cyan
        
        # Get service status
        $readyCondition = gcloud run services describe $SERVICE_NAME `
            --region $REGION `
            --project $PROJECT_ID `
            --format='value(status.conditions[0].status)'
        
        if ($readyCondition -eq "True") {
            Write-Host "   Status: ✅ READY" -ForegroundColor Green
        } else {
            Write-Host "   Status: ❌ NOT READY" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ Service '$SERVICE_NAME' not found" -ForegroundColor Red
        Write-Host "   The service may not be deployed yet" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Could not describe service" -ForegroundColor Red
}
Write-Host ""

# Test connectivity
Write-Host "4. API CONNECTIVITY TEST" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------------------------"
if ($serviceUrl) {
    Write-Host "Testing: $serviceUrl/health"
    
    try {
        $response = Invoke-WebRequest -Uri "$serviceUrl/health" -Method GET -ErrorAction SilentlyContinue
        $statusCode = $response.StatusCode
        
        switch ($statusCode) {
            200 {
                Write-Host "✅ Status Code: 200 OK" -ForegroundColor Green
                Write-Host ""
                Write-Host "Response:" -ForegroundColor Cyan
                Write-Host $response.Content
            }
            503 {
                Write-Host "❌ Status Code: 503 Service Unavailable" -ForegroundColor Red
                Write-Host "   The service is deployed but not responding" -ForegroundColor Yellow
                Write-Host "   This is the error you're seeing!" -ForegroundColor Yellow
            }
            404 {
                Write-Host "❌ Status Code: 404 Not Found" -ForegroundColor Red
                Write-Host "   The /health endpoint doesn't exist" -ForegroundColor Yellow
            }
            default {
                Write-Host "⚠️  Status Code: $statusCode" -ForegroundColor Yellow
            }
        }
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        if ($statusCode -eq 503) {
            Write-Host "❌ Status Code: 503 Service Unavailable" -ForegroundColor Red
            Write-Host "   This is the main problem!" -ForegroundColor Yellow
        } else {
            Write-Host "❌ Cannot connect to service" -ForegroundColor Red
            Write-Host "   Status: $statusCode" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "Testing: $serviceUrl/ (root)"
    try {
        $response = Invoke-WebRequest -Uri "$serviceUrl/" -Method GET -ErrorAction SilentlyContinue
        Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
    } catch {
        Write-Host "Status: Failed" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  No service URL available - skipping connectivity tests" -ForegroundColor Yellow
}
Write-Host ""

# Check Dockerfile
Write-Host "5. DOCKERFILE CONFIGURATION" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------------------------"
if (Test-Path "Dockerfile.cloudrun") {
    Write-Host "✅ Dockerfile.cloudrun exists" -ForegroundColor Green
    
    $cmdLine = Get-Content "Dockerfile.cloudrun" | Select-String "^CMD"
    Write-Host "   CMD: $cmdLine" -ForegroundColor Cyan
    
    if ($cmdLine -match "8000") {
        Write-Host "   ✅ Port 8000 configured" -ForegroundColor Green
    } elseif ($cmdLine -match "8080") {
        Write-Host "   ❌ Port 8080 found - should be 8000" -ForegroundColor Red
    } else {
        Write-Host "   ⚠️  No explicit port found" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ Dockerfile.cloudrun not found" -ForegroundColor Red
}
Write-Host ""

# Check API file
Write-Host "6. API CONFIGURATION" -ForegroundColor Yellow
Write-Host "-----------------------------------------------------------------------"
if (Test-Path "src/inference/api_fastapi.py") {
    Write-Host "✅ api_fastapi.py exists" -ForegroundColor Green
    
    $content = Get-Content "src/inference/api_fastapi.py" -Raw
    
    if ($content -match '@app\.get\("/health"') {
        Write-Host "   ✅ Health endpoint defined" -ForegroundColor Green
    } else {
        Write-Host "   ❌ Health endpoint missing" -ForegroundColor Red
    }
    
    if ($content -match 'allow_origins=\["\*"\]') {
        Write-Host "   ✅ CORS configured (allow all origins)" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  CORS may need configuration" -ForegroundColor Yellow
    }
} else {
    Write-Host "❌ api_fastapi.py not found" -ForegroundColor Red
}
Write-Host ""

# Summary
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "📊 DIAGNOSTIC SUMMARY" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

if ($statusCode -eq 200) {
    Write-Host "✅ API is working correctly!" -ForegroundColor Green
} elseif ($statusCode -eq 503) {
    Write-Host "❌ PRIMARY ISSUE: Service returning 503" -ForegroundColor Red
    Write-Host ""
    Write-Host "This means:" -ForegroundColor Yellow
    Write-Host "  - Service is deployed but crashing" -ForegroundColor Yellow
    Write-Host "  - Likely port mismatch or startup error" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Solutions to try:" -ForegroundColor Yellow
    Write-Host "  1. Run: .\fix_and_deploy.ps1" -ForegroundColor Cyan
    Write-Host "  2. Or follow manual steps in CORS_AND_API_FIX.md" -ForegroundColor Cyan
} else {
    Write-Host "⚠️  Issues detected - review sections above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "For complete fix instructions, see: CORS_AND_API_FIX.md" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan