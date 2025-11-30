# Automatic Website Update Script
# Makes all necessary changes to enable image upload with OCR

$websiteFile = "D:\vsCOde\ledgerx-mlops-final\website\index.html"

Write-Host "`n🔧 Updating website for image upload with OCR...`n" -ForegroundColor Cyan

# Backup
Copy-Item $websiteFile "$websiteFile.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Write-Host "✅ Backup created" -ForegroundColor Green

# Read file
$content = Get-Content $websiteFile -Raw

# Change 1: Update file input to accept images
$content = $content -replace 'accept="\.json"', 'accept=".jpg,.jpeg,.png,.pdf,.json" multiple'
Write-Host "✅ Updated file input to accept images" -ForegroundColor Green

# Change 2: Update API URL
$content = $content -replace 'https://ledgerx-api-671429123152\.us-central1\.run\.app', 'https://ledgerx-api-zyz6umftna-uc.a.run.app'
Write-Host "✅ Updated API URL" -ForegroundColor Green

# Change 3: Update upload text
$content = $content -replace 'Drop invoice JSON here', 'Drop invoice files here'
$content = $content -replace 'Supports JSON files with invoice data', 'Supports JPG, PNG, PDF, and JSON • OCR Auto-processing'
Write-Host "✅ Updated upload zone text" -ForegroundColor Green

# Change 4: Update file validation in handleFileUpload
$oldValidation = "if\(!file \|\| !file\.name\.endsWith\('\.json'\)\)\{[\s\S]*?return;[\s\S]*?\}"
$newValidation = @"
// Validate file type
    const validTypes = ['.json', '.jpg', '.jpeg', '.png', '.pdf'];
    const isValid = validTypes.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if(!file || !isValid){
        showToast('Invalid File', 'Please upload JPG, PNG, PDF, or JSON file', 'error');
        return;
    }
"@

$content = $content -replace $oldValidation, $newValidation
Write-Host "✅ Updated file validation" -ForegroundColor Green

# Change 5: Add image/PDF handling in handleFileUpload
$oldFetch = "const formData = new FormData\(\);[\s\S]*?formData\.append\('file', file\);[\s\S]*?const response = await fetch\(`\`\${STATE\.apiUrl}/validate/invoice-full`"

$newFetch = @"
// Determine file type
        const isImage = file.type.startsWith('image/');
        const isPDF = file.type === 'application/pdf';
        const isJSON = file.type === 'application/json';
        
        let response;
        
        if(isImage || isPDF){
            // Use image upload endpoint with OCR
            const formData = new FormData();
            formData.append('file', file);
            
            response = await fetch(`\${STATE.apiUrl}/upload/image`
"@

$content = $content -replace $oldFetch, $newFetch

# Save
$content | Out-File -FilePath $websiteFile -Encoding UTF8

Write-Host "`n✅ All changes applied!" -ForegroundColor Green
Write-Host "`n📤 Now deploying to GCP..." -ForegroundColor Yellow

# Deploy to GCP
gsutil -h "Cache-Control:no-cache, max-age=0" cp $websiteFile gs://ledgerx-frontend/

Write-Host "`n🎉 Website updated and deployed!" -ForegroundColor Green
Write-Host "`n🌐 Open in incognito to see changes:" -ForegroundColor Cyan
Write-Host "https://storage.googleapis.com/ledgerx-frontend/index.html`n" -ForegroundColor White

# Open in incognito
Start-Process msedge -ArgumentList "--inprivate","https://storage.googleapis.com/ledgerx-frontend/index.html"