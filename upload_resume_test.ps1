# PowerShell script to test resume upload
$resumeFile = "C:\ATS\profile Sathish Kumar ch Power platform resume.pdf"
$apiUrl = "http://localhost:8000/api/v1/upload-resume"

Write-Host "========================================"
Write-Host "  Testing Resume Upload"
Write-Host "========================================"
Write-Host ""
Write-Host "Resume File: $resumeFile"
Write-Host "API URL: $apiUrl"
Write-Host ""

# Check if resume file exists
if (-not (Test-Path $resumeFile)) {
    Write-Host "[ERROR] Resume file not found at: $resumeFile" -ForegroundColor Red
    Write-Host "Please check the file path."
    pause
    exit 1
}

# Check if API is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -Method GET -TimeoutSec 5 -ErrorAction Stop
    Write-Host "[INFO] API server is running" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] API server is not running or not accessible" -ForegroundColor Red
    Write-Host "Please start the API server using: start_api.bat"
    pause
    exit 1
}

Write-Host "[INFO] Uploading resume..." -ForegroundColor Yellow
Write-Host ""

try {
    # Prepare form data
    $form = @{
        file = Get-Item -Path $resumeFile
        candidate_name = "Sathish Kumar CH"
        job_role = "Power Platform Developer"
    }

    # Upload the file
    $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Form $form -ContentType "multipart/form-data"
    
    Write-Host "[SUCCESS] Resume uploaded successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Response:" -ForegroundColor Cyan
    $response | ConvertTo-Json -Depth 10 | Write-Host
    
} catch {
    Write-Host "[ERROR] Failed to upload resume" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response: $responseBody" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "  Test Complete"
Write-Host "========================================"
Write-Host ""
pause

