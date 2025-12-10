@echo off
REM Script to test resume upload with a sample resume file

set RESUME_FILE=C:\ATS\Sathish Kumar ch Power platform resume.pdf
set API_URL=http://localhost:8000/api/v1/upload-resume

echo ========================================
echo   Testing Resume Upload
echo ========================================
echo.
echo Resume File: %RESUME_FILE%
echo API URL: %API_URL%
echo.

REM Check if resume file exists
if not exist "%RESUME_FILE%" (
    echo [ERROR] Resume file not found at: %RESUME_FILE%
    echo Please check the file path.
    pause
    exit /b 1
)

REM Check if curl is available
where curl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] curl is not installed or not in PATH.
    echo Please install curl or use PowerShell to upload the file.
    echo.
    echo Alternative: Use PowerShell script upload_resume_test.ps1
    pause
    exit /b 1
)

echo [INFO] Uploading resume...
echo.

curl -X POST "%API_URL%" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@%RESUME_FILE%" ^
  -F "candidate_name=Sathish Kumar CH" ^
  -F "job_role=Power Platform Developer"

echo.
echo.
echo ========================================
echo   Upload Complete
echo ========================================
pause

