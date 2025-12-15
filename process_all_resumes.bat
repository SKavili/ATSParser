@echo off
REM Script to process all resumes from the Resumes directory
REM This script processes files one by one and extracts designation for each

echo ========================================
echo   Bulk Resume Processing
echo   Designation Extraction
echo ========================================
echo.
echo This script will process all resumes in:
echo C:\ATS\V200\ATSParser\app\Resumes
echo.
echo Each resume will be:
echo 1. Uploaded to the database
echo 2. Processed with OLLAMA to extract designation
echo 3. Designation saved to database
echo.
echo Processing will happen ONE BY ONE with detailed logs.
echo.
pause

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    pause
    exit /b 1
)

REM Run the Python script
echo.
echo Starting Python script...
echo.
python process_all_resumes.py

if errorlevel 1 (
    echo.
    echo [ERROR] Script execution failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Processing Complete
echo ========================================
pause

