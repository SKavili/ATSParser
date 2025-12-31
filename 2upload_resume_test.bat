@echo off
setlocal enabledelayedexpansion

REM Script to test resume upload with multiple resume files from a folder

REM Get directory where this .bat file exists
set BASE_DIR=%~dp0

REM Set resume folder relative to project
set RESUME_FOLDER=%BASE_DIR%Resumes
set API_URL=http://localhost:8000/api/v1/upload-resume
 
echo ========================================
echo   Testing Resume Upload (Multiple Files)
echo ========================================
echo.
echo Resume Folder: %RESUME_FOLDER%
echo API URL: %API_URL%
echo.
echo ========================================
echo   MODULE SELECTION MENU
echo ========================================
echo.
echo Select which modules to extract:
echo.
echo   0 - Extract ALL modules (default)
echo   1 - Designation only
echo   2 - Name only
echo   3 - Role only
echo   4 - Email only
echo   5 - Mobile only
echo   6 - Experience only
echo   7 - Domain only
echo   8 - Education only
echo   9 - Skills only
echo.
echo   You can also select multiple modules:
echo   Example: 1,2,3 or 1,9 or designation,skills,role
echo.
set /p MODULE_CHOICE="Enter your choice (0-9, comma-separated, or 'all'): "

REM Default to "all" if empty
if "!MODULE_CHOICE!"=="" set MODULE_CHOICE=0

REM Convert choice to extract_modules format
if "!MODULE_CHOICE!"=="0" (
    set EXTRACT_MODULES=all
) else (
    set EXTRACT_MODULES=!MODULE_CHOICE!
)

echo.
echo [INFO] Selected modules: !EXTRACT_MODULES!
echo.
 
REM Check if resume folder exists
if not exist "%RESUME_FOLDER%" (
    echo [ERROR] Resume folder not found at: %RESUME_FOLDER%
    echo Please check the folder path.
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
 
echo [INFO] Scanning folder for resume files...
echo.
 
REM Counter for processed files
set /a FILE_COUNT=0
set /a SUCCESS_COUNT=0
set /a FAILED_COUNT=0
 
REM Process PDF files
for %%F in ("%RESUME_FOLDER%\*.pdf") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

REM Process DOC files
for %%F in ("%RESUME_FOLDER%\*.doc") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

REM Process DOCX files
for %%F in ("%RESUME_FOLDER%\*.docx") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

REM Process HTML files
for %%F in ("%RESUME_FOLDER%\*.html") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

REM Process HTM files
for %%F in ("%RESUME_FOLDER%\*.htm") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

REM Process image files (JPG, PNG, etc.)
for %%F in ("%RESUME_FOLDER%\*.jpg") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

for %%F in ("%RESUME_FOLDER%\*.jpeg") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

for %%F in ("%RESUME_FOLDER%\*.png") do (
    set /a FILE_COUNT+=1
    call :ProcessFile "%%F" "!EXTRACT_MODULES!"
)

REM Note: Processing .pdf, .doc, .docx, .html, .htm, .jpg, .jpeg, .png files
 
echo.
echo ========================================
echo   Upload Summary
echo ========================================
echo Total files found: !FILE_COUNT!
echo Successfully uploaded: !SUCCESS_COUNT!
echo Failed: !FAILED_COUNT!
echo ========================================
pause
exit /b 0
 
:ProcessFile
setlocal enabledelayedexpansion
set "RESUME_FILE=%~1"
set "FILENAME=%~nx1"
set "EXTRACT_MODULES=%~2"
 
echo.
echo ========================================
echo [!FILE_COUNT!] Processing: !FILENAME!
echo ========================================
echo.
echo [INFO] This file will be processed ONE BY ONE
echo [INFO] Selected modules will be extracted using OLLAMA
echo [INFO] Results will be saved to database
echo.
 
REM Extract candidate name from filename (remove extension, replace underscores/hyphens with spaces)
REM Use delayed expansion to avoid parsing errors with special characters
set "CANDIDATE_NAME=!FILENAME!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.pdf=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.doc=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.docx=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.txt=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.html=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.htm=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.jpg=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.jpeg=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:.png=!"
set "CANDIDATE_NAME=!CANDIDATE_NAME:_= !"
set "CANDIDATE_NAME=!CANDIDATE_NAME:-= !"
 
REM Job role is now automatically extracted from resume using OLLAMA
REM You can optionally provide a job_role hint, but it will be extracted automatically
REM Set to empty to let the system extract it automatically
set "JOB_ROLE="
 
echo [INFO] Candidate Name: !CANDIDATE_NAME!
echo [INFO] Job Role: (will be extracted automatically from resume)
echo [INFO] Modules to extract: !EXTRACT_MODULES!
echo.
 
REM Upload the file - this triggers extraction based on selected modules
REM Note: If "role" is in extract_modules, job role will be extracted automatically
echo [INFO] Uploading to API and extracting selected modules...
if "!JOB_ROLE!"=="" (
    REM No job_role provided - let system extract it automatically
    curl -X POST "%API_URL%" ^
      -H "accept: application/json" ^
      -H "Content-Type: multipart/form-data" ^
      -F "file=@\"!RESUME_FILE!\"" ^
      -F "candidate_name=!CANDIDATE_NAME!" ^
      -F "extract_modules=!EXTRACT_MODULES!" > temp_response.txt 2>&1
) else (
    REM Optional job_role hint provided (will still be extracted if "role" module is selected)
    curl -X POST "%API_URL%" ^
      -H "accept: application/json" ^
      -H "Content-Type: multipart/form-data" ^
      -F "file=@\"!RESUME_FILE!\"" ^
      -F "candidate_name=!CANDIDATE_NAME!" ^
      -F "job_role=!JOB_ROLE!" ^
      -F "extract_modules=!EXTRACT_MODULES!" > temp_response.txt 2>&1
)
 
REM Check if curl command succeeded and if response contains error
set CURL_SUCCESS=0
if errorlevel 1 (
    set CURL_SUCCESS=1
) else (
    REM Check if response contains error message
    findstr /C:"\"detail\"" /C:"\"error\"" /C:"Failed" temp_response.txt >nul 2>&1
    if errorlevel 1 (
        REM No error found in response
        set CURL_SUCCESS=0
    ) else (
        REM Error found in response
        set CURL_SUCCESS=1
    )
)

if !CURL_SUCCESS!==1 (
    echo.
    echo [ERROR] Failed to upload or process: !FILENAME!
    echo [INFO] Check temp_response.txt for details
    type temp_response.txt
    set /a FAILED_COUNT+=1
) else (
    echo.
    echo [SUCCESS] Uploaded and processed: !FILENAME!
    echo [INFO] Extraction completed for selected modules (check server logs for details)
    set /a SUCCESS_COUNT+=1
)

REM Clean up temp file
if exist temp_response.txt del temp_response.txt
endlocal
echo.
echo Waiting 2 seconds before processing next file...
timeout /t 2 /nobreak >nul
 
goto :eof
