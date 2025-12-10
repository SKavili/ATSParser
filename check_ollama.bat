@echo off
REM Script to check OLLAMA status

echo ========================================
echo   Checking OLLAMA Status
echo ========================================
echo.

set OLLAMA_HOST=http://localhost:11434

echo Checking OLLAMA at: %OLLAMA_HOST%
echo.

REM Check if curl is available
where curl >nul 2>nul
if errorlevel 1 (
    echo [ERROR] curl is not installed or not in PATH.
    echo Please install curl or use PowerShell.
    echo.
    echo Trying PowerShell method...
    echo.
    powershell -Command "try { $response = Invoke-WebRequest -Uri '%OLLAMA_HOST%/api/tags' -Method GET -TimeoutSec 5; Write-Host '[SUCCESS] OLLAMA is running!' -ForegroundColor Green; Write-Host $response.Content } catch { Write-Host '[ERROR] OLLAMA is not running or not accessible' -ForegroundColor Red; Write-Host $_.Exception.Message }"
    pause
    exit /b 0
)

echo [INFO] Testing OLLAMA connection...
curl -s %OLLAMA_HOST%/api/tags

if errorlevel 1 (
    echo.
    echo [ERROR] OLLAMA is not running or not accessible at %OLLAMA_HOST%
    echo.
    echo To start OLLAMA:
    echo   1. Open a new terminal
    echo   2. Run: ollama serve
    echo   3. Wait for it to start
    echo   4. Then try again
) else (
    echo.
    echo [SUCCESS] OLLAMA is running!
    echo.
    echo Checking if llama3.1 model is installed...
    curl -s %OLLAMA_HOST%/api/tags | findstr /i "llama3.1"
    if errorlevel 1 (
        echo [WARNING] llama3.1 model not found. Install it with:
        echo   ollama pull llama3.1
    ) else (
        echo [SUCCESS] llama3.1 model is installed
    )
)

echo.
echo ========================================
pause

