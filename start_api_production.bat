@echo off
REM ATS Backend API Production Startup Script
REM This script starts the FastAPI server in production mode (no reload)

echo ========================================
echo   ATS Backend API - Production Mode
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please create it first by running: python -m venv venv
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo [ERROR] Dependencies not installed!
    echo Please run: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Start the FastAPI server in production mode
echo [INFO] Starting FastAPI server in production mode...
echo [INFO] Server will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

REM If we get here, the server stopped
echo.
echo [INFO] Server stopped.
pause

