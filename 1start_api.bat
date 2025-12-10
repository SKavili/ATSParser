@echo off
REM ATS Backend API Startup Script
REM This script starts the FastAPI server

echo ========================================
echo   ATS Backend API - Starting Server
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

REM Check if .env file exists
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo Please copy .env.example to .env and configure it.
    echo.
    pause
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

REM Set environment variables if .env exists
if exist ".env" (
    echo [INFO] Loading environment variables from .env...
)

REM Start the FastAPI server
echo [INFO] Starting FastAPI server...
echo [INFO] Server will be available at: http://localhost:8000
echo [INFO] API Documentation: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

REM If we get here, the server stopped
echo.
echo [INFO] Server stopped.
pause

