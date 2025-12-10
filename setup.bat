@echo off
REM ATS Backend Setup Script
REM This script sets up the development environment

echo ========================================
echo   ATS Backend - Setup Script
echo ========================================
echo.

REM Check Python version
echo [INFO] Checking Python version...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or higher.
    echo.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [INFO] Virtual environment already exists.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo [INFO] Creating .env file from .env.example...
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo [SUCCESS] .env file created. Please edit it with your configuration.
    ) else (
        echo [WARNING] .env.example not found. Please create .env manually.
    )
) else (
    echo [INFO] .env file already exists.
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your MySQL credentials and OLLAMA settings
echo 2. Ensure MySQL server is running
echo 3. Run database migrations: alembic upgrade head
echo 4. Ensure OLLAMA is running with required models
echo 5. Start the API server: start_api.bat
echo.
pause

