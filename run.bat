@echo off
REM Fake Medicine Detector - Quick Start Script
REM This script sets up and runs the Streamlit app

echo.
echo ====================================================
echo   💊 FAKE MEDICINE DETECTOR - QUICK START
echo ====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo ✅ Python found!
echo.

REM Check if virtual environment exists
if exist ".venv" (
    echo ✅ Virtual environment found
) else (
    echo 📦 Creating virtual environment...
    python -m venv .venv
    echo ✅ Virtual environment created
)

echo.
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo 📥 Installing dependencies from requirements.txt...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ❌ ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ All dependencies installed!

echo.
echo 🔍 Checking for medicine_model.pkl...
if exist "medicine_model.pkl" (
    echo ✅ Model file found!
) else (
    echo ⚠️  WARNING: medicine_model.pkl not found
    echo Please run fake_medicine_detection.ipynb first to generate the model
    echo.
    pause
    exit /b 1
)

echo.
echo ====================================================
echo   🚀 STARTING STREAMLIT APP
echo ====================================================
echo.
echo The app will open in your default browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the app
echo.

timeout /t 3

REM Run Streamlit
streamlit run app.py

pause
