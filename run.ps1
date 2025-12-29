#!/usr/bin/env pwsh
# Fake Medicine Detector - PowerShell Quick Start Script

Write-Host "`n"
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   💊 FAKE MEDICINE DETECTOR - QUICK START" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "`n"

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n"

# Check/Create virtual environment
if (Test-Path ".venv") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
}

Write-Host "`n"
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow

# Activate virtual environment
& ".\.venv\Scripts\Activate.ps1"

Write-Host "`n"
Write-Host "📥 Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -q -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ ERROR: Failed to install dependencies" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✅ All dependencies installed!" -ForegroundColor Green

Write-Host "`n"
Write-Host "🔍 Checking for medicine_model.pkl..." -ForegroundColor Yellow

if (Test-Path "medicine_model.pkl") {
    Write-Host "✅ Model file found!" -ForegroundColor Green
} else {
    Write-Host "⚠️  WARNING: medicine_model.pkl not found" -ForegroundColor Red
    Write-Host "Please run fake_medicine_detection.ipynb first to generate the model" -ForegroundColor Yellow
    Write-Host "`n"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n"
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   🚀 STARTING STREAMLIT APP" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "`n"
Write-Host "The app will open in your default browser at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8501" -ForegroundColor Yellow
Write-Host "`n"
Write-Host "Press Ctrl+C to stop the app" -ForegroundColor Yellow
Write-Host "`n"

Start-Sleep -Seconds 3

# Run Streamlit
streamlit run app.py
