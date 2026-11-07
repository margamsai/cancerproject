@echo off
echo ========================================================
echo   🎗️ OncoAI: React + Node.js + Python ML System
echo ========================================================
echo Launching React Frontend and Python Backend...
node start.js
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Node.js error, launching servers individually...
    start py backend/main.py
    cd frontend && npm run dev
)
pause
