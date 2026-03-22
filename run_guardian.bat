@echo off
title Guardian Net Launcher
color 0A
cls
echo ========================================
echo    GUARDIAN NET - LAUNCHER
echo ========================================
echo.
echo Choose what to run:
echo.
echo 1. Start Backend Server (Node.js)
echo 2. Run Fall Detection
echo 3. Run Gesture Detection
echo 4. Run Both (in separate windows)
echo 5. Test Alert System
echo 6. Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Starting Backend Server...
    cd backend
    start cmd /k "node server.js"
    cd ..
)

if "%choice%"=="2" (
    echo.
    echo Starting Fall Detection...
    cd detector
    python guardian_fall.py
)

if "%choice%"=="3" (
    echo.
    echo Starting Gesture Detection...
    cd detector
    python guardian_gesture.py
)

if "%choice%"=="4" (
    echo.
    echo Starting Backend Server...
    cd backend
    start cmd /k "node server.js"
    cd ..
    
    timeout /t 3
    
    echo.
    echo Starting Fall Detection...
    cd detector
    start cmd /k "python guardian_fall.py"
    
    timeout /t 2
    
    echo.
    echo Starting Gesture Detection...
    start cmd /k "python guardian_gesture.py"
)

if "%choice%"=="5" (
    echo.
    echo Testing Alert System...
    cd detector
    python test_alert.py
)

if "%choice%"=="6" (
    echo.
    echo Goodbye!
    exit
)

pause