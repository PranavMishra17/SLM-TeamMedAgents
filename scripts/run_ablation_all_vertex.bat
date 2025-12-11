@echo off
REM ============================================================================
REM Ablation Master Launcher for Vertex AI MedGemma (moved to scripts/)
REM Launches 3 parallel instances for seeds 111, 222, 333
REM Each instance runs all datasets x 6 configs with 50 questions
REM Usage: run_ablation_all_vertex.bat
REM ============================================================================

echo ============================================================================
echo ABLATION MASTER LAUNCHER - Vertex AI
echo Loading configuration from .env file...
echo ============================================================================

REM Load environment variables from .env file
call "%~dp0load_env.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

echo Configuration loaded successfully!
echo This will launch 3 parallel instances (Seeds: 111, 222, 333)
echo Each instance runs all 8 datasets x 6 configs sequentially
echo ============================================================================

REM Launch 3 instances, one for each seed
echo Launching Seed 111 instance...
start "Ablation Seed 111" cmd /k "%~dp0run_ablation_single_vertex.bat 111"
timeout /t 2 /nobreak >nul

echo Launching Seed 222 instance...
start "Ablation Seed 222" cmd /k "%~dp0run_ablation_single_vertex.bat 222"
timeout /t 2 /nobreak >nul

echo Launching Seed 333 instance...
start "Ablation Seed 333" cmd /k "%~dp0run_ablation_single_vertex.bat 333"

echo.
echo ============================================================================
echo All instances launched. Monitor the 3 windows that opened.
echo ============================================================================
pause
