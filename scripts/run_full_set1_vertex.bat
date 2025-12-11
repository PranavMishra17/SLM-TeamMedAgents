@echo off
REM ============================================================================
REM Full Dataset Benchmark - Set 1 (Vertex AI) (moved to scripts/)
REM Launches 4 datasets in parallel using run_full_single_vertex.bat
REM Datasets: medqa, path_vqa, medbullets, ddxplus
REM ============================================================================

REM Load environment variables from .env file
echo Loading configuration from .env file...
call "%~dp0load_env.bat"
if errorlevel 1 (
    pause
    exit /b 1
)

echo ============================================================================
echo FULL DATASET BENCHMARK - SET 1 (Vertex AI)
echo Configuration loaded successfully!
echo ============================================================================

echo Launching medqa...
start "Full - medqa" cmd /k "%~dp0run_full_single_vertex.bat medqa"
timeout /t 3 /nobreak >nul

echo Launching path_vqa...
start "Full - path_vqa" cmd /k "%~dp0run_full_single_vertex.bat path_vqa"
timeout /t 3 /nobreak >nul

echo Launching medbullets...
start "Full - medbullets" cmd /k "%~dp0run_full_single_vertex.bat medbullets"
timeout /t 3 /nobreak >nul

echo Launching ddxplus...
start "Full - ddxplus" cmd /k "%~dp0run_full_single_vertex.bat ddxplus"

echo.
echo ============================================================================
echo All 4 instances launched!
echo ============================================================================
pause
