@echo off
REM ============================================================================
REM Full Dataset Benchmark - Set 1 (Vertex AI) (moved to scripts/)
REM Launches 4 datasets in parallel using run_full_single_vertex.bat
REM Datasets: medqa, path_vqa, medbullets, ddxplus
REM ============================================================================

echo ============================================================================
echo FULL DATASET BENCHMARK - SET 1 (Vertex AI)
echo Ensure VERTEX_AI_ENDPOINT_ID and GOOGLE_CLOUD_PROJECT are set
echo ============================================================================

if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID not set
    pause
    exit /b 1
)

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
