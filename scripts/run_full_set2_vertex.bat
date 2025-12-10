@echo off
REM ============================================================================
REM Full Dataset Benchmark - Set 2 (Vertex AI) (moved to scripts/)
REM Launches 4 datasets in parallel using run_full_single_vertex.bat
REM Datasets: medmcqa, pmc_vqa, pubmedqa, mmlupro-med
REM ============================================================================

echo ============================================================================
echo FULL DATASET BENCHMARK - SET 2 (Vertex AI)
echo Ensure VERTEX_AI_ENDPOINT_ID and GOOGLE_CLOUD_PROJECT are set
echo ============================================================================

if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID not set
    pause
    exit /b 1
)

echo Launching medmcqa...
start "Full - medmcqa" cmd /k "%~dp0run_full_single_vertex.bat medmcqa"
timeout /t 3 /nobreak >nul

echo Launching pmc_vqa...
start "Full - pmc_vqa" cmd /k "%~dp0run_full_single_vertex.bat pmc_vqa"
timeout /t 3 /nobreak >nul

echo Launching pubmedqa...
start "Full - pubmedqa" cmd /k "%~dp0run_full_single_vertex.bat pubmedqa"
timeout /t 3 /nobreak >nul

echo Launching mmlupro-med...
start "Full - mmlupro" cmd /k "%~dp0run_full_single_vertex.bat mmlupro-med"

echo.
echo ============================================================================
echo All 4 instances launched!
echo ============================================================================
pause
