@echo off
REM ============================================================================
REM Master Teamwork Configuration Runner - Parallel Execution
REM
REM This script launches 3 parallel instances of teamwork configuration testing
REM Each instance runs with a different API key and seed to avoid rate limits
REM
REM Total Runs: 96 (32 runs × 3 instances)
REM   - 8 datasets
REM   - 4 configurations
REM   - 3 seeds
REM
REM Each instance will open in a separate command window
REM ============================================================================

echo ============================================================================
echo TEAMWORK CONFIGURATION BENCHMARK - PARALLEL LAUNCHER
echo ============================================================================
echo.
echo This will launch 3 parallel instances in separate windows:
echo.
echo   Instance 1: API Key #1, Seed 1 (32 runs)
echo   Instance 2: API Key #2, Seed 2 (32 runs)
echo   Instance 3: API Key #3, Seed 3 (32 runs)
echo.
echo Total: 96 runs across all instances
echo.
echo Configurations being tested:
echo   1. Team Orientation + Mutual Monitoring
echo   2. SMM + Trust
echo   3. Team Orientation + SMM + Leadership
echo   4. All Teamwork Components
echo.
echo Datasets: medqa, medmcqa, mmlupro-med, pubmedqa, medbullets, ddxplus, pmc_vqa, path_vqa
echo.
echo Output Directory: multi-agent-gemma/ablation/
echo.
echo ============================================================================
echo.
echo WARNING: Ensure all 3 API keys are set in your environment:
echo   - GOOGLE_API_KEY   (for instance 1)
echo   - GOOGLE_API_KEY2  (for instance 2)
echo   - GOOGLE_API_KEY3  (for instance 3)
echo.
echo ============================================================================
echo.
pause

REM Check if run_teamwork_parallel.bat exists
if not exist "run_teamwork_parallel.bat" (
    echo ERROR: run_teamwork_parallel.bat not found in current directory
    echo Please ensure both scripts are in the same folder
    pause
    exit /b 1
)

echo.
echo Launching parallel instances...
echo.

REM Launch Instance 1 (Key 1, Seed 1)
echo Starting Instance 1: Key #1, Seed 1
start "Teamwork Benchmark - Key 1 Seed 1" cmd /k "run_teamwork_parallel.bat 1 1"
timeout /t 3 /nobreak >nul

REM Launch Instance 2 (Key 2, Seed 2)
echo Starting Instance 2: Key #2, Seed 2
start "Teamwork Benchmark - Key 2 Seed 2" cmd /k "run_teamwork_parallel.bat 2 2"
timeout /t 3 /nobreak >nul

REM Launch Instance 3 (Key 3, Seed 3)
echo Starting Instance 3: Key #3, Seed 3
start "Teamwork Benchmark - Key 3 Seed 3" cmd /k "run_teamwork_parallel.bat 3 3"

echo.
echo ============================================================================
echo All 3 instances launched!
echo ============================================================================
echo.
echo Monitor progress in the 3 separate windows that just opened.
echo.
echo Each instance will create its own log file:
echo   - teamwork_key1_seed1_log.txt
echo   - teamwork_key2_seed2_log.txt
echo   - teamwork_key3_seed3_log.txt
echo.
echo Results will be saved in: multi-agent-gemma/ablation/
echo.
echo ============================================================================
echo.
echo Press any key to close this launcher window...
pause >nul
