@echo off
REM ============================================================================
REM Launch 3 Parallel Baseline Benchmark Instances
REM
REM This script launches 3 separate command windows running baseline benchmarks
REM simultaneously with different API keys and seeds to maximize throughput.
REM
REM Prerequisites:
REM   - .env file must contain:
REM     GOOGLE_API_KEY=...
REM     GOOGLE_API_KEY2=...
REM     GOOGLE_API_KEY3=...
REM
REM Each instance runs:
REM   - 8 datasets
REM   - 3 methods (zero_shot, few_shot, cot)
REM   - 50 questions per run
REM   = 24 runs per instance
REM
REM Total across all 3 instances: 72 runs
REM ============================================================================

echo ============================================================================
echo LAUNCHING 3 PARALLEL BASELINE BENCHMARK INSTANCES
echo ============================================================================
echo.
echo This will open 3 separate command windows running simultaneously:
echo.
echo   Window 1: API Key 1, Seed 1
echo   Window 2: API Key 2, Seed 2
echo   Window 3: API Key 3, Seed 3
echo.
echo Each window will run 24 benchmark runs (8 datasets x 3 methods).
echo Total: 72 runs across all 3 windows.
echo.
echo Make sure you have GOOGLE_API_KEY, GOOGLE_API_KEY2, and GOOGLE_API_KEY3
echo set in your .env file before proceeding.
echo.
echo ============================================================================
echo.
pause

echo Launching instance 1 (Key 1, Seed 1)...
start "Baseline Benchmarks - Key 1 Seed 1" cmd /k "run_baseline_parallel.bat 1 1"
timeout /t 2 /nobreak >nul

echo Launching instance 2 (Key 2, Seed 2)...
start "Baseline Benchmarks - Key 2 Seed 2" cmd /k "run_baseline_parallel.bat 2 2"
timeout /t 2 /nobreak >nul

echo Launching instance 3 (Key 3, Seed 3)...
start "Baseline Benchmarks - Key 3 Seed 3" cmd /k "run_baseline_parallel.bat 3 3"
timeout /t 2 /nobreak >nul

echo.
echo ============================================================================
echo All 3 instances launched successfully!
echo ============================================================================
echo.
echo Monitor progress in the 3 command windows.
echo Each instance will create its own log file:
echo   - baseline_key1_seed1_log.txt
echo   - baseline_key2_seed2_log.txt
echo   - baseline_key3_seed3_log.txt
echo.
echo Results will be saved in: SLM_Results/gemma3_4b/
echo.
echo ============================================================================
echo.
pause
