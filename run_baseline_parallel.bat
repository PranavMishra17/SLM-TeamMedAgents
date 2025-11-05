@echo off
REM ============================================================================
REM Parameterized Baseline Benchmark Runner
REM
REM Usage: run_baseline_parallel.bat <KEY_NUMBER> <SEED>
REM
REM Example:
REM   run_baseline_parallel.bat 1 1    (Uses GOOGLE_API_KEY with seed 1)
REM   run_baseline_parallel.bat 2 2    (Uses GOOGLE_API_KEY2 with seed 2)
REM   run_baseline_parallel.bat 3 3    (Uses GOOGLE_API_KEY3 with seed 3)
REM
REM Run 3 instances simultaneously in separate command windows:
REM   Window 1: run_baseline_parallel.bat 1 1
REM   Window 2: run_baseline_parallel.bat 2 2
REM   Window 3: run_baseline_parallel.bat 3 3
REM ============================================================================

REM Check if parameters are provided
if "%1"=="" (
    echo ERROR: KEY_NUMBER parameter is required
    echo.
    echo Usage: run_baseline_parallel.bat KEY_NUMBER SEED
    echo.
    echo Examples:
    echo   run_baseline_parallel.bat 1 1
    echo   run_baseline_parallel.bat 2 2
    echo   run_baseline_parallel.bat 3 3
    echo.
    pause
    exit /b 1
)

if "%2"=="" (
    echo ERROR: SEED parameter is required
    echo.
    echo Usage: run_baseline_parallel.bat KEY_NUMBER SEED
    echo.
    echo Examples:
    echo   run_baseline_parallel.bat 1 1
    echo   run_baseline_parallel.bat 2 2
    echo   run_baseline_parallel.bat 3 3
    echo.
    pause
    exit /b 1
)

setlocal enabledelayedexpansion

REM Set parameters
set KEY_NUMBER=%1
set SEED=%2
set MODEL=gemma3_4b

echo ============================================================================
echo BASELINE BENCHMARK RUNNER - PARALLEL INSTANCE
echo ============================================================================
echo.
echo Configuration:
echo - API Key Number: %KEY_NUMBER% (GOOGLE_API_KEY%KEY_NUMBER%)
echo - Random Seed: %SEED%
echo - Model: %MODEL%
echo - Questions per run: 50
echo.
echo Datasets: 8
echo Methods: 3 (zero_shot, few_shot, cot)
echo Total runs for this instance: 24
echo.
echo ============================================================================
echo.

REM Define datasets
set DATASETS=medqa medmcqa mmlupro-med pubmedqa medbullets ddxplus pmc_vqa path_vqa

REM Define methods
set METHODS=zero_shot few_shot cot

REM Counter for progress
set /a TOTAL=24
set /a CURRENT=0

REM Create log file for this instance
set LOG_FILE=baseline_key%KEY_NUMBER%_seed%SEED%_log.txt
echo Baseline Benchmark Log - Key %KEY_NUMBER%, Seed %SEED% > %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%

REM Loop through datasets
for %%D in (%DATASETS%) do (
    echo.
    echo ========================================
    echo Dataset: %%D
    echo ========================================
    echo.
    echo ========================================>> %LOG_FILE%
    echo Dataset: %%D >> %LOG_FILE%
    echo ========================================>> %LOG_FILE%
    echo. >> %LOG_FILE%

    REM Loop through methods
    for %%M in (%METHODS%) do (
        set /a CURRENT+=1
        echo.
        echo [!CURRENT!/%TOTAL%] Running: %%D - %%M - Key %KEY_NUMBER% - Seed %SEED%
        echo Command: python slm_runner.py --dataset %%D --method %%M --model %MODEL% --num_questions 50 --random_seed %SEED% --key %KEY_NUMBER%
        echo.
        echo [!CURRENT!/%TOTAL%] %%D - %%M >> %LOG_FILE%

        python slm_runner.py --dataset %%D --method %%M --model %MODEL% --num_questions 50 --random_seed %SEED% --key %KEY_NUMBER%

        if errorlevel 1 (
            echo ERROR: Run failed for %%D - %%M >> %LOG_FILE%
            echo ERROR: Run failed for %%D - %%M
            echo Continuing to next run...
        ) else (
            echo SUCCESS: Completed %%D - %%M >> %LOG_FILE%
            echo SUCCESS: Completed %%D - %%M
        )
        echo. >> %LOG_FILE%

        REM Small delay between runs
        timeout /t 2 /nobreak >nul
    )
)

echo.
echo ============================================================================
echo BENCHMARK INSTANCE COMPLETE
echo ============================================================================
echo.
echo Completed %TOTAL% runs with:
echo - API Key: #%KEY_NUMBER%
echo - Seed: %SEED%
echo.
echo Results saved in: SLM_Results/%MODEL%/
echo Log saved in: %LOG_FILE%
echo.
echo Completed: %date% %time% >> %LOG_FILE%
echo ============================================================================
echo.
pause
