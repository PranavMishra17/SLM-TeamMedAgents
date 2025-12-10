@echo off
REM ============================================================================
REM Single Full Dataset Runner
REM
REM Usage: run_full_single.bat <DATASET> <KEY_NUMBER>
REM
REM Example:
REM   run_full_single.bat medqa 1
REM   run_full_single.bat path_vqa 2
REM ============================================================================

REM Check parameters
if "%1"=="" (
    echo ERROR: DATASET parameter is required
    echo.
    echo Usage: run_full_single.bat DATASET KEY_NUMBER
    echo.
    echo Examples:
    echo   run_full_single.bat medqa 1
    echo   run_full_single.bat path_vqa 2
    echo.
    pause
    exit /b 1
)

if "%2"=="" (
    echo ERROR: KEY_NUMBER parameter is required
    echo.
    echo Usage: run_full_single.bat DATASET KEY_NUMBER
    echo.
    echo Examples:
    echo   run_full_single.bat medqa 1
    echo   run_full_single.bat path_vqa 2
    echo.
    pause
    exit /b 1
)

setlocal enabledelayedexpansion

REM Set parameters
set DATASET=%1
set KEY_NUMBER=%2
set MODEL=gemma3_4b
set N_QUESTIONS=500
set SEED=42
set OUTPUT_DIR=multi-agent-gemma/full

echo ============================================================================
echo FULL DATASET BENCHMARK - SINGLE INSTANCE
echo ============================================================================
echo.
echo Configuration:
echo - Dataset: %DATASET%
echo - API Key Number: %KEY_NUMBER% (GOOGLE_API_KEY%KEY_NUMBER%)
echo - Questions: %N_QUESTIONS%
echo - Agents: Dynamic (determined by algorithm)
echo - Model: %MODEL%
echo - Seed: %SEED%
echo - Teamwork: Best config from ablation study
echo - Output Directory: %OUTPUT_DIR%
echo.
echo ============================================================================
echo.

REM Create log file
set LOG_FILE=full_%DATASET%_key%KEY_NUMBER%_log.txt
echo Full Dataset Benchmark Log - %DATASET% (Key %KEY_NUMBER%) > %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%
echo. >> %LOG_FILE%
echo Configuration: >> %LOG_FILE%
echo - Dataset: %DATASET% >> %LOG_FILE%
echo - Questions: %N_QUESTIONS% >> %LOG_FILE%
echo - Agents: Dynamic >> %LOG_FILE%
echo - Model: %MODEL% >> %LOG_FILE%
echo - Seed: %SEED% >> %LOG_FILE%
echo - API Key: %KEY_NUMBER% >> %LOG_FILE%
echo. >> %LOG_FILE%

echo Running full dataset benchmark...
echo.

REM Determine best config flags based on ablation results
if /i "%DATASET%"=="ddxplus" (
    set CONFIG_FLAGS=--all-teamwork
    set CONFIG_NAME=All Teamwork Components
) else if /i "%DATASET%"=="medbullets" (
    set CONFIG_FLAGS=--smm --trust
    set CONFIG_NAME=SMM + Trust
) else if /i "%DATASET%"=="medmcqa" (
    set CONFIG_FLAGS=--smm --trust
    set CONFIG_NAME=SMM + Trust
) else if /i "%DATASET%"=="medqa" (
    set CONFIG_FLAGS=--team-orientation --smm --leadership
    set CONFIG_NAME=TO + SMM + Leadership
) else if /i "%DATASET%"=="path_vqa" (
    set CONFIG_FLAGS=--team-orientation --mutual-monitoring
    set CONFIG_NAME=TO + Mutual Monitoring
) else if /i "%DATASET%"=="pmc_vqa" (
    set CONFIG_FLAGS=--smm --trust
    set CONFIG_NAME=SMM + Trust
) else if /i "%DATASET%"=="pubmedqa" (
    set CONFIG_FLAGS=--smm --trust
    set CONFIG_NAME=SMM + Trust
) else if /i "%DATASET%"=="mmlupro" (
    set CONFIG_FLAGS=--smm --trust
    set CONFIG_NAME=SMM + Trust
) else (
    echo ERROR: Unknown dataset %DATASET%
    echo Using default: All Teamwork Components
    set CONFIG_FLAGS=--all-teamwork
    set CONFIG_NAME=All Teamwork Components
)

echo Using best config for %DATASET%: !CONFIG_NAME!
echo Command: python run_simulation_adk.py --dataset %DATASET% --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% !CONFIG_FLAGS!
echo.
echo. >> %LOG_FILE%
echo Best Config: !CONFIG_NAME! >> %LOG_FILE%
echo Command: python run_simulation_adk.py --dataset %DATASET% --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% !CONFIG_FLAGS! >> %LOG_FILE%
echo. >> %LOG_FILE%

REM Run the benchmark
python run_simulation_adk.py --dataset %DATASET% --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% !CONFIG_FLAGS!

if errorlevel 1 (
    echo. >> %LOG_FILE%
    echo ERROR: Benchmark failed for %DATASET% >> %LOG_FILE%
    echo ERROR: Benchmark failed! >> %LOG_FILE%
    echo. >> %LOG_FILE%
    echo.
    echo ============================================================================
    echo ERROR: Benchmark failed for %DATASET%
    echo ============================================================================
    echo.
    echo Check log file: %LOG_FILE%
    echo.
) else (
    echo. >> %LOG_FILE%
    echo SUCCESS: Benchmark completed for %DATASET% >> %LOG_FILE%
    echo. >> %LOG_FILE%
    echo.
    echo ============================================================================
    echo SUCCESS: Full benchmark completed for %DATASET%
    echo ============================================================================
    echo.
    echo Results saved in: %OUTPUT_DIR%/
    echo Log saved in: %LOG_FILE%
    echo.
)

echo Completed: %date% %time% >> %LOG_FILE%
echo ============================================================================ >> %LOG_FILE%

echo.
echo ============================================================================
echo.
pause
