@echo off
REM ============================================================================
REM Parameterized Teamwork Configuration Runner
REM
REM Usage: run_teamwork_parallel.bat <KEY_NUMBER> <SEED>
REM
REM Example:
REM   run_teamwork_parallel.bat 1 1    (Uses GOOGLE_API_KEY with seed 1)
REM   run_teamwork_parallel.bat 2 2    (Uses GOOGLE_API_KEY2 with seed 2)
REM   run_teamwork_parallel.bat 3 3    (Uses GOOGLE_API_KEY3 with seed 3)
REM
REM Run 3 instances simultaneously in separate command windows:
REM   Window 1: run_teamwork_parallel.bat 1 1
REM   Window 2: run_teamwork_parallel.bat 2 2
REM   Window 3: run_teamwork_parallel.bat 3 3
REM ============================================================================

REM Check if parameters are provided
if "%1"=="" (
    echo ERROR: KEY_NUMBER parameter is required
    echo.
    echo Usage: run_teamwork_parallel.bat KEY_NUMBER SEED
    echo.
    echo Examples:
    echo   run_teamwork_parallel.bat 1 1
    echo   run_teamwork_parallel.bat 2 2
    echo   run_teamwork_parallel.bat 3 3
    echo.
    pause
    exit /b 1
)

if "%2"=="" (
    echo ERROR: SEED parameter is required
    echo.
    echo Usage: run_teamwork_parallel.bat KEY_NUMBER SEED
    echo.
    echo Examples:
    echo   run_teamwork_parallel.bat 1 1
    echo   run_teamwork_parallel.bat 2 2
    echo   run_teamwork_parallel.bat 3 3
    echo.
    pause
    exit /b 1
)

setlocal enabledelayedexpansion

REM Set parameters
set KEY_NUMBER=%1
set SEED=%2
set MODEL=gemma3_4b
set N_QUESTIONS=50
set OUTPUT_DIR=multi-agent-gemma/ablation

echo ============================================================================
echo TEAMWORK CONFIGURATION BENCHMARK - PARALLEL INSTANCE
echo ============================================================================
echo.
echo Configuration:
echo - API Key Number: %KEY_NUMBER% (GOOGLE_API_KEY%KEY_NUMBER%)
echo - Random Seed: %SEED%
echo - Model: %MODEL%
echo - Questions per run: %N_QUESTIONS%
echo - Agents per run: Dynamic (determined by algorithm)
echo - Output Directory: %OUTPUT_DIR%
echo.
echo Datasets: 8
echo Configurations: 4
echo   1. Team Orientation + Mutual Monitoring
echo   2. SMM + Trust
echo   3. Team Orientation + SMM + Leadership
echo   4. All Teamwork Components
echo.
echo Total runs for this instance: 32 (8 datasets x 4 configs)
echo.
echo ============================================================================
echo.

REM Define datasets
set DATASETS=medqa medmcqa mmlupro-med pubmedqa medbullets ddxplus pmc_vqa path_vqa

REM Counter for progress
set /a TOTAL=32
set /a CURRENT=0

REM Create log file for this instance
set LOG_FILE=teamwork_key%KEY_NUMBER%_seed%SEED%_log.txt
echo Teamwork Configuration Benchmark Log - Key %KEY_NUMBER%, Seed %SEED% > %LOG_FILE%
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

    REM Configuration 1: Team Orientation + Mutual Monitoring
    set /a CURRENT+=1
    echo.
    echo [!CURRENT!/%TOTAL%] Config 1: Team Orientation + Mutual Monitoring
    echo Dataset: %%D - Key %KEY_NUMBER% - Seed %SEED%
    echo Command: python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --team-orientation --mutual-monitoring
    echo.
    echo [!CURRENT!/%TOTAL%] Config 1: TO+MM - %%D >> %LOG_FILE%

    python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --team-orientation --mutual-monitoring

    if errorlevel 1 (
        echo ERROR: Config 1 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 1 failed for %%D
    ) else (
        echo SUCCESS: Config 1 completed for %%D >> %LOG_FILE%
        echo SUCCESS: Config 1 completed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Configuration 2: SMM + Trust
    set /a CURRENT+=1
    echo.
    echo [!CURRENT!/%TOTAL%] Config 2: SMM + Trust
    echo Dataset: %%D - Key %KEY_NUMBER% - Seed %SEED%
    echo Command: python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --smm --trust
    echo.
    echo [!CURRENT!/%TOTAL%] Config 2: SMM+Trust - %%D >> %LOG_FILE%

    python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --smm --trust

    if errorlevel 1 (
        echo ERROR: Config 2 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 2 failed for %%D
    ) else (
        echo SUCCESS: Config 2 completed for %%D >> %LOG_FILE%
        echo SUCCESS: Config 2 completed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Configuration 3: Team Orientation + SMM + Leadership
    set /a CURRENT+=1
    echo.
    echo [!CURRENT!/%TOTAL%] Config 3: Team Orientation + SMM + Leadership
    echo Dataset: %%D - Key %KEY_NUMBER% - Seed %SEED%
    echo Command: python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --team-orientation --smm --leadership
    echo.
    echo [!CURRENT!/%TOTAL%] Config 3: TO+SMM+L - %%D >> %LOG_FILE%

    python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --team-orientation --smm --leadership

    if errorlevel 1 (
        echo ERROR: Config 3 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 3 failed for %%D
    ) else (
        echo SUCCESS: Config 3 completed for %%D >> %LOG_FILE%
        echo SUCCESS: Config 3 completed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Configuration 4: All Teamwork Components
    set /a CURRENT+=1
    echo.
    echo [!CURRENT!/%TOTAL%] Config 4: All Teamwork Components
    echo Dataset: %%D - Key %KEY_NUMBER% - Seed %SEED%
    echo Command: python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --all-teamwork
    echo.
    echo [!CURRENT!/%TOTAL%] Config 4: ALL - %%D >> %LOG_FILE%

    python run_simulation_adk.py --dataset %%D --n-questions %N_QUESTIONS% --model %MODEL% --seed %SEED% --key %KEY_NUMBER% --output-dir %OUTPUT_DIR% --all-teamwork

    if errorlevel 1 (
        echo ERROR: Config 4 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 4 failed for %%D
    ) else (
        echo SUCCESS: Config 4 completed for %%D >> %LOG_FILE%
        echo SUCCESS: Config 4 completed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul
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
echo Results saved in: %OUTPUT_DIR%/
echo Log saved in: %LOG_FILE%
echo.
echo Completed: %date% %time% >> %LOG_FILE%
echo ============================================================================
echo.
pause
