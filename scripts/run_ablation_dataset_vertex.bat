@echo off
REM ============================================================================
REM Ablation runner for a single dataset on Vertex AI (moved to scripts/)
REM Usage: run_ablation_dataset_vertex.bat DATASET SEED
REM Runs 6 teamwork configurations for the given dataset with 50 questions
REM ============================================================================

if "%1"=="" (
    echo ERROR: DATASET parameter is required
    echo.
    echo Usage: %~nx0 DATASET SEED
    pause
    exit /b 1
)

if "%2"=="" (
    echo ERROR: SEED parameter is required
    echo.
    echo Usage: %~nx0 DATASET SEED
    pause
    exit /b 1
)

set DATASET=%1
set SEED=%2
set N_QUESTIONS=50
set OUTPUT_DIR=multi-agent-gemma/ablation_vertex

if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID not set. See documentation/VERTEX_AI_SETUP.md
    pause
    exit /b 1
)

echo ============================================================================
echo ABLATION (Dataset) - Vertex AI
echo Dataset: %DATASET%
echo Seed: %SEED%
echo Questions per run: %N_QUESTIONS%
echo Endpoint: %VERTEX_AI_ENDPOINT_ID%
echo Output Dir: %OUTPUT_DIR%
echo ============================================================================

set LOG_FILE=ablation_%DATASET%_seed%SEED%_log.txt
echo Ablation Log - %DATASET% Seed %SEED% > %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%

REM Configurations to run
set CONFIGS=1 2 3 4 5 6

for %%C in (%CONFIGS%) do (
    if %%C==1 (
        set FLAGS=--team-orientation --mutual-monitoring
        set LABEL=TO+MM
    ) else if %%C==2 (
        set FLAGS=--smm --trust
        set LABEL=SMM+Trust
    ) else if %%C==3 (
        set FLAGS=--team-orientation --smm --leadership
        set LABEL=TO+SMM+L
    ) else if %%C==4 (
        set FLAGS=--all-teamwork
        set LABEL=ALL
    ) else if %%C==5 (
        set FLAGS=--leadership --trust
        set LABEL=L+Trust
    ) else if %%C==6 (
        set FLAGS=--smm --mutual-monitoring
        set LABEL=SMM+MM
    )

    echo Running Config %%C (%LABEL%) for %DATASET% (Seed %SEED%)
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %DATASET% --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% %FLAGS%
    echo [%DATE% %TIME%] Config %%C (%LABEL%) >> %LOG_FILE%

    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %DATASET% --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% %FLAGS%

    if errorlevel 1 (
        echo ERROR: Config %%C failed for %DATASET% >> %LOG_FILE%
        echo ERROR: Config %%C failed for %DATASET%
    ) else (
        echo SUCCESS: Config %%C completed for %DATASET% >> %LOG_FILE%
    )
    timeout /t 2 /nobreak >nul
)

echo Completed ablation for %DATASET% (Seed %SEED%) >> %LOG_FILE%
echo Completed: %date% %time% >> %LOG_FILE%
pause
