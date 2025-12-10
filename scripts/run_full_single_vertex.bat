@echo off
REM ============================================================================
REM Single Full Dataset Runner for Vertex AI MedGemma (moved to scripts/)
REM Usage: run_full_single_vertex.bat DATASET
REM Runs full dataset (default 500 questions) with best config from ablation
REM ============================================================================

if "%1"=="" (
    echo ERROR: DATASET parameter is required
    echo.
    echo Usage: %~nx0 DATASET
    pause
    exit /b 1
)

set DATASET=%1
set N_QUESTIONS=500
set SEED=42
set OUTPUT_DIR=multi-agent-gemma/full_vertex

if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID environment variable is not set
    echo Please set VERTEX_AI_ENDPOINT_ID (see documentation/VERTEX_AI_SETUP.md)
    pause
    exit /b 1
)

echo ============================================================================
echo FULL DATASET BENCHMARK - Vertex AI
echo Dataset: %DATASET%
echo Questions: %N_QUESTIONS%
echo Seed: %SEED%
echo Endpoint: %VERTEX_AI_ENDPOINT_ID%
echo Output Dir: %OUTPUT_DIR%
echo ============================================================================

REM Determine best config flags based on ablation results (defaults chosen)
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
) else if /i "%DATASET%"=="mmlupro-med" (
    set CONFIG_FLAGS=--smm --trust
    set CONFIG_NAME=SMM + Trust
) else (
    echo WARNING: Unknown dataset %DATASET%. Using default: All Teamwork Components
    set CONFIG_FLAGS=--all-teamwork
    set CONFIG_NAME=All Teamwork Components
)

set LOG_FILE=full_%DATASET%_vertex_log.txt
echo Full Dataset Vertex Log - %DATASET% > %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%

echo Using best config for %DATASET%: %CONFIG_NAME%
echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %DATASET% --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% %CONFIG_FLAGS%

python "%~dp0..\run_simulation_vertex_adk.py" --dataset %DATASET% --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% %CONFIG_FLAGS%

if errorlevel 1 (
    echo ERROR: Benchmark failed for %DATASET% >> %LOG_FILE%
    echo ERROR: Benchmark failed! >> %LOG_FILE%
    echo ERROR: Benchmark failed!
) else (
    echo SUCCESS: Benchmark completed for %DATASET% >> %LOG_FILE%
    echo SUCCESS: Full benchmark completed for %DATASET%
    echo Results saved in: %OUTPUT_DIR%/
    echo Log saved in: %LOG_FILE%
)

echo Completed: %date% %time% >> %LOG_FILE%
echo ============================================================================ >> %LOG_FILE%
pause
