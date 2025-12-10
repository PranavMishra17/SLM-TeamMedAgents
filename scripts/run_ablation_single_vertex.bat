@echo off
REM ============================================================================
REM Ablation Single Instance for Vertex AI MedGemma (moved to scripts/)
REM Usage: run_ablation_single_vertex.bat SEED
REM Runs 50-question ablation across 8 datasets and 6 teamwork configs
REM ============================================================================

if "%1"=="" (
    echo ERROR: SEED parameter is required
    echo.
    echo Usage: %~nx0 SEED
    pause
    exit /b 1
)

set SEED=%1
set N_QUESTIONS=50
set OUTPUT_DIR=multi-agent-gemma/ablation_vertex

REM Verify Vertex env vars
if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID environment variable is not set
    echo Please set VERTEX_AI_ENDPOINT_ID (see documentation/VERTEX_AI_SETUP.md)
    pause
    exit /b 1
)

if /i "%GOOGLE_GENAI_USE_VERTEXAI%" neq "TRUE" (
    echo WARNING: GOOGLE_GENAI_USE_VERTEXAI is not set to TRUE. Continuing anyway.
)

echo ============================================================================
echo ABLATION (Single Instance) - Vertex AI
echo Seed: %SEED%
echo Questions per run: %N_QUESTIONS%
echo Endpoint: %VERTEX_AI_ENDPOINT_ID%
echo Output Dir: %OUTPUT_DIR%
echo ============================================================================

set DATASETS=medqa medmcqa mmlupro-med pubmedqa medbullets ddxplus pmc_vqa path_vqa

REM Define 6 configurations
setlocal enabledelayedexpansion
set /a TOTAL=0
for %%D in (%DATASETS%) do (
    set /a TOTAL+=6
)

set /a CURRENT=0

set LOG_FILE=ablation_seed%SEED%_log.txt
echo Ablation Vertex Log - Seed %SEED% > %LOG_FILE%
echo Started: %date% %time% >> %LOG_FILE%

for %%D in (%DATASETS%) do (
    echo.
    echo ========================================
    echo Dataset: %%D
    echo ========================================
    echo.

    REM Config 1
    set /a CURRENT+=1
    echo [!CURRENT!/%TOTAL%] Config 1: Team Orientation + Mutual Monitoring - %%D
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --team-orientation --mutual-monitoring
    echo [!CURRENT!/%TOTAL%] Config 1: TO+MM - %%D >> %LOG_FILE%
    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --team-orientation --mutual-monitoring
    if errorlevel 1 (
        echo ERROR: Config 1 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 1 failed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Config 2
    set /a CURRENT+=1
    echo [!CURRENT!/%TOTAL%] Config 2: SMM + Trust - %%D
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --smm --trust
    echo [!CURRENT!/%TOTAL%] Config 2: SMM+Trust - %%D >> %LOG_FILE%
    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --smm --trust
    if errorlevel 1 (
        echo ERROR: Config 2 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 2 failed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Config 3
    set /a CURRENT+=1
    echo [!CURRENT!/%TOTAL%] Config 3: Team Orientation + SMM + Leadership - %%D
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --team-orientation --smm --leadership
    echo [!CURRENT!/%TOTAL%] Config 3: TO+SMM+L - %%D >> %LOG_FILE%
    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --team-orientation --smm --leadership
    if errorlevel 1 (
        echo ERROR: Config 3 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 3 failed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Config 4
    set /a CURRENT+=1
    echo [!CURRENT!/%TOTAL%] Config 4: All Teamwork Components - %%D
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --all-teamwork
    echo [!CURRENT!/%TOTAL%] Config 4: ALL - %%D >> %LOG_FILE%
    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --all-teamwork
    if errorlevel 1 (
        echo ERROR: Config 4 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 4 failed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Config 5
    set /a CURRENT+=1
    echo [!CURRENT!/%TOTAL%] Config 5: Leadership + Trust - %%D
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --leadership --trust
    echo [!CURRENT!/%TOTAL%] Config 5: L+Trust - %%D >> %LOG_FILE%
    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --leadership --trust
    if errorlevel 1 (
        echo ERROR: Config 5 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 5 failed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul

    REM Config 6
    set /a CURRENT+=1
    echo [!CURRENT!/%TOTAL%] Config 6: SMM + Mutual Monitoring - %%D
    echo Command: python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --smm --mutual-monitoring
    echo [!CURRENT!/%TOTAL%] Config 6: SMM+MM - %%D >> %LOG_FILE%
    python "%~dp0..\run_simulation_vertex_adk.py" --dataset %%D --n-questions %N_QUESTIONS% --endpoint-id %VERTEX_AI_ENDPOINT_ID% --project-id %GOOGLE_CLOUD_PROJECT% --location %GOOGLE_CLOUD_LOCATION% --seed %SEED% --output-dir %OUTPUT_DIR% --smm --mutual-monitoring
    if errorlevel 1 (
        echo ERROR: Config 6 failed for %%D >> %LOG_FILE%
        echo ERROR: Config 6 failed for %%D
    )
    echo. >> %LOG_FILE%
    timeout /t 2 /nobreak >nul
)

echo.
echo ============================================================================
echo ABLATION INSTANCE COMPLETE - Seed %SEED%
echo Results saved in: %OUTPUT_DIR%/
echo Log saved in: %LOG_FILE%
echo Completed: %date% %time% >> %LOG_FILE%
echo ============================================================================
echo.
pause
