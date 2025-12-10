@echo off
REM ============================================================================
REM Ablation Master Launcher for Vertex AI MedGemma (moved to scripts/)
REM Launches 3 parallel instances for seeds 111, 222, 333
REM Each instance runs all datasets x 6 configs with 50 questions
REM Usage: run_ablation_all_vertex.bat
REM ============================================================================

echo ============================================================================
echo ABLATION MASTER LAUNCHER - Vertex AI
echo This will launch 3 parallel instances (Seeds: 111,222,333)
echo Ensure the following env vars are set:
echo - VERTEX_AI_ENDPOINT_ID
echo - GOOGLE_CLOUD_PROJECT
echo - GOOGLE_CLOUD_LOCATION (optional)
echo - GOOGLE_GENAI_USE_VERTEXAI=TRUE
echo ============================================================================

if "%VERTEX_AI_ENDPOINT_ID%"=="" (
    echo ERROR: VERTEX_AI_ENDPOINT_ID not set. See documentation/VERTEX_AI_SETUP.md
    pause
    exit /b 1
)

if /i "%GOOGLE_GENAI_USE_VERTEXAI%" neq "TRUE" (
    echo WARNING: GOOGLE_GENAI_USE_VERTEXAI is not set to TRUE. Continuing anyway.
)

REM We'll run each seed across all 8 datasets but limit concurrency to 4 parallel dataset runs at a time.

set DATASETS_A=medqa medmcqa mmlupro-med pubmedqa
set DATASETS_B=medbullets ddxplus pmc_vqa path_vqa

REM Seed 111: run group A then group B (4 parallel each)
echo Launching Seed 111 - Group A (4 parallel)...
for %%D in (%DATASETS_A%) do (
    start "Ablation 111 - %%D" cmd /k "%~dp0run_ablation_dataset_vertex.bat %%D 111"
    timeout /t 1 /nobreak >nul
)
timeout /t 5 /nobreak >nul
echo Launching Seed 111 - Group B (4 parallel)...
for %%D in (%DATASETS_B%) do (
    start "Ablation 111 - %%D" cmd /k "%~dp0run_ablation_dataset_vertex.bat %%D 111"
    timeout /t 1 /nobreak >nul
)

REM Seed 222: run group A then group B
echo Launching Seed 222 - Group A (4 parallel)...
for %%D in (%DATASETS_A%) do (
    start "Ablation 222 - %%D" cmd /k "%~dp0run_ablation_dataset_vertex.bat %%D 222"
    timeout /t 1 /nobreak >nul
)
timeout /t 5 /nobreak >nul
echo Launching Seed 222 - Group B (4 parallel)...
for %%D in (%DATASETS_B%) do (
    start "Ablation 222 - %%D" cmd /k "%~dp0run_ablation_dataset_vertex.bat %%D 222"
    timeout /t 1 /nobreak >nul
)

REM Seed 333: run group A then group B
echo Launching Seed 333 - Group A (4 parallel)...
for %%D in (%DATASETS_A%) do (
    start "Ablation 333 - %%D" cmd /k "%~dp0run_ablation_dataset_vertex.bat %%D 333"
    timeout /t 1 /nobreak >nul
)
timeout /t 5 /nobreak >nul
echo Launching Seed 333 - Group B (4 parallel)...
for %%D in (%DATASETS_B%) do (
    start "Ablation 333 - %%D" cmd /k "%~dp0run_ablation_dataset_vertex.bat %%D 333"
    timeout /t 1 /nobreak >nul
)

echo.
echo ============================================================================
echo All instances launched. Monitor the 3 windows that opened.
echo ============================================================================
pause
