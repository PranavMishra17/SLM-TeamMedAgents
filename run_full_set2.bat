@echo off
REM ============================================================================
REM Full Dataset Benchmark - Set 2 (Parallel Execution)
REM
REM Runs 500 questions (or max available) for 4 datasets in parallel:
REM   1. medmcqa (Key 1)
REM   2. pmc_vqa (Key 2)
REM   3. pubmedqa (Key 3)
REM   4. mmlupro-med (Key 4)
REM
REM Each dataset runs with all teamwork components activated
REM Single seed (42) for reproducibility
REM ============================================================================

echo ============================================================================
echo FULL DATASET BENCHMARK - SET 2
echo ============================================================================
echo.
echo Running 4 datasets in parallel (500q each):
echo.
echo   Dataset 1: medmcqa      ^(API Key 1^) - Config: SMM+Trust
echo   Dataset 2: pmc_vqa      ^(API Key 2^) - Config: SMM+Trust
echo   Dataset 3: pubmedqa     ^(API Key 3^) - Config: SMM+Trust
echo   Dataset 4: mmlupro      ^(API Key 4^) - Config: SMM+Trust
echo.
echo Configuration:
echo - Questions: 500 per dataset
echo - Agents: Dynamic (determined by algorithm)
echo - Model: gemma3_4b
echo - Seed: 42
echo - Teamwork: Best config per dataset (from ablation study)
echo - Output: multi-agent-gemma/full/
echo.
echo ============================================================================
echo.
echo WARNING: Ensure all 4 API keys are set:
echo   - GOOGLE_API_KEY   ^(for medmcqa^)
echo   - GOOGLE_API_KEY2  ^(for pmc_vqa^)
echo   - GOOGLE_API_KEY3  ^(for pubmedqa^)
echo   - GOOGLE_API_KEY4  ^(for mmlupro-med^)
echo.
echo ============================================================================
echo.
echo This will take several hours to complete.
echo Each dataset will run in its own window.
echo.
pause

REM Check if run_full_single.bat exists (we'll use it for individual runs)
if not exist "run_full_single.bat" (
    echo ERROR: run_full_single.bat not found
    echo Please ensure the helper script is in the same directory
    pause
    exit /b 1
)

echo.
echo Launching 4 parallel instances...
echo.

REM Launch Instance 1: medmcqa with Key 1
echo Starting Instance 1: medmcqa ^(Key 1^)
start "Full Benchmark - medmcqa" cmd /k "run_full_single.bat medmcqa 1"
timeout /t 3 /nobreak >nul

REM Launch Instance 2: pmc_vqa with Key 2
echo Starting Instance 2: pmc_vqa ^(Key 2^)
start "Full Benchmark - pmc_vqa" cmd /k "run_full_single.bat pmc_vqa 2"
timeout /t 3 /nobreak >nul

REM Launch Instance 3: pubmedqa with Key 3
echo Starting Instance 3: pubmedqa ^(Key 3^)
start "Full Benchmark - pubmedqa" cmd /k "run_full_single.bat pubmedqa 3"
timeout /t 3 /nobreak >nul

REM Launch Instance 4: mmlupro with Key 4
echo Starting Instance 4: mmlupro ^(Key 4^)
start "Full Benchmark - mmlupro" cmd /k "run_full_single.bat mmlupro 4"

echo.
echo ============================================================================
echo All 4 instances launched!
echo ============================================================================
echo.
echo Monitor progress in the 4 separate windows.
echo.
echo Log files:
echo   - full_medmcqa_key1_log.txt
echo   - full_pmc_vqa_key2_log.txt
echo   - full_pubmedqa_key3_log.txt
echo   - full_mmlupro_key4_log.txt
echo.
echo Results: multi-agent-gemma/full/
echo.
echo ============================================================================
echo.
echo Press any key to close this launcher...
pause >nul
