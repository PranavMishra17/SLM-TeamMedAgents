@echo off
REM ============================================================================
REM Full Dataset Benchmark - Set 1 (Parallel Execution)
REM
REM Runs 500 questions (or max available) for 4 datasets in parallel:
REM   1. medqa (Key 1)
REM   2. path_vqa (Key 2)
REM   3. medbullets (Key 3)
REM   4. ddxplus (Key 4)
REM
REM Each dataset runs with all teamwork components activated
REM Single seed (42) for reproducibility
REM ============================================================================

echo ============================================================================
echo FULL DATASET BENCHMARK - SET 1
echo ============================================================================
echo.
echo Running 4 datasets in parallel (500q each):
echo.
echo   Dataset 1: medqa        ^(API Key 1^) - Config: TO+SMM+Leadership
echo   Dataset 2: path_vqa     ^(API Key 2^) - Config: TO+Mutual Monitoring
echo   Dataset 3: medbullets   ^(API Key 3^) - Config: SMM+Trust
echo   Dataset 4: ddxplus      ^(API Key 4^) - Config: All Teamwork
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
echo   - GOOGLE_API_KEY   ^(for medqa^)
echo   - GOOGLE_API_KEY2  ^(for path_vqa^)
echo   - GOOGLE_API_KEY3  ^(for medbullets^)
echo   - GOOGLE_API_KEY4  ^(for ddxplus^)
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

REM Launch Instance 1: medqa with Key 1
echo Starting Instance 1: medqa ^(Key 1^)
start "Full Benchmark - medqa" cmd /k "run_full_single.bat medqa 1"
timeout /t 3 /nobreak >nul

REM Launch Instance 2: path_vqa with Key 2
echo Starting Instance 2: path_vqa ^(Key 2^)
start "Full Benchmark - path_vqa" cmd /k "run_full_single.bat path_vqa 2"
timeout /t 3 /nobreak >nul

REM Launch Instance 3: medbullets with Key 3
echo Starting Instance 3: medbullets ^(Key 3^)
start "Full Benchmark - medbullets" cmd /k "run_full_single.bat medbullets 3"
timeout /t 3 /nobreak >nul

REM Launch Instance 4: ddxplus with Key 4
echo Starting Instance 4: ddxplus ^(Key 4^)
start "Full Benchmark - ddxplus" cmd /k "run_full_single.bat ddxplus 4"

echo.
echo ============================================================================
echo All 4 instances launched!
echo ============================================================================
echo.
echo Monitor progress in the 4 separate windows.
echo.
echo Log files:
echo   - full_medqa_key1_log.txt
echo   - full_path_vqa_key2_log.txt
echo   - full_medbullets_key3_log.txt
echo   - full_ddxplus_key4_log.txt
echo.
echo Results: multi-agent-gemma/full/
echo.
echo ============================================================================
echo.
echo Press any key to close this launcher...
pause >nul
