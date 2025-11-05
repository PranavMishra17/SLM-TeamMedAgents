@echo off
REM ============================================================================
REM Baseline Benchmarks: Zero-Shot, Few-Shot, and CoT Prompting
REM
REM Runs all 3 prompting methods across all 8 datasets with 3 different seeds
REM Total: 8 datasets x 3 methods x 3 seeds = 72 runs
REM Each run processes 50 questions
REM ============================================================================

echo ============================================================================
echo BASELINE BENCHMARK SUITE
echo ============================================================================
echo.
echo Configuration:
echo - Datasets: 8 (medqa, medmcqa, mmlupro-med, pubmedqa, medbullets, ddxplus, pmc_vqa, path_vqa)
echo - Methods: 3 (zero_shot, few_shot, cot)
echo - Seeds: 3 (1, 2, 3)
echo - Questions per run: 50
echo - Total runs: 72
echo.
echo ============================================================================
echo.

setlocal enabledelayedexpansion

REM Set base model
set MODEL=gemma3_4b

REM Define datasets
set DATASETS=medqa medmcqa mmlupro-med pubmedqa medbullets ddxplus pmc_vqa path_vqa

REM Define methods
set METHODS=zero_shot few_shot cot

REM Define seeds
set SEEDS=1 2 3

REM Counter for progress
set /a TOTAL=72
set /a CURRENT=0

REM Loop through datasets
for %%D in (%DATASETS%) do (
    echo.
    echo ========================================
    echo Dataset: %%D
    echo ========================================
    echo.

    REM Loop through methods
    for %%M in (%METHODS%) do (
        echo.
        echo --- Method: %%M ---

        REM Loop through seeds
        for %%S in (%SEEDS%) do (
            set /a CURRENT+=1
            echo.
            echo [!CURRENT!/%TOTAL%] Running: %%D - %%M - Seed %%S
            echo Command: python slm_runner.py --dataset %%D --method %%M --model %MODEL% --num_questions 50 --random_seed %%S
            echo.

            python slm_runner.py --dataset %%D --method %%M --model %MODEL% --num_questions 50 --random_seed %%S

            if errorlevel 1 (
                echo ERROR: Run failed for %%D - %%M - Seed %%S
                echo Continuing to next run...
            ) else (
                echo SUCCESS: Completed %%D - %%M - Seed %%S
            )

            REM Small delay between runs
            timeout /t 2 /nobreak >nul
        )
    )
)

echo.
echo ============================================================================
echo BENCHMARK SUITE COMPLETE
echo ============================================================================
echo.
echo Completed %TOTAL% runs
echo Results saved in: SLM_Results/%MODEL%/
echo.
echo Summary structure:
echo   SLM_Results/%MODEL%/
echo     - medqa/
echo       - zero_shot/
echo       - few_shot/
echo       - cot/
echo     - medmcqa/
echo       - zero_shot/
echo       - few_shot/
echo       - cot/
echo     [... etc for all datasets ...]
echo.
echo ============================================================================
pause
