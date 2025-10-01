@echo off
echo Starting comprehensive evaluation for MedGemma-4B model
echo This will run all dataset-method combinations
echo Default: 20 questions per dataset (customizable with --num_questions)
echo ================================================

python slm_runner.py --model medgemma_4b --all --num_questions 20

echo ================================================
echo Comprehensive evaluation complete!
echo Check SLM_Results/medgemma_4b/ for detailed results
echo.
echo To run with different number of questions, use:
echo python slm_runner.py --model medgemma_4b --all --num_questions 10
echo python slm_runner.py --model medgemma_4b --all --num_questions 100
pause