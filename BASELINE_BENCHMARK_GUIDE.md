# Baseline Benchmark Guide

## Overview

Comprehensive baseline evaluation system for zero-shot, few-shot, and Chain-of-Thought (CoT) prompting across all 8 medical datasets.

## Prerequisites

### API Keys Setup

Create or update `.env` file in the project root:

```env
GOOGLE_API_KEY=your_first_api_key_here
GOOGLE_API_KEY2=your_second_api_key_here
GOOGLE_API_KEY3=your_third_api_key_here
```

**Note**: All 3 keys are required for parallel execution to avoid rate limits.

## Execution Modes

### 1. Sequential Execution (Single Key)

**Use**: When you have only 1 API key or want to run benchmarks sequentially.

```bash
run_baseline_benchmarks.bat
```

**Configuration**:
- Uses GOOGLE_API_KEY (key 1)
- Runs 72 benchmarks sequentially
- 8 datasets x 3 methods x 3 seeds
- 50 questions per run
- Estimated time: 6-10 hours (depending on rate limits)

### 2. Parallel Execution (3 Keys)

**Use**: When you have 3 API keys and want maximum throughput.

#### Option A: Launch All at Once

```bash
launch_all_parallel.bat
```

Opens 3 command windows automatically:
- Window 1: Key 1, Seed 1 (24 runs)
- Window 2: Key 2, Seed 2 (24 runs)
- Window 3: Key 3, Seed 3 (24 runs)

**Total**: 72 runs across 3 windows
**Estimated time**: 2-4 hours (3x faster)

#### Option B: Manual Launch

Open 3 separate command prompts and run:

**Terminal 1**:
```bash
run_baseline_parallel.bat 1 1
```

**Terminal 2**:
```bash
run_baseline_parallel.bat 2 2
```

**Terminal 3**:
```bash
run_baseline_parallel.bat 3 3
```

## Benchmark Configuration

### Datasets (8 Total)
1. **medqa** - USMLE-style medical questions
2. **medmcqa** - Indian medical entrance exams
3. **mmlupro-med** - MMLU-Pro Health subset
4. **pubmedqa** - Biomedical research questions
5. **medbullets** - Clinical case questions
6. **ddxplus** - Differential diagnosis
7. **pmc_vqa** - Medical visual QA (with images)
8. **path_vqa** - Pathology image analysis

### Prompting Methods (3 Total)
1. **zero_shot** - No examples, direct question answering
2. **few_shot** - 2-3 example questions with answers
3. **cot** - Chain-of-Thought reasoning with step-by-step examples

### Seeds (3 Total)
- Seed 1, Seed 2, Seed 3
- Different random samples from each dataset
- Ensures robustness of results

## Results Structure

Results are saved in:

```
SLM_Results/
└── gemma3_4b/
    ├── medqa/
    │   ├── zero_shot/
    │   │   ├── seed_1_results.json
    │   │   ├── seed_2_results.json
    │   │   ├── seed_3_results.json
    │   │   └── aggregated_summary.json
    │   ├── few_shot/
    │   │   ├── seed_1_results.json
    │   │   ├── seed_2_results.json
    │   │   ├── seed_3_results.json
    │   │   └── aggregated_summary.json
    │   └── cot/
    │       ├── seed_1_results.json
    │       ├── seed_2_results.json
    │       ├── seed_3_results.json
    │       └── aggregated_summary.json
    ├── medmcqa/
    │   └── [same structure]
    └── [... other datasets]
```

### Per-Run Results

Each run creates:
- **{seed}_results.json** - Detailed question-by-question results
  - Question text
  - Ground truth answer
  - Predicted answer
  - Is correct (boolean)
  - Response time
  - Token counts (input/output/total)
  - Full model response

### Aggregated Results

**Method Level** (`aggregated_summary.json` in each method folder):
```json
{
  "dataset": "medqa",
  "method": "zero_shot",
  "model": "gemma3_4b",
  "runs": [
    {
      "seed": 1,
      "accuracy": 0.68,
      "total_questions": 50,
      "correct": 34
    },
    {
      "seed": 2,
      "accuracy": 0.70,
      "total_questions": 50,
      "correct": 35
    },
    {
      "seed": 3,
      "accuracy": 0.66,
      "total_questions": 50,
      "correct": 33
    }
  ],
  "mean_accuracy": 0.68,
  "std_accuracy": 0.016,
  "total_questions": 150,
  "total_correct": 102
}
```

**Dataset Level** (combines all 3 methods):
- Comparison across zero_shot, few_shot, cot
- Mean and std deviation for each method
- Best performing method

**Model Level** (combines all datasets):
- Overall accuracy across all benchmarks
- Per-dataset breakdown
- Per-method breakdown

## Metrics Tracked

### Per Question
- **Correctness**: Binary correct/incorrect
- **Response Time**: Seconds per question
- **Token Usage**:
  - Input tokens
  - Output tokens
  - Total tokens
- **Full Response**: Complete model output

### Per Run (50 questions)
- **Accuracy**: Correct/Total
- **Average Time**: Mean time per question
- **Total Time**: Full run duration
- **Token Statistics**:
  - Total tokens used
  - Average tokens per question
  - Input/output token breakdown

### Aggregated (Across Seeds)
- **Mean Accuracy**: Average across 3 seeds
- **Standard Deviation**: Accuracy variance
- **95% Confidence Interval**: Statistical confidence
- **Total Questions**: 150 per method (3 seeds × 50 questions)

## Progress Monitoring

### Log Files (Parallel Execution)

Each parallel instance creates a log file:
- `baseline_key1_seed1_log.txt`
- `baseline_key2_seed2_log.txt`
- `baseline_key3_seed3_log.txt`

### Console Output

Real-time progress shown:
```
[5/24] Running: medqa - cot - Key 1 - Seed 1
Command: python slm_runner.py --dataset medqa --method cot --model gemma3_4b --num_questions 50 --random_seed 1 --key 1

SUCCESS: Completed medqa - cot
```

## Troubleshooting

### Rate Limit Errors

**Problem**: "Rate limit exceeded" errors

**Solution**:
1. Ensure you're using parallel execution with 3 keys
2. Each key has separate rate limits
3. Verify all 3 keys are valid in `.env`

### Missing Results

**Problem**: Some runs failed or results missing

**Solution**:
1. Check log files for error messages
2. Re-run failed configurations manually:
   ```bash
   python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 50 --random_seed 1 --key 1
   ```

### API Key Not Found

**Problem**: "GOOGLE_API_KEY2 environment variable not set"

**Solution**:
1. Verify `.env` file exists in project root
2. Verify key naming: `GOOGLE_API_KEY2` (not `GOOGLE_API_KEY_2`)
3. Restart command prompt to reload environment

## Manual Execution

For fine-grained control, run individual configurations:

```bash
# Single dataset, method, seed
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 50 --random_seed 1 --key 1

# With custom output directory
python slm_runner.py --dataset medqa --method few_shot --model gemma3_4b --num_questions 50 --random_seed 2 --key 2 --output_dir custom_results/

# Different number of questions
python slm_runner.py --dataset medqa --method cot --model gemma3_4b --num_questions 100 --random_seed 3 --key 3
```

## Expected Runtime

### Sequential (Single Key)
- **Per run**: 5-8 minutes
- **72 runs**: 6-10 hours
- **Bottleneck**: Rate limits on single key

### Parallel (3 Keys)
- **Per run**: 5-8 minutes
- **24 runs per instance**: 2-3.5 hours
- **3 instances simultaneously**: 2-4 hours total
- **Speedup**: ~3x faster

## Analysis Scripts

After benchmarks complete, analyze results:

```bash
# Generate comparison tables
python scripts/analyze_baseline_results.py

# Compare with multi-agent results
python scripts/compare_baseline_vs_multiagent.py
```

## Key Findings Expected

Baseline benchmarks establish:
1. **Zero-shot performance**: Model capability without examples
2. **Few-shot improvement**: Benefit of example questions
3. **CoT effectiveness**: Value of step-by-step reasoning
4. **Dataset difficulty**: Which datasets are hardest
5. **Seed stability**: Result variance across different samples

These baselines are crucial for evaluating multi-agent system improvements.
