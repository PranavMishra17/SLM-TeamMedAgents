# Multi-Key Baseline Benchmark Setup - Complete

## Summary

Successfully added multi-key API support and parallel execution capabilities for baseline benchmarks to avoid rate limits and maximize throughput.

## Changes Made

### 1. Multi-Key API Support

**Modified Files**:
- `chat_instances.py` - Added `_get_api_key()` method to support GOOGLE_API_KEY, GOOGLE_API_KEY2, GOOGLE_API_KEY3
- `slm_runner.py` - Added `--key` parameter and `key_number` to SLMMethodRunner constructor

**Key Support**:
```python
# Key 1: GOOGLE_API_KEY or GEMINI_API_KEY
# Key 2: GOOGLE_API_KEY2
# Key 3: GOOGLE_API_KEY3
# ... and so on
```

**Usage**:
```bash
python slm_runner.py --dataset medqa --method zero_shot --key 1
python slm_runner.py --dataset medqa --method zero_shot --key 2
python slm_runner.py --dataset medqa --method zero_shot --key 3
```

### 2. Parallel Execution Scripts

**Created Files**:

#### A. run_baseline_parallel.bat
- Parameterized batch file
- Takes KEY_NUMBER and SEED as arguments
- Runs 24 benchmarks (8 datasets × 3 methods)
- Creates individual log files per instance

**Usage**:
```bash
run_baseline_parallel.bat 1 1  # Key 1, Seed 1
run_baseline_parallel.bat 2 2  # Key 2, Seed 2
run_baseline_parallel.bat 3 3  # Key 3, Seed 3
```

#### B. launch_all_parallel.bat
- Launches 3 instances automatically
- Opens 3 separate command windows
- Each uses different key and seed
- Total: 72 runs across all windows

**Usage**:
```bash
launch_all_parallel.bat  # One-click launch of all 3 instances
```

#### C. run_baseline_benchmarks.bat
- Original sequential runner
- Uses single key (Key 1)
- Runs all 72 benchmarks sequentially
- Takes 6-10 hours vs 2-4 hours for parallel

### 3. Comprehensive Metrics Tracking

**Already Implemented in slm_runner.py**:

#### Per-Question Metrics:
- Correctness (boolean)
- Response time (seconds)
- Token usage (input/output/total)
- Full model response
- Ground truth vs predicted answer

#### Per-Run Metrics (50 questions):
- Overall accuracy
- Total time and average time per question
- Total token counts
- Error count and details
- Timestamp

#### Aggregated Metrics:

**Method Level** (`aggregate_method_results()`):
- Combines all runs with same dataset + method
- Calculates mean accuracy across seeds
- Standard deviation
- Total questions answered

**Dataset Level** (`aggregate_dataset_results()`):
- Compares zero_shot vs few_shot vs cot
- Best performing method
- Overall dataset difficulty

**Model Level** (`aggregate_model_results()`):
- Overall performance across all datasets
- Per-dataset breakdown
- Per-method breakdown
- Comprehensive model evaluation

### 4. Results Structure

```
SLM_Results/
└── gemma3_4b/
    ├── medqa/
    │   ├── zero_shot/
    │   │   ├── seed_1_results.json          # Individual run
    │   │   ├── seed_2_results.json
    │   │   ├── seed_3_results.json
    │   │   ├── summary.json                  # Run summaries
    │   │   └── aggregated_summary.json       # Method aggregation
    │   ├── few_shot/
    │   │   └── [same structure]
    │   ├── cot/
    │   │   └── [same structure]
    │   └── dataset_summary.json              # Dataset aggregation
    ├── medmcqa/
    │   └── [same structure]
    ├── [... 6 more datasets]
    └── model_summary.json                    # Model-level aggregation
```

### 5. Execution Modes

#### Sequential (Original)
```bash
run_baseline_benchmarks.bat
```
- 72 runs sequentially
- Single API key
- 6-10 hours
- Use when you have only 1 key

#### Parallel (New)
```bash
launch_all_parallel.bat
```
- 3 instances simultaneously
- 24 runs per instance
- Different keys and seeds
- 2-4 hours (3x speedup)
- Use when you have 3 keys

#### Manual Parallel
```bash
# Terminal 1
run_baseline_parallel.bat 1 1

# Terminal 2
run_baseline_parallel.bat 2 2

# Terminal 3
run_baseline_parallel.bat 3 3
```

## Environment Setup

**Required .env file**:

```env
GOOGLE_API_KEY=your_first_key_here
GOOGLE_API_KEY2=your_second_key_here
GOOGLE_API_KEY3=your_third_key_here
```

All 3 keys required for parallel execution.

## Benchmark Configuration

**Fixed Parameters**:
- Model: gemma3_4b
- Questions per run: 50
- Datasets: 8 (medqa, medmcqa, mmlupro-med, pubmedqa, medbullets, ddxplus, pmc_vqa, path_vqa)
- Methods: 3 (zero_shot, few_shot, cot)
- Seeds: 3 (1, 2, 3)

**Total Runs**: 8 datasets × 3 methods × 3 seeds = 72 runs

## Progress Monitoring

### Log Files
Each parallel instance creates a log file:
- `baseline_key1_seed1_log.txt`
- `baseline_key2_seed2_log.txt`
- `baseline_key3_seed3_log.txt`

### Console Output
Real-time progress in each window:
```
[5/24] Running: medqa - cot - Key 1 - Seed 1
SUCCESS: Completed medqa - cot
```

### Results Files
- Saved immediately after each run
- Intermediate results every 10 questions
- Aggregated summaries updated after each method completes

## Key Advantages

### 1. Rate Limit Avoidance
- Each key has separate rate limits
- 3 keys = 3x throughput
- No waiting for rate limit cooldowns

### 2. Reproducibility
- Different seeds ensure robust evaluation
- Same configuration across all runs
- Consistent metrics tracking

### 3. Comprehensive Metrics
- Question-level details
- Run-level summaries
- Method-level aggregations
- Dataset-level comparisons
- Model-level overview

### 4. Fault Tolerance
- Each run is independent
- Failed runs don't affect others
- Individual log files for debugging
- Intermediate saves prevent data loss

## Verification Checklist

- [x] Multi-key API support implemented
- [x] --key parameter added to slm_runner.py
- [x] Parallel execution scripts created
- [x] Comprehensive metrics tracking verified
- [x] Aggregation at 3 levels (method, dataset, model)
- [x] Results structure documented
- [x] Log files for each instance
- [x] User guide created

## Quick Start

1. **Setup API keys**:
   ```bash
   # Edit .env file
   GOOGLE_API_KEY=key1
   GOOGLE_API_KEY2=key2
   GOOGLE_API_KEY3=key3
   ```

2. **Launch parallel benchmarks**:
   ```bash
   launch_all_parallel.bat
   ```

3. **Monitor progress**:
   - Watch 3 command windows
   - Check log files: `baseline_key*_seed*_log.txt`

4. **Check results**:
   ```bash
   # Results in:
   SLM_Results/gemma3_4b/
   ```

5. **Analyze**:
   - `aggregated_summary.json` in each method folder
   - `dataset_summary.json` in each dataset folder
   - `model_summary.json` in model root folder

## Expected Runtime

- **Sequential**: 6-10 hours (single key, rate limits)
- **Parallel**: 2-4 hours (3 keys, no waiting)
- **Per run**: 5-8 minutes (50 questions)

## Troubleshooting

**Rate limit errors?**
- Verify all 3 keys are set in .env
- Ensure each instance uses different key
- Check key quotas haven't been exhausted

**Missing results?**
- Check log files for errors
- Re-run failed configurations manually
- Verify output directory permissions

**API key not found?**
- Restart terminal after updating .env
- Check exact key names (GOOGLE_API_KEY2, not GOOGLE_API_KEY_2)
- Verify .env in project root

## Files Created

1. `run_baseline_parallel.bat` - Parameterized runner
2. `launch_all_parallel.bat` - Launch all 3 instances
3. `BASELINE_BENCHMARK_GUIDE.md` - Comprehensive user guide
4. `MULTI_KEY_BASELINE_SETUP.md` - This technical summary

## Files Modified

1. `chat_instances.py` - Added multi-key support
2. `slm_runner.py` - Added --key parameter

## Verification Commands

Test single run with each key:
```bash
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 5 --random_seed 1 --key 1
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 5 --random_seed 2 --key 2
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 5 --random_seed 3 --key 3
```

All 3 should complete successfully with different API keys.
