# Baseline Benchmark Guide

## Overview

Comprehensive baseline evaluation system for zero-shot, few-shot, and Chain-of-Thought (CoT) prompting across all 8 medical datasets with multi-key parallel execution support.

---

## Quick Start

### Setup API Keys

Create or update `.env` file in project root:

```env
GOOGLE_API_KEY=your_first_api_key_here
GOOGLE_API_KEY2=your_second_api_key_here
GOOGLE_API_KEY3=your_third_api_key_here
```

**Note:** All 3 keys required for parallel execution to avoid rate limits.

### Execute Benchmarks

**Sequential (Single Key):**
```bash
run_baseline_benchmarks.bat
```
- 72 runs sequentially
- 6-10 hours
- Use when you have only 1 key

**Parallel (3 Keys):**
```bash
launch_all_parallel.bat
```
- 3 instances simultaneously
- 2-4 hours (3x speedup)
- Use when you have 3 keys

---

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

**Total Runs:** 8 datasets × 3 methods × 3 seeds = **72 runs**
**Questions per run:** 50

---

## Execution Modes

### 1. Sequential Execution (Single Key)

**Command:**
```bash
run_baseline_benchmarks.bat
```

**Configuration:**
- Uses GOOGLE_API_KEY (key 1)
- Runs 72 benchmarks sequentially
- 50 questions per run
- Estimated time: 6-10 hours

### 2. Parallel Execution (3 Keys)

**Option A: Launch All at Once**
```bash
launch_all_parallel.bat
```

Opens 3 command windows automatically:
- Window 1: Key 1, Seed 1 (24 runs)
- Window 2: Key 2, Seed 2 (24 runs)
- Window 3: Key 3, Seed 3 (24 runs)

**Total:** 72 runs across 3 windows
**Estimated time:** 2-4 hours (3x faster)

**Option B: Manual Launch**

Open 3 separate command prompts:

```bash
# Terminal 1
run_baseline_parallel.bat 1 1

# Terminal 2
run_baseline_parallel.bat 2 2

# Terminal 3
run_baseline_parallel.bat 3 3
```

---

## Multi-Key API Support

### Key Configuration

The system supports multiple Google API keys for load balancing:

**Naming Convention:**
- First key: `GOOGLE_API_KEY`
- Additional keys: `GOOGLE_API_KEY2`, `GOOGLE_API_KEY3`, `GOOGLE_API_KEY4`, etc.

### Usage Examples

**Default Key (Key 1):**
```bash
python slm_runner.py --dataset medqa --method zero_shot --num_questions 50 --random_seed 1
```

**Specify Key Number:**
```bash
# Use key 2
python slm_runner.py --dataset medqa --method zero_shot --num_questions 50 --random_seed 1 --key 2

# Use key 3
python slm_runner.py --dataset medqa --method zero_shot --num_questions 50 --random_seed 2 --key 3
```

### Parallel Runs with Different Keys

**Scenario: Distribute Work Across Keys**
```bash
# Terminal 1 - Use key 1
python slm_runner.py --dataset medqa --method zero_shot --num_questions 50 --key 1

# Terminal 2 - Use key 2 simultaneously
python slm_runner.py --dataset medmcqa --method few_shot --num_questions 50 --key 2

# Terminal 3 - Use key 3 simultaneously
python slm_runner.py --dataset pubmedqa --method cot --num_questions 50 --key 3
```

### Error Handling

**Missing Key:**
```bash
python slm_runner.py --dataset medqa --num_questions 10 --key 5
```

**Output:**
```
ERROR: API key not found: GOOGLE_API_KEY5 environment variable not set.
Available keys: 1, 2, 3
```

---

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
    │   │   └── [same structure]
    │   └── cot/
    │       └── [same structure]
    ├── medmcqa/
    │   └── [same structure]
    └── [... other datasets]
```

### Per-Run Results

Each run creates `{seed}_results.json` with:
- Question text
- Ground truth answer
- Predicted answer
- Is correct (boolean)
- Response time
- Token counts (input/output/total)
- Full model response

### Aggregated Results

**Method Level** (`aggregated_summary.json`):
```json
{
  "dataset": "medqa",
  "method": "zero_shot",
  "model": "gemma3_4b",
  "runs": [
    {"seed": 1, "accuracy": 0.68, "total_questions": 50, "correct": 34},
    {"seed": 2, "accuracy": 0.70, "total_questions": 50, "correct": 35},
    {"seed": 3, "accuracy": 0.66, "total_questions": 50, "correct": 33}
  ],
  "mean_accuracy": 0.68,
  "std_accuracy": 0.016,
  "total_questions": 150,
  "total_correct": 102
}
```

---

## Metrics Tracked

### Per Question
- **Correctness:** Binary correct/incorrect
- **Response Time:** Seconds per question
- **Token Usage:** Input/output/total tokens
- **Full Response:** Complete model output

### Per Run (50 questions)
- **Accuracy:** Correct/Total
- **Average Time:** Mean time per question
- **Total Time:** Full run duration
- **Token Statistics:** Total and average tokens

### Aggregated (Across Seeds)
- **Mean Accuracy:** Average across 3 seeds
- **Standard Deviation:** Accuracy variance
- **95% Confidence Interval:** Statistical confidence
- **Total Questions:** 150 per method (3 seeds × 50)

---

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

---

## Manual Execution

For fine-grained control:

```bash
# Single dataset, method, seed
python slm_runner.py \
  --dataset medqa \
  --method zero_shot \
  --model gemma3_4b \
  --num_questions 50 \
  --random_seed 1 \
  --key 1

# With custom output directory
python slm_runner.py \
  --dataset medqa \
  --method few_shot \
  --model gemma3_4b \
  --num_questions 50 \
  --random_seed 2 \
  --key 2 \
  --output_dir custom_results/

# Different number of questions
python slm_runner.py \
  --dataset medqa \
  --method cot \
  --model gemma3_4b \
  --num_questions 100 \
  --random_seed 3 \
  --key 3
```

---

## Expected Runtime

### Sequential (Single Key)
- **Per run:** 5-8 minutes
- **72 runs:** 6-10 hours
- **Bottleneck:** Rate limits on single key

### Parallel (3 Keys)
- **Per run:** 5-8 minutes
- **24 runs per instance:** 2-3.5 hours
- **3 instances simultaneously:** 2-4 hours total
- **Speedup:** ~3x faster

---

## Troubleshooting

### Rate Limit Errors

**Problem:** "Rate limit exceeded" errors

**Solution:**
1. Ensure you're using parallel execution with 3 keys
2. Each key has separate rate limits
3. Verify all 3 keys are valid in `.env`

### Missing Results

**Problem:** Some runs failed or results missing

**Solution:**
1. Check log files for error messages
2. Re-run failed configurations manually:
   ```bash
   python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 50 --random_seed 1 --key 1
   ```

### API Key Not Found

**Problem:** "GOOGLE_API_KEY2 environment variable not set"

**Solution:**
1. Verify `.env` file exists in project root
2. Verify key naming: `GOOGLE_API_KEY2` (not `GOOGLE_API_KEY_2`)
3. Restart command prompt to reload environment

### Keys Not Loading from .env

**Solution:**
Ensure `.env` file is properly formatted:
```bash
GOOGLE_API_KEY=value_without_quotes
GOOGLE_API_KEY2=value_without_quotes
```

Check logs at startup:
```
INFO - Using API key: GOOGLE_API_KEY2
```

---

## Key Benefits

✅ **No Rate Limit Bottleneck** - Switch keys when one hits quota
✅ **Parallel Execution** - Run multiple experiments simultaneously
✅ **Easy Switching** - Single `--key` parameter
✅ **Scalable** - Add more keys as needed (GOOGLE_API_KEY4, GOOGLE_API_KEY5, etc.)
✅ **Error Reporting** - Shows available keys when one is missing
✅ **Reproducibility** - Different seeds ensure robust evaluation
✅ **Comprehensive Metrics** - Question, run, and aggregate level tracking

---

## Analysis Scripts

After benchmarks complete, analyze results:

```bash
# Generate comparison tables
python scripts/analyze_baseline_results.py

# Compare with multi-agent results
python scripts/compare_baseline_vs_multiagent.py
```

---

## Key Findings Expected

Baseline benchmarks establish:
1. **Zero-shot performance** - Model capability without examples
2. **Few-shot improvement** - Benefit of example questions
3. **CoT effectiveness** - Value of step-by-step reasoning
4. **Dataset difficulty** - Which datasets are hardest
5. **Seed stability** - Result variance across different samples

These baselines are crucial for evaluating multi-agent system improvements.

---

## Best Practices

1. **Rotate Keys:** Distribute work across keys to maximize throughput
2. **Name Consistently:** Always use GOOGLE_API_KEYN format
3. **Document Keys:** Note which key is used for which experiments
4. **Monitor Quotas:** Track usage at https://ai.dev/usage
5. **Keep Secure:** Never commit `.env` file to version control

---

**Last Updated:** 2025-10-30
**Scalability:** Supports unlimited keys (tested up to 10)
**Backward Compatible:** Works with existing code without changes
