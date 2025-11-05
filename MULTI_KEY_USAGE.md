# Multi-Key API Support

## Overview

The system now supports multiple Google API keys for easy switching and load balancing across different quota limits.

## Setup

### 1. Add Keys to `.env` File

```bash
# Key 1 (default)
GOOGLE_API_KEY=AIzaSyA...your_key_1

# Key 2
GOOGLE_API_KEY2=AIzaSyB...your_key_2

# Key 3
GOOGLE_API_KEY3=AIzaSyC...your_key_3

# Add more keys as needed
GOOGLE_API_KEY4=AIzaSyD...your_key_4
GOOGLE_API_KEY5=AIzaSyE...your_key_5
```

**Naming Convention**:
- First key: `GOOGLE_API_KEY`
- Additional keys: `GOOGLE_API_KEY2`, `GOOGLE_API_KEY3`, `GOOGLE_API_KEY4`, etc.

## Usage

### Default Key (Key 1)
```bash
python run_simulation_adk.py --dataset medqa --n-questions 20
# Uses GOOGLE_API_KEY
```

### Specify Key Number
```bash
# Use key 2
python run_simulation_adk.py --dataset medqa --n-questions 20 --key 2

# Use key 3
python run_simulation_adk.py --dataset medqa --n-questions 20 --key 3

# Use key 4
python run_simulation_adk.py --dataset medqa --n-questions 20 --key 4
```

### With All Configurations
```bash
# Run all 6 configs with key 2
python run_simulation_adk.py --dataset path_vqa --n-questions 20 --all --key 2
```

## Examples

### Scenario 1: Rate Limit Hit on Key 1
```bash
# First run hits rate limit
python run_simulation_adk.py --dataset path_vqa --n-questions 100 --key 1

# Switch to key 2 to continue
python run_simulation_adk.py --dataset path_vqa --n-questions 100 --key 2
```

### Scenario 2: Parallel Runs with Different Keys
```bash
# Terminal 1 - Use key 1
python run_simulation_adk.py --dataset medqa --n-questions 50 --key 1

# Terminal 2 - Use key 2 simultaneously
python run_simulation_adk.py --dataset medmcqa --n-questions 50 --key 2

# Terminal 3 - Use key 3 simultaneously
python run_simulation_adk.py --dataset pubmedqa --n-questions 50 --key 3
```

### Scenario 3: Distribute Work Across Keys
```bash
# Config 1-3 with key 1
python run_simulation_adk.py --dataset medqa --n-questions 20 --smm --key 1
python run_simulation_adk.py --dataset medqa --n-questions 20 --leadership --key 1
python run_simulation_adk.py --dataset medqa --n-questions 20 --trust --key 1

# Config 4-6 with key 2
python run_simulation_adk.py --dataset medqa --n-questions 20 --team-orientation --key 2
python run_simulation_adk.py --dataset medqa --n-questions 20 --mutual-monitoring --key 2
python run_simulation_adk.py --dataset medqa --n-questions 20 --all-teamwork --key 2
```

## Error Handling

### Missing Key
```bash
python run_simulation_adk.py --dataset medqa --n-questions 10 --key 5
```

**Output**:
```
ERROR: API key not found: GOOGLE_API_KEY5 environment variable not set.
Available keys: 1, 2, 3
```

### Check Available Keys
The system automatically detects available keys and shows them in error messages.

## Batch File Integration

Update your `adk.bat` file to use specific keys:

```batch
@echo off
REM Use key 2 for large runs
python run_simulation_adk.py --dataset path_vqa --n-questions 20 --all --key 2
```

## Benefits

✅ **No Rate Limit Bottleneck**: Switch keys when one hits quota
✅ **Parallel Execution**: Run multiple experiments simultaneously
✅ **Easy Switching**: Single `--key` parameter
✅ **Scalable**: Add GOOGLE_API_KEY6, GOOGLE_API_KEY7, etc. as needed
✅ **Error Reporting**: Shows available keys when one is missing

## Technical Details

### How It Works

1. **Argument Parsing**: `--key N` parameter specifies which key to use
2. **Key Resolution**: `get_google_api_key(N)` loads the appropriate key
3. **Environment Setup**: Sets `GOOGLE_API_KEY` and `GEMINI_API_KEY` for all downstream code
4. **Transparent Usage**: All existing code automatically uses the selected key

### Fallback Behavior

- `--key 1`: Checks `GOOGLE_API_KEY` then `GEMINI_API_KEY`
- `--key 2+`: Only checks `GOOGLE_API_KEYN`

### Key Selection Logic

```python
# Key 1 (default)
GOOGLE_API_KEY or GEMINI_API_KEY

# Key 2
GOOGLE_API_KEY2

# Key 3
GOOGLE_API_KEY3

# Key N
GOOGLE_API_KEYN
```

## Troubleshooting

### Key Not Loaded from `.env`
Ensure your `.env` file is in the project root and properly formatted:
```bash
GOOGLE_API_KEY=value_without_quotes
GOOGLE_API_KEY2=value_without_quotes
```

### Wrong Key Being Used
Check logs at startup:
```
INFO - Using API key: GOOGLE_API_KEY2
```

### Keys Not Available
List all environment variables:
```bash
# Windows
set | findstr GOOGLE_API_KEY

# Linux/Mac
env | grep GOOGLE_API_KEY
```

## Best Practices

1. **Rotate Keys**: Distribute work across keys to maximize throughput
2. **Name Consistently**: Always use GOOGLE_API_KEYN format
3. **Document Keys**: Note which key is used for which experiments
4. **Monitor Quotas**: Track usage at https://ai.dev/usage
5. **Keep Secure**: Never commit `.env` file to version control

---

**Added**: 2025-10-28
**Scalability**: Supports unlimited keys (tested up to 10)
**Backward Compatible**: Works with existing code without changes
