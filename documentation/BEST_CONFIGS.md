# Best Teamwork Configurations per Dataset

Based on ablation study results (50 questions × 3 seeds per configuration).

## Configuration Mapping

| Dataset | Best Config | Accuracy | Components |
|---------|-------------|----------|------------|
| **DDXPLUS** | All Teamwork | **67.33% ± 6.11** | SMM + Leadership + Team Orientation + Trust + Mutual Monitoring |
| **MEDBULLETS** | SMM + Trust | **33.33% ± 5.03** | Shared Mental Model + Trust Network |
| **MEDMCQA** | SMM + Trust | **52.00% ± 10.00** | Shared Mental Model + Trust Network |
| **MEDQA** | TO + SMM + Leadership | **49.33% ± 9.87** | Team Orientation + Shared Mental Model + Leadership |
| **PATH_VQA** | TO + Mutual Monitoring | **64.00% ± 3.46** | Team Orientation + Mutual Monitoring |
| **PMC_VQA** | SMM + Trust | **46.00% ± 3.46** | Shared Mental Model + Trust Network |
| **PUBMEDQA** | SMM + Trust | **72.00% ± 5.29** | Shared Mental Model + Trust Network |
| **MMLUPRO** | SMM + Trust | **32.00% ± 10.00** | Shared Mental Model + Trust Network |

---

## Full Dataset Run Configuration

The following bat files use these best configurations for 500-question runs:

### Set 1: `run_full_set1.bat`
- **medqa** → TO + SMM + Leadership
- **path_vqa** → TO + Mutual Monitoring
- **medbullets** → SMM + Trust
- **ddxplus** → All Teamwork

### Set 2: `run_full_set2.bat`
- **medmcqa** → SMM + Trust
- **pmc_vqa** → SMM + Trust
- **pubmedqa** → SMM + Trust
- **mmlupro** → SMM + Trust

---

## Key Insights

### 1. **SMM + Trust Dominates**
Wins on **5/8 datasets** (62.5%):
- MEDBULLETS, MEDMCQA, PMC_VQA, PUBMEDQA, MMLUPRO

**Why?**
- Lightweight: Only 2 components
- Synergy: Trust-weighted knowledge sharing
- Generalizes well across diverse medical domains

### 2. **All Components Rarely Optimal**
Only wins on **1/8 datasets** (DDXPLUS: 67.33%)

**Evidence of component interference:**
- PATH_VQA: All (54.00%) vs TO+MM (64.00%) - **10% drop!**
- PUBMEDQA: All (68.67%) vs SMM+Trust (72.00%) - **3.33% drop**

### 3. **Dataset-Specific Patterns**

**Visual reasoning (PATH_VQA):**
- Best: TO + MM (64.00%)
- Role specialization + validation works well for image interpretation

**Differential diagnosis (DDXPLUS):**
- Best: All Teamwork (67.33%)
- Complex reasoning benefits from full coordination

**Medical knowledge (MEDQA, MEDMCQA):**
- SMM + Trust or TO + SMM + L perform best
- Knowledge sharing crucial for factual recall

---

## Command-Line Flags

### SMM + Trust
```bash
--smm --trust
```

### TO + Mutual Monitoring
```bash
--team-orientation --mutual-monitoring
```

### TO + SMM + Leadership
```bash
--team-orientation --smm --leadership
```

### All Teamwork
```bash
--all-teamwork
```

---

## Implementation in run_full_single.bat

The script automatically selects the best config based on dataset name:

```batch
if /i "%DATASET%"=="ddxplus" (
    set CONFIG_FLAGS=--all-teamwork
) else if /i "%DATASET%"=="path_vqa" (
    set CONFIG_FLAGS=--team-orientation --mutual-monitoring
) else if /i "%DATASET%"=="medqa" (
    set CONFIG_FLAGS=--team-orientation --smm --leadership
) else (
    REM Most datasets use SMM + Trust
    set CONFIG_FLAGS=--smm --trust
)
```

---

## Reproducibility

All results based on:
- **Model**: gemma3_4b
- **Questions**: 50 per ablation run
- **Seeds**: 1, 2, 3 (3 runs per config)
- **Agents**: Dynamic (determined by algorithm)
- **Output**: multi-agent-gemma/ablation/

Results aggregated using:
```bash
python aggregate_ablation_results.py
```

---

Generated: 2025-11-10
