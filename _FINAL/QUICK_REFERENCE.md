# Quick Reference Card - Multi-Agent Medical Reasoning Results

## 🏆 Rankings

### By Accuracy (25q subset, seed=42)
```
1. PATH-VQA    60% ████████████░░░░░░░░ (Vision)
2. PMC-VQA     56% ███████████░░░░░░░░░ (Vision)
3. MedBullets  48% █████████░░░░░░░░░░░ (Text)
4. DDXPlus     44% ████████░░░░░░░░░░░░ (Text)
4. MedMCQA     44% ████████░░░░░░░░░░░░ (Text)
4. MMLU-Pro    44% ████████░░░░░░░░░░░░ (Text)
7. PubMedQA    40% ███████░░░░░░░░░░░░░ (Text)
8. MedQA       28% █████░░░░░░░░░░░░░░░ (Text)
```

### By Speed (seconds per question)
```
1. PATH-VQA     32.2s  ⚡⚡⚡⚡⚡
2. MedMCQA      36.0s  ⚡⚡⚡⚡⚡
3. PMC-VQA      42.8s  ⚡⚡⚡⚡
4. MMLU-Pro     48.2s  ⚡⚡⚡
5. MedQA        48.8s  ⚡⚡⚡
6. PubMedQA     64.3s  ⚡⚡
7. MedBullets   83.2s  ⚡
8. DDXPlus     107.8s  ⚡
```

### By Convergence (agent agreement)
```
1. PATH-VQA    100% ████████████████████
2. PMC-VQA      92% ██████████████████░░
2. PubMedQA     92% ██████████████████░░
4. MedBullets   84% ████████████████░░░░
5. DDXPlus      60% ████████████░░░░░░░░
5. MedMCQA      60% ████████████░░░░░░░░
7. MedQA        48% █████████░░░░░░░░░░░
8. MMLU-Pro     40% ████████░░░░░░░░░░░░
```

## 📊 Category Comparison

### Vision vs Text
```
                Vision    Text
Accuracy        58.0%    41.7%   +16.3pp 🏆
Convergence     96.0%    64.0%   +32.0pp 🏆
Agreement       98.5%    69.0%   +29.5pp 🏆
Time/Question   37.5s    64.6s   -27.1s  🏆
Agents/Question 2.68     3.20    -0.52   🏆
Output Tokens   1,758    3,338   -1,580  🏆

Winner: Vision across all metrics ✅
```

## 💰 Token Economics

### Average Tokens per Question
```
Input Tokens (Text):    282  ▓▓▓░░░░░░░
Input Tokens (Vision):  2,257 ▓▓▓▓▓▓▓▓▓▓ (includes ~2,000 image tokens)
Output Tokens (Text):   3,338 ▓▓▓▓▓▓▓▓▓▓
Output Tokens (Vision): 1,758 ▓▓▓▓▓░░░░░
```

### Cost Efficiency
```
Best:  PATH-VQA   103.2 tokens/sec
Good:  PMC-VQA    109.9 tokens/sec
       MedMCQA     86.5 tokens/sec
Fair:  MedQA       89.2 tokens/sec
Poor:  DDXPlus     35.2 tokens/sec
```

## 🎯 Performance Matrix

```
                 Accuracy  Convergence  Speed
PATH-VQA  🏆      60%       100%       32.2s
PMC-VQA   ⭐      56%       92%        42.8s
MedBullets        48%       84%        83.2s
DDXPlus           44%       60%       107.8s
MedMCQA           44%       60%        36.0s
MMLU-Pro          44%       40%        48.2s
PubMedQA  ⚠️      40%       92%        64.3s  (False consensus)
MedQA     ⚠️      28%       48%        48.8s  (Most challenging)
```

## 🔑 Key Insights

### ✅ What Works
- **Multimodal grounding** → +16.3% accuracy, 100% convergence
- **Vision tasks** → Faster despite image processing
- **Fewer agents on vision** → 2.7 avg vs 3.2 for text

### ⚠️ What Doesn't
- **Text-only USMLE (MedQA)** → Only 28% accuracy
- **High convergence ≠ accuracy** → PubMedQA: 92% agree, 40% correct
- **Differential diagnosis** → Slowest (107.8s per question)

### 💡 Surprising Findings
- Vision needs 47% fewer output tokens than text
- Perfect agent alignment only achieved with multimodal
- Agent variance is 6-10% for vision vs 18-65% for text

## 📈 Overall Stats

```
Questions:      200 total (8 datasets × 25 questions)
Correct:        91 (45.5% accuracy)
Total Tokens:   768,835
API Calls:      2,257
Total Time:     3h 11m
Avg Time/Q:     57.4 seconds
```

## 🚀 Recommendations

### Immediate
1. ✅ Deploy multimodal for all vision tasks
2. 🔧 Investigate MedQA low performance (28%)
3. ⚡ Optimize DDXPlus speed (107s → target 50s)

### Strategic
1. 📊 Scale vision datasets to full evaluation
2. 🔬 Analyze PubMedQA false consensus pattern
3. 🎯 Improve text-only methods (16% accuracy gap)

## 🎓 Dataset Profiles

### PATH-VQA 🏆 (Best Overall)
- **What**: Pathology microscopy images
- **Performance**: 60% accuracy, 100% convergence
- **Speed**: 32.2s per question
- **Best for**: Visual pathology diagnosis

### PMC-VQA ⭐ (Strong)
- **What**: Medical literature figures
- **Performance**: 56% accuracy, 92% convergence
- **Speed**: 42.8s per question
- **Best for**: Chart/diagram interpretation

### MedQA ⚠️ (Challenging)
- **What**: USMLE-style questions
- **Performance**: 28% accuracy, 48% convergence
- **Speed**: 48.8s per question
- **Best for**: Nothing yet - needs improvement

### PubMedQA ⚠️ (False Consensus)
- **What**: Yes/No/Maybe biomedical questions
- **Performance**: 40% accuracy, 92% convergence
- **Speed**: 64.3s per question
- **Warning**: High agreement on wrong answers

---

**Last Updated**: 2025-10-23
**Sample Size**: 25 questions per dataset (seed=42)
**Model**: Gemma-3-4b-it Multi-Agent System
