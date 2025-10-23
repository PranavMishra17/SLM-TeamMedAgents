# Consolidated Results - Final Reports

This folder contains the comprehensive consolidated results from all 8 datasets evaluated with the Gemma-3-4b-it multi-agent system.

## 📁 Files in This Folder

### 1. `consolidated_results.json`
**Comprehensive JSON export** containing all metrics and statistics in machine-readable format.

**Contains**:
- Complete results for all 8 datasets
- Aggregate statistics (overall, text-only, vision)
- Performance rankings by accuracy, convergence, efficiency
- Key findings and metadata

**Use for**: Data analysis, visualization, programmatic access

---

### 2. `CONSOLIDATED_RESULTS_TABLES.md`
**Detailed analysis document** with 3 comprehensive tables and cross-analysis.

**Tables**:
1. **Accuracy Performance** - Rankings, scores, type classification
2. **Token Usage & Efficiency** - Time, tokens, API calls, cost metrics
3. **Agent Disagreement & Resolution** - Convergence, agreement, variance

**Includes**:
- Performance matrix (Vision vs Text)
- Efficiency vs Accuracy trade-off analysis
- Cross-table insights
- Detailed interpretations

**Use for**: Deep analysis, research papers, technical presentations

---

### 3. `RESULTS_SUMMARY.md`
**Executive summary** with clean, focused tables and actionable insights.

**Features**:
- 3 simplified tables (Accuracy, Efficiency, Consensus)
- Key findings highlighted
- Immediate recommendations
- Quick reference format

**Use for**: Presentations, quick reviews, stakeholder updates

---

## 📊 Quick Stats

| Metric | Value |
|--------|-------|
| **Total Questions** | 200 (25 per dataset) |
| **Overall Accuracy** | 45.5% (91/200 correct) |
| **Vision Accuracy** | 58.0% (29/50 correct) |
| **Text Accuracy** | 41.7% (62/150 correct) |
| **Total Tokens** | 768,835 |
| **Total API Calls** | 2,257 |
| **Total Time** | 3.19 hours |
| **Best Performer** | PATH-VQA (60% accuracy) |
| **Most Efficient** | PATH-VQA (32.2s/question) |

## 🎯 Top 3 Insights

1. **Multimodal Advantage**: Vision datasets achieve 16.3% higher accuracy than text-only
2. **Perfect Convergence**: PATH-VQA shows 100% agent agreement with multimodal support
3. **Efficiency Win**: Vision processing is faster despite including image analysis

## 📖 How to Use These Reports

### For Quick Reviews
→ Start with `RESULTS_SUMMARY.md`

### For Deep Analysis
→ Read `CONSOLIDATED_RESULTS_TABLES.md`

### For Data Processing
→ Use `consolidated_results.json`

## 🗂️ Source Data

Results consolidated from:
- `multi-agent-gemma/results/ddxplus/ddxplus_25q_run1/summary_report.json`
- `multi-agent-gemma/results/medbullets/medbullets_25q_run1/summary_report.json`
- `multi-agent-gemma/results/medmcqa/medmcqa_25q_run2/summary_report.json`
- `multi-agent-gemma/results/medqa/medqa_25q_run1/summary_report.json`
- `multi-agent-gemma/results/mmlupro/mmlupro_25q_run1/summary_report.json`
- `multi-agent-gemma/results/pmc_vqa/pmc_vqa_25q_run1/summary_report.json`
- `multi-agent-gemma/results/path_vqa/path_vqa_25q_run1/summary_report.json`
- `multi-agent-gemma/results/pubmedqa/pubmedqa_25q_run2/summary_report.json`

## 🔬 Methodology

- **Model**: Gemma-3-4b-it via Google AI Studio
- **Framework**: Google ADK Multi-Agent System
- **Method**: Three-Round Debate + Dynamic Recruitment + Borda Count
- **Sample**: 25 questions per dataset (random subset, seed=42)
- **Date**: October 23, 2025

## 📝 Notes

- Vision datasets (PATH-VQA, PMC-VQA) use multimodal capabilities
- Image tokens counted separately (258 tokens per image per agent per round)
- All metrics based on 25-question subsets; full dataset results may vary
- Convergence rate = % of questions where all agents agree on final answer
- Pairwise agreement = average agreement rate across all agent pairs

---

**Generated**: 2025-10-23
**Author**: Multi-Agent Medical Reasoning System
**Contact**: See project root for details
