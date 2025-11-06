# SLM-TeamMedAgents: Modular Multi-Agent Medical Reasoning System

> A comprehensive multi-agent collaborative reasoning framework using Small Language Models (SLMs) for medical question answering, featuring five independently toggleable teamwork components for ablation studies.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Production-success.svg)]()

---

## Abstract

We present SLM-TeamMedAgents, a modular multi-agent framework for collaborative medical reasoning featuring five pluggable teamwork mechanisms: Shared Mental Model (SMM), Leadership, Team Orientation, Trust Network, and Mutual Monitoring. The system supports 2-4 dynamically recruited specialist agents engaging in structured three-round deliberation with final aggregation via weighted voting schemes. Built on Google's Agent Development Kit (ADK), the system achieves comprehensive ablation study support through independent component toggles while maintaining production-grade reliability and observability.

**Key Contributions:**
- Modular teamwork architecture with independently toggleable components
- Hierarchical role specialization with medical domain expertise
- Trust-based dynamic weighting for improved consensus
- Inter-round mutual monitoring for quality control
- Comprehensive evaluation framework across 8 medical datasets

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Documentation](#documentation)
- [Citation](#citation)

---

## Installation

### Prerequisites
- Python 3.9+
- Google Generative AI API key
- 16GB+ RAM (recommended)

### Setup

1. **Clone repository:**
```bash
git clone https://github.com/yourusername/SLM-TeamMedAgents
cd SLM-TeamMedAgents
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install google-adk google-genai
```

3. **Configure API keys:**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or create `.env` file:
```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_API_KEY2=your_second_key  # Optional for parallel runs
GOOGLE_API_KEY3=your_third_key   # Optional for parallel runs
```

4. **Verify installation:**
```bash
python run_simulation_adk.py --help
```

---

## Quick Start

### Basic Execution

```bash
# Base system (no teamwork components)
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --n-agents 3

# Full system (all components enabled)
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --all-teamwork
```

### Ablation Study

```bash
# Test individual components
python run_simulation_adk.py --dataset medqa --n-questions 10 --smm
python run_simulation_adk.py --dataset medqa --n-questions 10 --leadership
python run_simulation_adk.py --dataset medqa --n-questions 10 --trust

# Combined components
python run_simulation_adk.py --dataset medqa --n-questions 10 --leadership --team-orientation
```

---

## System Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│           Multi-Agent System Pipeline                    │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Phase 1: RECRUITMENT (2 API calls)                     │
│  ├─ Dynamic/Fixed agent recruitment (2-4 agents)        │
│  ├─ [SMM] Question analysis & trick detection           │
│  ├─ [Leadership] Recruiter → Leader designation         │
│  └─ [TeamO] Medical specialty assignment + weights      │
│                                                           │
│  Phase 2: ROUND 2 - Initial Prediction (N+1 calls)     │
│  ├─ Parallel agent predictions                          │
│  ├─ [SMM] Inject shared context                         │
│  └─ Post-R2: Extract facts, create report, evaluate     │
│                                                           │
│  Phase 3: ROUND 3 - Collaborative Discussion            │
│  ├─ Multi-turn discussion (2-3 turns)                   │
│  ├─ [Leadership] Mediation after each turn              │
│  ├─ [MM] Challenge weakest agent (between turns)        │
│  └─ Final turn: Extract rankings                        │
│                                                           │
│  Phase 4: AGGREGATION                                   │
│  ├─ [Trust] Trust-weighted Borda count                  │
│  ├─ [TeamO] Hierarchical-weighted voting                │
│  └─ [Leadership] Tie-breaking with correction power     │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Teamwork Components

| Component | Purpose | Impact | Dependencies |
|-----------|---------|--------|--------------|
| **Shared Mental Model** | Passive knowledge repository | Shared context reduces redundancy | None |
| **Leadership** | Active orchestration | Structured coordination & quality control | Enhances SMM/TO/Trust |
| **Team Orientation** | Role specialization | Medical expertise with weighted voting | Best with Leadership |
| **Trust Network** | Dynamic reliability scoring | Reliable agents weighted more | Leadership can evaluate |
| **Mutual Monitoring** | Inter-round validation | Fact-checking & reasoning improvement | Requires Leadership |

### Directory Structure

```
SLM-TeamMedAgents/
├── adk_agents/                  # Core ADK agent implementations
│   ├── multi_agent_system_adk.py       # Root coordinator
│   ├── dynamic_recruiter_adk.py        # Agent recruitment
│   ├── three_round_debate_adk.py       # 3-round reasoning
│   ├── decision_aggregator_adk.py      # Voting methods
│   └── gemma_agent_adk.py              # Gemma agent factory
│
├── teamwork_components/          # Modular teamwork mechanisms
│   ├── config.py                       # TeamworkConfig class
│   ├── shared_mental_model.py          # SMM implementation
│   ├── leadership.py                   # Leadership coordinator
│   ├── team_orientation.py             # Role specialization
│   ├── trust_network.py                # Trust scoring
│   └── mutual_monitoring.py            # Inter-round validation
│
├── medical_datasets/             # Dataset loaders & formatters
│   ├── dataset_loader.py               # Text datasets
│   ├── vision_dataset_loader.py        # Image datasets
│   ├── dataset_formatters.py           # Text formatters
│   └── vision_dataset_formatters.py    # Image formatters
│
├── utils/                        # Utilities
│   ├── simulation_logger.py            # Logging system
│   ├── results_storage.py              # Results management
│   ├── metrics_calculator.py           # Performance metrics
│   ├── prompts.py                      # Prompt templates
│   └── rate_limiter.py                 # API rate limiting
│
├── documentation/                # Documentation
│   ├── SYSTEM_ARCHITECTURE.md          # Algorithm specification
│   ├── ADK_GUIDE.md                    # ADK system guide
│   ├── BASELINE_BENCHMARKS.md          # Baseline evaluation
│   ├── QUICKSTART.md                   # Quick start guide
│   ├── TOKEN_SUMMARY_GUIDE.md          # Metrics guide
│   ├── PROMPT_IMPROVEMENTS.md          # Prompt details
│   └── KNOWLEDGE.md                    # System knowledge base
│
└── run_simulation_adk.py         # Main entry point
```

---

## Usage

### Command-Line Interface

```bash
python run_simulation_adk.py [OPTIONS]
```

**Required:**
- `--dataset DATASET` - Dataset: medqa, medmcqa, pubmedqa, mmlupro, ddxplus, medbullets, pmc_vqa, path_vqa

**Optional:**
- `--n-questions N` - Number of questions (default: 10)
- `--n-agents N` - Fixed agent count (default: dynamic 2-4)
- `--model MODEL` - Model: gemma3_4b, gemma2_9b, gemma2_27b, medgemma_4b
- `--output-dir DIR` - Output directory
- `--seed N` - Random seed for reproducibility
- `--key N` - API key number (1, 2, 3, etc.)

**Teamwork Components:**
- `--smm` - Enable Shared Mental Model
- `--leadership` - Enable Leadership
- `--team-orientation` - Enable Team Orientation
- `--trust` - Enable Trust Network
- `--mutual-monitoring` - Enable Mutual Monitoring
- `--all-teamwork` - Enable all components
- `--n-turns N` - Discussion turns (2 or 3)

### Supported Models

| Model | Size | Vision | Specialization |
|-------|------|--------|----------------|
| gemma3_4b | 4B | ✓ | General-purpose |
| gemma2_9b | 9B | ✓ | General-purpose |
| gemma2_27b | 27B | ✓ | High-capacity |
| medgemma_4b | 4B | ✓ | Medical-specialized |

### Supported Datasets

**Text Datasets:**
- **medqa** - USMLE-style medical questions (271k questions)
- **medmcqa** - Indian medical entrance exams (194k questions)
- **pubmedqa** - Biomedical research questions (1k questions)
- **mmlupro** - MMLU-Pro Health subset
- **medbullets** - Clinical case questions
- **ddxplus** - Differential diagnosis

**Vision Datasets:**
- **pmc_vqa** - Medical visual QA (227k images)
- **path_vqa** - Pathology image analysis (33k images)

---

## Experimental Setup

### Reproducibility

**Fixed Parameters:**
- Model: gemma3_4b (default)
- Temperature: 0.7 (reasoning), 0.5 (recruitment)
- Max tokens: 2048
- Random seed: Configurable via `--seed`

**API Call Management:**
- Exponential backoff retry (5 attempts)
- Base delay: 60s for rate limits
- Jitter: ±20% to prevent thundering herd

### Ablation Study Configuration

Test individual component impact:

```bash
# Baseline (no teamwork)
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42

# +SMM
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42 --smm

# +Leadership
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42 --leadership

# +Team Orientation (with Leadership)
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42 --leadership --team-orientation

# +Trust
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42 --trust

# +Mutual Monitoring (with Leadership + Trust)
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42 --leadership --trust --mutual-monitoring

# Full system
python run_simulation_adk.py --dataset medqa --n-questions 50 --seed 42 --all-teamwork
```

### Baseline Benchmarks

Evaluate baseline performance across prompting methods:

```bash
# Sequential execution (single key)
run_baseline_benchmarks.bat

# Parallel execution (3 keys)
launch_all_parallel.bat
```

**Configuration:**
- **Datasets:** 8 (all medical datasets)
- **Methods:** 3 (zero-shot, few-shot, CoT)
- **Seeds:** 3 (for robustness)
- **Total runs:** 72 (8 × 3 × 3)
- **Questions per run:** 50

---

## Results

### Output Structure

Results saved to: `{output_dir}/{dataset}_{n_questions}q_run{X}/`

```
medqa_50q_run1/
├── questions/                    # Individual question results
│   ├── q001_results.json
│   ├── q002_results.json
│   └── ...
├── summary_report.json           # Aggregate metrics
├── accuracy_summary.json         # Accuracy breakdown
├── convergence_analysis.json     # Convergence metrics
├── agent_performance.json        # Per-agent statistics
├── simulation.log                # Detailed execution log
└── config.json                   # Run configuration
```

### Key Metrics

**Performance Metrics:**
- Overall accuracy
- Per-method accuracy (Borda, Majority, Weighted)
- Per-task-type accuracy

**Teamwork Metrics:**
- Convergence rate (Round 1 → Round 3)
- Agent agreement (pairwise, full)
- Trust score evolution
- Opinion change frequency

**Efficiency Metrics:**
- API calls per question
- Execution time (total, per-phase)
- Token usage (input/output/total)

### Expected Performance

**API Calls (N=3, n_turns=2):**

| Configuration | Calls/Question |
|---------------|----------------|
| Base | 9 |
| +SMM | 9 |
| +Leadership | 11 |
| +Trust | 9 |
| +MM | 12 |
| **ALL ON** | 14 |

**Timing (N=3, n_turns=2):**
- Recruitment: ~2-5 seconds
- Round 2: ~5-8 seconds
- Round 3: ~6-10 seconds
- Aggregation: < 1 second
- **Total:** ~15-25 seconds per question

---

## Documentation

### Core Documentation
- **[SYSTEM_ARCHITECTURE.md](documentation/SYSTEM_ARCHITECTURE.md)** - Algorithm specification & complexity analysis
- **[ADK_GUIDE.md](documentation/ADK_GUIDE.md)** - ADK system guide & API reference
- **[BASELINE_BENCHMARKS.md](documentation/BASELINE_BENCHMARKS.md)** - Baseline evaluation & multi-key usage
- **[QUICKSTART.md](documentation/QUICKSTART.md)** - Quick start guide with examples

### Additional Resources
- **[TOKEN_SUMMARY_GUIDE.md](documentation/TOKEN_SUMMARY_GUIDE.md)** - Token usage & cost analysis
- **[PROMPT_IMPROVEMENTS.md](documentation/PROMPT_IMPROVEMENTS.md)** - Prompt engineering details
- **[KNOWLEDGE.md](documentation/KNOWLEDGE.md)** - System knowledge base & design patterns

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{slm_teammedagents,
  title={SLM-TeamMedAgents: Modular Multi-Agent Medical Reasoning System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/SLM-TeamMedAgents},
  note={Built on Google's Agent Development Kit (ADK)}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Acknowledgments

- **Google DeepMind** - Gemma and MedGemma models
- **Google ADK Team** - Agent Development Kit framework
- **Dataset Providers** - MedQA, MedMCQA, PubMedQA, PMC-VQA, Path-VQA

---

## Contact

For questions, issues, or collaboration:
- **GitHub Issues:** [Link]
- **Email:** [Your Email]
- **Documentation:** See `documentation/` folder

---

**Version:** 1.0.0 | **Status:** Production Ready | **Last Updated:** 2025-11-06
