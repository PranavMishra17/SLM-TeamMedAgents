# ADK Multi-Agent Medical Reasoning System

A modular multi-agent system for medical question answering built on Google's Agent Development Kit (ADK), featuring pluggable teamwork mechanisms for collaborative reasoning.

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Teamwork Components](#teamwork-components)
- [Reproducibility](#reproducibility)
- [Results & Logging](#results--logging)

---

## Overview

This system implements a multi-agent collaborative reasoning framework for medical question answering with five modular teamwork components:

- **Shared Mental Model (SMM)**: Passive knowledge repository for shared context
- **Leadership**: Active coordination with correction authority
- **Team Orientation**: Medical role specialization with hierarchical weighting
- **Trust Network**: Dynamic agent reliability scoring
- **Mutual Monitoring**: Inter-round validation and challenges

**Key Features**:
- ✅ Modular design: Enable/disable components independently
- ✅ Ablation study support: Test individual component impact
- ✅ Backward compatible: Base system works without teamwork
- ✅ Production-ready: Comprehensive error handling & retry logic
- ✅ Multi-dataset support: MedQA, MedMCQA, PubMedQA, PMC-VQA, Path-VQA, etc.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│           ADK Multi-Agent System Pipeline               │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Phase 1: RECRUITMENT                                   │
│  ├─ Dynamic/Fixed agent recruitment (2-4 agents)        │
│  ├─ [SMM] Question analysis & trick detection           │
│  ├─ [Leadership] Recruiter → Leader designation         │
│  └─ [TeamO] Medical specialty assignment + weights      │
│                                                           │
│  Phase 2: ROUND 2 (R2) - Initial Prediction            │
│  ├─ Parallel agent predictions (N calls)                │
│  ├─ [SMM] Inject shared context in prompts              │
│  └─ Post-R2 Processing:                                 │
│      ├─ [SMM] Extract verified facts                    │
│      ├─ [TeamO] Create formal medical report            │
│      └─ [Trust] Evaluate response quality               │
│                                                           │
│  Phase 3: ROUND 3 (R3) - Collaborative Discussion      │
│  ├─ Multi-turn discussion (2-3 turns)                   │
│  ├─ [SMM + Trust + TeamO] Inject all context            │
│  ├─ [Leadership] Mediate after each turn                │
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

---

## Installation

### Prerequisites
- Python 3.9+
- Google Generative AI API key
- Google ADK library

### Setup

1. **Clone repository**:
```bash
git clone <repository-url>
cd SLM-TeamMedAgents
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install Google ADK**:
```bash
pip install google-adk google-genai
```

4. **Set API key**:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### Verify Installation
```bash
python run_simulation_adk.py --help
```

---

## Usage

### Basic Syntax
```bash
python run_simulation_adk.py --dataset DATASET --n-questions N [OPTIONS]
```

### Command-Line Arguments

#### Required Arguments
| Argument | Type | Choices | Description |
|----------|------|---------|-------------|
| `--dataset` | str | medqa, medmcqa, pubmedqa, mmlupro, ddxplus, medbullets, pmc_vqa, path_vqa | Dataset to use |

#### Optional Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-questions` | int | 10 | Number of questions to process |
| `--n-agents` | int | None | Fixed number of agents (default: dynamic 2-4) |
| `--model` | str | gemma3_4b | Model: gemma3_4b, gemma2_9b, gemma2_27b, medgemma_4b |
| `--output-dir` | str | multi-agent-gemma/results | Output directory for results |

#### Teamwork Component Flags
| Flag | Description | Dependencies |
|------|-------------|--------------|
| `--smm` | Enable Shared Mental Model | None |
| `--leadership` | Enable Leadership coordination | Enhances SMM/TeamO/Trust |
| `--team-orientation` | Enable medical role specialization | Best with Leadership |
| `--trust` | Enable dynamic trust scoring | None |
| `--mutual-monitoring` | Enable inter-round validation | Requires Leadership |
| `--all-teamwork` | Enable ALL components at once | N/A |
| `--n-turns` | Number of R3 discussion turns (2 or 3) | Default: 2 |

### Usage Examples

#### 1. Base System (No Teamwork)
```bash
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --n-agents 3 \
  --model gemma3_4b
```

#### 2. Single Component (Ablation Study)
```bash
# Test SMM only
python run_simulation_adk.py --dataset medqa --n-questions 10 --smm

# Test Leadership only
python run_simulation_adk.py --dataset medqa --n-questions 10 --leadership

# Test Trust only
python run_simulation_adk.py --dataset medqa --n-questions 10 --trust

# Test Team Orientation only
python run_simulation_adk.py --dataset medqa --n-questions 10 --team-orientation
```

#### 3. Combined Components
```bash
# Leadership + Team Orientation (recommended)
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --leadership \
  --team-orientation

# Leadership + SMM + Trust
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --leadership \
  --smm \
  --trust
```

#### 4. Full System (All Components)
```bash
# With 2 turns (default)
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 50 \
  --all-teamwork \
  --model gemma3_4b

# With 3 turns (more thorough)
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 50 \
  --all-teamwork \
  --n-turns 3
```

#### 5. Vision Datasets
```bash
# PMC-VQA (medical images)
python run_simulation_adk.py \
  --dataset pmc_vqa \
  --n-questions 20 \
  --all-teamwork

# Path-VQA (pathology images)
python run_simulation_adk.py \
  --dataset path_vqa \
  --n-questions 20 \
  --all-teamwork
```

#### 6. Dynamic vs Fixed Agent Count
```bash
# Dynamic (2-4 agents based on question complexity)
python run_simulation_adk.py --dataset medqa --n-questions 10 --all-teamwork

# Fixed (always 3 agents)
python run_simulation_adk.py --dataset medqa --n-questions 10 --n-agents 3 --all-teamwork
```

---

## Project Structure

```
SLM-TeamMedAgents/
│
├── adk_agents/                          # Core ADK agent implementations
│   ├── __init__.py
│   ├── multi_agent_system_adk.py        # Root coordinator
│   ├── dynamic_recruiter_adk.py         # Agent recruitment
│   ├── three_round_debate_adk.py        # 3-round collaborative reasoning
│   ├── decision_aggregator_adk.py       # Voting & aggregation
│   └── gemma_agent_adk.py               # Gemma agent factory
│
├── teamwork_components/                 # Modular teamwork mechanisms
│   ├── __init__.py
│   ├── config.py                        # TeamworkConfig class
│   ├── shared_mental_model.py           # SMM implementation
│   ├── leadership.py                    # Leadership coordinator
│   ├── team_orientation.py              # Role specialization
│   ├── trust_network.py                 # Trust scoring
│   └── mutual_monitoring.py             # Inter-round validation
│
├── medical_datasets/                    # Dataset loaders & formatters
│   ├── dataset_loader.py
│   ├── vision_dataset_loader.py
│   ├── dataset_formatters.py
│   └── vision_dataset_formatters.py
│
├── utils/                               # Utilities
│   ├── simulation_logger.py            # Logging system
│   ├── results_storage.py              # Results management
│   ├── metrics_calculator.py           # Performance metrics
│   ├── prompts.py                       # Prompt templates
│   └── rate_limiter.py                  # API rate limiting
│
├── run_simulation_adk.py                # Main entry point
├── ALGO.md                              # Algorithm specification
├── readme_ADK.md                        # This file
├── IMPLEMENTATION_COMPLETE.md           # Implementation details
└── requirements.txt                     # Python dependencies
```

---

## Teamwork Components

### 1. Shared Mental Model (SMM)
**Purpose**: Passive knowledge repository for shared context

**Stores**:
- Question analysis (trick detection)
- Verified facts (consensus from R2)
- Debated points (controversies from R3)

**Impact**: Agents receive consistent shared knowledge, reducing redundant reasoning

### 2. Leadership
**Purpose**: Active orchestration with correction authority

**Powers**:
- Extract consensus facts from agent responses
- Create formal medical reports
- Mediate discussions after each turn
- Resolve ties with override capability

**Impact**: Structured coordination and quality control

### 3. Team Orientation
**Purpose**: Medical role specialization with hierarchical weighting

**Features**:
- Assigns specific medical specialties (Cardiologist, Radiologist, etc.)
- Hierarchical weights (0.5, 0.3, 0.2) hidden from agents
- Formal medical report generation

**Impact**: Specialized expertise with weighted aggregation

### 4. Trust Network
**Purpose**: Dynamic agent reliability scoring

**Mechanism**:
- Trust scores range: 0.4-1.0 (default: 0.8)
- Updated after R2, MM, and R3
- Evaluation criteria: fact accuracy, reasoning quality

**Impact**: Reliable agents weighted more in final vote

### 5. Mutual Monitoring
**Purpose**: Inter-round validation and quality control

**Protocol**:
1. Leader identifies weakest reasoning
2. Raises specific concern
3. Agent responds (accept/justify)
4. Updates Trust scores and SMM

**Impact**: Fact-checking and reasoning improvement

### Component Dependencies
```
SMM                → No dependencies (standalone)
Leadership         → Enhances SMM, TeamO, Trust
Team Orientation   → Works best with Leadership
Trust Network      → Can be evaluated by Leadership
Mutual Monitoring  → Requires Leadership + (optionally) Trust & SMM
```

---

## Reproducibility

### Experimental Setup

1. **Fixed Random Seed**: Set in dataset loaders for consistent sampling
```python
questions = DatasetLoader.load_medqa(n_questions=50, random_seed=42)
```

2. **Model Configuration**:
   - Model: `gemma3_4b` (or specified via `--model`)
   - Temperature: 0.7 (reasoning), 0.5 (recruitment)
   - Max tokens: 2048

3. **API Call Management**:
   - Exponential backoff retry (5 attempts)
   - Base delay: 60s for rate limits
   - Jitter: ±20% to prevent thundering herd

### Running Ablation Studies

To measure individual component impact, run experiments with single components enabled:

```bash
# Baseline
python run_simulation_adk.py --dataset medqa --n-questions 50

# +SMM
python run_simulation_adk.py --dataset medqa --n-questions 50 --smm

# +Leadership
python run_simulation_adk.py --dataset medqa --n-questions 50 --leadership

# +TeamO (with Leadership)
python run_simulation_adk.py --dataset medqa --n-questions 50 --leadership --team-orientation

# +Trust
python run_simulation_adk.py --dataset medqa --n-questions 50 --trust

# +MM (with Leadership + Trust)
python run_simulation_adk.py --dataset medqa --n-questions 50 --leadership --trust --mutual-monitoring

# Full system
python run_simulation_adk.py --dataset medqa --n-questions 50 --all-teamwork
```

### Expected API Call Counts

For N=3 agents, n_turns=2:

| Configuration | API Calls | Formula |
|---------------|-----------|---------|
| Base | 9 | 2N+3 |
| +SMM | 9 | 2N+3 |
| +Leadership | 11 | 2N+5 |
| +TeamO | 11 | 2N+5 |
| +Trust | 9 | 2N+3 |
| +MM | 12 | 2N+6 |
| **ALL ON** | 14 | 2N+8 |

---

## Results & Logging

### Output Structure
Results are saved to: `{output_dir}/{dataset}_{n_questions}q_run{X}/`

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

### Result Format

**Per-Question Result** (`q001_results.json`):
```json
{
  "question_id": "q001",
  "question": "...",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
  "recruited_agents": [
    {"agent_id": "agent_1", "role": "Cardiologist", "weight": 0.5},
    {"agent_id": "agent_2", "role": "Radiologist", "weight": 0.3},
    {"agent_id": "agent_3", "role": "Internist", "weight": 0.2}
  ],
  "round1_results": {...},
  "round2_results": {...},
  "round3_results": {...},
  "final_decision": {
    "primary_answer": "B",
    "borda_count": {...},
    "convergence": {...}
  },
  "ground_truth": "B",
  "is_correct": true,
  "metadata": {
    "total_time": 45.2,
    "api_calls": 14,
    "n_agents": 3
  },
  "teamwork_metrics": {
    "smm": {"verified_facts_count": 5, ...},
    "trust": {"trust_scores": {...}, ...},
    "leadership": {"mediations": 2},
    "mutual_monitoring": {"challenges": 1}
  }
}
```

### Key Metrics

- **Accuracy**: Overall correctness rate
- **Convergence**: Agreement rate among agents
- **Trust Scores**: Per-agent reliability
- **Timing**: Phase-wise execution time
- **API Calls**: Total API call count
- **Token Usage**: Input/output token counts

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{adk_multiagent_medical,
  title={ADK Multi-Agent Medical Reasoning System with Modular Teamwork Components},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo}
}
```

---

## License

[Your License Here]

---

## Contact

For questions or issues:
- GitHub Issues: [Link]
- Email: [Your Email]

---

## Acknowledgments

- Built on Google's Agent Development Kit (ADK)
- Medical datasets: MedQA, MedMCQA, PubMedQA, etc.
- Gemma models by Google DeepMind
