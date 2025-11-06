# Multi-Agent Gemma System

A modular multi-agent system for collaborative medical reasoning using Gemma models via Google AI Studio.

## Overview

This system enables 2-4 specialized medical expert agents to collaboratively solve medical questions through a three-round process:

1. **Round 1: Independent Analysis** - Each agent analyzes independently
2. **Round 2: Collaborative Discussion** - Agents see others' analyses and discuss
3. **Round 3: Final Ranking** - Agents provide final ranked decisions

Decisions are aggregated using Borda count, majority voting, and weighted consensus methods.

## Architecture

```
multi-agent-gemma/
├── config.py                    # System configuration
├── components/
│   ├── gemma_agent.py          # Role-specific agent wrapper
│   ├── agent_recruiter.py      # Dynamic agent recruitment
│   ├── simulation_rounds.py    # Round 1, 2, 3 execution
│   ├── decision_aggregator.py  # Ranking aggregation
│   ├── multi_agent_system.py   # Main orchestrator
│   └── rate_limit_manager.py   # Rate limiting coordinator
├── utils/
│   ├── prompts.py              # Centralized prompts
│   ├── simulation_logger.py    # Logging system
│   ├── metrics_calculator.py   # Metrics computation
│   └── results_storage.py      # Results organization
├── run_simulation.py           # Main entry point
└── results/                    # Output directory
```

## Installation

### Prerequisites

- Python 3.8+
- Existing SLM-TeamMedAgents environment
- Google AI Studio API key (or Hugging Face token)

### Setup

1. The multi-agent system is already integrated with your existing infrastructure:
   - Uses `chat_instances.py` for API calls
   - Uses `slm_config.py` for model configurations
   - Uses `utils/rate_limiter.py` for rate limiting
   - Uses `medical_datasets/` for dataset loading

2. Ensure environment variables are set:
   ```bash
   export GEMINI_API_KEY="your_api_key"
   # or
   export GOOGLE_API_KEY="your_api_key"
   ```

## Quick Start

### Basic Usage

```python
from multi_agent_gemma import MultiAgentSystem

# Initialize system
system = MultiAgentSystem(
    model_name="gemma3_4b",
    n_agents=3  # Or None for dynamic recruitment
)

# Run simulation
result = system.run_simulation(
    question="A 65-year-old man presents with chest pain...",
    options=[
        "A. Left anterior descending artery",
        "B. Right coronary artery",
        "C. Left circumflex artery",
        "D. Left main coronary artery"
    ],
    task_type="mcq"
)

# Get result
print(f"Answer: {result['final_decision']['primary_answer']}")
print(f"Confidence: {result['final_decision']['borda_count']}")
```

### Dynamic Agent Recruitment

```python
# Let the system decide optimal number of agents (2-4)
system = MultiAgentSystem(model_name="gemma3_4b", n_agents=None)
result = system.run_simulation(question, options)

# System analyzes complexity and recruits appropriate number of specialized agents
print(f"Recruited {len(result['recruited_agents'])} agents:")
for agent in result['recruited_agents']:
    print(f"  - {agent['role']}: {agent['expertise']}")
```

### Batch Processing (Coming Soon)

```python
from multi_agent_gemma import BatchSimulationRunner

runner = BatchSimulationRunner(
    model_name="gemma3_4b",
    n_agents=3,
    dataset_name="medqa",
    n_questions=50
)

summary = runner.run()
print(f"Accuracy: {summary['accuracy']:.2%}")
```

## Configuration

### config.py Settings

```python
# Agent configuration
DEFAULT_N_AGENTS = None  # None = dynamic, or 2-4 for fixed
MAX_AGENTS = 4
MIN_AGENTS = 2

# Decision aggregation
PRIMARY_DECISION_METHOD = "borda_count"  # or "majority_vote", "weighted_consensus"

# Rate limiting (inherited from slm_config.py)
RATE_LIMITS = {
    "gemma3_4b": {
        "rpm": 30,      # Requests per minute
        "tpm": 15000,   # Tokens per minute
        "rpd": 14400,   # Requests per day
    }
}
```

## Features

### Implemented ✅

- **Dynamic Agent Recruitment**: LLM determines optimal 2-4 agents based on question complexity
- **Three-Round Collaboration**: Independent → Collaborative → Final Ranking
- **Multiple Aggregation Methods**: Borda count, majority voting, weighted consensus
- **Rate Limiting**: Coordinates API calls across multiple agents
- **Comprehensive Logging**: Console + file logging with separate error logs
- **Modular Design**: Easy to extend with new components

### Planned 🔄

- **Leadership**: One agent coordinates discussion
- **Trust Network**: Agents weight others' opinions based on agreement history
- **Mutual Monitoring**: Agents check each other's reasoning
- **Team Orientation**: Emphasis on shared goals and consensus
- **Batch Processing**: Run on large datasets (MedQA, MedMCQA, etc.)
- **Metrics Calculation**: Accuracy, convergence, disagreement analysis
- **Results Export**: CSV/JSON exports with visualizations

## Components

### GemmaAgent

Role-specific medical expert agent that wraps `SLMAgent`:

```python
from components import GemmaAgent

agent = GemmaAgent(
    agent_id="agent_1",
    role="Cardiologist",
    expertise="Expert in cardiovascular diseases and ECG interpretation"
)

response = agent.analyze_question(prompt, round_number=1)
answer = agent.extract_answer(response, task_type="mcq")
```

### AgentRecruiter

Dynamically recruits specialized agents:

```python
from components import AgentRecruiter

recruiter = AgentRecruiter(model_name="gemma3_4b")
agents = recruiter.recruit_agents(question, options, n_agents=None)  # Dynamic
```

### SimulationRounds

Executes three rounds of collaboration:

```python
from components import RoundOrchestrator

orchestrator = RoundOrchestrator(agents)
results = orchestrator.execute_all_rounds(question, options)
```

### DecisionAggregator

Aggregates agent decisions:

```python
from components import DecisionAggregator

aggregator = DecisionAggregator()
final_decision = aggregator.aggregate_decisions(agent_decisions)
```

## Prompts

All prompts are centralized in `utils/prompts.py`:

```python
from utils.prompts import (
    RECRUITMENT_PROMPTS,
    ROUND1_PROMPTS,
    ROUND2_PROMPTS,
    ROUND3_PROMPTS
)
```

Modify prompts to customize agent behavior and collaboration style.

## Rate Limiting

The system coordinates API calls across multiple agents to respect rate limits:

```python
from components import MultiAgentRateLimiter

rate_limiter = MultiAgentRateLimiter(model_name="gemma3_4b", n_agents=3)

# Estimate time for batch processing
estimate = rate_limiter.estimate_time(n_questions=50)
print(f"Estimated time: {estimate['estimated_hours']:.1f} hours")

# Automatic rate limiting during simulation
rate_limiter.wait_if_needed()
```

**Example Estimates** (for gemma3_4b with RPM=30):
- 3 agents × 3 rounds = 9 API calls per question
- Can process ~3.3 questions/minute
- 50 questions ≈ 15 minutes
- 100 questions ≈ 30 minutes

## Results Structure

```
results/
└── run_20250105_143022/
    ├── config.json              # Run configuration
    ├── questions/               # Individual question results
    │   ├── q001_results.json
    │   ├── q002_results.json
    │   └── ...
    ├── metrics/                 # Aggregated metrics
    │   ├── accuracy_summary.csv
    │   ├── convergence_analysis.json
    │   ├── disagreement_matrix.csv
    │   └── agent_performance.json
    ├── logs/                    # Detailed logs
    │   ├── simulation.log
    │   ├── rate_limiting.log
    │   └── errors.log
    └── summary_report.json      # High-level overview
```

## Example Results

```json
{
  "question": "A 65-year-old man presents with chest pain...",
  "recruited_agents": [
    {"agent_id": "agent_1", "role": "Cardiologist", "expertise": "..."},
    {"agent_id": "agent_2", "role": "Emergency Medicine Physician", "expertise": "..."},
    {"agent_id": "agent_3", "role": "Radiologist", "expertise": "..."}
  ],
  "final_decision": {
    "borda_count": {
      "scores": {"A": 2, "B": 7, "C": 1, "D": 0},
      "winner": "B"
    },
    "majority_vote": {
      "winner": "B",
      "vote_percentage": 0.67
    },
    "primary_answer": "B"
  },
  "agreement_metrics": {
    "full_agreement": false,
    "partial_agreement_rate": 0.67
  },
  "is_correct": true,
  "metadata": {
    "total_time": 12.5,
    "n_agents": 3,
    "model_name": "gemma3_4b"
  }
}
```

## Extending the System

### Add Custom Aggregation Method

```python
# In decision_aggregator.py
def custom_voting(self, rankings: Dict[str, List[str]]) -> Dict[str, Any]:
    """Your custom aggregation logic."""
    # Implement your method
    pass
```

### Add Custom Round

```python
# In simulation_rounds.py
class Round2_5ErrorChecking(SimulationRound):
    """Agents check each other for errors."""
    def execute(self, **kwargs):
        # Implement error checking logic
        pass
```

### Customize Prompts

```python
# In utils/prompts.py
ROUND2_PROMPTS["aggressive_debate"] = """
You are a {role}. CHALLENGE your teammates' reasoning aggressively...
"""
```

## Troubleshooting

### Rate Limit Errors

- Decrease `n_agents` or `n_questions`
- Add `--delay` between questions
- Check `rate_limiting.log` for details

### Out of Memory

- Use `chat_instance_type="google_ai_studio"` (remote API)
- Reduce `n_agents`
- Clear conversation history between questions

### Import Errors

- Ensure you're running from parent directory: `cd e:/SLM-TeamMedAgents`
- Check Python path includes parent directory

## Citation

If you use this system in your research, please cite:

```bibtex
@software{multiagent_gemma2025,
  title={Multi-Agent Gemma System for Medical Reasoning},
  author={Your Team},
  year={2025},
  url={https://github.com/yourrepo/SLM-TeamMedAgents}
}
```

## License

[Your License Here]

## Contact

[Your Contact Information]
