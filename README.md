# SLM-TeamMedAgents: Multi-Agent Medical Reasoning System

A comprehensive multi-agent collaborative reasoning system using Small Language Models (SLMs) for medical question answering. Features dynamic agent recruitment, three-round deliberation, and multiple decision aggregation methods.

## Overview

This system implements a sophisticated multi-agent framework where specialized AI agents collaborate to solve medical questions through structured rounds of independent analysis, collaborative discussion, and final ranking. The system supports both single-question simulations and large-scale batch processing with comprehensive metrics tracking.

## Key Features

### Multi-Agent Collaboration
- **Dynamic Agent Recruitment**: LLM determines optimal agent count (2-4) and specializations
- **Fixed Agent Count**: Manual specification of agent count for controlled experiments
- **Specialized Agents**: Each agent has unique medical expertise (e.g., cardiologist, radiologist)
- **Three-Round Deliberation**:
  - **Round 1**: Independent analysis and initial ranking
  - **Round 2**: Collaborative discussion and perspective sharing
  - **Round 3**: Final ranking with revised reasoning

### Decision Aggregation Methods
- **Borda Count**: Point-based ranking system (1st=3pts, 2nd=2pts, 3rd=1pt)
- **Majority Vote**: Most frequently ranked #1 answer
- **Weighted Consensus**: Trust-based weighting (future enhancement)

### Comprehensive Metrics & Analysis
- **Accuracy Metrics**: Overall, by method, by task type
- **Convergence Analysis**: Round 1 vs Round 3 agreement rates
- **Disagreement Matrix**: Pairwise agent agreement tracking
- **Opinion Change Tracking**: How often agents revise rankings
- **Agent Performance**: Individual agent accuracy profiles

### Rate Limiting & Logging
- **Multi-Agent Rate Limiter**: Coordinates API calls across all agents
- **Time Estimation**: Batch processing time predictions
- **Comprehensive Logging**: Detailed simulation logs with timestamps
- **Results Storage**: JSON results + CSV exports for analysis

### Modular Chat Instances
- **Google AI Studio**: Managed API access (default)
- **Hugging Face**: Local model execution
- **Extensible**: Easy to add new providers

### Supported Models
- **Gemma3-4B-IT**: General-purpose model with vision capabilities
- **MedGemma-4B-IT**: Medical-specialized model with vision capabilities

## Architecture

```
Root Directory
├── Core System
│   ├── slm_runner.py               # Base SLM agent wrapper
│   ├── slm_config.py               # Model configuration & rate limits
│   ├── chat_instances.py           # Modular chat provider system
│   └── config.py                   # Multi-agent system configuration
│
├── Multi-Agent Components (components/)
│   ├── gemma_agent.py              # Individual agent with memory
│   ├── agent_recruiter.py          # Dynamic/fixed agent recruitment
│   ├── simulation_rounds.py        # Three-round orchestration
│   ├── decision_aggregator.py      # Borda/Majority/Weighted methods
│   ├── multi_agent_system.py       # Main orchestrator
│   └── rate_limit_manager.py       # Multi-agent rate limiting
│
├── Utilities (utils/)
│   ├── prompts.py                  # Round-specific prompt templates
│   ├── rate_limiter.py             # Base rate limiting logic
│   ├── results_logger.py           # Token tracking & logging
│   ├── simulation_logger.py        # Multi-agent simulation logs
│   ├── results_storage.py          # JSON/CSV results storage
│   └── metrics_calculator.py       # Comprehensive metrics analysis
│
├── Medical Datasets (medical_datasets/)
│   ├── dataset_loader.py           # Dataset loading utilities
│   ├── dataset_formatters.py       # Text dataset formatting
│   ├── vision_dataset_formatters.py # Image dataset formatting
│   └── dataset_runner.py           # Batch dataset processing
│
├── Entry Points
│   ├── run_simulation.py           # Multi-agent simulation runner
│   └── test_system.py              # System validation tests
│
└── Results Output
    ├── results/                    # Multi-agent simulation results
    │   ├── {model_name}/
    │   │   ├── {dataset_name}/
    │   │   │   ├── results_{timestamp}.json
    │   │   │   ├── accuracy_summary.csv
    │   │   │   ├── disagreement_matrix.csv
    │   │   │   └── summary_report.json
    │   │   └── metrics/
    │   └── logs/
    │       └── simulation_{timestamp}.log
    │
    └── SLM_Results/               # Single-agent baseline results
        ├── gemma3_4b/
        └── medgemma_4b/
```

## Installation

### Prerequisites
```bash
# Python 3.8+
pip install google-genai transformers torch pillow tqdm datasets
```

### Environment Setup

**For Google AI Studio (Default)**:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

**For Hugging Face Local**:
```bash
export HUGGINGFACE_TOKEN="your_hf_token_here"
huggingface-cli login  # For gated models
```

## Usage

### Single Question Simulation

```bash
# Quick test with dynamic recruitment
python run_simulation.py \
  --question "A 65-year-old patient presents with chest pain..." \
  --options "A) Myocardial infarction" "B) Angina pectoris" "C) GERD" "D) Panic attack" \
  --ground_truth "A"

# Fixed agent count
python run_simulation.py \
  --question "..." \
  --options "A) ..." "B) ..." \
  --n_agents 3 \
  --model medgemma_4b
```

### Batch Processing

```bash
# Process 10 questions from MedQA
python run_simulation.py \
  --dataset medqa \
  --n_questions 10 \
  --n_agents 3 \
  --model gemma3_4b

# Process 50 questions from MedMCQA with dynamic recruitment
python run_simulation.py \
  --dataset medmcqa \
  --n_questions 50 \
  --model medgemma_4b

# Full dataset evaluation
python run_simulation.py \
  --dataset medqa \
  --n_questions 100 \
  --n_agents 2 \
  --output_dir results/full_eval
```

### Supported Datasets

- **medqa**: USMLE-style medical questions
- **medmcqa**: Indian medical entrance exam questions
- **pubmedqa**: Biomedical research questions
- **mmlu_medical**: MMLU medical subset
- **pmc_vqa**: Medical visual question answering (with images)
- **path_vqa**: Pathology image analysis

### Command-Line Options

```bash
python run_simulation.py --help

Options:
  --question TEXT           Single question to process
  --options TEXT            Answer options (multiple)
  --ground_truth TEXT       Correct answer for evaluation
  --dataset TEXT            Dataset name for batch processing
  --n_questions INT         Number of questions (default: 10)
  --n_agents INT            Fixed agent count (2-4) or None for dynamic
  --model TEXT              Model name (gemma3_4b, medgemma_4b)
  --chat_instance TEXT      Chat instance type (google_ai_studio, huggingface)
  --output_dir TEXT         Output directory (default: results/)
  --enable_logging          Enable detailed logging (default: True)
```

## Configuration

### Multi-Agent System Settings ([config.py](config.py))

```python
# Agent Configuration
DEFAULT_MODEL = "gemma3_4b"
DEFAULT_N_AGENTS = 3
MIN_AGENTS = 2
MAX_AGENTS = 4

# Round Configuration
ENABLE_ROUND_1 = True  # Independent analysis
ENABLE_ROUND_2 = True  # Collaborative discussion
ENABLE_ROUND_3 = True  # Final ranking

# Decision Aggregation
PRIMARY_DECISION_METHOD = "borda_count"
DEFAULT_AGGREGATION_METHODS = ["borda_count", "majority_vote", "weighted_consensus"]

# Rate Limiting
ENABLE_RATE_LIMITING = True
ESTIMATE_TOKENS_PER_ROUND = 1000

# Logging
ENABLE_DETAILED_LOGGING = True
LOG_AGENT_REASONING = True
```

### Model Configuration ([slm_config.py](slm_config.py))

```python
# Rate Limits (per model)
RATE_LIMITS = {
    "gemma3_4b": {
        "rpm": 30,      # Requests per minute
        "tpm": 15000,   # Tokens per minute
        "rpd": 14400    # Requests per day
    },
    "medgemma_4b": {
        "rpm": 30,
        "tpm": 15000,
        "rpd": 14400
    }
}
```

## Output Structure

### Simulation Results

Each simulation produces comprehensive JSON results:

```json
{
  "question": "...",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "task_type": "mcq",
  "recruited_agents": [
    {
      "agent_id": "Agent_1",
      "role": "Cardiologist",
      "expertise": "Cardiovascular diseases, ECG interpretation"
    }
  ],
  "round1_results": {
    "Agent_1": {
      "ranking": ["A", "B", "C", "D"],
      "reasoning": "...",
      "response_time": 2.34
    }
  },
  "round2_results": { /* collaborative discussion */ },
  "round3_results": { /* final rankings */ },
  "final_decision": {
    "borda_count": {
      "winner": "A",
      "scores": {"A": 12, "B": 6, "C": 3, "D": 0},
      "confidence": 0.92
    },
    "majority_vote": {
      "winner": "A",
      "vote_counts": {"A": 3, "B": 0}
    },
    "primary_answer": "A"
  },
  "agreement_metrics": {
    "kendall_w": 0.87,
    "avg_pairwise_agreement": 0.82
  },
  "is_correct": true,
  "metadata": {
    "timestamp": "2025-10-06T15:30:00",
    "total_time": 12.45,
    "n_agents": 3,
    "model_name": "gemma3_4b"
  }
}
```

### Batch Results

Batch processing generates:

1. **Individual Results**: `results_{timestamp}.json` (all questions)
2. **Accuracy Summary**: `accuracy_summary.csv`
   ```csv
   Metric,Value,Correct,Total
   Overall Accuracy,0.850,17,20
   Borda Count Accuracy,0.850,17,20
   Majority Vote Accuracy,0.800,16,20
   ```

3. **Disagreement Matrix**: `disagreement_matrix.csv`
   ```csv
   ,Agent_1,Agent_2,Agent_3
   Agent_1,1.000,0.850,0.820
   Agent_2,0.850,1.000,0.780
   Agent_3,0.820,0.780,1.000
   ```

4. **Summary Report**: `summary_report.json` (comprehensive metrics)

5. **Simulation Logs**: `logs/simulation_{timestamp}.log`

## Testing

### System Validation

```bash
# Run all validation tests
python test_system.py

# Tests include:
# - Import validation
# - Prompt generation
# - Single simulation (if API key available)
```

### Example Output

```
======================================================================
MULTI-AGENT GEMMA SYSTEM - TEST SUITE
======================================================================

Testing imports...
  [OK] config
  [OK] components
  [OK] utils

[PASS] All imports successful


======================================================================
TESTING PROMPT GENERATION
======================================================================

Testing Round 1 prompts...
  [OK] Prompt generated (length: 450)

[PASS] All prompt tests passed


======================================================================
TESTING SIMPLE SIMULATION
======================================================================

Running 3-agent simulation...
  [OK] Recruited 3 agents
  [OK] Round 1 complete (3 responses)
  [OK] Round 2 complete (3 responses)
  [OK] Round 3 complete (3 rankings)
  [OK] Final decision: A (Borda Count)
  [OK] Simulation time: 8.23s

[PASS] Simulation test passed


======================================================================
TEST SUMMARY
======================================================================
Imports             : [PASS] PASSED
Prompts             : [PASS] PASSED
Simulation          : [PASS] PASSED

Total: 3 passed, 0 failed, 0 skipped

[PASS] ALL TESTS PASSED
```

## Performance Metrics

### Time Estimates

The system provides automatic time estimation for batch processing:

```
======================================================================
BATCH PROCESSING TIME ESTIMATE
======================================================================
Questions to process: 100
Agents per question: 3
Total API calls: 900 (9 per question)

Rate Limits:
  - Requests per minute: 30
  - Tokens per minute: 15000
  - Requests per day: 14400
  - Limiting factor: RPM (requests per minute)

Processing Rate:
  - Questions per minute: 3.33
  - Questions per hour: 200.0
  - Questions per day: 1600

Estimated Time:
  [TIME] 36.0 minutes

(Estimate includes 20% buffer for retries and overhead)
======================================================================
```

### Typical Performance

- **Single Question**: 8-15 seconds (3 agents, 3 rounds)
- **Batch (10 questions)**: 2-3 minutes
- **Batch (100 questions)**: 30-40 minutes
- **Large Dataset (1000 questions)**: 5-6 hours

*Times vary based on model, rate limits, and network conditions*

## Metrics & Analysis

### Calculated Metrics

1. **Accuracy Metrics**
   - Overall accuracy (all methods)
   - Per-method accuracy (Borda, Majority, Weighted)
   - Per-task-type accuracy

2. **Convergence Analysis**
   - Round 1 → Round 3 agreement rate
   - Opinion change frequency
   - Convergence direction (toward correct answer)

3. **Disagreement Analysis**
   - Pairwise agent agreement matrix
   - Most/least agreeable agent pairs
   - Disagreement patterns

4. **Agent Performance**
   - Individual agent accuracy
   - Agent specialization effectiveness
   - Best performing agent identification

### Accessing Metrics

```python
from utils.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator()
calculator.add_simulation_result(result)  # Add results

# Calculate metrics
accuracy = calculator.calculate_accuracy(ground_truth_answers)
convergence = calculator.calculate_convergence()
disagreement = calculator.calculate_disagreement_matrix()
summary = calculator.generate_summary_report(ground_truth_answers)
```

## Advanced Usage

### Custom Agent Configuration

```python
from components import MultiAgentSystem

# Create system with custom settings
system = MultiAgentSystem(
    model_name="medgemma_4b",
    n_agents=4,  # Fixed 4 agents
    chat_instance_type="google_ai_studio",
    enable_dynamic_recruitment=False  # Disable dynamic recruitment
)

# Run simulation
result = system.run_simulation(
    question="...",
    options=["A) ...", "B) ...", "C) ...", "D) ..."],
    task_type="mcq",
    ground_truth="A"
)
```

### Batch Processing with Custom Settings

```python
from run_simulation import BatchSimulationRunner

runner = BatchSimulationRunner(
    model_name="gemma3_4b",
    n_agents=3,
    output_dir="results/custom_experiment",
    dataset_name="medqa",
    n_questions=50
)

results = runner.run()
```

### Rate Limit Management

```python
from components.rate_limit_manager import MultiAgentRateLimiter

limiter = MultiAgentRateLimiter(model_name="gemma3_4b", n_agents=3)

# Get time estimate
estimate = limiter.estimate_time(n_questions=100)
print(f"Estimated time: {estimate['estimated_hours']:.1f} hours")

# Check current status
status = limiter.get_rate_status()
print(f"RPM utilization: {status['rpm_utilization']:.1%}")
```

## Future Enhancements

### Planned Features (Phase 2)

1. **Leadership Component**: One agent coordinates discussion and breaks ties
2. **Trust Network**: Dynamic trust scores based on historical accuracy
3. **Mutual Monitoring**: Agents check each other's reasoning for errors
4. **Team Orientation**: Emphasis on consensus-building

### Extensibility

The system is designed for easy extension:

- **New Aggregation Methods**: Add to [decision_aggregator.py](components/decision_aggregator.py)
- **New Round Types**: Extend [simulation_rounds.py](components/simulation_rounds.py)
- **Custom Metrics**: Add to [metrics_calculator.py](utils/metrics_calculator.py)
- **New Chat Instances**: Register in [chat_instances.py](chat_instances.py)

## Troubleshooting

### Common Issues

**Import Errors**:
- Ensure all files are in root directory (not subdirectories)
- Check `__init__.py` files exist in `components/` and `utils/`

**API Rate Limits**:
- Adjust rate limits in [slm_config.py](slm_config.py)
- Enable rate limiting: `ENABLE_RATE_LIMITING = True` in [config.py](config.py)

**Missing API Keys**:
```bash
# Check if key is set
echo $GEMINI_API_KEY

# Set key
export GEMINI_API_KEY="your_key_here"
```

**Memory Issues (HuggingFace Local)**:
- Reduce batch size
- Use smaller model
- Enable gradient checkpointing

### Debug Mode

```bash
# Enable verbose logging
python run_simulation.py --dataset medqa --n_questions 5 --verbose

# Check system configuration
python -c "from config import *; print(f'Model: {DEFAULT_MODEL}, Agents: {DEFAULT_N_AGENTS}')"
```

## Citation

If you use this system in your research, please cite:

```bibtex
@software{slm_teammedagents,
  title={SLM-TeamMedAgents: Multi-Agent Medical Reasoning System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/SLM-TeamMedAgents}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Acknowledgments

- Built on Google's Gemma and MedGemma models
- Uses Anthropic's multi-agent collaboration principles
- Inspired by medical team decision-making research

---

**Last Updated**: October 6, 2025
**Version**: 2.0.0 (Multi-Agent System Complete)
