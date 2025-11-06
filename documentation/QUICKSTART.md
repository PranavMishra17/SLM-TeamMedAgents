# Quick Start Guide - Multi-Agent Gemma System

## 🚀 Get Started in 5 Minutes

### Prerequisites

1. **Python 3.8+** installed
2. **Google AI Studio API Key**:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. **Existing SLM-TeamMedAgents environment** set up

### Installation Check

From the `SLM-TeamMedAgents` directory:

```bash
cd e:/SLM-TeamMedAgents
python multi-agent-gemma/test_system.py
```

Expected output:
```
✅ All imports successful!
✅ Prompt generation working!
✅ TEST PASSED - System is working!
```

---

## 📚 Basic Usage

### Example 1: Single Question with Fixed Agents

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))  # Add current directory to path

from multi_agent_gemma import MultiAgentSystem

# Initialize system with 3 agents
system = MultiAgentSystem(
    model_name="gemma3_4b",
    n_agents=3  # Fixed: 3 agents
)

# Medical question
question = """
A 65-year-old man presents with chest pain and shortness of breath.
ECG shows ST elevation in leads II, III, and aVF.
Which coronary artery is most likely occluded?
"""

options = [
    "A. Left anterior descending artery",
    "B. Right coronary artery",
    "C. Left circumflex artery",
    "D. Left main coronary artery"
]

# Run simulation
result = system.run_simulation(
    question=question,
    options=options,
    task_type="mcq",
    ground_truth="B"
)

# View results
print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")

print(f"\nRecruited Agents:")
for agent in result['recruited_agents']:
    print(f"  • {agent['role']}: {agent['expertise']}")

print(f"\nFinal Decision:")
print(f"  Answer: {result['final_decision']['primary_answer']}")
print(f"  Method: Borda Count")
print(f"  Scores: {result['final_decision']['borda_count']['scores']}")

print(f"\nEvaluation:")
print(f"  Ground Truth: {result['ground_truth']}")
print(f"  Correct: {'✅ YES' if result['is_correct'] else '❌ NO'}")

print(f"\nTiming:")
print(f"  Total Time: {result['metadata']['total_time']:.2f}s")
print(f"  Agents: {result['metadata']['n_agents']}")
```

---

### Example 2: Dynamic Agent Recruitment

```python
from multi_agent_gemma import MultiAgentSystem

# Let system decide optimal number of agents (2-4)
system = MultiAgentSystem(
    model_name="gemma3_4b",
    n_agents=None,  # Dynamic recruitment
    enable_dynamic_recruitment=True
)

# Simple question (may get 2 agents)
simple_question = "What is the most common cause of community-acquired pneumonia?"
simple_options = ["A. Streptococcus pneumoniae", "B. Haemophilus influenzae",
                 "C. Mycoplasma pneumoniae", "D. Staphylococcus aureus"]

result = system.run_simulation(simple_question, simple_options)
print(f"Simple question → {len(result['recruited_agents'])} agents recruited")

# Complex question (may get 4 agents)
complex_question = """
A 45-year-old woman with systemic lupus erythematosus presents with
progressive dyspnea, chest pain, and lower extremity edema.
Echocardiography shows pericardial effusion and reduced ejection fraction.
"""
complex_options = ["A. Acute myocardial infarction", "B. Lupus pericarditis with myocarditis",
                  "C. Pulmonary embolism", "D. Congestive heart failure"]

result = system.run_simulation(complex_question, complex_options)
print(f"Complex question → {len(result['recruited_agents'])} agents recruited")
```

---

### Example 3: Exploring Results

```python
from multi_agent_gemma import MultiAgentSystem
import json

system = MultiAgentSystem(model_name="gemma3_4b", n_agents=3)

result = system.run_simulation(question, options)

# View Round 1 analyses
print("\n📝 ROUND 1: Independent Analysis")
for agent_id, analysis in result['round1_results'].items():
    print(f"\n{agent_id}:")
    print(f"  {analysis[:200]}...")

# View Round 2 discussion
print("\n💬 ROUND 2: Collaborative Discussion")
for agent_id, discussion in result['round2_results'].items():
    print(f"\n{agent_id}:")
    print(f"  {discussion[:200]}...")

# View Round 3 decisions
print("\n🎯 ROUND 3: Final Rankings")
for agent_id, decision in result['round3_results'].items():
    print(f"\n{agent_id}:")
    print(f"  Ranking: {decision.get('ranking', 'N/A')}")
    print(f"  Confidence: {decision.get('confidence', 'N/A')}")

# View agreement metrics
if result.get('agreement_metrics'):
    print("\n🤝 Agreement Metrics:")
    metrics = result['agreement_metrics']
    print(f"  Full Agreement: {metrics['full_agreement']}")
    print(f"  Partial Agreement: {metrics['partial_agreement_rate']:.1%}")
    print(f"  Most Common Choice: {metrics['most_common_first_choice']}")

# Save full result to JSON
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print("\n💾 Full result saved to result.json")
```

---

## ⚙️ Configuration

### Change Models

```python
# Use MedGemma (medical specialist model)
system = MultiAgentSystem(model_name="medgemma_4b", n_agents=3)

# Use different model for recruitment vs agents
from components import AgentRecruiter, GemmaAgent

recruiter = AgentRecruiter(model_name="gemma3_4b")  # Fast general model
agents = recruiter.recruit_agents(
    question, options,
    agent_model_name="medgemma_4b"  # Specialized medical model
)
```

### Customize Aggregation

```python
# Change primary decision method
import config as multi_agent_config
multi_agent_config.PRIMARY_DECISION_METHOD = "majority_vote"

# Or access all methods
result = system.run_simulation(question, options)
print(f"Borda: {result['final_decision']['borda_count']['winner']}")
print(f"Majority: {result['final_decision']['majority_vote']['winner']}")
print(f"Weighted: {result['final_decision']['weighted_consensus']['winner']}")
```

### Customize Prompts

```python
# Modify prompts in utils/prompts.py
from utils import prompts

# View current Round 1 prompt
print(prompts.ROUND1_PROMPTS["independent_analysis_mcq"])

# Add custom prompt
prompts.ROUND2_PROMPTS["aggressive_debate"] = """
You are a {role}. CHALLENGE your teammates' reasoning...
"""
```

---

## 🔍 Monitoring & Debugging

### Enable Detailed Logging

```python
import logging

# Set logging level
logging.basicConfig(
    level=logging.DEBUG,  # Shows all details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run simulation (will see all API calls, prompts, etc.)
result = system.run_simulation(question, options)
```

### Check Rate Limits

```python
from components import MultiAgentRateLimiter

# Create rate limiter
rate_limiter = MultiAgentRateLimiter(model_name="gemma3_4b", n_agents=3)

# Estimate time for batch
estimate = rate_limiter.estimate_time(n_questions=50)
print(f"50 questions will take ~{estimate['estimated_minutes']:.1f} minutes")

# Check current status
status = rate_limiter.get_rate_status()
print(f"Requests this minute: {status['requests_this_minute']}/{status['limits']['rpm']}")
```

### View Logs

After running simulations, check logs in:
```
results/run_YYYYMMDD_HHMMSS/logs/
  - simulation.log       # Main events
  - rate_limiting.log    # Rate limit events
  - errors.log           # Errors only
```

---

## ⚡ Performance Tips

### Speed Up Simulations

1. **Use fewer agents**: `n_agents=2` instead of 4
2. **Fixed recruitment**: `n_agents=3` (skips complexity analysis)
3. **Parallel Round 1** (future): Enable parallel execution
4. **Lower temperature**: Faster token generation

```python
system = MultiAgentSystem(
    model_name="gemma3_4b",
    n_agents=2,  # Fewer agents = faster
    enable_dynamic_recruitment=False  # Skip recruitment step
)

# Modify agent temperature
import config as multi_agent_config
multi_agent_config.AGENT_TEMPERATURE = 0.1  # Faster generation
```

### Reduce API Calls

Each question uses: **n_agents × 3 rounds** API calls

- 2 agents: 6 calls/question
- 3 agents: 9 calls/question
- 4 agents: 12 calls/question

With RPM=30:
- 2 agents: ~5 questions/minute
- 3 agents: ~3.3 questions/minute
- 4 agents: ~2.5 questions/minute

---

## 🐛 Troubleshooting

### "Import Error: No module named..."

```python
# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))  # If in multi-agent-gemma/
sys.path.append(str(Path.cwd()))         # If in SLM-TeamMedAgents/
```

### "API Key Not Found"

```bash
# Set environment variable
export GEMINI_API_KEY="your_key"

# Or in Python
import os
os.environ['GEMINI_API_KEY'] = "your_key"

# Or in .env file (if using python-dotenv)
# GEMINI_API_KEY=your_key
```

### "Rate Limit Exceeded"

- **Solution 1**: Wait 60 seconds
- **Solution 2**: Use fewer agents
- **Solution 3**: Process fewer questions per batch

```python
# The system will automatically wait, but you can check status
rate_limiter.log_rate_status()
```

### "Response Parsing Error"

If agents' responses can't be parsed:

1. Check logs in `results/run_*/logs/errors.log`
2. View raw response in `result['round3_results'][agent_id]['raw']`
3. Adjust extraction patterns in `gemma_agent.py`

---

## 📊 Next Steps

1. **Run your first simulation** (Example 1 above)
2. **Try dynamic recruitment** (Example 2)
3. **Explore results** (Example 3)
4. **Customize for your use case** (Configuration section)
5. **Read full documentation** (README.md)

## 🎓 Learn More

- **Architecture**: [README.md](README.md)
- **Implementation Status**: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- **Prompts**: [utils/prompts.py](utils/prompts.py)
- **Configuration**: [config.py](config.py)

## 💬 Get Help

- Run `python test_system.py` to verify setup
- Check logs in `results/run_*/logs/`
- Review `IMPLEMENTATION_STATUS.md` for known issues

---

**Happy Multi-Agent Reasoning! 🚀**
