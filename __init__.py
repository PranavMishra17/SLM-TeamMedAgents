"""
Multi-Agent Gemma System

A modular multi-agent system for medical reasoning using Gemma models via Google AI Studio.

Quick Start:
    from multi_agent_gemma import MultiAgentSystem

    system = MultiAgentSystem(model_name="gemma3_4b", n_agents=3)
    result = system.run_simulation(
        question="What is the diagnosis?",
        options=["A. ...", "B. ...", "C. ...", "D. ..."]
    )

    print(f"Answer: {result['final_decision']['primary_answer']}")
"""

__version__ = "0.1.0"

from .components import (
    MultiAgentSystem,
    GemmaAgent,
    AgentRecruiter,
    DecisionAggregator,
    quick_simulate
)

__all__ = [
    "MultiAgentSystem",
    "GemmaAgent",
    "AgentRecruiter",
    "DecisionAggregator",
    "quick_simulate",
]
