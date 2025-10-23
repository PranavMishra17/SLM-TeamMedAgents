"""
Multi-Agent Gemma Components Package

Core components for multi-agent collaborative reasoning.
"""

from .gemma_agent import GemmaAgent, create_agent, create_agents_from_roles
from .agent_recruiter import AgentRecruiter, create_recruiter, quick_recruit
from .simulation_rounds import (
    Round1Independent,
    Round2Collaborative,
    Round3Ranking,
    RoundOrchestrator,
    run_three_rounds
)
from .decision_aggregator import (
    DecisionAggregator,
    aggregate_with_all_methods,
    get_final_answer
)
from .multi_agent_system import MultiAgentSystem, quick_simulate
from .rate_limit_manager import MultiAgentRateLimiter, create_rate_limiter

__all__ = [
    # Agents
    "GemmaAgent",
    "create_agent",
    "create_agents_from_roles",

    # Recruitment
    "AgentRecruiter",
    "create_recruiter",
    "quick_recruit",

    # Rounds
    "Round1Independent",
    "Round2Collaborative",
    "Round3Ranking",
    "RoundOrchestrator",
    "run_three_rounds",

    # Aggregation
    "DecisionAggregator",
    "aggregate_with_all_methods",
    "get_final_answer",

    # System
    "MultiAgentSystem",
    "quick_simulate",

    # Rate limiting
    "MultiAgentRateLimiter",
    "create_rate_limiter",
]
