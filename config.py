"""
Multi-Agent System Configuration

Central configuration for multi-agent Gemma-based medical reasoning system.
Reuses rate limits and model configs from existing slm_config.py.
"""

import os
from pathlib import Path

# ============= MODEL CONFIGURATION =============
# Reuse existing model configurations from slm_config.py
DEFAULT_MODEL = "gemma3_4b"  # Primary model for agents
RECRUITMENT_MODEL = "gemma3_4b"  # Model used by recruiter to determine agent count/roles

# ============= AGENT CONFIGURATION =============
# Agent count settings
DEFAULT_N_AGENTS = None  # None = dynamic recruitment (2-4 agents based on complexity)
MAX_AGENTS = 4
MIN_AGENTS = 2

# Agent behavior
AGENT_TEMPERATURE = 0.3  # Lower temperature for more consistent medical reasoning
AGENT_MAX_TOKENS = 8192

# ============= RECRUITMENT CONFIGURATION =============
# Dynamic recruitment settings
ENABLE_DYNAMIC_RECRUITMENT = True  # If False, always use DEFAULT_N_AGENTS
COMPLEXITY_THRESHOLD_SIMPLE = 0.3  # Below this: 2 agents
COMPLEXITY_THRESHOLD_MODERATE = 0.6  # Below this: 3 agents, above: 4 agents

# ============= SIMULATION CONFIGURATION =============
# Round settings
NUM_ROUNDS = 3  # Fixed: Round 1 (independent), Round 2 (collaborative), Round 3 (ranking)
ENABLE_PARALLEL_ROUND1 = False  # If True, agents analyze in parallel (requires async)
CONTEXT_SHARING_MODE = "full_text"  # Options: "full_text", "summary" (future)

# Task types
SUPPORTED_TASK_TYPES = ["mcq", "yes_no_maybe", "ranking", "open_ended"]

# ============= DECISION AGGREGATION =============
# Default aggregation methods
DEFAULT_AGGREGATION_METHODS = ["borda_count", "majority_vote", "weighted_consensus"]
PRIMARY_DECISION_METHOD = "borda_count"  # Used as final answer

# Borda count scoring (for ranking aggregation)
BORDA_SCORES = {
    1: 3,  # 1st choice gets 3 points
    2: 2,  # 2nd choice gets 2 points
    3: 1,  # 3rd choice gets 1 point
    4: 0   # 4th choice gets 0 points (if 4 options)
}

# ============= OUTPUT CONFIGURATION =============
# Directory structure
OUTPUT_BASE_DIR = Path("multi-agent-gemma/results")
LOGS_DIR = Path("multi-agent-gemma/results")  # Logs saved per run

# Results organization
SAVE_INDIVIDUAL_QUESTIONS = True
SAVE_ROUND_BY_ROUND_DETAILS = True
SAVE_AGENT_CONVERSATIONS = True

# ============= LOGGING CONFIGURATION =============
# Console output
CONSOLE_LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
SHOW_PROGRESS_BAR = True  # Use tqdm for batch processing

# File logging
FILE_LOG_LEVEL = "DEBUG"
SEPARATE_ERROR_LOG = True  # Create separate errors.log
SEPARATE_RATE_LIMIT_LOG = True  # Create separate rate_limiting.log

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============= RATE LIMITING CONFIGURATION =============
# Import rate limits from existing slm_config.py
# These are used by MultiAgentRateLimiter
try:
    import sys
    _parent_dir = str(Path(__file__).parent.parent)
    if _parent_dir not in sys.path:
        sys.path.insert(0, _parent_dir)
    from slm_config import RATE_LIMITS, RETRY_CONFIG

    # Multi-agent specific settings
    ENABLE_RATE_LIMITING = True
    ESTIMATE_TOKENS_PER_ROUND = 1000  # Rough estimate for time calculations
    BUFFER_TIME_SECONDS = 5  # Extra buffer between requests

except ImportError:
    print("Warning: Could not import RATE_LIMITS from slm_config.py")
    # Fallback rate limits (match Google AI Studio free tier)
    RATE_LIMITS = {
        "gemma3_4b": {
            "rpm": 30,      # Requests per minute
            "tpm": 15000,   # Tokens per minute
            "rpd": 14400,   # Requests per day
        },
        "medgemma_4b": {
            "rpm": 30,
            "tpm": 15000,
            "rpd": 14400,
        }
    }

    RETRY_CONFIG = {
        "max_retries": 5,
        "initial_delay": 1.0,
        "max_delay": 60.0,
        "exponential_base": 2.0,
        "jitter": True,
        "rate_limit_delay": 65.0
    }

    ENABLE_RATE_LIMITING = True
    ESTIMATE_TOKENS_PER_ROUND = 1000
    BUFFER_TIME_SECONDS = 5

# ============= METRICS CONFIGURATION =============
# Which metrics to calculate
CALCULATE_ACCURACY = True
CALCULATE_CONVERGENCE = True
CALCULATE_DISAGREEMENT_MATRIX = True
CALCULATE_OPINION_CHANGES = True
CALCULATE_AGENT_PERFORMANCE = True
CALCULATE_METHOD_COMPARISON = True

# Convergence definitions
FULL_AGREEMENT_THRESHOLD = 1.0  # All agents must agree (100%)
PARTIAL_AGREEMENT_THRESHOLD = 0.5  # Majority agreement (50%+)

# ============= FUTURE FEATURES (PLACEHOLDERS) =============
# Team dynamics features (not yet implemented)
ENABLE_LEADERSHIP = False  # One agent coordinates
ENABLE_TRUST_NETWORK = False  # Agents weight others' opinions
ENABLE_MUTUAL_MONITORING = False  # Agents check each other
ENABLE_TEAM_ORIENTATION = False  # Shared goals emphasis

# Leadership settings (future)
LEADER_SELECTION_METHOD = "random"  # Options: "random", "best_performing", "designated"

# Trust network settings (future)
INITIAL_TRUST_SCORE = 0.8  # Starting trust between all agents
TRUST_UPDATE_RATE = 0.1  # How quickly trust changes based on agreement

# ============= VALIDATION =============
def validate_config():
    """Validate configuration settings."""
    errors = []

    # Validate agent counts
    if DEFAULT_N_AGENTS is not None:
        if not (MIN_AGENTS <= DEFAULT_N_AGENTS <= MAX_AGENTS):
            errors.append(f"DEFAULT_N_AGENTS ({DEFAULT_N_AGENTS}) must be between {MIN_AGENTS} and {MAX_AGENTS}")

    # Validate directories
    try:
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Could not create output directories: {e}")

    # Validate task types
    if PRIMARY_DECISION_METHOD not in DEFAULT_AGGREGATION_METHODS:
        errors.append(f"PRIMARY_DECISION_METHOD ({PRIMARY_DECISION_METHOD}) not in DEFAULT_AGGREGATION_METHODS")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

# ============= HELPER FUNCTIONS =============
def get_model_config(model_name: str = None):
    """Get model configuration, delegating to slm_config.py."""
    try:
        from slm_config import get_model_config as slm_get_model_config
        return slm_get_model_config(model_name or DEFAULT_MODEL)
    except ImportError:
        raise ImportError("Could not import get_model_config from slm_config.py. Ensure slm_config.py is in parent directory.")

def get_chat_instance_type():
    """Get default chat instance type from slm_config.py."""
    try:
        from slm_config import DEFAULT_CHAT_INSTANCE
        return DEFAULT_CHAT_INSTANCE
    except ImportError:
        return "google_ai_studio"  # Fallback

def get_output_dir_for_run(run_id: str) -> Path:
    """Get output directory path for a specific run."""
    return OUTPUT_BASE_DIR / run_id

# ============= EXPORTS =============
__all__ = [
    # Model config
    "DEFAULT_MODEL",
    "RECRUITMENT_MODEL",

    # Agent config
    "DEFAULT_N_AGENTS",
    "MAX_AGENTS",
    "MIN_AGENTS",
    "AGENT_TEMPERATURE",
    "AGENT_MAX_TOKENS",

    # Simulation config
    "NUM_ROUNDS",
    "ENABLE_PARALLEL_ROUND1",
    "SUPPORTED_TASK_TYPES",

    # Aggregation config
    "DEFAULT_AGGREGATION_METHODS",
    "PRIMARY_DECISION_METHOD",
    "BORDA_SCORES",

    # Output config
    "OUTPUT_BASE_DIR",
    "LOGS_DIR",

    # Rate limiting config
    "RATE_LIMITS",
    "RETRY_CONFIG",
    "ENABLE_RATE_LIMITING",

    # Metrics config
    "CALCULATE_ACCURACY",
    "CALCULATE_CONVERGENCE",

    # Helper functions
    "validate_config",
    "get_model_config",
    "get_chat_instance_type",
    "get_output_dir_for_run",
]

# Auto-validate on import
if __name__ != "__main__":
    try:
        validate_config()
    except Exception as e:
        print(f"Config validation warning: {e}")
