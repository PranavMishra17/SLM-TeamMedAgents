"""
Multi-Agent Rate Limit Manager

Coordinates rate limiting across multiple agents to avoid exceeding API limits.
Wraps the existing RateLimiter from utils/rate_limiter.py with multi-agent awareness.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time

# Add directories to path FIRST (must be before imports)
_parent_dir = str(Path(__file__).parent.parent.parent)  # For slm_runner, slm_config
_multi_agent_dir = str(Path(__file__).parent.parent)  # For components, utils, config

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _multi_agent_dir not in sys.path:
    sys.path.insert(0, _multi_agent_dir)

from utils.rate_limiter import RateLimiter
from slm_config import RATE_LIMITS, RETRY_CONFIG
import config as multi_agent_config


class MultiAgentRateLimiter:
    """
    Rate limiter aware of multi-agent API call patterns.

    Key considerations:
    - Each agent makes 3 API calls per question (one per round)
    - N agents × 3 rounds × M questions = total API calls
    - Must coordinate all calls through single RateLimiter instance
    - Provides time estimates for batch processing
    """

    def __init__(self, model_name: str, n_agents: int):
        """
        Initialize multi-agent rate limiter.

        Args:
            model_name: Model name (for rate limits lookup)
            n_agents: Number of agents in the system
        """
        self.model_name = model_name
        self.n_agents = n_agents
        self.calls_per_question = n_agents * 3  # 3 rounds per agent

        # Get rate limits for this model
        if model_name not in RATE_LIMITS:
            logging.warning(f"Rate limits not found for {model_name}, using default")
            self.rate_limits = RATE_LIMITS.get("gemma3_4b")
        else:
            self.rate_limits = RATE_LIMITS[model_name]

        # Create single shared RateLimiter instance (from existing infrastructure)
        self.rate_limiter = RateLimiter(
            model_name=model_name,
            rate_limits=self.rate_limits,
            retry_config=RETRY_CONFIG
        )

        logging.info(f"Initialized MultiAgentRateLimiter: model={model_name}, "
                    f"n_agents={n_agents}, calls_per_question={self.calls_per_question}")
        logging.info(f"Rate limits: RPM={self.rate_limits.get('rpm')}, "
                    f"TPM={self.rate_limits.get('tpm')}, "
                    f"RPD={self.rate_limits.get('rpd')}")

    def estimate_time(self, n_questions: int) -> Dict[str, float]:
        """
        Estimate time to process n questions given rate limits.

        Args:
            n_questions: Number of questions to process

        Returns:
            {
                "total_calls": int,
                "estimated_minutes": float,
                "estimated_hours": float,
                "rpm_limit": int,
                "tpm_limit": int,
                "rpd_limit": int,
                "batch_size_per_minute": int,
                "questions_per_minute": float,
                "questions_per_hour": float
            }
        """
        total_calls = n_questions * self.calls_per_question

        rpm_limit = self.rate_limits.get('rpm', 30)
        tpm_limit = self.rate_limits.get('tpm', 15000)
        rpd_limit = self.rate_limits.get('rpd', 14400)

        # Calculate constraints
        # RPM constraint: How many questions per minute?
        questions_per_minute_rpm = rpm_limit / self.calls_per_question

        # TPM constraint (estimate ~1000 tokens per call)
        estimated_tokens_per_call = multi_agent_config.ESTIMATE_TOKENS_PER_ROUND
        tokens_per_question = self.calls_per_question * estimated_tokens_per_call
        questions_per_minute_tpm = tpm_limit / tokens_per_question

        # RPD constraint: How many questions per day?
        questions_per_day_rpd = rpd_limit / self.calls_per_question

        # Use most restrictive limit
        questions_per_minute = min(questions_per_minute_rpm, questions_per_minute_tpm)
        questions_per_hour = questions_per_minute * 60
        questions_per_day = min(questions_per_day_rpd, questions_per_hour * 24)

        # Calculate time needed
        if n_questions <= questions_per_day:
            # Can complete within a day
            estimated_minutes = n_questions / questions_per_minute
            estimated_hours = estimated_minutes / 60
        else:
            # Spans multiple days
            estimated_hours = n_questions / (questions_per_hour)
            estimated_minutes = estimated_hours * 60

        # Add buffer for retry delays and processing overhead
        buffer_factor = 1.2  # 20% buffer
        estimated_minutes *= buffer_factor
        estimated_hours *= buffer_factor

        return {
            "total_calls": total_calls,
            "estimated_minutes": estimated_minutes,
            "estimated_hours": estimated_hours,
            "estimated_days": estimated_hours / 24 if estimated_hours > 24 else 0,
            "rpm_limit": rpm_limit,
            "tpm_limit": tpm_limit,
            "rpd_limit": rpd_limit,
            "calls_per_question": self.calls_per_question,
            "questions_per_minute": questions_per_minute,
            "questions_per_hour": questions_per_hour,
            "questions_per_day": questions_per_day,
            "limiting_factor": self._get_limiting_factor(
                questions_per_minute_rpm,
                questions_per_minute_tpm
            )
        }

    def _get_limiting_factor(self, rpm_rate: float, tpm_rate: float) -> str:
        """Determine which rate limit is most restrictive."""
        if rpm_rate < tpm_rate:
            return "RPM (requests per minute)"
        else:
            return "TPM (tokens per minute)"

    def wait_if_needed(self, estimated_tokens: int = 0) -> float:
        """
        Check rate limits and wait if necessary before making next API call.

        This delegates to the existing RateLimiter's wait_if_needed method.

        Args:
            estimated_tokens: Estimated tokens for upcoming request

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        if not multi_agent_config.ENABLE_RATE_LIMITING:
            return 0.0

        return self.rate_limiter.wait_if_needed(estimated_tokens)

    def exponential_backoff_retry(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry.

        Delegates to existing RateLimiter's retry logic.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result from func()
        """
        return self.rate_limiter.exponential_backoff_retry(func, *args, **kwargs)

    def get_rate_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status for logging/monitoring.

        Returns:
            {
                "requests_this_minute": int,
                "requests_today": int,
                "tokens_this_minute": int,
                "rpm_utilization": float,  # 0.0 to 1.0
                "tpm_utilization": float,
                "rpd_utilization": float,
                "can_make_request": bool,
                "time_until_reset": float  # seconds
            }
        """
        # Get stats from underlying rate limiter
        stats = self.rate_limiter.get_stats()

        # Check if we can make a request
        can_proceed, wait_time = self.rate_limiter.can_make_request(
            estimated_tokens=multi_agent_config.ESTIMATE_TOKENS_PER_ROUND
        )

        return {
            "model": self.model_name,
            "n_agents": self.n_agents,
            "requests_this_minute": stats.get("current_rpm", 0),
            "requests_today": stats.get("current_rpd", 0),
            "tokens_this_minute": stats.get("current_tpm", 0),
            "rpm_utilization": stats.get("rpm_utilization", 0.0),
            "tpm_utilization": stats.get("tpm_utilization", 0.0),
            "rpd_utilization": stats.get("rpd_utilization", 0.0),
            "can_make_request": can_proceed,
            "time_until_reset": wait_time if not can_proceed else 0.0,
            "limits": stats.get("limits", {})
        }

    def log_rate_status(self):
        """Log current rate limit status (for monitoring)."""
        status = self.get_rate_status()

        logging.info(f"Rate Status: "
                    f"RPM={status['requests_this_minute']}/{status['limits'].get('rpm', 'N/A')} "
                    f"({status['rpm_utilization']:.1%}), "
                    f"TPM={status['tokens_this_minute']}/{status['limits'].get('tpm', 'N/A')} "
                    f"({status['tpm_utilization']:.1%}), "
                    f"RPD={status['requests_today']}/{status['limits'].get('rpd', 'N/A')} "
                    f"({status['rpd_utilization']:.1%})")

    def print_time_estimate(self, n_questions: int):
        """
        Print formatted time estimate for user.

        Args:
            n_questions: Number of questions to process
        """
        estimate = self.estimate_time(n_questions)

        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING TIME ESTIMATE")
        print(f"{'='*70}")
        print(f"Questions to process: {n_questions}")
        print(f"Agents per question: {self.n_agents}")
        print(f"Total API calls: {estimate['total_calls']} "
              f"({estimate['calls_per_question']} per question)")
        print(f"\nRate Limits:")
        print(f"  - Requests per minute: {estimate['rpm_limit']}")
        print(f"  - Tokens per minute: {estimate['tpm_limit']}")
        print(f"  - Requests per day: {estimate['rpd_limit']}")
        print(f"  - Limiting factor: {estimate['limiting_factor']}")
        print(f"\nProcessing Rate:")
        print(f"  - Questions per minute: {estimate['questions_per_minute']:.2f}")
        print(f"  - Questions per hour: {estimate['questions_per_hour']:.1f}")
        print(f"  - Questions per day: {estimate['questions_per_day']:.0f}")
        print(f"\nEstimated Time:")

        if estimate['estimated_hours'] < 1:
            print(f"  [TIME] {estimate['estimated_minutes']:.1f} minutes")
        elif estimate['estimated_hours'] < 24:
            print(f"  [TIME] {estimate['estimated_hours']:.1f} hours "
                  f"({estimate['estimated_minutes']:.0f} minutes)")
        else:
            print(f"  [TIME] {estimate['estimated_days']:.1f} days "
                  f"({estimate['estimated_hours']:.1f} hours)")

        print(f"\n(Estimate includes 20% buffer for retries and overhead)")
        print(f"{'='*70}\n")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_rate_limiter(model_name: str, n_agents: int) -> MultiAgentRateLimiter:
    """
    Create a MultiAgentRateLimiter instance.

    Args:
        model_name: Model name
        n_agents: Number of agents

    Returns:
        Initialized MultiAgentRateLimiter
    """
    return MultiAgentRateLimiter(model_name, n_agents)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MultiAgentRateLimiter",
    "create_rate_limiter",
]
