"""
Simulation Logger - Comprehensive Logging System

Provides console + file logging with progress tracking for multi-agent simulations.
Creates separate logs for main events, rate limiting, and errors.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


class SimulationLogger:
    """
    Comprehensive logging system for multi-agent simulations.

    Features:
    - Console output with INFO level
    - File logging with DEBUG level
    - Separate error log
    - Separate rate limiting log
    - Rich formatting for readability
    """

    def __init__(self, output_dir: str, run_id: str = None):
        """
        Initialize simulation logger.

        Args:
            output_dir: Base directory for this run
            run_id: Unique identifier for this run (auto-generated if None)
        """
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir) / self.run_id
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create loggers
        self.main_logger = self._create_logger("simulation", "simulation.log", logging.DEBUG)
        self.rate_logger = self._create_logger("rate_limiting", "rate_limiting.log", logging.INFO)
        self.error_logger = self._create_logger("errors", "errors.log", logging.ERROR)

    def _create_logger(self, name: str, filename: str, level=logging.INFO) -> logging.Logger:
        """Create a logger with file and console handlers."""
        logger = logging.getLogger(f"{self.run_id}_{name}")
        logger.setLevel(level)
        logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(self.log_dir / filename, encoding='utf-8')
        fh.setLevel(level)

        # Console handler (only for main logger)
        if name == "simulation":
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def log_run_start(self, config: Dict):
        """Log simulation run start with configuration."""
        self.main_logger.info("="*80)
        self.main_logger.info(f"SIMULATION RUN START: {self.run_id}")
        self.main_logger.info("="*80)
        self.main_logger.info(f"Configuration: {config}")

    def log_question_start(self, question_id: str, question_text: str):
        """Log start of question processing."""
        self.main_logger.info(f"\n{'='*60}")
        self.main_logger.info(f"QUESTION {question_id}: {question_text[:100]}...")
        self.main_logger.info(f"{'='*60}")

    def log_agent_recruitment(self, n_agents: int, agent_roles: List[str]):
        """Log agent recruitment results."""
        self.main_logger.info(f"Recruited {n_agents} agents:")
        for i, role in enumerate(agent_roles, 1):
            self.main_logger.info(f"  Agent {i}: {role}")

    def log_round_start(self, round_number: int):
        """Log round start."""
        round_names = {1: "Independent Analysis", 2: "Collaborative Discussion", 3: "Final Ranking"}
        self.main_logger.info(f"\n--- ROUND {round_number}: {round_names.get(round_number, 'Unknown')} ---")

    def log_agent_response(self, agent_id: str, round_number: int, response_preview: str):
        """Log agent response preview."""
        self.main_logger.info(f"  {agent_id} (R{round_number}): {response_preview[:150]}...")

    def log_decision_aggregation(self, method: str, result: Any):
        """Log decision aggregation results."""
        self.main_logger.info(f"Decision Method: {method} → Result: {result}")

    def log_question_complete(self, question_id: str, final_answer: str,
                             is_correct: bool = None, time_taken: float = None):
        """Log question completion."""
        status = "[CORRECT]" if is_correct else "[INCORRECT]" if is_correct is not None else "[COMPLETED]"
        self.main_logger.info(f"\n{status} - Question {question_id}: {final_answer}")
        if time_taken:
            self.main_logger.info(f"Time taken: {time_taken:.2f}s")

    def log_rate_limit_event(self, event_type: str, details: Dict):
        """Log rate limiting events."""
        self.rate_logger.info(f"{event_type}: {details}")

    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """Log errors with context."""
        self.error_logger.error(f"{error_type}: {error_message}")
        if context:
            self.error_logger.error(f"Context: {context}")

    def log_run_complete(self, summary: Dict):
        """Log simulation run completion with summary."""
        self.main_logger.info("\n" + "="*80)
        self.main_logger.info("SIMULATION RUN COMPLETE")
        self.main_logger.info("="*80)

        # Extract timing info
        timing = summary.get('timing', {})
        total_time = timing.get('total_time', 0)
        questions_processed = timing.get('questions_processed', 0)

        # Extract accuracy info
        accuracy_data = summary.get('accuracy', {})
        if isinstance(accuracy_data, dict):
            overall_accuracy = accuracy_data.get('overall', {}).get('accuracy', 0)
        else:
            overall_accuracy = 0

        self.main_logger.info(f"Total questions: {questions_processed}")
        self.main_logger.info(f"Overall accuracy: {overall_accuracy:.2%}")
        self.main_logger.info(f"Total time: {total_time:.2f}s")
        if questions_processed > 0:
            self.main_logger.info(f"Avg time per question: {total_time/questions_processed:.2f}s")
        self.main_logger.info(f"Results saved to: {self.output_dir}")


__all__ = ["SimulationLogger"]
