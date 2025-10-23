"""
Simulation Rounds - Multi-Agent Round Execution

Implements the three-round collaborative reasoning process:
- Round 1: Independent analysis (no context sharing)
- Round 2: Collaborative discussion (with Round 1 context)
- Round 3: Final ranking decisions (with full context)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import time

# Add parent directory to path FIRST (must be before imports)
_multi_agent_dir = str(Path(__file__).parent.parent)
if _multi_agent_dir not in sys.path:
    sys.path.insert(0, _multi_agent_dir)

from components.gemma_agent import GemmaAgent
from utils.prompts import (
    get_round1_prompt,
    get_round2_prompt,
    get_round3_prompt,
    format_options
)
import config as multi_agent_config


class SimulationRound:
    """Base class for simulation rounds."""

    def __init__(self, agents: List[GemmaAgent], round_number: int):
        """
        Initialize round executor.

        Args:
            agents: List of GemmaAgent instances
            round_number: Round number (1, 2, or 3)
        """
        self.agents = agents
        self.round_number = round_number
        self.results = {}

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the round. Implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")


class Round1Independent(SimulationRound):
    """
    Round 1: Independent Analysis

    Each agent analyzes the question independently without seeing
    other agents' responses. No context sharing.
    """

    def __init__(self, agents: List[GemmaAgent]):
        super().__init__(agents, round_number=1)

    def execute(self,
               question: str,
               options: List[str] = None,
               task_type: str = "mcq",
               image_path: Optional[str] = None) -> Dict[str, str]:
        """
        Execute Round 1 independent analysis.

        Args:
            question: The medical question
            options: Answer options (for MCQ tasks)
            task_type: Type of task (mcq, yes_no_maybe, ranking, open_ended)
            image_path: Optional path to medical image

        Returns:
            Dictionary mapping agent_id → response text
            {
                "agent_1": "analysis text...",
                "agent_2": "analysis text...",
                ...
            }
        """
        logging.info(f"=== ROUND 1: INDEPENDENT ANALYSIS ({len(self.agents)} agents) ===")

        results = {}
        start_time = time.time()

        # Validate image availability
        has_valid_image = image_path is not None and str(image_path).strip() != ""
        question_mentions_image = any(term in question.lower() for term in
                                     ['image', 'shown', 'figure', 'photograph', 'picture'])

        if question_mentions_image and not has_valid_image:
            logging.warning("Question mentions image but no valid image_path provided - adding constraint to prompt")

        # Each agent analyzes independently (sequential execution)
        for agent in self.agents:
            agent_start = time.time()

            # Create Round 1 prompt for this agent
            prompt = get_round1_prompt(
                task_type=task_type,
                role=agent.role,
                expertise=agent.expertise,
                question=question,
                options=options,
                has_image=has_valid_image,
                image_mentioned_but_missing=(question_mentions_image and not has_valid_image)
            )

            # Get agent's analysis
            try:
                response = agent.analyze_question(
                    prompt=prompt,
                    round_number=1,
                    image_path=image_path if has_valid_image else None
                )
                results[agent.agent_id] = response

                agent_time = time.time() - agent_start
                logging.info(f"  {agent.agent_id} ({agent.role}): "
                           f"Response received ({agent_time:.2f}s)")
                logging.debug(f"  Response preview: {response[:150]}...")

            except Exception as e:
                logging.error(f"Error in Round 1 for {agent.agent_id}: {e}")
                results[agent.agent_id] = f"ERROR: {str(e)}"

        total_time = time.time() - start_time
        logging.info(f"Round 1 completed in {total_time:.2f}s "
                    f"(avg: {total_time/len(self.agents):.2f}s per agent)")

        self.results = results
        return results


class Round2Collaborative(SimulationRound):
    """
    Round 2: Collaborative Discussion

    Agents see each other's Round 1 analyses and engage in collaborative
    discussion. They can debate, clarify, and build on each other's insights.
    """

    def __init__(self, agents: List[GemmaAgent]):
        super().__init__(agents, round_number=2)

    def execute(self,
               question: str,
               options: List[str],
               round1_results: Dict[str, str],
               task_type: str = "mcq") -> Dict[str, str]:
        """
        Execute Round 2 collaborative discussion.

        Args:
            question: The medical question
            options: Answer options
            round1_results: Dictionary of Round 1 responses {agent_id: response}
            task_type: Type of task

        Returns:
            Dictionary mapping agent_id → Round 2 discussion text
        """
        logging.info(f"=== ROUND 2: COLLABORATIVE DISCUSSION ===")

        results = {}
        start_time = time.time()

        # Each agent discusses after seeing others' Round 1 analyses
        for agent in self.agents:
            agent_start = time.time()

            # Get this agent's Round 1 analysis
            agent_round1 = round1_results.get(agent.agent_id, "No Round 1 response")

            # Get other agents' Round 1 analyses
            other_analyses = {
                other_id: analysis
                for other_id, analysis in round1_results.items()
                if other_id != agent.agent_id
            }

            # Create Round 2 prompt with full context
            prompt = get_round2_prompt(
                role=agent.role,
                expertise=agent.expertise,
                question=question,
                options=options,
                your_round1=agent_round1,
                other_analyses=other_analyses
            )

            # Get agent's collaborative discussion
            try:
                response = agent.analyze_question(
                    prompt=prompt,
                    round_number=2
                )
                results[agent.agent_id] = response

                agent_time = time.time() - agent_start
                logging.info(f"  {agent.agent_id} ({agent.role}): "
                           f"Discussion received ({agent_time:.2f}s)")
                logging.debug(f"  Discussion preview: {response[:150]}...")

            except Exception as e:
                logging.error(f"Error in Round 2 for {agent.agent_id}: {e}")
                results[agent.agent_id] = f"ERROR: {str(e)}"

        total_time = time.time() - start_time
        logging.info(f"Round 2 completed in {total_time:.2f}s "
                    f"(avg: {total_time/len(self.agents):.2f}s per agent)")

        self.results = results
        return results


class Round3Ranking(SimulationRound):
    """
    Round 3: Final Ranking Decisions

    After independent analysis (Round 1) and collaborative discussion (Round 2),
    each agent provides their final ranking of all answer options.
    """

    def __init__(self, agents: List[GemmaAgent]):
        super().__init__(agents, round_number=3)

    def execute(self,
               question: str,
               options: List[str],
               round1_results: Dict[str, str],
               round2_results: Dict[str, str],
               task_type: str = "mcq",
               image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute Round 3 final ranking decisions.

        Args:
            question: The medical question
            options: Answer options
            round1_results: Dictionary of Round 1 responses
            round2_results: Dictionary of Round 2 discussions
            task_type: Type of task
            image_path: Optional path to medical image

        Returns:
            Dictionary mapping agent_id → extracted answer structure
            {
                "agent_1": {
                    "ranking": ["B", "A", "C", "D"],
                    "confidence": "High",
                    "raw": "full response text..."
                },
                ...
            }
        """
        logging.info(f"=== ROUND 3: FINAL RANKING DECISIONS ===")

        results = {}
        start_time = time.time()

        # Validate image availability
        has_valid_image = image_path is not None and str(image_path).strip() != ""
        question_mentions_image = any(term in question.lower() for term in
                                     ['image', 'shown', 'figure', 'photograph', 'picture'])

        # Each agent makes final decision with full context
        for agent in self.agents:
            agent_start = time.time()

            # Get this agent's previous rounds
            agent_round1 = round1_results.get(agent.agent_id, "No Round 1 response")

            # Collect Round 2 discussion summary (all agents' R2 including this one)
            round2_summary = self._create_round2_summary(round2_results)

            # Create Round 3 prompt with full context
            prompt = get_round3_prompt(
                task_type=task_type,
                role=agent.role,
                expertise=agent.expertise,
                question=question,
                options=options,
                your_round1=agent_round1,
                round2_discussion=round2_summary,
                has_image=has_valid_image,
                image_mentioned_but_missing=(question_mentions_image and not has_valid_image)
            )

            # Get agent's final decision
            try:
                response = agent.analyze_question(
                    prompt=prompt,
                    round_number=3
                )

                # Extract structured answer from response
                extracted = agent.extract_answer(response, task_type)
                results[agent.agent_id] = extracted

                agent_time = time.time() - agent_start

                # Log extracted answer
                if task_type == "mcq":
                    answer_preview = f"Answer: {extracted.get('answer', 'N/A')}"
                elif task_type == "ranking":
                    ranking = extracted.get('ranking', [])
                    answer_preview = f"Ranking: {ranking[:3]}..." if ranking else "Ranking: N/A"
                else:
                    answer_preview = f"Answer: {extracted.get('answer', 'N/A')}"

                logging.info(f"  {agent.agent_id} ({agent.role}): "
                           f"{answer_preview} "
                           f"(Confidence: {extracted.get('confidence', 'N/A')}, "
                           f"{agent_time:.2f}s)")

            except Exception as e:
                logging.error(f"Error in Round 3 for {agent.agent_id}: {e}")
                results[agent.agent_id] = {
                    "error": str(e),
                    "raw": f"ERROR: {str(e)}"
                }

        total_time = time.time() - start_time
        logging.info(f"Round 3 completed in {total_time:.2f}s "
                    f"(avg: {total_time/len(self.agents):.2f}s per agent)")

        self.results = results
        return results

    def _create_round2_summary(self, round2_results: Dict[str, str]) -> str:
        """
        Create summary of Round 2 discussion from all agents.

        Args:
            round2_results: Dict of Round 2 responses {agent_id: discussion}

        Returns:
            Formatted summary string
        """
        summary_parts = []
        for agent_id, discussion in round2_results.items():
            summary_parts.append(f"=== {agent_id} ===\n{discussion}\n")

        return "\n".join(summary_parts)


# ============================================================================
# ROUND ORCHESTRATOR
# ============================================================================

class RoundOrchestrator:
    """
    Orchestrates the execution of all three rounds in sequence.

    Convenience class to manage the complete three-round process.
    """

    def __init__(self, agents: List[GemmaAgent]):
        """
        Initialize round orchestrator.

        Args:
            agents: List of GemmaAgent instances
        """
        self.agents = agents
        self.round1_executor = Round1Independent(agents)
        self.round2_executor = Round2Collaborative(agents)
        self.round3_executor = Round3Ranking(agents)

        self.round1_results = None
        self.round2_results = None
        self.round3_results = None

    def execute_all_rounds(self,
                          question: str,
                          options: List[str],
                          task_type: str = "mcq",
                          image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute all three rounds in sequence.

        Args:
            question: The medical question
            options: Answer options
            task_type: Type of task
            image_path: Optional path to medical image

        Returns:
            Complete results from all rounds
            {
                "round1_results": {...},
                "round2_results": {...},
                "round3_results": {...},
                "metadata": {...}
            }
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"STARTING MULTI-AGENT SIMULATION: {len(self.agents)} agents")
        logging.info(f"{'='*80}\n")

        simulation_start = time.time()

        # Round 1: Independent Analysis
        self.round1_results = self.round1_executor.execute(
            question=question,
            options=options,
            task_type=task_type,
            image_path=image_path
        )

        # Round 2: Collaborative Discussion
        self.round2_results = self.round2_executor.execute(
            question=question,
            options=options,
            round1_results=self.round1_results,
            task_type=task_type
        )

        # Round 3: Final Ranking
        self.round3_results = self.round3_executor.execute(
            question=question,
            options=options,
            round1_results=self.round1_results,
            round2_results=self.round2_results,
            task_type=task_type,
            image_path=image_path
        )

        simulation_time = time.time() - simulation_start

        logging.info(f"\n{'='*80}")
        logging.info(f"SIMULATION COMPLETE: {simulation_time:.2f}s total")
        logging.info(f"{'='*80}\n")

        return {
            "round1_results": self.round1_results,
            "round2_results": self.round2_results,
            "round3_results": self.round3_results,
            "metadata": {
                "n_agents": len(self.agents),
                "agent_roles": [a.role for a in self.agents],
                "total_time": simulation_time,
                "task_type": task_type
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_three_rounds(agents: List[GemmaAgent],
                    question: str,
                    options: List[str],
                    **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run all three rounds.

    Args:
        agents: List of GemmaAgent instances
        question: Medical question
        options: Answer options
        **kwargs: Additional arguments (task_type, image_path)

    Returns:
        Complete round results
    """
    orchestrator = RoundOrchestrator(agents)
    return orchestrator.execute_all_rounds(question, options, **kwargs)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "SimulationRound",
    "Round1Independent",
    "Round2Collaborative",
    "Round3Ranking",
    "RoundOrchestrator",
    "run_three_rounds",
]
