"""
Multi-Agent System - Main Orchestrator

Coordinates the entire multi-agent simulation process:
1. Agent recruitment
2. Round 1: Independent analysis
3. Round 2: Collaborative discussion
4. Round 3: Final ranking
5. Decision aggregation

This is the primary entry point for running multi-agent simulations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

# Add directories to path FIRST (must be before imports)
_parent_dir = str(Path(__file__).parent.parent.parent)  # For slm_runner, slm_config
_multi_agent_dir = str(Path(__file__).parent.parent)  # For components, utils, config

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _multi_agent_dir not in sys.path:
    sys.path.insert(0, _multi_agent_dir)

from components.agent_recruiter import AgentRecruiter
from components.simulation_rounds import RoundOrchestrator
from components.decision_aggregator import DecisionAggregator, get_final_answer
from components.gemma_agent import GemmaAgent
import config as multi_agent_config


class MultiAgentSystem:
    """
    Main multi-agent simulator orchestrating the entire collaborative process.

    Manages:
    - Agent recruitment (dynamic or fixed)
    - Three-round simulation execution
    - Decision aggregation
    - Results compilation
    - Future: Leadership, trust networks, mutual monitoring
    """

    def __init__(self,
                 model_name: str = None,
                 n_agents: int = None,
                 chat_instance_type: str = "google_ai_studio",
                 enable_dynamic_recruitment: bool = True,
                 enable_modular_components: bool = False):
        """
        Initialize multi-agent system.

        Args:
            model_name: Which Gemma model to use (gemma3_4b or medgemma_4b)
            n_agents: Fixed agent count (2-4) or None for dynamic recruitment
            chat_instance_type: Chat instance type (google_ai_studio or huggingface)
            enable_dynamic_recruitment: If True, recruiter determines agent count
            enable_modular_components: Enable future teamwork features (placeholder)
        """
        self.model_name = model_name or multi_agent_config.DEFAULT_MODEL
        self.n_agents = n_agents
        self.chat_instance_type = chat_instance_type
        self.enable_dynamic_recruitment = enable_dynamic_recruitment

        # Initialize components
        self.recruiter = AgentRecruiter(
            model_name=self.model_name,
            chat_instance_type=chat_instance_type,
            enable_dynamic_recruitment=enable_dynamic_recruitment
        )

        self.aggregator = DecisionAggregator()

        # Placeholder flags for future teamwork components
        self.enable_modular_components = enable_modular_components
        self.leadership_enabled = False
        self.trust_network_enabled = False
        self.mutual_monitoring_enabled = False
        self.team_orientation_enabled = False

        logging.info(f"Initialized MultiAgentSystem: model={self.model_name}, "
                    f"n_agents={'dynamic' if n_agents is None else n_agents}")

    def run_simulation(self,
                      question: str,
                      options: List[str],
                      task_type: str = "mcq",
                      image_path: Optional[str] = None,
                      ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete multi-agent simulation on a single question.

        Args:
            question: The medical question text
            options: List of answer options (for MCQ)
            task_type: Type of task (mcq, yes_no_maybe, ranking, open_ended)
            image_path: Optional path to medical image
            ground_truth: Optional correct answer for evaluation

        Returns:
            Comprehensive simulation results dict:
            {
                "question": str,
                "options": List[str],
                "task_type": str,
                "recruited_agents": List[Dict],
                "round1_results": Dict,
                "round2_results": Dict,
                "round3_results": Dict,
                "final_decision": {
                    "borda_count": {...},
                    "majority_vote": {...},
                    "weighted_consensus": {...},
                    "primary_answer": str,
                    "all_rankings": Dict
                },
                "agreement_metrics": Dict,
                "is_correct": bool (if ground_truth provided),
                "metadata": {
                    "timestamp": str,
                    "total_time": float,
                    "n_agents": int,
                    "model_name": str,
                    ...
                }
            }
        """
        simulation_start = time.time()
        timestamp = datetime.now().isoformat()

        logging.info(f"\n{'='*80}")
        logging.info(f"MULTI-AGENT SIMULATION START")
        logging.info(f"Question: {question[:100]}...")
        logging.info(f"Task Type: {task_type}")
        logging.info(f"{'='*80}\n")

        # ==================================================================
        # STEP 1: RECRUIT AGENTS
        # ==================================================================
        logging.info("STEP 1: RECRUITING AGENTS")
        recruit_start = time.time()

        agents = self.recruiter.recruit_agents(
            question=question,
            options=options,
            n_agents=self.n_agents,
            agent_model_name=self.model_name
        )

        recruit_time = time.time() - recruit_start

        agent_info = [
            {
                "agent_id": agent.agent_id,
                "role": agent.role,
                "expertise": agent.expertise
            }
            for agent in agents
        ]

        logging.info(f"Recruited {len(agents)} agents in {recruit_time:.2f}s:")
        for info in agent_info:
            logging.info(f"  - {info['agent_id']}: {info['role']} - {info['expertise']}")

        # ==================================================================
        # STEP 2-4: RUN THREE ROUNDS
        # ==================================================================
        logging.info("\nSTEP 2-4: EXECUTING THREE ROUNDS")
        rounds_start = time.time()

        orchestrator = RoundOrchestrator(agents)
        rounds_results = orchestrator.execute_all_rounds(
            question=question,
            options=options,
            task_type=task_type,
            image_path=image_path
        )

        rounds_time = time.time() - rounds_start

        # ==================================================================
        # STEP 5: AGGREGATE DECISIONS
        # ==================================================================
        logging.info("STEP 5: AGGREGATING DECISIONS")
        aggregate_start = time.time()

        final_decision = self.aggregator.aggregate_decisions(
            agent_decisions=rounds_results["round3_results"],
            methods=multi_agent_config.DEFAULT_AGGREGATION_METHODS
        )

        # Get primary answer using configured method
        primary_answer = get_final_answer(
            final_decision,
            method=multi_agent_config.PRIMARY_DECISION_METHOD
        )

        final_decision["primary_answer"] = primary_answer

        aggregate_time = time.time() - aggregate_start

        logging.info(f"Primary decision ({multi_agent_config.PRIMARY_DECISION_METHOD}): "
                    f"{primary_answer}")

        # ==================================================================
        # STEP 6: CALCULATE AGREEMENT METRICS
        # ==================================================================
        agreement_metrics = None
        if "all_rankings" in final_decision:
            # Extract rankings for agreement calculation
            rankings = {}
            for agent_id, decision in final_decision["all_rankings"].items():
                if "ranking" in decision and decision["ranking"]:
                    rankings[agent_id] = decision["ranking"]

            if rankings:
                agreement_metrics = self.aggregator.calculate_agreement_metrics(rankings)

        # ==================================================================
        # STEP 7: EVALUATE CORRECTNESS (if ground truth provided)
        # ==================================================================
        is_correct = None
        if ground_truth is not None:
            is_correct = (primary_answer == ground_truth)
            logging.info(f"Correctness: {'[CORRECT]' if is_correct else '[INCORRECT]'} "
                        f"(Answer: {primary_answer}, Ground Truth: {ground_truth})")

        # ==================================================================
        # COMPILE RESULTS
        # ==================================================================
        simulation_time = time.time() - simulation_start

        results = {
            "question": question,
            "options": options,
            "task_type": task_type,
            "image_path": image_path,

            # Agent information
            "recruited_agents": agent_info,

            # Round results
            "round1_results": rounds_results["round1_results"],
            "round2_results": rounds_results["round2_results"],
            "round3_results": rounds_results["round3_results"],

            # Decision aggregation
            "final_decision": final_decision,

            # Metrics
            "agreement_metrics": agreement_metrics,

            # Evaluation
            "ground_truth": ground_truth,
            "is_correct": is_correct,

            # Metadata
            "metadata": {
                "timestamp": timestamp,
                "total_time": simulation_time,
                "recruit_time": recruit_time,
                "rounds_time": rounds_time,
                "aggregate_time": aggregate_time,
                "n_agents": len(agents),
                "model_name": self.model_name,
                "chat_instance_type": self.chat_instance_type,
                "dynamic_recruitment": (self.n_agents is None),
                "teamwork_features": {
                    "leadership": self.leadership_enabled,
                    "trust_network": self.trust_network_enabled,
                    "mutual_monitoring": self.mutual_monitoring_enabled,
                    "team_orientation": self.team_orientation_enabled
                }
            }
        }

        logging.info(f"\n{'='*80}")
        logging.info(f"SIMULATION COMPLETE: {simulation_time:.2f}s")
        logging.info(f"  - Recruitment: {recruit_time:.2f}s")
        logging.info(f"  - Rounds: {rounds_time:.2f}s")
        logging.info(f"  - Aggregation: {aggregate_time:.2f}s")
        if is_correct is not None:
            logging.info(f"  - Result: {'CORRECT' if is_correct else 'INCORRECT'}")
        logging.info(f"{'='*80}\n")

        return results

    # ======================================================================
    # FUTURE: TEAMWORK COMPONENT METHODS (Placeholders)
    # ======================================================================

    def enable_leadership(self, leader_agent_id: str = None,
                         selection_method: str = "random"):
        """
        Enable leadership component (FUTURE FEATURE).

        One agent coordinates the discussion and makes final decisions.

        Args:
            leader_agent_id: Specific agent to designate as leader (or None for auto)
            selection_method: How to select leader (random, best_performing, designated)
        """
        self.leadership_enabled = True
        logging.info(f"Leadership component enabled (placeholder): "
                    f"leader={leader_agent_id}, method={selection_method}")
        # TODO: Implement leadership logic
        # - Leader gets final say in Round 3
        # - Leader synthesizes team discussion
        # - Leader breaks ties

    def enable_trust_network(self, initial_trust: float = 0.8):
        """
        Enable trust-based weighting (FUTURE FEATURE).

        Agents develop trust scores based on agreement/correctness history.
        Higher trust agents have more weight in decisions.

        Args:
            initial_trust: Starting trust score for all agent pairs
        """
        self.trust_network_enabled = True
        logging.info(f"Trust network component enabled (placeholder): "
                    f"initial_trust={initial_trust}")
        # TODO: Implement trust network
        # - Track pairwise trust scores
        # - Update based on agreement and correctness
        # - Use in weighted_consensus aggregation

    def enable_mutual_monitoring(self):
        """
        Enable mutual monitoring (FUTURE FEATURE).

        Agents check each other's reasoning for errors or oversights.
        """
        self.mutual_monitoring_enabled = True
        logging.info("Mutual monitoring component enabled (placeholder)")
        # TODO: Implement mutual monitoring
        # - Add Round 2.5: Error checking
        # - Agents review others' Round 1 analyses for errors
        # - Flag potential mistakes or oversights

    def enable_team_orientation(self, shared_goal_weight: float = 0.3):
        """
        Enable team orientation (FUTURE FEATURE).

        Emphasize shared goals and collaborative consensus-building.

        Args:
            shared_goal_weight: How much to weight team consensus vs individual opinion
        """
        self.team_orientation_enabled = True
        logging.info(f"Team orientation component enabled (placeholder): "
                    f"shared_goal_weight={shared_goal_weight}")
        # TODO: Implement team orientation
        # - Modify Round 2 prompts to emphasize consensus
        # - Add team goal prompts
        # - Track team cohesion metrics

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system configuration and status."""
        return {
            "model_name": self.model_name,
            "n_agents": "dynamic" if self.n_agents is None else self.n_agents,
            "chat_instance_type": self.chat_instance_type,
            "dynamic_recruitment_enabled": self.enable_dynamic_recruitment,
            "teamwork_features": {
                "leadership": self.leadership_enabled,
                "trust_network": self.trust_network_enabled,
                "mutual_monitoring": self.mutual_monitoring_enabled,
                "team_orientation": self.team_orientation_enabled
            }
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_simulate(question: str, options: List[str], **kwargs) -> Dict[str, Any]:
    """
    Quick simulation with default settings.

    Args:
        question: Medical question
        options: Answer options
        **kwargs: Additional arguments for MultiAgentSystem or run_simulation

    Returns:
        Simulation results
    """
    # Separate system args from simulation args
    system_args = {
        "model_name": kwargs.pop("model_name", None),
        "n_agents": kwargs.pop("n_agents", None),
        "chat_instance_type": kwargs.pop("chat_instance_type", "google_ai_studio"),
        "enable_dynamic_recruitment": kwargs.pop("enable_dynamic_recruitment", True)
    }

    system = MultiAgentSystem(**system_args)
    return system.run_simulation(question, options, **kwargs)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MultiAgentSystem",
    "quick_simulate",
]
