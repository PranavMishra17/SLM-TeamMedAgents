"""
Decision Aggregator - Ranking & Consensus Methods

Aggregates individual agent rankings/decisions into team decisions using
various methods: Borda count, majority voting, weighted consensus.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from collections import Counter, defaultdict

# Add parent directory to path
_multi_agent_dir = str(Path(__file__).parent.parent)
if _multi_agent_dir not in sys.path:
    sys.path.insert(0, _multi_agent_dir)

import config as multi_agent_config


class DecisionAggregator:
    """
    Aggregates agent decisions using various voting/ranking methods.

    Supports:
    - Borda count: Point-based ranking aggregation
    - Majority voting: Most popular first choice
    - Weighted consensus: Weighted Borda (for future trust networks)
    """

    def __init__(self):
        """Initialize decision aggregator."""
        self.borda_scores = multi_agent_config.BORDA_SCORES

    def aggregate_decisions(self,
                          agent_decisions: Dict[str, Any],
                          methods: List[str] = None) -> Dict[str, Any]:
        """
        Aggregate agent decisions using multiple methods.

        Args:
            agent_decisions: Dict of agent decisions from Round 3
                {
                    "agent_1": {"ranking": ["B", "A", "C", "D"], "confidence": "High"},
                    "agent_2": {"ranking": ["B", "C", "A", "D"], "confidence": "Medium"},
                    ...
                }
            methods: List of aggregation methods to use (default: all)

        Returns:
            Aggregated results for each method
            {
                "borda_count": {...},
                "majority_vote": {...},
                "weighted_consensus": {...},
                "all_rankings": agent_decisions
            }
        """
        if methods is None:
            methods = multi_agent_config.DEFAULT_AGGREGATION_METHODS

        results = {}

        # Extract rankings from decisions
        rankings = {}
        for agent_id, decision in agent_decisions.items():
            if "error" not in decision and "ranking" in decision:
                rankings[agent_id] = decision["ranking"]
            elif "error" not in decision and "answer" in decision:
                # For MCQ single answer, treat as 1-item ranking
                rankings[agent_id] = [decision["answer"]]

        if not rankings:
            logging.warning("No valid rankings found in agent decisions")
            return {
                "error": "No valid rankings found",
                "all_rankings": agent_decisions
            }

        # Apply each aggregation method
        if "borda_count" in methods:
            results["borda_count"] = self.borda_count(rankings)

        if "majority_vote" in methods:
            results["majority_vote"] = self.majority_voting(rankings)

        if "weighted_consensus" in methods:
            # Extract weights from confidence levels
            weights = self._extract_weights(agent_decisions)
            results["weighted_consensus"] = self.weighted_consensus(rankings, weights)

        # Include all original rankings
        results["all_rankings"] = agent_decisions

        return results

    def borda_count(self, rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Borda count aggregation: Point-based ranking system.

        Scoring (configurable via BORDA_SCORES):
        - 1st choice: 3 points
        - 2nd choice: 2 points
        - 3rd choice: 1 point
        - 4th choice: 0 points

        Args:
            rankings: Dictionary {agent_id: [ranked_options]}

        Returns:
            {
                "scores": {"A": 5, "B": 8, "C": 3, "D": 0},
                "winner": "B",
                "ranked_options": ["B", "A", "C", "D"]
            }
        """
        scores = defaultdict(int)

        # Calculate Borda scores
        for agent_id, ranking in rankings.items():
            if not ranking:
                logging.warning(f"{agent_id} has empty ranking")
                continue

            for position, option in enumerate(ranking, start=1):
                # Get points for this position (default to 0 if position > max)
                points = self.borda_scores.get(position, 0)
                scores[option] += points

        # Convert to regular dict and sort
        scores_dict = dict(scores)
        ranked_options = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)

        winner = ranked_options[0] if ranked_options else None

        logging.info(f"Borda Count: Winner={winner}, Scores={scores_dict}")

        return {
            "scores": scores_dict,
            "winner": winner,
            "ranked_options": ranked_options
        }

    def majority_voting(self, rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Majority voting: Count first-choice votes only.

        Args:
            rankings: Dictionary {agent_id: [ranked_options]}

        Returns:
            {
                "first_choice_votes": {"A": 1, "B": 2, "C": 0},
                "winner": "B",
                "vote_percentage": 0.67
            }
        """
        first_choices = []

        for agent_id, ranking in rankings.items():
            if ranking and len(ranking) > 0:
                first_choices.append(ranking[0])
            else:
                logging.warning(f"{agent_id} has no first choice")

        # Count votes
        vote_counts = Counter(first_choices)
        total_votes = len(first_choices)

        if not vote_counts:
            logging.warning("No first choice votes found")
            return {
                "first_choice_votes": {},
                "winner": None,
                "vote_percentage": 0.0
            }

        # Get winner (most common)
        winner, winner_votes = vote_counts.most_common(1)[0]
        vote_percentage = winner_votes / total_votes if total_votes > 0 else 0.0

        logging.info(f"Majority Vote: Winner={winner} "
                    f"({winner_votes}/{total_votes} = {vote_percentage:.1%})")

        return {
            "first_choice_votes": dict(vote_counts),
            "winner": winner,
            "winner_votes": winner_votes,
            "total_votes": total_votes,
            "vote_percentage": vote_percentage
        }

    def weighted_consensus(self,
                          rankings: Dict[str, List[str]],
                          weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Weighted Borda count using agent confidence or trust scores.

        Args:
            rankings: Dictionary {agent_id: [ranked_options]}
            weights: Dictionary {agent_id: weight_value}
                     If None, uses equal weights

        Returns:
            {
                "weighted_scores": {"A": 4.5, "B": 7.2, "C": 2.1},
                "winner": "B",
                "ranked_options": ["B", "A", "C"],
                "weights_used": {"agent_1": 1.0, "agent_2": 0.5}
            }
        """
        # Default to equal weights if not provided
        if weights is None:
            weights = {agent_id: 1.0 for agent_id in rankings.keys()}

        weighted_scores = defaultdict(float)

        # Calculate weighted Borda scores
        for agent_id, ranking in rankings.items():
            if not ranking:
                continue

            agent_weight = weights.get(agent_id, 1.0)

            for position, option in enumerate(ranking, start=1):
                points = self.borda_scores.get(position, 0)
                weighted_scores[option] += points * agent_weight

        # Convert to regular dict and sort
        scores_dict = dict(weighted_scores)
        ranked_options = sorted(scores_dict.keys(), key=lambda x: scores_dict[x], reverse=True)

        winner = ranked_options[0] if ranked_options else None

        logging.info(f"Weighted Consensus: Winner={winner}, "
                    f"Weighted Scores={scores_dict}, Weights={weights}")

        return {
            "weighted_scores": scores_dict,
            "winner": winner,
            "ranked_options": ranked_options,
            "weights_used": weights
        }

    def _extract_weights(self, agent_decisions: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract weights from agent confidence levels.

        Confidence → Weight mapping:
        - High: 1.0
        - Medium: 0.7
        - Low: 0.4
        - None: 0.5 (default)

        Args:
            agent_decisions: Agent decisions dict with confidence levels

        Returns:
            Dictionary {agent_id: weight}
        """
        confidence_weights = {
            "High": 1.0,
            "Medium": 0.7,
            "Low": 0.4
        }

        weights = {}
        for agent_id, decision in agent_decisions.items():
            if "error" in decision:
                weights[agent_id] = 0.0  # No weight for errors
            else:
                confidence = decision.get("confidence", "Medium")
                weights[agent_id] = confidence_weights.get(confidence, 0.5)

        return weights

    def calculate_agreement_metrics(self, rankings: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Calculate inter-agent agreement metrics.

        Args:
            rankings: Dictionary {agent_id: [ranked_options]}

        Returns:
            {
                "full_agreement": bool,  # All agents have same 1st choice
                "partial_agreement": float,  # % agents with majority 1st choice
                "kendall_tau": float,  # Rank correlation (future)
                "disagreement_pairs": [...]  # Pairs with different 1st choices
            }
        """
        if not rankings:
            return {"error": "No rankings to analyze"}

        first_choices = [ranking[0] for ranking in rankings.values() if ranking]

        if not first_choices:
            return {"error": "No first choices found"}

        # Full agreement: all first choices are the same
        full_agreement = len(set(first_choices)) == 1

        # Partial agreement: percentage with most common first choice
        most_common_choice, most_common_count = Counter(first_choices).most_common(1)[0]
        partial_agreement = most_common_count / len(first_choices)

        # Find disagreement pairs
        disagreement_pairs = []
        agent_list = list(rankings.keys())
        for i, agent1 in enumerate(agent_list):
            for agent2 in agent_list[i+1:]:
                if rankings[agent1][0] != rankings[agent2][0]:
                    disagreement_pairs.append((agent1, agent2))

        return {
            "full_agreement": full_agreement,
            "partial_agreement_rate": partial_agreement,
            "most_common_first_choice": most_common_choice,
            "disagreement_pairs": disagreement_pairs,
            "n_agents": len(rankings)
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def aggregate_with_all_methods(agent_decisions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick aggregation using all available methods.

    Args:
        agent_decisions: Agent decisions from Round 3

    Returns:
        Complete aggregation results
    """
    aggregator = DecisionAggregator()
    return aggregator.aggregate_decisions(agent_decisions)


def get_final_answer(aggregation_results: Dict[str, Any],
                     method: str = None) -> str:
    """
    Extract final answer from aggregation results.

    Args:
        aggregation_results: Results from aggregate_decisions()
        method: Which method to use (default: PRIMARY_DECISION_METHOD from config)

    Returns:
        Final answer (option letter/text)
    """
    if method is None:
        method = multi_agent_config.PRIMARY_DECISION_METHOD

    if method in aggregation_results:
        return aggregation_results[method].get("winner")

    logging.warning(f"Method {method} not found in results, using first available")
    for key, value in aggregation_results.items():
        if isinstance(value, dict) and "winner" in value:
            return value["winner"]

    return None


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "DecisionAggregator",
    "aggregate_with_all_methods",
    "get_final_answer",
]
