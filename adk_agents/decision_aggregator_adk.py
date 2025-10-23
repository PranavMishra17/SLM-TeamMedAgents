"""
Decision Aggregator for ADK Multi-Agent System

Utility functions for aggregating agent decisions using various voting methods:
- Borda count (weighted ranking)
- Majority voting
- Confidence-weighted voting

This is NOT an ADK Agent - it's a utility module for the main coordinator.

Usage:
    from adk_agents.decision_aggregator_adk import aggregate_rankings

    rankings = {
        'agent_1': ['B', 'A', 'C', 'D'],
        'agent_2': ['B', 'C', 'A', 'D'],
        'agent_3': ['A', 'B', 'D', 'C']
    }

    final_answer = aggregate_rankings(rankings, method='borda')
"""

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional


def aggregate_rankings(
    rankings: Dict[str, List[str]],
    confidences: Optional[Dict[str, str]] = None,
    method: str = 'borda'
) -> Dict[str, Any]:
    """
    Aggregate agent rankings using specified voting method.

    Args:
        rankings: Dict mapping agent_id → ranked list of options
        confidences: Optional dict mapping agent_id → confidence level
        method: Voting method ('borda', 'majority', 'confidence_weighted')

    Returns:
        Dict with:
        - winner: Final answer
        - scores: Score breakdown by option
        - method: Method used
        - agreement_rate: Fraction of agents agreeing with winner
    """
    if not rankings:
        return {
            'winner': 'A',
            'scores': {},
            'method': method,
            'agreement_rate': 0.0
        }

    if method == 'borda':
        result = _borda_count(rankings)
    elif method == 'majority':
        result = _majority_voting(rankings)
    elif method == 'confidence_weighted':
        result = _confidence_weighted(rankings, confidences or {})
    else:
        logging.warning(f"Unknown method '{method}', using borda")
        result = _borda_count(rankings)

    return result


def _borda_count(rankings: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Borda count: Points based on ranking position.

    For 4 options: 1st=3pts, 2nd=2pts, 3rd=1pt, 4th=0pts
    """
    scores = Counter()

    for agent_id, ranking in rankings.items():
        if not ranking:
            continue

        n_options = len(ranking)

        for position, option in enumerate(ranking):
            # Points = (n_options - position - 1)
            # First place gets (n-1) points, last gets 0
            points = n_options - position - 1
            scores[option] += points

            logging.debug(f"{agent_id} ranks {option} at position {position+1}: {points} points")

    # Get winner
    if scores:
        winner, top_score = scores.most_common(1)[0]

        # Calculate agreement rate (how many agents ranked winner first)
        first_choices = [r[0] for r in rankings.values() if r and len(r) > 0]
        agreement_rate = first_choices.count(winner) / len(first_choices) if first_choices else 0

        return {
            'winner': winner,
            'scores': dict(scores),
            'method': 'borda',
            'agreement_rate': agreement_rate
        }

    return {'winner': 'A', 'scores': {}, 'method': 'borda', 'agreement_rate': 0}


def _majority_voting(rankings: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Simple majority: Most common first-choice wins.
    """
    first_choices = []

    for ranking in rankings.values():
        if ranking and len(ranking) > 0:
            first_choices.append(ranking[0])

    if not first_choices:
        return {'winner': 'A', 'scores': {}, 'method': 'majority', 'agreement_rate': 0}

    vote_counts = Counter(first_choices)
    winner, count = vote_counts.most_common(1)[0]

    agreement_rate = count / len(first_choices)

    return {
        'winner': winner,
        'scores': dict(vote_counts),
        'method': 'majority',
        'agreement_rate': agreement_rate
    }


def _confidence_weighted(
    rankings: Dict[str, List[str]],
    confidences: Dict[str, str]
) -> Dict[str, Any]:
    """
    Confidence-weighted: Borda count with confidence multipliers.

    Weights: High=1.5, Medium=1.0, Low=0.5
    """
    confidence_weights = {
        'High': 1.5,
        'Medium': 1.0,
        'Low': 0.5
    }

    scores = defaultdict(float)

    for agent_id, ranking in rankings.items():
        if not ranking:
            continue

        # Get confidence weight
        confidence = confidences.get(agent_id, 'Medium')
        weight = confidence_weights.get(confidence, 1.0)

        n_options = len(ranking)

        for position, option in enumerate(ranking):
            points = (n_options - position - 1) * weight
            scores[option] += points

            logging.debug(f"{agent_id} ({confidence}) ranks {option}: {points} weighted points")

    if scores:
        winner = max(scores, key=scores.get)

        # Agreement rate
        first_choices = [r[0] for r in rankings.values() if r and len(r) > 0]
        agreement_rate = first_choices.count(winner) / len(first_choices) if first_choices else 0

        return {
            'winner': winner,
            'scores': dict(scores),
            'method': 'confidence_weighted',
            'agreement_rate': agreement_rate
        }

    return {'winner': 'A', 'scores': {}, 'method': 'confidence_weighted', 'agreement_rate': 0}


def calculate_convergence(rankings: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate convergence metrics for agent rankings.

    Returns:
        Dict with:
        - converged: True if all agents agree on first choice
        - first_choice_agreement: Fraction agreeing on most common first choice
        - ranking_similarity: Average Kendall tau distance (simplified)
    """
    if not rankings:
        return {
            'converged': False,
            'first_choice_agreement': 0,
            'ranking_similarity': 0
        }

    first_choices = [r[0] for r in rankings.values() if r and len(r) > 0]

    if not first_choices:
        return {'converged': False, 'first_choice_agreement': 0, 'ranking_similarity': 0}

    # Check convergence
    converged = len(set(first_choices)) == 1

    # Agreement rate
    most_common = Counter(first_choices).most_common(1)[0]
    agreement_rate = most_common[1] / len(first_choices)

    return {
        'converged': converged,
        'first_choice_agreement': agreement_rate,
        'ranking_similarity': agreement_rate  # Simplified metric
    }


__all__ = [
    'aggregate_rankings',
    'calculate_convergence'
]
