"""
Trust Network Component

Dynamic trust scoring system for multi-agent reasoning.
Trust scores represent agent reliability and are used for weighted voting.

Features:
- Per-agent trust scores (0.4-1.0 range, default 0.8)
- Update triggers: Post-R2, Post-MM, Post-R3
- Evaluation criteria: Fact accuracy, reasoning quality, response completeness
- Affects aggregation weights only, not agent behavior (agents don't see trust scores)

Design Pattern:
- When OFF: All agents have equal weight (0.8), simple majority voting
- When ON: Dynamic trust scores influence final vote weighting

Usage:
    trust = TrustNetwork(config=teamwork_config)

    # Initialize agents
    trust.initialize_agents(['agent_1', 'agent_2', 'agent_3'])

    # After R2
    trust.update_after_round2(agent_responses, evaluation_criteria)

    # During aggregation
    weighted_votes = trust.apply_trust_weighted_voting(agent_votes)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class AgentTrustProfile:
    """Trust profile for a single agent."""
    agent_id: str
    trust_score: float = 0.8              # Current trust score (0.4-1.0)
    history: List[float] = field(default_factory=list)  # Score history
    last_updated: datetime = field(default_factory=datetime.now)
    evaluation_count: int = 0

    def update_score(self, new_score: float, min_score: float = 0.4, max_score: float = 1.0):
        """Update trust score with bounds checking."""
        # Clamp to valid range
        new_score = max(min_score, min(max_score, new_score))

        # Store history
        self.history.append(self.trust_score)

        # Update
        self.trust_score = new_score
        self.last_updated = datetime.now()
        self.evaluation_count += 1

        logging.debug(f"[Trust] {self.agent_id} score updated: {self.history[-1]:.3f} → {self.trust_score:.3f}")


class TrustNetwork:
    """
    Trust network for dynamic agent reliability scoring.

    Thread-safe implementation for concurrent updates.
    """

    def __init__(self, config: Any):
        """
        Initialize trust network.

        Args:
            config: TeamworkConfig instance
        """
        self.config = config
        self.profiles: Dict[str, AgentTrustProfile] = {}
        self._lock = threading.RLock()  # RLock allows reentrant locking

        # Trust bounds
        self.min_trust = config.trust_range[0]
        self.max_trust = config.trust_range[1]
        self.default_trust = config.trust_default

        logging.info(f"[Trust] Initialized with range [{self.min_trust}, {self.max_trust}], default={self.default_trust}")

    def initialize_agents(self, agent_ids: List[str]) -> None:
        """
        Initialize trust profiles for all agents.

        Args:
            agent_ids: List of agent identifiers
        """
        with self._lock:
            for agent_id in agent_ids:
                self.profiles[agent_id] = AgentTrustProfile(
                    agent_id=agent_id,
                    trust_score=self.default_trust
                )

            logging.info(f"[Trust] Initialized {len(agent_ids)} agent profiles with default score {self.default_trust}")

    def get_trust_score(self, agent_id: str) -> float:
        """
        Get current trust score for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Trust score (0.4-1.0)
        """
        with self._lock:
            profile = self.profiles.get(agent_id)
            return profile.trust_score if profile else self.default_trust

    def get_all_trust_scores(self) -> Dict[str, float]:
        """Get trust scores for all agents."""
        with self._lock:
            return {agent_id: profile.trust_score for agent_id, profile in self.profiles.items()}

    def update_after_round2(
        self,
        agent_responses: Dict[str, Any],
        ground_truth: Optional[str] = None
    ) -> None:
        """
        Update trust scores after Round 2 based on response quality.

        Evaluation criteria:
        - Fact accuracy (if ground truth available)
        - Reasoning completeness (length, structure)
        - Confidence alignment with correctness

        Args:
            agent_responses: Dict mapping agent_id to R2 response dict
            ground_truth: Ground truth answer (optional)
        """
        with self._lock:
            for agent_id, response in agent_responses.items():
                if agent_id not in self.profiles:
                    logging.warning(f"[Trust] Unknown agent {agent_id}, skipping")
                    continue

                # Evaluate response quality
                quality_score = self._evaluate_response_quality(response, ground_truth)

                # Update trust score (weighted average: 70% old, 30% new)
                current_score = self.profiles[agent_id].trust_score
                new_score = 0.7 * current_score + 0.3 * quality_score

                self.profiles[agent_id].update_score(new_score, self.min_trust, self.max_trust)

            logging.info(f"[Trust] Updated trust scores after R2 for {len(agent_responses)} agents")

    def update_after_mutual_monitoring(
        self,
        challenged_agent_id: str,
        response_quality: str,
        adjustment: float = 0.05
    ) -> None:
        """
        Update trust score after Mutual Monitoring challenge.

        Args:
            challenged_agent_id: Agent who was challenged
            response_quality: 'strong', 'weak', or 'disputed'
            adjustment: Trust adjustment magnitude
        """
        with self._lock:
            if challenged_agent_id not in self.profiles:
                logging.warning(f"[Trust] Unknown agent {challenged_agent_id}, skipping MM update")
                return

            current_score = self.profiles[challenged_agent_id].trust_score

            # Adjust based on MM outcome
            if response_quality == 'strong':
                # Good defense → increase trust
                new_score = current_score + adjustment
            elif response_quality == 'weak':
                # Weak defense → decrease trust
                new_score = current_score - adjustment
            else:  # 'disputed'
                # Neutral outcome → small decrease
                new_score = current_score - (adjustment * 0.5)

            self.profiles[challenged_agent_id].update_score(new_score, self.min_trust, self.max_trust)

            logging.info(f"[Trust] MM update for {challenged_agent_id}: {current_score:.3f} → {new_score:.3f} ({response_quality})")

    def apply_trust_weighted_voting(
        self,
        agent_votes: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Apply trust-weighted voting to agent votes.

        Args:
            agent_votes: Dict mapping agent_id to voted option

        Returns:
            Dict mapping option to weighted vote count
        """
        weighted_votes = {}

        with self._lock:
            for agent_id, vote in agent_votes.items():
                trust_score = self.get_trust_score(agent_id)

                if vote not in weighted_votes:
                    weighted_votes[vote] = 0.0

                weighted_votes[vote] += trust_score

        logging.info(f"[Trust] Weighted votes: {weighted_votes}")
        return weighted_votes

    def calculate_trust_weighted_borda(
        self,
        rankings: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Calculate Borda count with trust-based weighting.

        Args:
            rankings: Dict mapping agent_id to ranked list

        Returns:
            Dict mapping option to weighted Borda score
        """
        scores = {}

        with self._lock:
            for agent_id, ranking in rankings.items():
                trust_score = self.get_trust_score(agent_id)
                n_options = len(ranking)

                # Borda points weighted by trust
                for position, option in enumerate(ranking):
                    points = (n_options - position - 1) * trust_score

                    if option not in scores:
                        scores[option] = 0.0

                    scores[option] += points

        logging.info(f"[Trust] Trust-weighted Borda scores: {scores}")
        return scores

    def get_trust_hints_for_prompt(self) -> str:
        """
        Generate trust hints for agent prompts.

        Returns human-readable trust context without revealing exact scores.

        Returns:
            Formatted trust hints string
        """
        with self._lock:
            if not self.profiles:
                return ""

            hints = ["Team Trust Context:"]
            for agent_id, profile in sorted(self.profiles.items()):
                # Categorize trust level
                if profile.trust_score >= 0.85:
                    level = "highly trusted"
                elif profile.trust_score >= 0.7:
                    level = "trusted"
                elif profile.trust_score >= 0.55:
                    level = "moderate trust"
                else:
                    level = "developing trust"

                hints.append(f"  {agent_id}: {level}")

            return "\n".join(hints)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of trust network state."""
        with self._lock:
            return {
                'agent_count': len(self.profiles),
                'trust_scores': {aid: profile.trust_score for aid, profile in self.profiles.items()},
                'avg_trust': sum(p.trust_score for p in self.profiles.values()) / len(self.profiles) if self.profiles else 0.0,
                'min_trust_threshold': self.min_trust,
                'max_trust_threshold': self.max_trust
            }

    def _evaluate_response_quality(
        self,
        response: Any,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Evaluate response quality for trust scoring.

        Criteria:
        1. Reasoning length and structure
        2. Fact count and specificity
        3. Answer correctness (if ground truth available)

        Args:
            response: Agent response dict
            ground_truth: Correct answer (optional)

        Returns:
            Quality score (0.4-1.0)
        """
        score = 0.5  # Base score

        # Check reasoning completeness
        justification = response.get('justification', '')
        if len(justification) > 50:
            score += 0.1
        if len(justification) > 100:
            score += 0.1

        # Check fact count
        facts = response.get('facts', [])
        if len(facts) >= 2:
            score += 0.1
        if len(facts) >= 4:
            score += 0.1

        # Check answer correctness (if available)
        if ground_truth:
            answer = response.get('answer', '')
            if str(answer).strip().upper() == str(ground_truth).strip().upper():
                score += 0.2
            else:
                score -= 0.1

        # Clamp to valid range
        return max(self.min_trust, min(self.max_trust, score))

    def clear(self) -> None:
        """Clear all trust profiles."""
        with self._lock:
            self.profiles.clear()
            logging.debug("[Trust] Cleared all profiles")


__all__ = ['TrustNetwork', 'AgentTrustProfile']
