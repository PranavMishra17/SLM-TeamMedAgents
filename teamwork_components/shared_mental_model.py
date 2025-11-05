"""
Shared Mental Model (SMM) Component

Passive knowledge repository that stores shared understanding across agents.
Acts as a context layer without decision-making authority.

Content:
- question_analysis: Trick detection, complexity assessment (1-line)
- verified_facts: Consensus facts extracted from agent responses
- debated_points: Key controversies and resolutions from discussions

Access Pattern:
- Write: Leader (if enabled) or automated system
- Read: All agents receive SMM in their prompts

Usage:
    smm = SharedMentalModel()
    smm.set_question_analysis("Complex multi-system disorder, watch for except clause")
    smm.add_verified_facts(["Symptom X confirms diagnosis Y", "Lab Z rules out condition W"])
    smm.add_debated_point("Agent 1 vs Agent 2 on treatment protocol - resolved via evidence")
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class SharedMentalModel:
    """
    Shared Mental Model - Passive knowledge repository for multi-agent reasoning.

    Thread-safe implementation for concurrent access.
    """

    # Core content
    question_analysis: Optional[str] = None          # Recruiter's trick detection
    verified_facts: List[str] = field(default_factory=list)      # Consensus facts from R2
    debated_points: List[str] = field(default_factory=list)      # Controversies from R3

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    update_count: int = 0

    # Thread safety (RLock allows reentrant locking - same thread can acquire multiple times)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def set_question_analysis(self, analysis: str) -> None:
        """
        Set question analysis from Recruiter (R1).

        Args:
            analysis: 1-2 sentence trick detection or complexity note
        """
        with self._lock:
            self.question_analysis = analysis
            self.last_updated = datetime.now()
            self.update_count += 1
            logging.info(f"[SMM] Question analysis set: {analysis}")

    def add_verified_facts(self, facts: List[str]) -> None:
        """
        Add verified facts from Post-R2 processing.

        Facts are extracted via consensus or intersection logic.

        Args:
            facts: List of verified medical facts
        """
        with self._lock:
            if not facts:
                logging.debug("[SMM] No verified facts to add")
                return

            for fact in facts:
                if fact and fact not in self.verified_facts:
                    self.verified_facts.append(fact)
                    logging.debug(f"[SMM] Added verified fact: {fact}")

            self.last_updated = datetime.now()
            self.update_count += 1
            logging.info(f"[SMM] Added {len(facts)} verified facts (total: {len(self.verified_facts)})")

    def add_debated_point(self, point: str) -> None:
        """
        Add a debated point from Mutual Monitoring or R3 discussion.

        Args:
            point: Description of controversy and resolution
        """
        with self._lock:
            if point and point not in self.debated_points:
                self.debated_points.append(point)
                self.last_updated = datetime.now()
                self.update_count += 1
                logging.debug(f"[SMM] Added debated point: {point}")

    def get_context_string(self) -> str:
        """
        Generate formatted SMM context for agent prompts.

        Returns:
            Formatted string with all SMM content
        """
        with self._lock:
            if not self.has_content():
                return ""

            lines = ["=== SHARED MENTAL MODEL ==="]

            if self.question_analysis:
                lines.append(f"\nQuestion Analysis:\n{self.question_analysis}")

            if self.verified_facts:
                lines.append(f"\nVerified Facts ({len(self.verified_facts)}):")
                for i, fact in enumerate(self.verified_facts, 1):
                    lines.append(f"  {i}. {fact}")

            if self.debated_points:
                lines.append(f"\nDebated Points ({len(self.debated_points)}):")
                for i, point in enumerate(self.debated_points, 1):
                    lines.append(f"  {i}. {point}")

            lines.append("=" * 29)

            return "\n".join(lines)

    def has_content(self) -> bool:
        """Check if SMM has any content."""
        with self._lock:
            return bool(
                self.question_analysis or
                self.verified_facts or
                self.debated_points
            )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of SMM content for logging.

        Returns:
            Dictionary with SMM statistics
        """
        with self._lock:
            return {
                'has_question_analysis': bool(self.question_analysis),
                'question_analysis': self.question_analysis,
                'verified_facts_count': len(self.verified_facts),
                'verified_facts': self.verified_facts.copy(),
                'debated_points_count': len(self.debated_points),
                'debated_points': self.debated_points.copy(),
                'update_count': self.update_count,
                'last_updated': self.last_updated.isoformat()
            }

    def clear(self) -> None:
        """Clear all SMM content (for new question)."""
        with self._lock:
            self.question_analysis = None
            self.verified_facts.clear()
            self.debated_points.clear()
            self.update_count = 0
            logging.debug("[SMM] Cleared all content")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return self.get_summary()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedMentalModel':
        """Deserialize from dictionary."""
        smm = cls()
        if data.get('question_analysis'):
            smm.set_question_analysis(data['question_analysis'])
        if data.get('verified_facts'):
            smm.add_verified_facts(data['verified_facts'])
        for point in data.get('debated_points', []):
            smm.add_debated_point(point)
        return smm

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            return (f"SharedMentalModel(analysis={bool(self.question_analysis)}, "
                   f"facts={len(self.verified_facts)}, debates={len(self.debated_points)})")


# Helper functions for automated SMM updates (when Leadership is OFF)

def extract_facts_intersection(agent_responses: Dict[str, Any]) -> List[str]:
    """
    Extract consensus facts via intersection logic.

    Finds facts mentioned by multiple agents (2+ agents for N=3, 3+ for N=4).

    Args:
        agent_responses: Dict mapping agent_id to response dict containing 'facts'

    Returns:
        List of consensus facts
    """
    from collections import Counter

    if not agent_responses:
        return []

    # Extract all facts from all agents
    all_facts = []
    for agent_id, response in agent_responses.items():
        facts = response.get('facts', [])
        if isinstance(facts, list):
            all_facts.extend(facts)

    # Count fact occurrences (simple string matching)
    fact_counts = Counter(all_facts)

    # Consensus threshold: 2+ agents for N<=3, 3+ for N>=4
    n_agents = len(agent_responses)
    threshold = 2 if n_agents <= 3 else 3

    # Filter to consensus facts
    consensus_facts = [fact for fact, count in fact_counts.items() if count >= threshold]

    logging.debug(f"[SMM Auto] Extracted {len(consensus_facts)} consensus facts from {n_agents} agents")
    return consensus_facts


def detect_question_tricks(question: str, options: List[str]) -> str:
    """
    Simple rule-based trick detection for SMM (when Leadership is OFF).

    Args:
        question: Question text
        options: Answer options

    Returns:
        1-2 sentence analysis
    """
    tricks = []

    # Check for EXCEPT questions
    if any(term in question.lower() for term in ['except', 'not true', 'false', 'incorrect', 'is not']):
        tricks.append("EXCEPT question - rank from most false to most true")

    # Check for image mentions
    if any(term in question.lower() for term in ['image', 'shown', 'figure', 'photograph']):
        tricks.append("visual analysis required")

    # Check for multi-step reasoning
    if any(term in question.lower() for term in ['first', 'next step', 'most appropriate', 'best']):
        tricks.append("multi-step clinical reasoning")

    # Check complexity
    if len(options) > 5:
        tricks.append("complex case with many options")

    if tricks:
        return "Complexity: " + "; ".join(tricks)
    else:
        return "Standard medical reasoning question"


__all__ = ['SharedMentalModel', 'extract_facts_intersection', 'detect_question_tricks']
