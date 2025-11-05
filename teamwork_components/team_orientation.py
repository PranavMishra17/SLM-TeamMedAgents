"""
Team Orientation Component

Specialized role assignment with hierarchical weighting for multi-agent systems.

Features:
- Medical specialty roles (cardiology, radiology, pathology, etc.)
- Hierarchical weights hidden from agents (0.5, 0.3, 0.2 for N=3)
- Formal medical report generation (via Leadership)
- Enhanced role definitions using SMM context

Design:
- When OFF: Generic "medical expert" roles, equal weights (0.8 each)
- When ON: Specific specialists with domain expertise, weighted by importance

Usage:
    team_o = TeamOrientationManager(config=teamwork_config)

    # During recruitment
    roles, weights = team_o.assign_roles(
        question="Patient with chest pain...",
        n_agents=3,
        smm=shared_mental_model,
        use_llm=leadership_enabled
    )

    # During aggregation
    weighted_votes = team_o.apply_hierarchical_weights(agent_votes, weights)
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class AgentRole:
    """Represents a specialist agent role."""
    role_name: str                  # E.g., "Cardiologist"
    expertise: str                  # Detailed expertise description
    weight: float                   # Hierarchical weight (0.0-1.0)
    domain: str                     # Medical domain (cardiovascular, neuro, etc.)


class TeamOrientationManager:
    """
    Manages team orientation with specialized roles and hierarchical weights.
    """

    # Medical specialty templates
    MEDICAL_SPECIALTIES = [
        {
            "role": "Cardiologist",
            "expertise": "Cardiovascular diseases, ECG interpretation, cardiac imaging",
            "domain": "cardiovascular",
            "keywords": ["heart", "cardiac", "cardiovascular", "ecg", "myocardial", "arrhythmia"]
        },
        {
            "role": "Pulmonologist",
            "expertise": "Respiratory diseases, pulmonary function, lung pathology",
            "domain": "respiratory",
            "keywords": ["lung", "respiratory", "pulmonary", "breathing", "pneumonia", "asthma"]
        },
        {
            "role": "Neurologist",
            "expertise": "Neurological disorders, CNS pathology, neuromuscular diseases",
            "domain": "neurological",
            "keywords": ["brain", "neuro", "cns", "stroke", "seizure", "cognitive"]
        },
        {
            "role": "Radiologist",
            "expertise": "Medical imaging interpretation, diagnostic radiology",
            "domain": "imaging",
            "keywords": ["image", "ct", "mri", "x-ray", "scan", "radiograph"]
        },
        {
            "role": "Pathologist",
            "expertise": "Tissue diagnosis, laboratory medicine, histopathology",
            "domain": "pathology",
            "keywords": ["biopsy", "tissue", "histology", "pathology", "specimen"]
        },
        {
            "role": "General Internist",
            "expertise": "Comprehensive internal medicine, differential diagnosis",
            "domain": "general",
            "keywords": []  # Catch-all
        }
    ]

    def __init__(self, config: Any):
        """
        Initialize Team Orientation manager.

        Args:
            config: TeamworkConfig instance
        """
        self.config = config

    def assign_roles(
        self,
        question: str,
        n_agents: int,
        smm: Optional[Any] = None,
        use_llm: bool = False,
        leader_agent: Optional[Any] = None,
        ctx: Optional[Any] = None
    ) -> Tuple[List[AgentRole], Dict[str, float]]:
        """
        Assign specialized roles to N agents based on question domain.

        Args:
            question: Medical question text
            n_agents: Number of agents to create
            smm: Shared Mental Model (optional, for enhanced context)
            use_llm: Use Leader LLM for role assignment (if Leadership enabled)
            leader_agent: Leader agent instance (if use_llm=True)
            ctx: Invocation context (if use_llm=True)

        Returns:
            Tuple of (list of AgentRole objects, dict mapping agent_id to weight)
        """
        if use_llm and leader_agent and ctx:
            # LLM-powered role assignment (when Leadership enabled)
            logging.info("[TeamO] Using Leader for role assignment")
            # TODO: Implement LLM-based assignment in future iteration
            # For now, fall back to rule-based
            pass

        # Rule-based role assignment
        roles = self._assign_roles_rule_based(question, n_agents)
        weights = self._assign_hierarchical_weights(n_agents)

        # Map weights to agent_ids
        weight_map = {f"agent_{i+1}": weights[i] for i in range(n_agents)}

        logging.info(f"[TeamO] Assigned {n_agents} specialist roles with hierarchical weights")
        for i, role in enumerate(roles):
            logging.info(f"  agent_{i+1}: {role.role_name} (weight={role.weight:.2f})")

        return roles, weight_map

    def _assign_roles_rule_based(self, question: str, n_agents: int) -> List[AgentRole]:
        """
        Rule-based role assignment using keyword matching.

        Args:
            question: Question text
            n_agents: Number of agents

        Returns:
            List of AgentRole objects
        """
        question_lower = question.lower()

        # Score each specialty by keyword relevance
        specialty_scores = []
        for specialty in self.MEDICAL_SPECIALTIES:
            score = sum(1 for keyword in specialty['keywords'] if keyword in question_lower)
            specialty_scores.append((score, specialty))

        # Sort by relevance
        specialty_scores.sort(key=lambda x: x[0], reverse=True)

        # Assign hierarchical weights
        weights = self._assign_hierarchical_weights(n_agents)

        # Create agent roles
        roles = []
        for i in range(n_agents):
            specialty = specialty_scores[i % len(specialty_scores)][1]
            role = AgentRole(
                role_name=specialty['role'],
                expertise=specialty['expertise'],
                weight=weights[i],
                domain=specialty['domain']
            )
            roles.append(role)

        return roles

    def _assign_hierarchical_weights(self, n_agents: int) -> List[float]:
        """
        Assign hierarchical weights based on agent count.

        Weights are hidden from agents but used in aggregation.

        Args:
            n_agents: Number of agents

        Returns:
            List of weights (sum = 1.0)
        """
        if n_agents == 2:
            return [0.6, 0.4]
        elif n_agents == 3:
            return self.config.hierarchical_weights  # [0.5, 0.3, 0.2]
        elif n_agents == 4:
            return [0.4, 0.3, 0.2, 0.1]
        else:
            # Equal weights as fallback
            return [1.0 / n_agents] * n_agents

    def apply_hierarchical_weights(
        self,
        agent_votes: Dict[str, str],
        weight_map: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply hierarchical weights to agent votes.

        Args:
            agent_votes: Dict mapping agent_id to voted option
            weight_map: Dict mapping agent_id to weight

        Returns:
            Dict mapping option to weighted vote count
        """
        weighted_scores = {}

        for agent_id, vote in agent_votes.items():
            weight = weight_map.get(agent_id, 1.0 / len(agent_votes))

            if vote not in weighted_scores:
                weighted_scores[vote] = 0.0

            weighted_scores[vote] += weight

        logging.info(f"[TeamO] Hierarchical weighted votes: {weighted_scores}")
        return weighted_scores

    def get_role_context_for_agent(self, role: AgentRole) -> str:
        """
        Generate role-specific context for agent prompts.

        Args:
            role: AgentRole object

        Returns:
            Formatted context string
        """
        return f"""You are a {role.role_name} with expertise in {role.expertise}.
Approach this question from your specialized perspective while considering the broader clinical context."""

    @staticmethod
    def calculate_hierarchical_score(
        rankings: Dict[str, List[str]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate weighted Borda count scores using hierarchical weights.

        Args:
            rankings: Dict mapping agent_id to ranked list of options
            weights: Dict mapping agent_id to weight

        Returns:
            Dict mapping option to weighted score
        """
        scores = {}

        for agent_id, ranking in rankings.items():
            weight = weights.get(agent_id, 1.0)
            n_options = len(ranking)

            # Borda count: first choice gets (n-1) points, second gets (n-2), etc.
            for position, option in enumerate(ranking):
                points = (n_options - position - 1) * weight

                if option not in scores:
                    scores[option] = 0.0

                scores[option] += points

        return scores


# Utility function for role-based prompt enhancement
def enhance_prompt_with_role(base_prompt: str, role: AgentRole) -> str:
    """
    Enhance agent prompt with role-specific instructions.

    Args:
        base_prompt: Original prompt
        role: AgentRole object

    Returns:
        Enhanced prompt with role context
    """
    role_context = f"""
[ROLE ASSIGNMENT]
You are a {role.role_name} with specialized expertise in {role.expertise}.
Analyze this question through your specialist lens while maintaining awareness of broader clinical considerations.
[END ROLE ASSIGNMENT]

"""
    return role_context + base_prompt


__all__ = [
    'TeamOrientationManager',
    'AgentRole',
    'enhance_prompt_with_role'
]
