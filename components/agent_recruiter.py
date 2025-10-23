"""
Agent Recruiter - Dynamic Agent Recruitment & Role Assignment

Analyzes questions to determine optimal number of agents (2-4) and creates
specialized agents with appropriate roles and expertise.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re

# Add directories to path FIRST (must be before imports)
_parent_dir = str(Path(__file__).parent.parent.parent)  # For slm_runner, slm_config
_multi_agent_dir = str(Path(__file__).parent.parent)  # For components, utils, config

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _multi_agent_dir not in sys.path:
    sys.path.insert(0, _multi_agent_dir)

from slm_runner import SLMAgent
from slm_config import get_model_config
from components.gemma_agent import GemmaAgent
from utils.prompts import RECRUITMENT_PROMPTS, format_options
import config as multi_agent_config


class AgentRecruiter:
    """
    Dynamically recruits specialized agents based on question analysis.

    The recruiter uses an LLM to:
    1. Analyze question complexity
    2. Determine optimal number of agents (2-4)
    3. Assign specialized medical roles to each agent
    4. Create GemmaAgent instances with appropriate expertise
    """

    def __init__(self,
                 model_name: str = None,
                 chat_instance_type: str = "google_ai_studio",
                 enable_dynamic_recruitment: bool = True):
        """
        Initialize the agent recruiter.

        Args:
            model_name: Model to use for recruitment decisions (default: from config)
            chat_instance_type: Chat instance type
            enable_dynamic_recruitment: If False, uses DEFAULT_N_AGENTS from config
        """
        self.model_name = model_name or multi_agent_config.RECRUITMENT_MODEL
        self.chat_instance_type = chat_instance_type
        self.enable_dynamic_recruitment = enable_dynamic_recruitment

        # Initialize recruiter LLM (for complexity analysis and role assignment)
        model_config = get_model_config(self.model_name, chat_instance_type)
        self.recruiter_llm = SLMAgent(model_config, chat_instance_type)

        logging.info(f"Initialized AgentRecruiter with {self.model_name} "
                    f"(dynamic={enable_dynamic_recruitment})")

    def recruit_agents(self,
                      question: str,
                      options: List[str] = None,
                      n_agents: int = None,
                      agent_model_name: str = None) -> List[GemmaAgent]:
        """
        Recruit specialized agents for this question.

        Args:
            question: The medical question to analyze
            options: Answer options (for MCQ tasks)
            n_agents: Fixed number of agents (overrides dynamic recruitment if provided)
            agent_model_name: Model name for created agents (default: same as recruiter)

        Returns:
            List of initialized GemmaAgent instances with specialized roles
        """
        # Determine number of agents
        if n_agents is not None:
            # Fixed agent count (user override)
            num_agents = self._validate_agent_count(n_agents)
            logging.info(f"Using fixed agent count: {num_agents}")
        elif not self.enable_dynamic_recruitment:
            # Use default from config
            num_agents = multi_agent_config.DEFAULT_N_AGENTS or 3
            logging.info(f"Using default agent count from config: {num_agents}")
        else:
            # Dynamic recruitment based on question complexity
            num_agents = self.determine_optimal_count(question, options)
            logging.info(f"Dynamically determined optimal agent count: {num_agents}")

        # Assign roles to agents
        agent_roles = self.assign_roles(question, options, num_agents)

        # Create GemmaAgent instances
        agent_model = agent_model_name or self.model_name
        agents = []

        for i, role_info in enumerate(agent_roles):
            agent = GemmaAgent(
                agent_id=role_info['agent_id'],
                role=role_info['role'],
                expertise=role_info['expertise'],
                model_name=agent_model,
                chat_instance_type=self.chat_instance_type
            )
            agents.append(agent)

        logging.info(f"Recruited {len(agents)} agents: "
                    f"{[a.role for a in agents]}")

        return agents

    def determine_optimal_count(self,
                               question: str,
                               options: List[str] = None) -> int:
        """
        Use LLM to determine optimal number of agents (2-4) for this question.

        Args:
            question: The medical question
            options: Answer options

        Returns:
            Optimal number of agents (2, 3, or 4)
        """
        options_str = format_options(options) if options else "N/A"

        # Create complexity analysis prompt
        prompt = RECRUITMENT_PROMPTS["complexity_analysis"].format(
            question=question,
            options=options_str
        )

        try:
            # Get LLM response
            response = self.recruiter_llm.simple_chat(prompt)

            # Extract number from response
            num_agents = self._extract_agent_count(response)

            if num_agents is None:
                logging.warning(f"Could not extract agent count from response: {response}")
                num_agents = 3  # Default to 3 agents

            num_agents = self._validate_agent_count(num_agents)

            logging.debug(f"Complexity analysis result: {num_agents} agents")
            return num_agents

        except Exception as e:
            logging.error(f"Error in complexity analysis: {e}")
            # Fallback to default
            return multi_agent_config.DEFAULT_N_AGENTS or 3

    def assign_roles(self,
                    question: str,
                    options: List[str],
                    n_agents: int) -> List[Dict[str, str]]:
        """
        Use LLM to assign specialized medical roles to agents.

        Args:
            question: The medical question
            options: Answer options
            n_agents: Number of agents to create roles for

        Returns:
            List of role dictionaries with keys: agent_id, role, expertise
        """
        options_str = format_options(options) if options else "N/A"

        # Create role assignment prompt
        prompt = RECRUITMENT_PROMPTS["role_assignment"].format(
            n_agents=n_agents,
            question=question,
            options=options_str
        )

        try:
            # Get LLM response
            response = self.recruiter_llm.simple_chat(prompt)

            # Parse roles from response
            roles = self._parse_role_assignments(response, n_agents)

            if not roles or len(roles) < n_agents:
                logging.warning(f"Could not parse sufficient roles from response, using fallback")
                roles = self._get_fallback_roles(n_agents)

            logging.debug(f"Assigned roles: {[r['role'] for r in roles]}")
            return roles

        except Exception as e:
            logging.error(f"Error in role assignment: {e}")
            # Fallback to generic roles
            return self._get_fallback_roles(n_agents)

    def _extract_agent_count(self, response: str) -> Optional[int]:
        """Extract agent count (2-4) from LLM response."""
        # Look for standalone number
        patterns = [
            r"(?:Number of agents needed|Agent count|Total agents):\s*(\d)",
            r"^(\d)\s*agents?",
            r"optimal(?:ly)?\s+(\d)\s+agents?",
            r"\b([2-4])\b"  # Any number 2-4
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                num = int(match.group(1))
                if 2 <= num <= 4:
                    return num

        return None

    def _parse_role_assignments(self, response: str, expected_count: int) -> List[Dict[str, str]]:
        """
        Parse role assignments from LLM response.

        Expected format:
        AGENT_1: Cardiologist - Expert in cardiovascular diseases
        AGENT_2: Emergency Medicine Physician - Specialist in acute care
        """
        roles = []

        # Pattern: AGENT_1: Role - Expertise
        pattern = r"AGENT_(\d+):\s*([^-]+?)\s*-\s*(.+?)(?=\n|$)"
        matches = re.findall(pattern, response, re.MULTILINE)

        for agent_num, role, expertise in matches:
            roles.append({
                'agent_id': f"agent_{agent_num}",
                'role': role.strip(),
                'expertise': expertise.strip()
            })

        # Ensure we have the expected count
        if len(roles) < expected_count:
            # Try alternative parsing
            # Look for lines with role titles
            lines = response.split('\n')
            for line in lines:
                if ':' in line and '-' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        role_parts = parts[1].split('-', 1)
                        if len(role_parts) == 2:
                            agent_num = len(roles) + 1
                            roles.append({
                                'agent_id': f"agent_{agent_num}",
                                'role': role_parts[0].strip(),
                                'expertise': role_parts[1].strip()
                            })
                            if len(roles) >= expected_count:
                                break

        return roles[:expected_count]

    def _get_fallback_roles(self, n_agents: int) -> List[Dict[str, str]]:
        """
        Get generic fallback roles if LLM-based role assignment fails.

        Args:
            n_agents: Number of agents

        Returns:
            List of generic role dictionaries
        """
        fallback_roles = [
            {
                'agent_id': 'agent_1',
                'role': 'General Internist',
                'expertise': 'Broad medical knowledge and diagnostic reasoning'
            },
            {
                'agent_id': 'agent_2',
                'role': 'Clinical Specialist',
                'expertise': 'Specialized clinical knowledge and treatment planning'
            },
            {
                'agent_id': 'agent_3',
                'role': 'Medical Researcher',
                'expertise': 'Evidence-based medicine and current research'
            },
            {
                'agent_id': 'agent_4',
                'role': 'Critical Care Specialist',
                'expertise': 'Acute and complex case management'
            }
        ]

        return fallback_roles[:n_agents]

    def _validate_agent_count(self, n_agents: int) -> int:
        """
        Validate and clamp agent count to allowed range.

        Args:
            n_agents: Requested agent count

        Returns:
            Valid agent count (clamped to MIN_AGENTS - MAX_AGENTS)
        """
        min_agents = multi_agent_config.MIN_AGENTS
        max_agents = multi_agent_config.MAX_AGENTS

        if n_agents < min_agents:
            logging.warning(f"Agent count {n_agents} < min {min_agents}, using {min_agents}")
            return min_agents
        elif n_agents > max_agents:
            logging.warning(f"Agent count {n_agents} > max {max_agents}, using {max_agents}")
            return max_agents

        return n_agents


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_recruiter(**kwargs) -> AgentRecruiter:
    """
    Create an AgentRecruiter with default settings.

    Args:
        **kwargs: Override default settings

    Returns:
        Initialized AgentRecruiter instance
    """
    return AgentRecruiter(**kwargs)


def quick_recruit(question: str,
                 options: List[str] = None,
                 n_agents: int = None,
                 **kwargs) -> List[GemmaAgent]:
    """
    Quick recruitment - creates recruiter and recruits agents in one call.

    Args:
        question: Medical question
        options: Answer options
        n_agents: Fixed agent count (optional)
        **kwargs: Additional arguments for AgentRecruiter

    Returns:
        List of recruited GemmaAgent instances
    """
    recruiter = AgentRecruiter(**kwargs)
    return recruiter.recruit_agents(question, options, n_agents)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "AgentRecruiter",
    "create_recruiter",
    "quick_recruit",
]
