"""
Gemma Agent - Role-Specific Agent Wrapper

Wraps the existing SLMAgent with role-specific context and conversation tracking.
Does NOT modify SLMAgent - simply adds multi-agent functionality on top.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import re

# Add parent directory to path FIRST (must be before imports that need it)
_parent_dir = str(Path(__file__).parent.parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from slm_runner import SLMAgent
from slm_config import get_model_config


class GemmaAgent:
    """
    Role-specific medical expert agent for multi-agent collaboration.

    Wraps SLMAgent with:
    - Role and expertise context
    - Conversation history tracking per round
    - Answer extraction logic
    - Multi-round interaction management
    """

    def __init__(self,
                 agent_id: str,
                 role: str,
                 expertise: str,
                 model_name: str = "gemma3_4b",
                 chat_instance_type: str = "google_ai_studio"):
        """
        Initialize a Gemma agent with specific role and expertise.

        Args:
            agent_id: Unique identifier for this agent (e.g., "agent_1", "cardiologist")
            role: Medical specialty or role (e.g., "Cardiologist")
            expertise: Brief description of expertise (e.g., "Expert in cardiovascular diseases")
            model_name: Which Gemma model to use (gemma3_4b or medgemma_4b)
            chat_instance_type: Chat instance type (google_ai_studio or huggingface)
        """
        self.agent_id = agent_id
        self.role = role
        self.expertise = expertise
        self.model_name = model_name
        self.chat_instance_type = chat_instance_type

        # Initialize underlying SLM agent
        model_config = get_model_config(model_name, chat_instance_type)
        self.slm_agent = SLMAgent(model_config, chat_instance_type)

        # Conversation tracking per round
        self.conversation_history = {
            "round1": [],
            "round2": [],
            "round3": []
        }

        # Response storage
        self.responses = {
            "round1": None,
            "round2": None,
            "round3": None
        }

        logging.info(f"Initialized {self.agent_id}: {self.role} - {self.expertise}")

    def analyze_question(self,
                        prompt: str,
                        round_number: int,
                        image_path: Optional[str] = None) -> str:
        """
        Analyze a question for a specific round.

        Args:
            prompt: Full prompt including role context and question
            round_number: Which round (1, 2, or 3)
            image_path: Optional path to medical image for vision tasks

        Returns:
            Agent's response text
        """
        round_key = f"round{round_number}"

        # Add to conversation history
        self.conversation_history[round_key].append({
            "role": "user",
            "content": prompt
        })

        # Get response from underlying SLM agent
        try:
            response = self.slm_agent.simple_chat(prompt, image_path)

            # Store response
            self.conversation_history[round_key].append({
                "role": "assistant",
                "content": response
            })
            self.responses[round_key] = response

            logging.debug(f"{self.agent_id} Round {round_number} response: {response[:200]}...")

            return response

        except Exception as e:
            logging.error(f"Error in {self.agent_id} Round {round_number} analysis: {e}")
            raise

    def extract_answer(self, response: str, task_type: str = "mcq") -> Optional[Dict[str, Any]]:
        """
        Extract structured answer from agent's response.

        Args:
            response: Agent's text response
            task_type: Type of task (mcq, yes_no_maybe, ranking, open_ended)

        Returns:
            Structured answer dict with keys depending on task_type:
                - mcq: {"answer": "B", "confidence": "High", "raw": response}
                - ranking: {"ranking": ["B", "A", "C", "D"], "confidence": "High", "raw": response}
                - yes_no_maybe: {"answer": "yes", "confidence": "Medium", "raw": response}
        """
        if task_type == "mcq":
            return self._extract_mcq_answer(response)
        elif task_type == "ranking":
            return self._extract_ranking(response)
        elif task_type == "yes_no_maybe":
            return self._extract_yes_no_maybe(response)
        else:
            return {"answer": response, "raw": response}

    def _extract_mcq_answer(self, response: str) -> Dict[str, Any]:
        """Extract MCQ answer (single letter: A, B, C, D, etc.)."""
        # Patterns to match answer
        patterns = [
            r"(?:Final\s+)?Answer:\s*([A-J])\b",  # "Answer: B" or "Final Answer: B"
            r"(?:Final\s+)?Answer:\s*\(?([A-J])\)",  # "Answer: (B)"
            r"^([A-J])\.",  # "B. Some text" at start of line
            r"RANKING:\s*\n\s*1\.\s*([A-J])",  # First in ranking
            r"\b([A-J])\b(?=\s*[-\.\)]|\s*$)"  # Standalone letter
        ]

        answer = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                break

        # Extract confidence if present
        confidence = self._extract_confidence(response)

        return {
            "answer": answer,
            "confidence": confidence,
            "raw": response
        }

    def _extract_ranking(self, response: str) -> Dict[str, Any]:
        """
        Extract ranking from response.

        Expected format:
        RANKING:
        1. B - Explanation
        2. A - Explanation
        3. C - Explanation
        4. D - Explanation
        """
        ranking = []

        # Try to find RANKING: section
        ranking_section = re.search(r"RANKING:\s*(.*?)(?:CONFIDENCE:|EXPLANATION:|$)",
                                   response, re.DOTALL | re.IGNORECASE)

        if ranking_section:
            ranking_text = ranking_section.group(1)

            # Extract ranked options
            # Pattern: "1. B" or "1. Option B" or "1. B -"
            rank_patterns = [
                r"(\d+)\.\s*([A-J])\b",  # "1. B"
                r"(\d+)\.\s*(?:Option\s+)?([A-J])\s*[-:]",  # "1. Option B -"
            ]

            for pattern in rank_patterns:
                matches = re.findall(pattern, ranking_text, re.IGNORECASE)
                if matches:
                    # Sort by rank number and extract letters
                    sorted_matches = sorted(matches, key=lambda x: int(x[0]))
                    ranking = [match[1].upper() for match in sorted_matches]
                    break

        # Fallback: if no RANKING section, try to find first occurrence of options
        if not ranking:
            # Look for sequence of option letters
            letters = re.findall(r"\b([A-J])\b", response)
            if letters:
                # Remove duplicates while preserving order
                seen = set()
                ranking = [l.upper() for l in letters if not (l.upper() in seen or seen.add(l.upper()))]

        # Extract confidence
        confidence = self._extract_confidence(response)

        return {
            "ranking": ranking if ranking else None,
            "confidence": confidence,
            "raw": response
        }

    def _extract_yes_no_maybe(self, response: str) -> Dict[str, Any]:
        """Extract yes/no/maybe answer."""
        patterns = [
            r"(?:Final\s+)?Answer:\s*(yes|no|maybe)",
            r"FINAL\s+ANSWER:\s*(yes|no|maybe)",
            r"\b(yes|no|maybe)\b"
        ]

        answer = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).lower()
                break

        confidence = self._extract_confidence(response)

        return {
            "answer": answer,
            "confidence": confidence,
            "raw": response
        }

    def _extract_confidence(self, response: str) -> Optional[str]:
        """Extract confidence level (High/Medium/Low) from response."""
        confidence_pattern = r"CONFIDENCE:\s*(High|Medium|Low)"
        match = re.search(confidence_pattern, response, re.IGNORECASE)

        if match:
            return match.group(1).capitalize()

        # Alternative patterns
        alt_patterns = [
            r"confidence\s+(?:is\s+)?(?:level:\s*)?(high|medium|low)",
            r"I am (highly|moderately|somewhat) confident",
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                conf_text = match.group(1).lower()
                if conf_text in ["highly", "high"]:
                    return "High"
                elif conf_text in ["moderately", "medium"]:
                    return "Medium"
                elif conf_text in ["somewhat", "low"]:
                    return "Low"

        return None

    def get_round_response(self, round_number: int) -> Optional[str]:
        """Get the stored response for a specific round."""
        round_key = f"round{round_number}"
        return self.responses.get(round_key)

    def get_conversation_history(self, round_number: int = None) -> Dict[str, List]:
        """
        Get conversation history for specific round or all rounds.

        Args:
            round_number: Round to retrieve (1, 2, or 3), or None for all rounds

        Returns:
            Conversation history dict
        """
        if round_number is not None:
            round_key = f"round{round_number}"
            return self.conversation_history.get(round_key, [])
        return self.conversation_history

    def reset_history(self):
        """Reset all conversation history (for reusing agent)."""
        self.conversation_history = {
            "round1": [],
            "round2": [],
            "round3": []
        }
        self.responses = {
            "round1": None,
            "round2": None,
            "round3": None
        }
        logging.info(f"Reset conversation history for {self.agent_id}")

    def get_agent_info(self) -> Dict[str, str]:
        """Get agent identification and role information."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "expertise": self.expertise,
            "model_name": self.model_name,
            "chat_instance_type": self.chat_instance_type
        }

    def __str__(self) -> str:
        return f"GemmaAgent(id={self.agent_id}, role={self.role})"

    def __repr__(self) -> str:
        return f"GemmaAgent(agent_id='{self.agent_id}', role='{self.role}', expertise='{self.expertise}')"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_agent(agent_id: str, role: str, expertise: str, **kwargs) -> GemmaAgent:
    """
    Convenience function to create a GemmaAgent.

    Args:
        agent_id: Unique agent identifier
        role: Medical role/specialty
        expertise: Expertise description
        **kwargs: Additional arguments passed to GemmaAgent constructor

    Returns:
        Initialized GemmaAgent instance
    """
    return GemmaAgent(agent_id, role, expertise, **kwargs)


def create_agents_from_roles(roles: List[Dict[str, str]], **kwargs) -> List[GemmaAgent]:
    """
    Create multiple agents from a list of role dictionaries.

    Args:
        roles: List of dicts with keys: 'agent_id', 'role', 'expertise'
        **kwargs: Additional arguments passed to each GemmaAgent constructor

    Returns:
        List of initialized GemmaAgent instances
    """
    agents = []
    for role_info in roles:
        agent = GemmaAgent(
            agent_id=role_info['agent_id'],
            role=role_info['role'],
            expertise=role_info['expertise'],
            **kwargs
        )
        agents.append(agent)
    return agents


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "GemmaAgent",
    "create_agent",
    "create_agents_from_roles",
]
