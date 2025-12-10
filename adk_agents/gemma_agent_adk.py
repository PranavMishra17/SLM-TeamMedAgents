"""
Gemma Agent Factory for Google ADK

Wraps Gemma models via Google AI Studio API with ADK's Agent class.
Provides factory methods for creating specialized medical reasoning agents.

Key Features:
- Uses Google AI Studio API only (same as existing system)
- Automatic rate limiting via model configuration
- Answer extraction from agent responses
- Memory management via session.state

Usage:
    agent = GemmaAgentFactory.create_agent(
        name="cardiologist",
        role="Cardiologist",
        expertise="Cardiovascular disease and ECG interpretation",
        model_name="gemma3_4b"
    )
"""

import os
import logging
import re
from typing import Optional

try:
    from google.adk.agents import Agent
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.warning("Google ADK not installed. Install with: pip install google-adk")

# Import Gemma model (for Google AI Studio)
try:
    from google.adk.models import Gemma
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    logging.warning("Gemma model not available in ADK")

# Optional: Vertex AI agent factory fallback. If the environment indicates
# Vertex AI should be used, we will delegate agent creation to
# VertexAIAgentFactory defined in `gemma_agent_vertex_adk.py`.
try:
    from .gemma_agent_vertex_adk import VertexAIAgentFactory
    VERTEX_FACTORY_AVAILABLE = True
except Exception:
    VERTEX_FACTORY_AVAILABLE = False


class GemmaAgentFactory:
    """
    Factory for creating ADK-based Gemma agents using Google AI Studio.

    This factory configures Gemma models via Google AI Studio API,
    matching the existing system's backend while leveraging ADK framework.
    """

    # Model name mappings (internal name → Google AI Studio model name)
    # Available models: gemma-3-1b-it, gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it
    # Note: Gemma models can handle vision tasks through ADK artifact system
    MODEL_MAPPINGS = {
        'gemma3_4b': 'gemma-3-4b-it',      # Gemma 3 4B - PRIMARY SLM for research
        'gemma3_1b': 'gemma-3-1b-it',      # Gemma 3 1B - smallest SLM
        'gemma3_12b': 'gemma-3-12b-it',    # Gemma 3 12B
        'gemma3_27b': 'gemma-3-27b-it',    # Gemma 3 27B
        'gemma3n_e4b': 'gemma-3n-e4b-it',  # Gemma 3 Nano 4B efficient
        'gemma3n_e2b': 'gemma-3n-e2b-it',  # Gemma 3 Nano 2B efficient
        'gemma2_9b': 'gemma-3-12b-it',     # Fallback: Gemma 2 9B -> Gemma 3 12B
        'gemma2_27b': 'gemma-3-27b-it',    # Fallback: Gemma 2 27B -> Gemma 3 27B
    }

    @staticmethod
    def create_agent(
        name: str,
        role: str,
        expertise: str,
        model_name: str = 'gemma3_4b',
        temperature: float = 0.7,
        max_tokens: int = 2048,
        has_image: bool = False,
        **kwargs
    ) -> 'Agent':
        """
        Create an ADK Agent configured with Gemma model via Google AI Studio.

        Args:
            name: Unique agent identifier
            role: Agent's role (e.g., "Cardiologist")
            expertise: Domain expertise description
            model_name: Model to use (gemma3_4b, gemma2_9b, gemma2_27b, medgemma_4b)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            has_image: If True, agent will use ADK artifacts for image access
            **kwargs: Additional model-specific parameters

        Returns:
            Agent instance configured with Gemma model

        Raises:
            RuntimeError: If ADK not installed or API key not set
        """
        if not ADK_AVAILABLE:
            raise RuntimeError("Google ADK not installed. Run: pip install google-adk")

        # If the environment requests Vertex AI usage, delegate creation to VertexAIAgentFactory.
        use_vertex = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper() == 'TRUE'
        if use_vertex and VERTEX_FACTORY_AVAILABLE:
            logging.info("GOOGLE_GENAI_USE_VERTEXAI=TRUE -> delegating agent creation to VertexAIAgentFactory")
            # Vertex factory will read endpoint/project from env vars if not provided
            return VertexAIAgentFactory.create_agent(
                name=name,
                role=role,
                expertise=expertise,
                endpoint_id=None,
                project_id=None,
                location=None,
                temperature=temperature,
                max_tokens=max_tokens,
                has_image=has_image,
                **kwargs
            )

        # Get API key
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable required"
            )

        # Map model name to actual Gemma model
        mapped_name = GemmaAgentFactory.MODEL_MAPPINGS.get(model_name, model_name)

        if has_image:
            logging.info(f"Creating agent with Gemma model (artifact image support): {model_name} -> {mapped_name}")
        else:
            logging.info(f"Creating agent with Gemma model: {model_name} -> {mapped_name}")

        # Create Gemma model instance
        if GEMMA_AVAILABLE:
            model = Gemma(
                model=mapped_name
            )
        else:
            raise RuntimeError("Gemma model not available in ADK. Check ADK installation.")

        # Create instruction
        instruction = GemmaAgentFactory._build_instruction(role, expertise, has_image)

        # Create and return Agent
        agent = Agent(
            name=name,
            model=model,
            description=f"{role} - {expertise}",
            instruction=instruction
        )

        # Store model configuration on agent for multimodal bypass
        # This allows three_round_debate_adk.py to use google.genai client directly
        agent._gemma_config = {
            'model_name': mapped_name,
            'api_key': api_key,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        logging.info(f"Created ADK agent '{name}' with GEMMA model: {mapped_name}")
        logging.debug(f"Stored model config on agent for multimodal support")
        return agent

    @staticmethod
    def _build_instruction(role: str, expertise: str, has_image: bool = False) -> str:
        """Build system instruction for medical reasoning agent."""
        base_instruction = f"""You are a {role} with expertise in {expertise}. Your job is to collaborate with other medical experts in a team.

Guidelines:
- Provide evidence-based medical reasoning
- Be precise with medical terminology
- Acknowledge uncertainty when appropriate
- Consider patient safety and best practices
- Collaborate constructively with other specialists
- Actively engage with peer opinions and reasoning
- Deliver your opinions in a way to convince other experts with clear reasoning
- Focus on clinical accuracy

When analyzing medical questions:
1. Consider only information explicitly provided
2. Apply systematic clinical reasoning
3. Evaluate all options carefully
4. Provide clear justification for your conclusions
"""

        if has_image:
            vision_addendum = """
IMAGE ANALYSIS INSTRUCTIONS:
- An image artifact is available for this medical case
- Analyze the image artifact thoroughly when provided
- Describe relevant visual findings systematically
- Integrate visual observations with clinical knowledge
- Be specific about anatomical structures and pathology
- Do NOT fabricate or hallucinate visual details
- If the image quality is poor, acknowledge limitations
"""
            return base_instruction + vision_addendum

        return base_instruction

    @staticmethod
    def extract_answer(response: str, task_type: str = "mcq") -> Optional[str]:
        """
        Extract answer from agent response text.

        Args:
            response: Full agent response text
            task_type: Type of task (mcq, yes_no_maybe, ranking)

        Returns:
            Extracted answer (letter for MCQ, yes/no/maybe, or ranking list)
        """
        if task_type == "mcq":
            # Look for patterns like "Answer: A" or "ANSWER: B"
            patterns = [
                r"(?:Final\s+)?(?:ANSWER|Answer):\s*([A-J])\b",
                r"(?:Final\s+)?(?:ANSWER|Answer):\s*\(?([A-J])\)",
                r"^([A-J])\.",
                r"\b([A-J])\b(?=\s*[-\.\)]|\s*is\s+(?:correct|the|most))"
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()

        elif task_type == "yes_no_maybe":
            pattern = r"(?:Final\s+)?(?:ANSWER|Answer):\s*(yes|no|maybe)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        elif task_type == "ranking":
            # Extract ranked list
            pattern = r'RANKING:\s*\n((?:\d+\.\s*[A-Z].*\n?)+)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                lines = match.group(1).strip().split('\n')
                ranking = []
                for line in lines:
                    letter_match = re.search(r'([A-Z])', line)
                    if letter_match:
                        ranking.append(letter_match.group(1))
                return ranking if ranking else None

        return None

    @staticmethod
    def extract_confidence(response: str) -> Optional[str]:
        """Extract confidence level from agent response."""
        pattern = r'(?:CONFIDENCE|Confidence):\s*(High|Medium|Low)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        return None


# Convenience function for backward compatibility
def create_gemma_agent(
    name: str,
    role: str,
    expertise: str,
    **kwargs
) -> 'Agent':
    """
    Convenience function to create Gemma agent with default settings.

    This function provides a simple interface matching the old GemmaAgent API.
    """
    return GemmaAgentFactory.create_agent(
        name=name,
        role=role,
        expertise=expertise,
        **kwargs
    )


__all__ = ['GemmaAgentFactory', 'create_gemma_agent']
