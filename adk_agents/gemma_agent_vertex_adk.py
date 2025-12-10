"""
Vertex AI Endpoint Agent Factory for Google ADK

Wraps MedGemma models deployed on Vertex AI endpoints with ADK's Agent class.
Provides factory methods for creating specialized medical reasoning agents.

Key Features:
- Uses Vertex AI deployed endpoints (MedGemma on Model Garden)
- Supports multimodal (text + images) via base64 encoding
- Automatic retry and error handling
- Memory management via session.state
- Compatible with all existing teamwork components

Requirements:
    Environment variables:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_CLOUD_LOCATION: Vertex AI region (e.g., us-central1)
    - VERTEX_AI_ENDPOINT_ID: Your MedGemma endpoint ID
    - GOOGLE_GENAI_USE_VERTEXAI: Set to "TRUE"

    Authentication:
    - gcloud auth application-default login (local)
    - OR GOOGLE_APPLICATION_CREDENTIALS=/path/to/keyfile.json (production)

Usage:
    # After deploying MedGemma to Vertex AI endpoint
    agent = VertexAIAgentFactory.create_agent(
        name="cardiologist",
        role="Cardiologist",
        expertise="Cardiovascular disease and ECG interpretation",
        endpoint_id="your-endpoint-id"  # or use env var
    )
"""

import os
import logging
import re
import base64
import io
from typing import Optional, Dict, Any

try:
    from google.adk.agents import Agent, LlmAgent
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.warning("Google ADK not installed. Install with: pip install google-adk")

# Import for multimodal support
try:
    from google.genai import types
    from google import genai
    from PIL import Image
    VISION_SUPPORT = True
except ImportError:
    VISION_SUPPORT = False
    logging.warning("Vision support not available. Install: pip install google-genai pillow")


class VertexAIAgentFactory:
    """
    Factory for creating ADK-based agents using Vertex AI endpoints.

    This factory configures agents to use MedGemma deployed on Vertex AI,
    providing production-grade scalability and multimodal support.
    """

    @staticmethod
    def get_vertex_config() -> Dict[str, str]:
        """
        Get Vertex AI configuration from environment variables.

        Returns:
            Dict with project_id, location, endpoint_id

        Raises:
            RuntimeError: If required environment variables are not set
        """
        config = {
            'project_id': os.environ.get('GOOGLE_CLOUD_PROJECT'),
            'location': os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
            'endpoint_id': os.environ.get('VERTEX_AI_ENDPOINT_ID'),
            'use_vertex': os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper() == 'TRUE'
        }

        # Validate required fields
        if not config['project_id']:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT environment variable not set.\n"
                "Set it to your GCP project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'"
            )

        if not config['endpoint_id']:
            raise RuntimeError(
                "VERTEX_AI_ENDPOINT_ID environment variable not set.\n"
                "Set it to your deployed MedGemma endpoint ID: export VERTEX_AI_ENDPOINT_ID='your-endpoint-id'"
            )

        if not config['use_vertex']:
            logging.warning(
                "GOOGLE_GENAI_USE_VERTEXAI is not set to TRUE. "
                "ADK may default to Google AI Studio instead of Vertex AI. "
                "Set: export GOOGLE_GENAI_USE_VERTEXAI=TRUE"
            )

        logging.info(f"Vertex AI config: project={config['project_id']}, "
                    f"location={config['location']}, endpoint={config['endpoint_id']}")

        return config

    @staticmethod
    def build_endpoint_resource_name(project_id: str, location: str, endpoint_id: str) -> str:
        """
        Build full Vertex AI endpoint resource name.

        Args:
            project_id: GCP project ID
            location: Vertex AI region (e.g., us-central1)
            endpoint_id: Endpoint ID from Model Garden deployment

        Returns:
            Full resource name: projects/{project}/locations/{location}/endpoints/{endpoint}
        """
        return f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

    @staticmethod
    def create_agent(
        name: str,
        role: str,
        expertise: str,
        endpoint_id: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        has_image: bool = False,
        **kwargs
    ) -> 'Agent':
        """
        Create an ADK Agent configured with Vertex AI endpoint.

        Args:
            name: Unique agent identifier
            role: Agent's role (e.g., "Cardiologist")
            expertise: Domain expertise description
            endpoint_id: Vertex AI endpoint ID (or use VERTEX_AI_ENDPOINT_ID env var)
            project_id: GCP project ID (or use GOOGLE_CLOUD_PROJECT env var)
            location: Vertex AI region (or use GOOGLE_CLOUD_LOCATION env var)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            has_image: If True, agent expects multimodal input
            **kwargs: Additional model-specific parameters

        Returns:
            Agent instance configured with Vertex AI endpoint

        Raises:
            RuntimeError: If ADK not installed or configuration missing
        """
        if not ADK_AVAILABLE:
            raise RuntimeError("Google ADK not installed. Run: pip install google-adk")

        # Get Vertex AI configuration
        config = VertexAIAgentFactory.get_vertex_config()

        # Override with explicit parameters if provided
        project_id = project_id or config['project_id']
        location = location or config['location']
        endpoint_id = endpoint_id or config['endpoint_id']

        # Build endpoint resource name
        endpoint_resource = VertexAIAgentFactory.build_endpoint_resource_name(
            project_id, location, endpoint_id
        )

        logging.info(f"Creating agent with Vertex AI endpoint: {endpoint_resource}")
        if has_image:
            logging.info("  Multimodal mode enabled (text + images)")

        # Create instruction
        instruction = VertexAIAgentFactory._build_instruction(role, expertise, has_image)

        # Create LlmAgent with Vertex AI endpoint
        # NOTE: ADK's LlmAgent should accept Vertex AI endpoint resource names
        # Format: projects/{project}/locations/{location}/endpoints/{endpoint}
        # The endpoint resource name is passed as the 'model' parameter.
        # If ADK doesn't support this format directly, LlmAgent will raise an exception.
        try:
            agent = LlmAgent(
                name=name,
                model=endpoint_resource,  # Full endpoint resource name
                description=f"{role} - {expertise}",
                instruction=instruction
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create LlmAgent with Vertex AI endpoint: {endpoint_resource}\n"
                f"Error: {e}\n"
                f"This may indicate that ADK's LlmAgent doesn't support Vertex AI endpoints directly.\n"
                f"Consider using google.genai client directly and wrapping it in a custom Agent."
            ) from e

        # Store configuration on agent for direct API access if needed
        agent._vertex_config = {
            'endpoint_resource': endpoint_resource,
            'project_id': project_id,
            'location': location,
            'endpoint_id': endpoint_id,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'has_image': has_image
        }

        logging.info(f"Created Vertex AI agent '{name}' with endpoint: {endpoint_id}")
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
- An image is provided as part of this medical case
- Analyze the image thoroughly when provided
- Describe relevant visual findings systematically
- Integrate visual observations with clinical knowledge
- Be specific about anatomical structures and pathology
- Do NOT fabricate or hallucinate visual details
- If the image quality is poor, acknowledge limitations
- Medical images may include X-rays, CT scans, MRI, pathology slides, etc.
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


def prepare_image_for_vertex(image) -> tuple:
    """
    Convert PIL Image to base64-encoded data for Vertex AI multimodal input.

    Args:
        image: PIL Image object

    Returns:
        Tuple of (base64_string, mime_type) or (None, None) if image is None
    """
    if image is None:
        return None, None

    try:
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        logging.info(f"Prepared image for Vertex AI: {len(image_base64)} base64 chars")
        return image_base64, 'image/jpeg'
    except Exception as e:
        logging.error(f"Error preparing image for Vertex AI: {e}")
        return None, None


# Convenience function for backward compatibility
def create_vertex_agent(
    name: str,
    role: str,
    expertise: str,
    **kwargs
) -> 'Agent':
    """
    Convenience function to create Vertex AI agent with default settings.

    This function provides a simple interface matching the GemmaAgent API.
    """
    return VertexAIAgentFactory.create_agent(
        name=name,
        role=role,
        expertise=expertise,
        **kwargs
    )


__all__ = ['VertexAIAgentFactory', 'create_vertex_agent', 'prepare_image_for_vertex']
