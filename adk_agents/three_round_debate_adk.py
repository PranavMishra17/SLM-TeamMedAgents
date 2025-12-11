"""
Three Round Debate Orchestrator for Google ADK

Custom BaseAgent implementing 3-round collaborative medical reasoning:
- Round 1: Independent analysis (parallel)
- Round 2: Collaborative discussion (sequential with shared state)
- Round 3: Final ranking (sequential with full context)

Key Features:
- Uses ADK ParallelAgent for Round 1 concurrency
- State sharing via session.state
- Token optimization (context summarization)
- Handles "EXCEPT" questions properly
- Integrates prompt templates from utils/prompts.py

Usage:
    debate = ThreeRoundDebateAgent()

    # Agents must be in session.state from recruiter
    session.state['recruited_agents'] = [...]
    session.state['question'] = "..."
    session.state['options'] = [...]

    async for event in debate.run_async(session):
        print(event.content)

    # Results stored in session.state
    round1_results = session.state['round1_results']
    round3_rankings = session.state['round3_rankings']
"""

import logging
import re
import asyncio
import random
import base64
import io
import os
from typing import AsyncGenerator, Dict, List, Any
from pathlib import Path

try:
    from google.adk.agents import BaseAgent, ParallelAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.error("Google ADK not installed")

try:
    from google.genai import types
    from google import genai
    from PIL import Image
    VISION_SUPPORT = True
except ImportError:
    VISION_SUPPORT = False
    logging.warning("Vision support not available")

# Import prompts from existing system
try:
    from utils.prompts import (
        get_round1_prompt,
        get_round2_prompt,
        get_round3_prompt
    )
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    logging.warning("utils.prompts not available, using fallback prompts")


def encode_image_to_base64(image) -> str:
    """
    Encode PIL Image to base64 string for ADK.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded image string
    """
    if image is None:
        return None

    try:
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)

        # Encode to base64
        image_bytes = buffer.read()
        base64_str = base64.b64encode(image_bytes).decode('utf-8')

        return base64_str
    except Exception as e:
        logging.error(f"Error encoding image: {e}")
        return None


def prepare_image_for_multimodal(image) -> tuple:
    """
    Convert PIL Image to base64-encoded data for multimodal Gemma models.

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

        logging.info(f"Prepared image for multimodal input: {len(image_base64)} base64 chars")
        return image_base64, 'image/jpeg'
    except Exception as e:
        logging.error(f"Error preparing image for multimodal: {e}")
        return None, None


def save_image_as_artifact(ctx: InvocationContext, image, artifact_name: str = "medical_image.jpg") -> bool:
    """
    Save PIL Image as ADK artifact using tool_context.

    Args:
        ctx: ADK InvocationContext
        image: PIL Image object
        artifact_name: Filename for the artifact

    Returns:
        True if successful, False otherwise
    """
    if image is None:
        return False

    try:
        # Convert PIL Image to bytes
        buffer = io.BytesIO()
        # Convert to RGB if needed (JPEG doesn't support RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # Create a Part object from image bytes
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg'
        )

        # Save as artifact using tool_context
        # Note: ADK agents can access artifacts through tool_context
        if hasattr(ctx, 'tool_context') and ctx.tool_context:
            ctx.tool_context.save_artifact(artifact_name, image_part)
            logging.info(f"Saved image as artifact: {artifact_name}")
            return True
        else:
            # Fallback: save to session state for agents to access
            ctx.session.state['image_artifact'] = {
                'name': artifact_name,
                'data': image_bytes,
                'mime_type': 'image/jpeg'
            }
            logging.info(f"Saved image data in session state (artifact fallback): {artifact_name}")
            return True

    except Exception as e:
        logging.error(f"Error saving image as artifact: {e}")
        return False


async def execute_multimodal_gemma(
    model_config: Dict[str, Any],
    text_prompt: str,
    image_base64: str,
    image_mime_type: str = 'image/jpeg'
) -> str:
    """
    Direct multimodal call to Gemma/MedGemma using google.genai client, bypassing ADK Agent wrapper.

    This function works around ADK's Agent interface which doesn't properly expose
    multimodal capabilities. We use google.genai client directly for both Google AI Studio
    and Vertex AI endpoints.

    Args:
        model_config: Dictionary containing:
            - model_name: Model name or endpoint resource name
            - api_key: API key (Google AI Studio only, None for Vertex AI)
            - temperature: Sampling temperature
            - max_tokens: Max output tokens
            - use_vertex: True if using Vertex AI endpoint
        text_prompt: Text prompt for the agent
        image_base64: Base64-encoded image data
        image_mime_type: MIME type of the image (default: 'image/jpeg')

    Returns:
        Response text from the model

    Raises:
        Exception: If multimodal execution fails
    """
    if not VISION_SUPPORT:
        raise RuntimeError("Vision support not available - install google-genai")

    client = None
    try:
        model_name = model_config.get('model_name')
        use_vertex = model_config.get('use_vertex', False)

        logging.debug(f"[MULTIMODAL DIRECT] Calling {'Vertex AI' if use_vertex else 'Google AI'} model: {model_name}")
        logging.debug(f"[MULTIMODAL DIRECT] Text prompt length: {len(text_prompt)} chars")
        logging.debug(f"[MULTIMODAL DIRECT] Image base64 length: {len(image_base64)} chars")

        # Create google.genai client
        if use_vertex:
            # Vertex AI: Use Application Default Credentials
            logging.debug(f"[MULTIMODAL DIRECT] Using Vertex AI with ADC")
            # For Vertex AI, genai.Client will automatically use ADC when no api_key is provided
            # and GOOGLE_GENAI_USE_VERTEXAI=TRUE is set in environment
            client = genai.Client(
                vertexai=True,
                project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
                location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
            )
        else:
            # Google AI Studio: Use API key
            api_key = model_config.get('api_key')
            if not api_key:
                raise RuntimeError("API key not found in model_config for Google AI Studio")
            client = genai.Client(api_key=api_key)

        # Decode base64 image to bytes
        image_bytes = base64.b64decode(image_base64)

        # Create multimodal content with image + text
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(
                        mime_type=image_mime_type,
                        data=image_bytes
                    ),
                    types.Part.from_text(text=text_prompt)
                ]
            )
        ]

        # Create generation config
        config = types.GenerateContentConfig(
            temperature=model_config.get('temperature', 0.7),
            max_output_tokens=model_config.get('max_tokens', 2048)
        )

        # Call model with multimodal content
        logging.debug(f"[MULTIMODAL DIRECT] Sending request to model...")
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=contents,
            config=config
        )

        # Extract text from response
        response_text = response.text if hasattr(response, 'text') else str(response)

        logging.info(f"[MULTIMODAL DIRECT] Successfully received response: {len(response_text)} chars")
        return response_text

    except Exception as e:
        logging.error(f"[MULTIMODAL DIRECT] Error in multimodal execution: {type(e).__name__}: {e}")
        logging.error(f"[MULTIMODAL DIRECT] Model: {model_config.get('model_name')}")
        logging.error(f"[MULTIMODAL DIRECT] Use Vertex: {model_config.get('use_vertex')}")
        logging.error(f"[MULTIMODAL DIRECT] Image MIME: {image_mime_type}")
        raise

    finally:
        # Clean up client resources to avoid unclosed session warnings
        if client:
            try:
                await client.aio.close()
            except Exception as cleanup_error:
                logging.debug(f"[MULTIMODAL DIRECT] Client cleanup note: {cleanup_error}")
                pass  # Ignore cleanup errors


async def retry_with_exponential_backoff(async_gen_func, max_retries=5):
    """
    Wrapper to retry async generator functions with exponential backoff for rate limits.

    Args:
        async_gen_func: Async generator function to retry
        max_retries: Maximum number of retry attempts

    Yields:
        Events from the async generator
    """
    for attempt in range(max_retries + 1):
        try:
            async for event in async_gen_func():
                yield event
            return  # Success, exit
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = any(term in error_msg for term in ['429', 'resource_exhausted', 'quota', 'rate limit'])

            if is_rate_limit and attempt < max_retries:
                # Extract retry delay from error message
                retry_delay = None
                if 'retry in' in error_msg:
                    match = re.search(r'retry in ([\d.]+)s', error_msg)
                    if match:
                        retry_delay = float(match.group(1))

                if retry_delay:
                    delay = retry_delay
                    logging.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries+1}), API suggested waiting {delay:.1f}s")
                else:
                    # Exponential backoff: 60s base
                    delay = min(60 * (2 ** attempt), 300)
                    jitter = random.uniform(0.8, 1.2)
                    delay *= jitter
                    logging.warning(f"Rate limit error (attempt {attempt+1}/{max_retries+1}), waiting {delay:.1f}s")

                await asyncio.sleep(delay)
                continue
            else:
                # Non-retryable error or max retries exceeded
                if is_rate_limit:
                    logging.error(f"Rate limit error after {max_retries+1} attempts, giving up")
                raise


def extract_text_from_content(content) -> str:
    """Extract text from ADK Content object."""
    if isinstance(content, str):
        return content
    if hasattr(content, 'parts') and content.parts:
        text_parts = [part.text for part in content.parts if hasattr(part, 'text') and part.text]
        return ''.join(text_parts)
    return str(content)


class ThreeRoundDebateAgent(BaseAgent):
    """
    Custom ADK BaseAgent orchestrating 3-round collaborative medical reasoning.

    This agent coordinates the entire debate process:
    1. Round 1: Each agent analyzes independently (parallel)
    2. Round 2: Agents discuss with awareness of others' R1 views (sequential)
    3. Round 3: Final ranking with full debate history (sequential)

    All communication happens via session.state, following ADK patterns.
    """

    name: str = "three_round_debate"
    description: str = "Orchestrates 3-round collaborative medical reasoning"
    model_config = {'extra': 'allow'}  # Allow custom attributes

    def __init__(self, teamwork_config=None, **kwargs):
        super().__init__(**kwargs)
        self.teamwork_config = teamwork_config
        logging.info(f"Initialized ThreeRoundDebateAgent (teamwork: {teamwork_config.get_active_components() if teamwork_config else 'None'})")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Main execution logic for 3-round debate.

        Expects session.state to contain:
        - recruited_agents: List of agent dicts with 'agent', 'agent_id', 'role', 'expertise'
        - question: Question text
        - options: List of options
        - task_type: Type of task (mcq, yes_no_maybe, etc.)
        """
        # Initialize API call counter
        if 'api_call_count' not in ctx.session.state:
            ctx.session.state['api_call_count'] = 0

        # Get inputs from session state
        recruited_agents = ctx.session.state.get('recruited_agents', [])
        question = ctx.session.state.get('question', '')
        options = ctx.session.state.get('options', [])
        task_type = ctx.session.state.get('task_type', 'mcq')
        image = ctx.session.state.get('image')  # PIL Image object
        dataset = ctx.session.state.get('dataset', '')

        if not recruited_agents or not question:
            logging.error("ERROR: Missing recruited_agents or question in session.state")
            return

        # Detect if this is an "EXCEPT" question
        is_except_question = any(term in question.lower() for term in
                                ['except', 'not true', 'false', 'incorrect', 'is not'])

        if is_except_question:
            logging.info("[DETECTED: 'EXCEPT' question - will rank from MOST FALSE to MOST TRUE]")
            ctx.session.state['is_except_question'] = True
        else:
            ctx.session.state['is_except_question'] = False

        # Validate image availability (for hallucination prevention)
        has_valid_image = image is not None
        question_mentions_image = any(term in question.lower() for term in
                                     ['image', 'shown', 'figure', 'photograph', 'picture'])

        # Expanded list of vision datasets
        vision_datasets = ['path-vqa', 'pmc-vqa', 'pathvqa', 'pmcvqa', 'slake', 'rad', 'vqa-rad', 'vqarad']

        # Only log warning for datasets that actually have images
        if question_mentions_image and not has_valid_image:
            if dataset and any(d in dataset.lower() for d in vision_datasets):
                logging.warning(f"Question mentions image but no valid image provided for vision dataset '{dataset}' - adding constraint")
            else:
                logging.debug(f"Question mentions 'image' but dataset '{dataset}' is text-only - no constraint needed")

        # Prepare image for multimodal Gemma models
        image_base64 = None
        image_mime_type = None
        if has_valid_image:
            logging.info(f"[MULTIMODAL MODE] Preparing image for dataset: {dataset}")
            image_base64, image_mime_type = prepare_image_for_multimodal(image)
            if image_base64:
                logging.info(f"Image prepared for multimodal processing: {len(image_base64)} chars")
                # Store base64 image in session state for agents to access
                ctx.session.state['image_base64'] = image_base64
                ctx.session.state['image_mime_type'] = image_mime_type
            else:
                logging.warning("Failed to prepare image for multimodal processing")
                has_valid_image = False

        # Execute 3 rounds with modular teamwork integration
        # Round 1 & 2: Initial predictions
        await self._execute_round1(ctx, recruited_agents, question, options, task_type,
                                   has_valid_image, question_mentions_image, dataset)

        await self._execute_round2(ctx, recruited_agents, question, options, task_type)

        # ========== PHASE 2: POST-R2 PROCESSING ==========
        # [SMM] Extract verified facts
        # [TeamO] Create formal medical report
        # [Trust] Evaluate R2 quality
        await self._post_r2_processing(ctx, recruited_agents)

        # ========== PHASE 3: COLLABORATIVE DISCUSSION (R3) ==========
        await self._execute_round3(ctx, recruited_agents, question, options, task_type,
                                   is_except_question, has_valid_image, question_mentions_image, dataset)

        logging.info("=== THREE ROUND DEBATE COMPLETE ===")

        # Yield event to trigger session state persistence
        from google.genai import types
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text="Three-round debate complete")])
        )

    async def _post_r2_processing(
        self,
        ctx: InvocationContext,
        recruited_agents: List[Dict]
    ) -> None:
        """
        Post-R2 Processing: Extract facts, create reports, evaluate trust.

        Combined API call handling:
        - [SMM] Extract verified facts from R2 responses
        - [TeamO] Leader creates formal medical report
        - [Trust] Evaluate R2 response quality

        Args:
            ctx: Invocation context
            recruited_agents: List of recruited agent dicts
        """
        if not self.teamwork_config:
            logging.debug("[Post-R2] No teamwork config, skipping")
            return

        logging.info("\n=== POST-R2 PROCESSING ===")

        round2_results = ctx.session.state.get('round2_results', {})
        if not round2_results:
            logging.warning("[Post-R2] No R2 results to process")
            return

        # Get teamwork components
        smm = ctx.session.state.get('smm', None)
        leadership_coord = ctx.session.state.get('leadership_coordinator', None)
        team_o_manager = ctx.session.state.get('team_orientation_manager', None)
        trust_network = ctx.session.state.get('trust_network', None)

        # Convert R2 results to response format for processing
        # (Parse facts from responses - simplified extraction)
        agent_responses = {}
        for agent_id, response_text in round2_results.items():
            # Extract facts from response text (simple regex)
            import re
            facts = []
            # Look for "FACTS:" section or numbered lists
            facts_section = re.search(r'FACTS?:\s*\n((?:.+\n?)+)', response_text, re.IGNORECASE)
            if facts_section:
                fact_lines = facts_section.group(1).strip().split('\n')
                for line in fact_lines:
                    if line.strip():
                        facts.append(line.strip())

            agent_responses[agent_id] = {
                'justification': response_text[:200],  # First 200 chars
                'facts': facts[:5],  # Up to 5 facts
                'answer': None  # Not extracted in R2
            }

        # [SMM] Extract verified facts
        if self.teamwork_config.smm and smm:
            logging.info("[SMM] Extracting verified facts from R2 responses...")

            if self.teamwork_config.leadership and leadership_coord:
                # Leader extracts facts
                verified_facts = await leadership_coord.extract_verified_facts_from_responses(
                    ctx, agent_responses, smm
                )
            else:
                # Automated extraction
                from teamwork_components import extract_facts_intersection
                verified_facts = extract_facts_intersection(agent_responses)

            if verified_facts:
                smm.add_verified_facts(verified_facts)
                logging.info(f"[SMM] Added {len(verified_facts)} verified facts")

        # [TeamO] Create formal medical report
        if self.teamwork_config.team_orientation and self.teamwork_config.leadership:
            if leadership_coord and team_o_manager:
                logging.info("[TeamO] Creating formal medical report...")
                formal_report = await leadership_coord.create_formal_medical_report(
                    ctx, agent_responses, smm
                )
                ctx.session.state['formal_medical_report'] = formal_report
                logging.info(f"[TeamO] Formal report created ({len(formal_report)} chars)")

        # [Trust] Evaluate R2 quality
        if self.teamwork_config.trust and trust_network:
            logging.info("[Trust] Evaluating R2 response quality...")
            ground_truth = ctx.session.state.get('ground_truth', None)
            trust_network.update_after_round2(agent_responses, ground_truth)

            trust_scores = trust_network.get_all_trust_scores()
            logging.info(f"[Trust] Updated scores: {trust_scores}")

        logging.info("=== POST-R2 PROCESSING COMPLETE ===")

    async def _execute_agent_with_image(
        self,
        ctx: InvocationContext,
        agent,
        prompt: str,
        image_base64: str = None,
        image_mime_type: str = None
    ) -> str:
        """
        Execute an agent with optional multimodal image input.

        This method detects if the agent has multimodal capability (via _gemma_config)
        and uses direct google.genai client to bypass ADK's Agent wrapper, which
        doesn't properly expose multimodal capabilities.

        Args:
            ctx: Invocation context
            agent: ADK Agent instance
            prompt: Text prompt for the agent
            image_base64: Base64-encoded image data (optional)
            image_mime_type: MIME type of the image (optional)

        Returns:
            Agent's response text
        """
        # Increment API call counter
        ctx.session.state['api_call_count'] = ctx.session.state.get('api_call_count', 0) + 1
        logging.debug(f"API Call #{ctx.session.state['api_call_count']}: {agent.name}")

        response_text = ""

        # Check if this is a multimodal call and agent has config for direct API access
        has_image = image_base64 and image_mime_type
        has_gemma_config = hasattr(agent, '_gemma_config') and agent._gemma_config
        has_vertex_config = hasattr(agent, '_vertex_config') and agent._vertex_config

        if has_image and (has_gemma_config or has_vertex_config):
            # MULTIMODAL PATH: Bypass ADK Agent wrapper and use google.genai client directly
            model_config = agent._gemma_config if has_gemma_config else agent._vertex_config

            # For Vertex AI, convert _vertex_config to format expected by execute_multimodal_gemma
            if has_vertex_config:
                logging.info(f"[{agent.name}] Using direct multimodal Vertex AI API (bypassing ADK)")
                # Convert Vertex AI config to model_config format
                model_config = {
                    'model_name': model_config.get('endpoint_resource'),
                    'api_key': None,  # Vertex AI uses ADC, not API key
                    'temperature': model_config.get('temperature', 0.7),
                    'max_tokens': model_config.get('max_tokens', 2048),
                    'use_vertex': True
                }
            else:
                logging.info(f"[{agent.name}] Using direct multimodal Gemma API (bypassing ADK)")

            try:
                # Use direct multimodal call with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response_text = await execute_multimodal_gemma(
                            model_config=model_config,
                            text_prompt=prompt,
                            image_base64=image_base64,
                            image_mime_type=image_mime_type
                        )
                        logging.info(f"[{agent.name}] Multimodal response received: {len(response_text)} chars")
                        break  # Success

                    except Exception as e:
                        error_msg = str(e).lower()
                        is_rate_limit = any(term in error_msg for term in ['429', 'resource_exhausted', 'quota', 'rate limit'])

                        if is_rate_limit and attempt < max_retries - 1:
                            # Retry with backoff
                            delay = min(60 * (2 ** attempt), 300)
                            logging.warning(f"[{agent.name}] Rate limit in multimodal call (attempt {attempt+1}/{max_retries}), waiting {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            # Non-retryable or max retries
                            raise

            except Exception as e:
                # Multimodal call failed completely - fall back to text-only
                logging.error(f"[{agent.name}] Multimodal execution failed after retries: {type(e).__name__}: {e}")
                logging.warning(f"[{agent.name}] Falling back to text-only ADK agent execution")

                try:
                    async for event in retry_with_exponential_backoff(lambda: agent.run_async(ctx)):
                        if hasattr(event, 'content') and event.content:
                            response_text += extract_text_from_content(event.content)
                except Exception as text_error:
                    logging.error(f"[{agent.name}] Text-only fallback also failed: {text_error}")
                    response_text = f"ERROR: Could not execute agent - {str(text_error)}"

        elif has_image and not (has_gemma_config or has_vertex_config):
            # Image provided but no direct API config - warn and fall back to text
            logging.warning(f"[{agent.name}] Image provided but agent lacks _gemma_config/_vertex_config, falling back to text-only")
            logging.warning(f"[{agent.name}] This may result in hallucinated visual observations!")

            try:
                async for event in retry_with_exponential_backoff(lambda: agent.run_async(ctx)):
                    if hasattr(event, 'content') and event.content:
                        response_text += extract_text_from_content(event.content)
            except Exception as e:
                logging.error(f"[{agent.name}] Text-only execution failed: {e}")
                response_text = f"ERROR: Could not execute agent - {str(e)}"

        else:
            # TEXT-ONLY PATH: Standard ADK agent execution
            logging.debug(f"[{agent.name}] Using standard text-only ADK agent execution")

            try:
                async for event in retry_with_exponential_backoff(lambda: agent.run_async(ctx)):
                    if hasattr(event, 'content') and event.content:
                        response_text += extract_text_from_content(event.content)
            except Exception as e:
                logging.error(f"[{agent.name}] Text-only execution failed: {e}")
                response_text = f"ERROR: Could not execute agent - {str(e)}"

        return response_text

    async def _execute_round1(
        self,
        ctx: InvocationContext,
        recruited_agents: List[Dict],
        question: str,
        options: List[str],
        task_type: str,
        has_valid_image: bool,
        image_mentioned_but_missing: bool,
        dataset: str
    ) -> None:
        """Execute Round 1: Independent Analysis (Parallel)."""
        # Get image data from session state
        image_base64 = ctx.session.state.get('image_base64')
        image_mime_type = ctx.session.state.get('image_mime_type')

        if has_valid_image and image_base64:
            logging.info(f"\n=== ROUND 1: INDEPENDENT ANALYSIS (WITH MULTIMODAL IMAGE - PARALLEL) ===")
        else:
            logging.info("\n=== ROUND 1: INDEPENDENT ANALYSIS (PARALLEL) ===")

        # Prepare tasks for parallel execution
        tasks = []
        agent_ids = []

        logging.info(f"Building tasks for {len(recruited_agents)} agents...")

        for agent_data in recruited_agents:
            agent = agent_data['agent']
            agent_id = agent_data['agent_id']
            role = agent_data['role']
            expertise = agent_data['expertise']

            # Build Round 1 prompt
            if PROMPTS_AVAILABLE:
                prompt = get_round1_prompt(
                    task_type=task_type,
                    role=role,
                    expertise=expertise,
                    question=question,
                    options=options,
                    has_image=has_valid_image,
                    image_mentioned_but_missing=image_mentioned_but_missing,
                    dataset=dataset
                )
            else:
                prompt = self._fallback_round1_prompt(role, expertise, question, options,
                                                      image_mentioned_but_missing)

            # Update agent's instruction
            agent.instruction = prompt

            # Log multimodal usage
            if has_valid_image and image_base64:
                logging.info(f"[{agent_id}] Creating R1 task with multimodal image ({len(image_base64)} chars)")

            # Create task for parallel execution
            task = self._execute_agent_with_image(
                ctx, agent, prompt, image_base64, image_mime_type
            )
            tasks.append(task)
            agent_ids.append(agent_id)

        logging.info(f"All {len(tasks)} tasks built, ready to execute in parallel")

        # Execute all agents in parallel with error handling
        logging.info(f"Executing {len(tasks)} agents in parallel...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        round1_results = {}
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                logging.error(f"[{agent_id}] Failed with error: {type(result).__name__}: {result}")
                round1_results[agent_id] = f"ERROR: Agent execution failed - {str(result)}"
            else:
                round1_results[agent_id] = result
                logging.debug(f"  [{agent_id}] {result[:100]}...")

        logging.info(f"Round 1 parallel execution complete: {len(round1_results)} agents")

        # Store R1 results in session state
        ctx.session.state['round1_results'] = round1_results

    async def _execute_round2(
        self,
        ctx: InvocationContext,
        recruited_agents: List[Dict],
        question: str,
        options: List[str],
        task_type: str
    ) -> None:
        """Execute Round 2: Collaborative Discussion (Parallel)."""
        # Get image data from session state
        image_base64 = ctx.session.state.get('image_base64')
        image_mime_type = ctx.session.state.get('image_mime_type')
        has_image = image_base64 is not None

        if has_image:
            logging.info(f"\n=== ROUND 2: COLLABORATIVE DISCUSSION (WITH MULTIMODAL IMAGE - PARALLEL) ===")
        else:
            logging.info("\n=== ROUND 2: COLLABORATIVE DISCUSSION (PARALLEL) ===")

        round1_results = ctx.session.state.get('round1_results', {})

        # Get SMM for context injection
        smm = ctx.session.state.get('smm', None)
        smm_context = ""
        if self.teamwork_config and self.teamwork_config.smm and smm and smm.has_content():
            smm_context = "\n\n" + smm.get_context_string() + "\n"
            logging.debug("[SMM] Injecting SMM context into R2 prompts")

        # Prepare tasks for parallel execution
        tasks = []
        agent_ids = []

        logging.info(f"Building tasks for {len(recruited_agents)} agents...")

        for agent_data in recruited_agents:
            agent = agent_data['agent']
            agent_id = agent_data['agent_id']
            role = agent_data['role']
            expertise = agent_data['expertise']

            logging.debug(f"[{agent_id}] Building R2 task...")

            # Get this agent's R1
            your_round1 = round1_results.get(agent_id, '')

            # Get other agents' R1 analyses (excluding this agent)
            other_analyses = {k: v for k, v in round1_results.items() if k != agent_id}

            # Build Round 2 prompt
            logging.debug(f"[{agent_id}] Building prompt...")
            if PROMPTS_AVAILABLE:
                prompt = get_round2_prompt(
                    role=role,
                    expertise=expertise,
                    question=question,
                    options=options,
                    your_round1=your_round1,
                    other_analyses=other_analyses
                )
            else:
                prompt = self._fallback_round2_prompt(role, expertise, question, options,
                                                      your_round1, other_analyses)

            # [SMM] Inject SMM context
            if smm_context:
                prompt = smm_context + prompt

            agent.instruction = prompt

            # Log multimodal usage
            if has_image:
                logging.info(f"[{agent_id}] Creating R2 task with multimodal image ({len(image_base64)} chars)")

            # Create task for parallel execution
            logging.debug(f"[{agent_id}] Creating coroutine...")
            task = self._execute_agent_with_image(
                ctx, agent, prompt, image_base64, image_mime_type
            )
            tasks.append(task)
            agent_ids.append(agent_id)
            logging.debug(f"[{agent_id}] Task created successfully")

        logging.info(f"All {len(tasks)} tasks built, ready to execute in parallel")

        # Execute all agents in parallel with error handling
        logging.info(f"Executing {len(tasks)} agents in parallel...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        round2_results = {}
        for agent_id, result in zip(agent_ids, results):
            if isinstance(result, Exception):
                logging.error(f"[{agent_id}] Failed with error: {type(result).__name__}: {result}")
                round2_results[agent_id] = f"ERROR: Agent execution failed - {str(result)}"
            else:
                round2_results[agent_id] = result
                logging.debug(f"  [{agent_id}] {result[:100]}...")

        logging.info(f"Round 2 parallel execution complete: {len(round2_results)} agents")

        # Store R2 results
        ctx.session.state['round2_results'] = round2_results

    async def _execute_round3(
        self,
        ctx: InvocationContext,
        recruited_agents: List[Dict],
        question: str,
        options: List[str],
        task_type: str,
        is_except_question: bool,
        has_valid_image: bool,
        image_mentioned_but_missing: bool,
        dataset: str
    ) -> None:
        """
        Execute Round 3: Multi-turn Collaborative Discussion with Final Ranking.

        Phase 3 Integration:
        - Multiple turns (n_turns from config, default 2)
        - [SMM] Inject SMM context in all prompts
        - [Trust] Inject trust hints in prompts
        - [TeamO] Inject formal medical report
        - [Leadership] Mediation after each turn
        - [MM] Mutual Monitoring between turns (not after final)
        - Final turn: Extract rankings
        """
        # Get image data from session state
        image_base64 = ctx.session.state.get('image_base64')
        image_mime_type = ctx.session.state.get('image_mime_type')
        has_image = image_base64 is not None

        # Get teamwork components
        n_turns = self.teamwork_config.n_turns if self.teamwork_config else 2
        smm = ctx.session.state.get('smm', None)
        trust_network = ctx.session.state.get('trust_network', None)
        leadership_coord = ctx.session.state.get('leadership_coordinator', None)
        mm_coordinator = ctx.session.state.get('mm_coordinator', None)
        formal_report = ctx.session.state.get('formal_medical_report', '')

        if has_valid_image and has_image:
            logging.info(f"\n=== ROUND 3: COLLABORATIVE DISCUSSION & RANKING ({n_turns} turns, WITH MULTIMODAL IMAGE) ===")
        else:
            logging.info(f"\n=== ROUND 3: COLLABORATIVE DISCUSSION & RANKING ({n_turns} turns) ===")

        round1_results = ctx.session.state.get('round1_results', {})
        round2_results = ctx.session.state.get('round2_results', {})

        # Store discourse for each turn
        turn_discourses = {}  # {turn_num: {agent_id: discourse}}

        # ========== MULTI-TURN DISCUSSION ==========
        for turn_num in range(1, n_turns + 1):
            is_final_turn = (turn_num == n_turns)

            logging.info(f"\n--- Turn {turn_num}/{n_turns} {'(FINAL)' if is_final_turn else ''} ---")

            turn_discourses[turn_num] = {}

            # Build shared context for this turn
            shared_context = self._build_r3_shared_context(
                ctx, round1_results, round2_results, turn_discourses,
                smm, trust_network, formal_report, turn_num
            )

            # Round-robin discourse
            for agent_data in recruited_agents:
                agent = agent_data['agent']
                agent_id = agent_data['agent_id']
                role = agent_data['role']
                expertise = agent_data['expertise']

                # Build agent-specific prompt
                if is_final_turn:
                    # Final turn: Request ranking
                    prompt = self._build_r3_final_prompt(
                        role, expertise, question, options, task_type,
                        shared_context, is_except_question,
                        has_valid_image, image_mentioned_but_missing, dataset
                    )
                else:
                    # Intermediate turn: Discussion
                    prompt = self._build_r3_discussion_prompt(
                        role, expertise, question, options,
                        shared_context, turn_num
                    )

                agent.instruction = prompt

                # Execute agent
                response_text = await self._execute_agent_with_image(
                    ctx, agent, prompt, image_base64, image_mime_type
                )

                turn_discourses[turn_num][agent_id] = response_text
                logging.debug(f"  [{agent_id}] Turn {turn_num}: {response_text[:80]}...")

            # [Leadership] Mediation after this turn
            if self.teamwork_config and self.teamwork_config.leadership and leadership_coord:
                logging.info(f"[Leadership] Mediating Turn {turn_num}...")
                mediation = await leadership_coord.mediate_discussion(
                    ctx, turn_num, turn_discourses[turn_num], smm
                )
                ctx.session.state[f'turn{turn_num}_mediation'] = mediation
                logging.info(f"[Leadership] Mediation: {mediation[:100]}...")

            # [MM] Mutual Monitoring BETWEEN turns (not after final)
            if not is_final_turn:
                if self.teamwork_config and self.teamwork_config.mutual_monitoring and mm_coordinator:
                    logging.info(f"[MM] Executing Mutual Monitoring after Turn {turn_num}...")
                    mm_result = await mm_coordinator.execute_monitoring(
                        ctx, turn_num, turn_discourses[turn_num],
                        recruited_agents, trust_network, smm
                    )

                    if mm_result:
                        logging.info(f"[MM] Challenged {mm_result.challenged_agent_id}, quality: {mm_result.response_quality}")
                        # Trust and SMM already updated by MM coordinator

        # ========== EXTRACT FINAL RANKINGS ==========
        logging.info("\n--- Extracting Final Rankings ---")

        round3_results = {}
        final_turn_responses = turn_discourses.get(n_turns, {})

        for agent_id, response_text in final_turn_responses.items():
            # Extract ranking from final turn response
            ranking = self._extract_ranking(response_text, task_type)
            confidence = self._extract_confidence(response_text)

            round3_results[agent_id] = {
                'raw': response_text,
                'ranking': ranking,
                'confidence': confidence,
                'answer': ranking[0] if ranking and len(ranking) > 0 else None
            }

            logging.info(f"  [{agent_id}] Final Ranking: {ranking}, Confidence: {confidence}")

        # Store R3 results
        ctx.session.state['round3_results'] = round3_results
        ctx.session.state['turn_discourses'] = turn_discourses  # Store all turn data

    def _build_r3_shared_context(
        self,
        ctx: InvocationContext,
        round1_results: Dict,
        round2_results: Dict,
        turn_discourses: Dict,
        smm: Any,
        trust_network: Any,
        formal_report: str,
        current_turn: int
    ) -> str:
        """
        Build shared context for R3 turns with teamwork components.

        Includes:
        - [SMM] Shared Mental Model
        - [Trust] Trust hints
        - [TeamO] Formal medical report
        - Previous turn discourses
        - Mediations from previous turns
        """
        context_parts = []

        # [SMM] Shared Mental Model
        if self.teamwork_config and self.teamwork_config.smm and smm and smm.has_content():
            context_parts.append(smm.get_context_string())

        # [Trust] Trust hints
        if self.teamwork_config and self.teamwork_config.trust and trust_network:
            trust_hints = trust_network.get_trust_hints_for_prompt()
            if trust_hints:
                context_parts.append(trust_hints)

        # [TeamO] Formal Medical Report
        if self.teamwork_config and self.teamwork_config.team_orientation and formal_report:
            context_parts.append(f"\n=== FORMAL MEDICAL REPORT ===\n{formal_report}\n")

        # R2 Summary
        r2_summary = "\n".join([
            f"[{aid}] {text[:150]}..." if len(text) > 150 else f"[{aid}] {text}"
            for aid, text in round2_results.items()
        ])
        context_parts.append(f"\n=== ROUND 2 ANALYSIS ===\n{r2_summary}\n")

        # Previous R3 turns
        for turn_num in range(1, current_turn):
            if turn_num in turn_discourses:
                turn_summary = "\n".join([
                    f"[{aid}] {disc[:100]}..." if len(disc) > 100 else f"[{aid}] {disc}"
                    for aid, disc in turn_discourses[turn_num].items()
                ])
                context_parts.append(f"\n=== TURN {turn_num} DISCUSSION ===\n{turn_summary}\n")

                # Add mediation if present
                mediation = ctx.session.state.get(f'turn{turn_num}_mediation', '')
                if mediation:
                    context_parts.append(f"[Leader Mediation] {mediation}\n")

        return "\n".join(context_parts)

    def _build_r3_discussion_prompt(
        self,
        role: str,
        expertise: str,
        question: str,
        options: List[str],
        shared_context: str,
        turn_num: int
    ) -> str:
        """Build prompt for intermediate R3 discussion turns."""
        return f"""You are a {role} with expertise in {expertise}.

{shared_context}

Question: {question}

Options: {', '.join(options) if options else 'N/A'}

This is Turn {turn_num} of the collaborative discussion. Review the shared context above and provide your perspective:
- Agree or disagree with consensus points
- Raise new insights or concerns
- Challenge reasoning that seems flawed
- Be concise (2-3 sentences)

Your Turn {turn_num} response:"""

    def _build_r3_final_prompt(
        self,
        role: str,
        expertise: str,
        question: str,
        options: List[str],
        task_type: str,
        shared_context: str,
        is_except_question: bool,
        has_valid_image: bool,
        image_mentioned_but_missing: bool,
        dataset: str
    ) -> str:
        """Build prompt for final R3 turn requesting ranking."""
        # Get existing prompt templates if available
        if PROMPTS_AVAILABLE:
            base_prompt = get_round3_prompt(
                task_type=task_type,
                role=role,
                expertise=expertise,
                question=question,
                options=options,
                your_round1="",  # Not needed, in shared context
                round2_discussion="",  # Not needed, in shared context
                has_image=has_valid_image,
                image_mentioned_but_missing=image_mentioned_but_missing,
                dataset=dataset
            )
            # Prepend shared context
            return shared_context + "\n\n" + base_prompt
        else:
            # Fallback prompt
            rank_instruction = "Rank from MOST FALSE (1) to MOST TRUE" if is_except_question else "Rank from MOST LIKELY (1) to LEAST LIKELY"

            return f"""{shared_context}

You are a {role} with expertise in {expertise}.

Question: {question}

Options: {', '.join(options) if options else 'N/A'}

Based on all team discussion above, provide your FINAL RANKING.

{rank_instruction}.

Format:
RANKING:
1. [Letter] - [brief reasoning]
2. [Letter] - [brief reasoning]
...

CONFIDENCE: High/Medium/Low

RANKING:"""

    def _extract_ranking(self, response: str, task_type: str) -> List[str]:
        """Extract ranking from agent response."""
        if task_type == "mcq":
            # Look for RANKING: section
            pattern = r'RANKING:\s*\n((?:\d+\.\s*[A-Z].*\n?)+)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                lines = match.group(1).strip().split('\n')
                ranking = []
                for line in lines:
                    letter_match = re.search(r'([A-Z])', line)
                    if letter_match:
                        ranking.append(letter_match.group(1))
                return ranking if ranking else []

            # Fallback: look for any letter mentions
            letters = re.findall(r'\b([A-J])\b', response)
            return list(dict.fromkeys(letters))  # Remove duplicates, preserve order

        elif task_type == "yes_no_maybe":
            pattern = r'(?:ANSWER|Answer):\s*(yes|no|maybe)'
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return [match.group(1).lower()]

        return []

    def _extract_confidence(self, response: str) -> str:
        """Extract confidence level."""
        pattern = r'(?:CONFIDENCE|Confidence):\s*(High|Medium|Low)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()
        return "Medium"

    # Fallback prompts (if utils.prompts not available)
    def _fallback_round1_prompt(self, role, expertise, question, options, image_missing):
        constraint = ""
        if image_missing:
            constraint = "**CRITICAL**: Question mentions image but NO IMAGE PROVIDED. Do NOT fabricate observations.\n\n"

        return f"""{constraint}You are a {role} with expertise in {expertise}.

Analyze independently:
{question}

Options: {', '.join(options) if options else 'N/A'}

Provide your preliminary answer with reasoning."""

    def _fallback_round2_prompt(self, role, expertise, question, options, your_r1, others):
        others_text = "\n".join([f"{k}: {v[:150]}" for k, v in others.items()])
        return f"""You are a {role}.

Your R1: {your_r1[:150]}

Teammates' R1:
{others_text}

In 2-3 sentences, agree/disagree with consensus or provide new insights."""

    def _fallback_round3_prompt(self, role, expertise, question, options, r2_summary, is_except):
        rank_instruction = "Rank from MOST FALSE (1) to MOST TRUE" if is_except else "Rank from MOST LIKELY (1) to LEAST LIKELY"

        return f"""You are a {role}.

Team consensus: {r2_summary[:250]}

{rank_instruction}.

Format:
RANKING:
1. A - [reasoning]
2. B - [reasoning]
...

CONFIDENCE: High/Medium/Low"""


__all__ = ['ThreeRoundDebateAgent']
