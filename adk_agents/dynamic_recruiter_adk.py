"""
Dynamic Agent Recruiter for Google ADK

Custom BaseAgent that dynamically creates 2-4 specialized medical agents at runtime
based on question complexity and domain.

Key Features:
- LLM determines optimal agent count (2-4)
- LLM generates specialized roles and expertise
- Creates fresh ADK Agent instances per question
- Stores recruited agents in session.state

Usage:
    recruiter = DynamicRecruiterAgent(model_name='gemma3_4b')

    session.state['question'] = "Medical question..."
    session.state['options'] = ["A. ...", "B. ...", ...]

    async for event in recruiter.run_async(session):
        print(event.content)

    # Access recruited agents
    agents = session.state['recruited_agents']
"""

import logging
import json
import re
import asyncio
import random
from typing import AsyncGenerator, List, Dict

try:
    from google.adk.agents import BaseAgent, Agent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.error("Google ADK not installed")

from .gemma_agent_adk import GemmaAgentFactory


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


class DynamicRecruiterAgent(BaseAgent):
    """
    Custom ADK BaseAgent that dynamically recruits specialized medical agents.

    This agent analyzes the medical question and determines:
    1. How many agents are needed (2-4)
    2. What specialized roles are required
    3. What specific expertise each agent should have

    The recruited agents are stored in session.state for use by other agents.
    """

    name: str = "dynamic_recruiter"
    description: str = "Recruits specialized medical agents dynamically based on question"
    model_config = {'extra': 'allow'}  # Allow custom attributes

    def __init__(self, model_name: str = 'gemma3_4b', **kwargs):
        """
        Initialize dynamic recruiter.

        Args:
            model_name: Gemma model to use for recruitment planning
            **kwargs: Additional parameters for BaseAgent
        """
        super().__init__(**kwargs)
        self.model_name = model_name

        # Create planner agent for recruitment decisions
        self.planner = GemmaAgentFactory.create_agent(
            name='recruitment_planner',
            role='Medical Education Coordinator',
            expertise='Determining optimal specialist composition for medical case analysis',
            model_name=model_name,
            temperature=0.5  # Lower temperature for more consistent recruitment
        )

        logging.info(f"Initialized DynamicRecruiterAgent with {model_name}")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Main execution logic for dynamic recruitment.

        Reads question from session.state, determines optimal agents,
        creates them, and stores in session.state.
        """
        # Get question from session state
        question = ctx.session.state.get('question', '')
        options = ctx.session.state.get('options', [])
        image = ctx.session.state.get('image', None)  # PIL Image object
        dataset = ctx.session.state.get('dataset', '')

        if not question:
            logging.error("No question found in session.state")
            return

        # Determine if image artifact handling is needed
        # Expanded list of vision datasets
        vision_datasets = ['path-vqa', 'pmc-vqa', 'pathvqa', 'pmcvqa', 'slake', 'rad', 'vqa-rad', 'vqarad']
        has_image = image is not None and any(d in dataset.lower() for d in vision_datasets)

        if has_image:
            logging.info(f"=== DYNAMIC AGENT RECRUITMENT (WITH MULTIMODAL IMAGE - Dataset: {dataset}) ===")
        else:
            logging.info("=== DYNAMIC AGENT RECRUITMENT ===")

        # STEP 1: Determine agent count
        logging.info("Analyzing question complexity...")

        agent_count = await self._determine_agent_count(ctx, question, options)

        logging.info(f"Determined optimal agent count: {agent_count}")

        # STEP 2: Generate roles for each agent
        logging.info("Generating specialist roles...")

        recruited_agents = []
        for i in range(agent_count):
            role, expertise = await self._generate_role(ctx, question, options, i + 1, agent_count)

            # Create ADK Agent (with artifact image support if needed)
            agent = GemmaAgentFactory.create_agent(
                name=f"agent_{i+1}",
                role=role,
                expertise=expertise,
                model_name=self.model_name,
                has_image=has_image
            )

            recruited_agents.append({
                'agent': agent,
                'agent_id': f"agent_{i+1}",
                'role': role,
                'expertise': expertise
            })

            model_info = " (IMAGE ARTIFACT)" if has_image else ""
            logging.info(f"  Agent {i+1}: {role} - {expertise}{model_info}")

        # STEP 3: Store in session state
        ctx.session.state['recruited_agents'] = recruited_agents
        ctx.session.state['n_agents'] = agent_count

        logging.info(f"=== RECRUITMENT COMPLETE: {agent_count} AGENTS ===")

        # Yield event to trigger session state persistence
        from google.genai import types
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Recruited {agent_count} agents")])
        )

    async def _determine_agent_count(
        self,
        ctx: InvocationContext,
        question: str,
        options: List[str]
    ) -> int:
        """
        Use LLM to determine optimal agent count based on question complexity.

        Returns:
            Integer between 2 and 4
        """
        prompt = f"""Analyze this medical question and determine the optimal number of specialist agents needed (2-4).

Question: {question}

Options: {', '.join(options) if options else 'Open-ended'}

Consider:
- Question complexity (simple → 2 agents, complex → 4 agents)
- Number of medical domains involved
- Interdisciplinary analysis requirements

Respond with ONLY a single number between 2 and 4.

Optimal agent count:"""

        # Update planner instruction temporarily
        original_instruction = self.planner.instruction
        self.planner.instruction = prompt

        # Get LLM response with retry logic
        response_text = ""
        async for event in retry_with_exponential_backoff(lambda: self.planner.run_async(ctx)):
            if hasattr(event, 'content') and event.content:
                response_text += extract_text_from_content(event.content)

        # Restore original instruction
        self.planner.instruction = original_instruction

        # Extract number
        match = re.search(r'\b([2-4])\b', response_text)
        if match:
            count = int(match.group(1))
            logging.info(f"LLM determined agent count: {count}")
            return count

        # Fallback
        logging.warning(f"Could not parse agent count from: {response_text[:100]}, defaulting to 3")
        return 3

    async def _generate_role(
        self,
        ctx: InvocationContext,
        question: str,
        options: List[str],
        agent_num: int,
        total_agents: int
    ) -> tuple:
        """
        Use LLM to generate specialized role and expertise for an agent.

        Returns:
            Tuple of (role, expertise)
        """
        prompt = f"""Generate a specialized medical role for Agent #{agent_num} of {total_agents} agents analyzing this question.

Question: {question}

The {total_agents} agents should have complementary expertise covering different aspects of this question.

Respond in this EXACT format:
ROLE: [Specific medical specialty/role]
EXPERTISE: [Detailed area of expertise]

Example:
ROLE: Pediatric Cardiologist
EXPERTISE: Congenital heart defects, pediatric arrhythmias, and fetal cardiology

Generate Agent #{agent_num}:"""

        # Update planner instruction
        original_instruction = self.planner.instruction
        self.planner.instruction = prompt

        # Get LLM response with retry logic
        response_text = ""
        async for event in retry_with_exponential_backoff(lambda: self.planner.run_async(ctx)):
            if hasattr(event, 'content') and event.content:
                response_text += extract_text_from_content(event.content)

        # Restore instruction
        self.planner.instruction = original_instruction

        # Parse role and expertise
        role_match = re.search(r'ROLE:\s*(.+)', response_text, re.IGNORECASE)
        expertise_match = re.search(r'EXPERTISE:\s*(.+)', response_text, re.IGNORECASE)

        if role_match and expertise_match:
            role = role_match.group(1).strip()
            expertise = expertise_match.group(1).strip()
            logging.info(f"Generated Agent {agent_num}: {role}")
            return role, expertise

        # Fallback roles
        fallback_roles = [
            ("General Internist", "Broad medical knowledge and diagnostic reasoning"),
            ("Medical Specialist", "Disease pathophysiology and treatment protocols"),
            ("Clinical Researcher", "Evidence-based medicine and current literature"),
            ("Diagnostician", "Differential diagnosis and systematic clinical reasoning")
        ]

        role, expertise = fallback_roles[(agent_num - 1) % len(fallback_roles)]
        logging.warning(f"Using fallback role for Agent {agent_num}: {role}")
        return role, expertise


class FixedAgentRecruiter(BaseAgent):
    """
    Alternative recruiter that creates a fixed number of generic agents.

    Useful for consistent testing and comparison with dynamic recruitment.
    """

    name: str = "fixed_recruiter"
    description: str = "Creates fixed number of medical specialist agents"
    model_config = {'extra': 'allow'}  # Allow custom attributes

    def __init__(self, n_agents: int = 3, model_name: str = 'gemma3_4b', **kwargs):
        super().__init__(**kwargs)
        self.n_agents = n_agents
        self.model_name = model_name
        logging.info(f"Initialized FixedAgentRecruiter with {n_agents} agents")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Create fixed number of agents with predefined roles."""
        # Check if image artifact handling is needed
        image = ctx.session.state.get('image', None)  # PIL Image object
        dataset = ctx.session.state.get('dataset', '')

        # Expanded list of vision datasets
        vision_datasets = ['path-vqa', 'pmc-vqa', 'pathvqa', 'pmcvqa', 'slake', 'rad', 'vqa-rad', 'vqarad']
        has_image = image is not None and any(d in dataset.lower() for d in vision_datasets)

        if has_image:
            logging.info(f"=== FIXED RECRUITMENT: {self.n_agents} AGENTS (WITH MULTIMODAL IMAGE - Dataset: {dataset}) ===")
        else:
            logging.info(f"=== FIXED RECRUITMENT: {self.n_agents} AGENTS ===")

        # Predefined roles for consistent agents
        predefined_roles = [
            ("General Internist", "Broad medical knowledge and diagnostic reasoning"),
            ("Medical Specialist", "Disease pathophysiology and treatment protocols"),
            ("Clinical Researcher", "Evidence-based medicine and systematic reviews"),
            ("Diagnostician", "Differential diagnosis and pattern recognition"),
            ("Clinical Pharmacologist", "Drug interactions and pharmacotherapy")
        ]

        recruited_agents = []
        for i in range(self.n_agents):
            role, expertise = predefined_roles[i % len(predefined_roles)]

            agent = GemmaAgentFactory.create_agent(
                name=f"agent_{i+1}",
                role=role,
                expertise=expertise,
                model_name=self.model_name,
                has_image=has_image
            )

            recruited_agents.append({
                'agent': agent,
                'agent_id': f"agent_{i+1}",
                'role': role,
                'expertise': expertise
            })

            model_info = " (IMAGE ARTIFACT)" if has_image else ""
            logging.info(f"  Agent {i+1}: {role}{model_info}")

        # Store in session state
        ctx.session.state['recruited_agents'] = recruited_agents
        ctx.session.state['n_agents'] = self.n_agents

        logging.info("=== RECRUITMENT COMPLETE ===")

        # Yield event to trigger session state persistence
        from google.genai import types
        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Recruited {self.n_agents} agents")])
        )


__all__ = ['DynamicRecruiterAgent', 'FixedAgentRecruiter']
