"""
Multi-Agent System Root Coordinator for Google ADK

Root BaseAgent that coordinates the entire multi-agent medical reasoning pipeline:
1. Agent recruitment (dynamic or fixed)
2. Three-round collaborative debate
3. Decision aggregation
4. Results packaging

This is the main entry point for the ADK-based system.

Usage:
    from google.adk.sessions import Session
    from adk_agents import MultiAgentSystemADK

    system = MultiAgentSystemADK(
        model_name='gemma3_4b',
        n_agents=3  # or None for dynamic
    )

    session = Session()
    session.state['question'] = "Medical question..."
    session.state['options'] = ["A. ...", "B. ...", ...]
    session.state['task_type'] = "mcq"

    async for event in system.run_async(session):
        print(event.content)

    # Get results
    final_answer = session.state['final_answer']
    round1_results = session.state['round1_results']
    round3_results = session.state['round3_results']
"""

import logging
import time
from typing import AsyncGenerator, Optional

try:
    from google.adk.agents import BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.error("Google ADK not installed")

from .dynamic_recruiter_adk import DynamicRecruiterAgent, FixedAgentRecruiter
from .three_round_debate_adk import ThreeRoundDebateAgent
from .decision_aggregator_adk import aggregate_rankings, calculate_convergence


class MultiAgentSystemADK(BaseAgent):
    """
    Root coordinator for ADK-based multi-agent medical reasoning system.

    This agent orchestrates the complete pipeline:
    - Dynamic or fixed agent recruitment
    - Three-round collaborative debate
    - Borda count aggregation
    - Comprehensive results with timing and metrics

    All intermediate results are stored in session.state for inspection.
    """

    name: str = "multi_agent_system"
    description: str = "Root coordinator for multi-agent medical reasoning"
    model_config = {'extra': 'allow'}  # Allow custom attributes

    def __init__(
        self,
        model_name: str = 'gemma3_4b',
        n_agents: Optional[int] = None,
        aggregation_method: str = 'borda',
        **kwargs
    ):
        """
        Initialize multi-agent system.

        Args:
            model_name: Gemma model to use for all agents
            n_agents: Fixed number of agents, or None for dynamic (2-4)
            aggregation_method: Decision aggregation method ('borda', 'majority', 'confidence_weighted')
            **kwargs: Additional BaseAgent parameters
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.n_agents = n_agents
        self.aggregation_method = aggregation_method

        # Initialize sub-agents
        if n_agents is None:
            self.recruiter = DynamicRecruiterAgent(model_name=model_name)
            logging.info("Initialized with DynamicRecruiterAgent")
        else:
            self.recruiter = FixedAgentRecruiter(n_agents=n_agents, model_name=model_name)
            logging.info(f"Initialized with FixedAgentRecruiter ({n_agents} agents)")

        self.debate_agent = ThreeRoundDebateAgent()

        logging.info(f"MultiAgentSystemADK initialized with {model_name}")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Main execution logic for multi-agent system.

        Expects session.state to contain:
        - question: Question text
        - options: List of options
        - task_type: Task type (mcq, yes_no_maybe, etc.)
        - ground_truth: (optional) for evaluation
        - image_path: (optional) for image-based questions

        Produces session.state outputs:
        - recruited_agents: List of agents
        - round1_results: Round 1 analyses
        - round2_results: Round 2 discussions
        - round3_results: Round 3 rankings
        - final_answer: Aggregated decision
        - timing: Timing breakdown
        - convergence: Convergence metrics
        """
        start_time = time.time()

        question = ctx.session.state.get('question', '')
        options = ctx.session.state.get('options', [])

        if not question:
            logging.error("No question provided in session.state")
            return

        logging.info(f"MultiAgentSystemADK starting for question: {question[:100]}")

        # PHASE 1: Agent Recruitment
        recruit_start = time.time()
        logging.info("Phase 1: Agent Recruitment")

        async for event in self.recruiter.run_async(ctx):
            pass  # Just execute, don't yield progress events

        recruit_time = time.time() - recruit_start
        logging.info(f"Recruitment complete: {recruit_time:.2f}s")

        # PHASE 2: Three-Round Debate
        debate_start = time.time()
        logging.info("Phase 2: Three-Round Debate")

        async for event in self.debate_agent.run_async(ctx):
            pass  # Just execute

        debate_time = time.time() - debate_start
        logging.info(f"Debate complete: {debate_time:.2f}s")

        # PHASE 3: Decision Aggregation
        agg_start = time.time()
        logging.info("Phase 3: Decision Aggregation")

        final_answer, aggregation_result = await self._aggregate_decisions(ctx)

        agg_time = time.time() - agg_start
        logging.info(f"Aggregation complete: {agg_time:.2f}s")

        # Store results
        ctx.session.state['final_answer'] = final_answer
        ctx.session.state['aggregation_result'] = aggregation_result

        # Calculate timing
        total_time = time.time() - start_time
        ctx.session.state['timing'] = {
            'recruit_time': recruit_time,
            'debate_time': debate_time,
            'round1_time': debate_time * 0.4,  # Approximate
            'round2_time': debate_time * 0.3,
            'round3_time': debate_time * 0.3,
            'aggregation_time': agg_time,
            'total_time': total_time
        }

        # Calculate convergence
        round3_results = ctx.session.state.get('round3_results', {})
        rankings = {k: v.get('ranking', []) for k, v in round3_results.items()}
        convergence = calculate_convergence(rankings)
        ctx.session.state['convergence'] = convergence

        # Log final results
        logging.info(f"FINAL ANSWER: {final_answer}")
        logging.info(f"Borda Scores: {aggregation_result['scores']}")
        logging.info(f"Agreement Rate: {aggregation_result['agreement_rate']:.1%}")
        logging.info(f"Converged: {convergence['converged']}")
        logging.info(f"Total Time: {total_time:.2f}s")

        # Check correctness if ground truth provided
        ground_truth = ctx.session.state.get('ground_truth')
        if ground_truth:
            is_correct = str(final_answer).strip().upper() == str(ground_truth).strip().upper()
            ctx.session.state['is_correct'] = is_correct

            result_symbol = "CORRECT" if is_correct else "INCORRECT"
            logging.info(f"{result_symbol} - Answer: {final_answer}, Ground Truth: {ground_truth}")

        # DEBUG: Log session state before returning
        logging.debug(f"Session state keys before return: {list(ctx.session.state.keys())}")
        logging.debug(f"recruited_agents count: {len(ctx.session.state.get('recruited_agents', []))}")
        logging.debug(f"round1_results count: {len(ctx.session.state.get('round1_results', {}))}")
        logging.debug(f"round2_results count: {len(ctx.session.state.get('round2_results', {}))}")
        logging.debug(f"round3_results count: {len(ctx.session.state.get('round3_results', {}))}")

        # Yield a completion event with embedded results (workaround for session state not persisting)
        import json
        from google.genai import types

        # Package all results as JSON to pass through event
        results_payload = {
            'final_answer': final_answer,
            'aggregation_result': aggregation_result,
            'timing': ctx.session.state.get('timing', {}),
            'convergence': convergence,
            'is_correct': ctx.session.state.get('is_correct', False),
            'recruited_agents': ctx.session.state.get('recruited_agents', []),
            'round1_results': ctx.session.state.get('round1_results', {}),
            'round2_results': ctx.session.state.get('round2_results', {}),
            'round3_results': ctx.session.state.get('round3_results', {})
        }

        results_json = json.dumps(results_payload, default=str)

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Analysis complete. Final answer: {final_answer}\nRESULTS:{results_json}")])
        )

    async def _aggregate_decisions(self, ctx: InvocationContext) -> tuple:
        """
        Aggregate agent rankings using configured method.

        Returns:
            Tuple of (final_answer, aggregation_result_dict)
        """
        round3_results = ctx.session.state.get('round3_results', {})

        if not round3_results:
            logging.warning("No Round 3 results to aggregate")
            return 'A', {'winner': 'A', 'scores': {}, 'method': self.aggregation_method, 'agreement_rate': 0}

        # Extract rankings and confidences
        rankings = {}
        confidences = {}

        for agent_id, result in round3_results.items():
            ranking = result.get('ranking', [])
            confidence = result.get('confidence', 'Medium')

            if ranking:
                rankings[agent_id] = ranking
                confidences[agent_id] = confidence

        # Aggregate
        aggregation_result = aggregate_rankings(
            rankings=rankings,
            confidences=confidences,
            method=self.aggregation_method
        )

        final_answer = aggregation_result['winner']

        logging.info(f"Aggregation complete: {final_answer} (method={self.aggregation_method})")

        return final_answer, aggregation_result


__all__ = ['MultiAgentSystemADK']
