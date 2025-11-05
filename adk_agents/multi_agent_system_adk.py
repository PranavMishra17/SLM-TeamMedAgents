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
        teamwork_config=None,
        **kwargs
    ):
        """
        Initialize multi-agent system with modular teamwork components.

        Args:
            model_name: Gemma model to use for all agents
            n_agents: Fixed number of agents, or None for dynamic (2-4)
            aggregation_method: Decision aggregation method ('borda', 'majority', 'confidence_weighted',
                                'trust_weighted', 'hierarchical_weighted')
            teamwork_config: TeamworkConfig instance for modular components
            **kwargs: Additional BaseAgent parameters
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.n_agents = n_agents
        self.aggregation_method = aggregation_method
        self.teamwork_config = teamwork_config

        # Initialize sub-agents with teamwork config
        if n_agents is None:
            self.recruiter = DynamicRecruiterAgent(model_name=model_name, teamwork_config=teamwork_config)
            logging.info("Initialized with DynamicRecruiterAgent")
        else:
            self.recruiter = FixedAgentRecruiter(n_agents=n_agents, model_name=model_name, teamwork_config=teamwork_config)
            logging.info(f"Initialized with FixedAgentRecruiter ({n_agents} agents)")

        self.debate_agent = ThreeRoundDebateAgent(teamwork_config=teamwork_config)

        # Log teamwork configuration
        if teamwork_config:
            active_components = teamwork_config.get_active_components()
            if active_components:
                logging.info(f"Teamwork components enabled: {', '.join(active_components)}")
            else:
                logging.info("Base system mode (all teamwork components OFF)")
        else:
            logging.info("No teamwork config provided, using base system")

        logging.info(f"MultiAgentSystemADK initialized with {model_name}")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Main execution logic for multi-agent system with modular teamwork integration.

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
        - teamwork_metrics: (if teamwork enabled) SMM, Trust, MM results
        """
        start_time = time.time()

        question = ctx.session.state.get('question', '')
        options = ctx.session.state.get('options', [])

        if not question:
            logging.error("No question provided in session.state")
            return

        logging.info(f"MultiAgentSystemADK starting for question: {question[:100]}")

        # ========== INITIALIZE TEAMWORK COMPONENTS ==========
        await self._initialize_teamwork_components(ctx)

        # PHASE 1: Agent Recruitment (with SMM, Leadership, TeamO)
        recruit_start = time.time()
        logging.info("Phase 1: Agent Recruitment")

        async for event in self.recruiter.run_async(ctx):
            pass  # Just execute, don't yield progress events

        recruit_time = time.time() - recruit_start
        logging.info(f"Recruitment complete: {recruit_time:.2f}s")

        # Post-recruitment setup for Leadership and MM
        if self.teamwork_config:
            await self._post_recruitment_setup(ctx)

        # PHASE 2 & 3: Three-Round Debate (with all teamwork components)
        debate_start = time.time()
        logging.info("Phase 2 & 3: Three-Round Debate (R1, R2 + Post-R2, R3 + MM)")

        async for event in self.debate_agent.run_async(ctx):
            pass  # Just execute

        debate_time = time.time() - debate_start
        logging.info(f"Debate complete: {debate_time:.2f}s")

        # PHASE 4: Decision Aggregation (with Trust/Hierarchical weighting + Leadership tie-breaking)
        agg_start = time.time()
        logging.info("Phase 4: Decision Aggregation")

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

        # Store teamwork metrics if enabled
        if self.teamwork_config:
            await self._store_teamwork_metrics(ctx)

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
            'round3_results': ctx.session.state.get('round3_results', {}),
            'teamwork_interactions': ctx.session.state.get('teamwork_interactions', {}),
            'api_call_count': ctx.session.state.get('api_call_count', 0)  # Dynamic count
        }

        results_json = json.dumps(results_payload, default=str)

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=f"Analysis complete. Final answer: {final_answer}\nRESULTS:{results_json}")])
        )

    async def _initialize_teamwork_components(self, ctx: InvocationContext) -> None:
        """
        Initialize all teamwork components and store in session.state.

        Components initialized based on teamwork_config:
        - SMM (Shared Mental Model)
        - Leadership (LeadershipCoordinator)
        - Team Orientation (TeamOrientationManager)
        - Trust Network (TrustNetwork)
        - Mutual Monitoring (MutualMonitoringCoordinator)
        """
        if not self.teamwork_config:
            logging.debug("No teamwork config, skipping component initialization")
            return

        logging.info("Initializing teamwork components...")

        # [SMM] Shared Mental Model
        if self.teamwork_config.smm:
            from teamwork_components import SharedMentalModel
            smm = SharedMentalModel()
            ctx.session.state['smm'] = smm
            logging.info("[SMM] Initialized Shared Mental Model")

        # [Leadership] Leadership Coordinator
        if self.teamwork_config.leadership:
            from teamwork_components import LeadershipCoordinator
            # Leader agent will be set by recruiter
            # We'll create coordinator after recruitment
            ctx.session.state['leadership_enabled'] = True
            logging.info("[Leadership] Leadership mode enabled")

        # [TeamO] Team Orientation Manager
        if self.teamwork_config.team_orientation:
            from teamwork_components import TeamOrientationManager
            team_o_manager = TeamOrientationManager(config=self.teamwork_config)
            ctx.session.state['team_orientation_manager'] = team_o_manager
            logging.info("[TeamO] Initialized Team Orientation Manager")

        # [Trust] Trust Network
        if self.teamwork_config.trust:
            from teamwork_components import TrustNetwork
            trust_network = TrustNetwork(config=self.teamwork_config)
            ctx.session.state['trust_network'] = trust_network
            logging.info(f"[Trust] Initialized Trust Network (range: {self.teamwork_config.trust_range})")

        # [MM] Mutual Monitoring Coordinator
        if self.teamwork_config.mutual_monitoring:
            from teamwork_components import MutualMonitoringCoordinator
            # Will be set up after leader is designated
            ctx.session.state['mm_enabled'] = True
            logging.info("[MM] Mutual Monitoring enabled")

        # After recruiter runs, set up Leadership and MM coordinators
        # This is done in a post-recruitment hook below

    async def _post_recruitment_setup(self, ctx: InvocationContext) -> None:
        """Set up Leadership and MM coordinators after recruitment."""
        if not self.teamwork_config:
            return

        # [Leadership] Create coordinator with leader agent
        if self.teamwork_config.leadership:
            leader_agent = ctx.session.state.get('leader_agent')
            if leader_agent:
                from teamwork_components import LeadershipCoordinator
                leadership_coord = LeadershipCoordinator(
                    leader_agent=leader_agent,
                    config=self.teamwork_config
                )
                ctx.session.state['leadership_coordinator'] = leadership_coord
                logging.info("[Leadership] Created Leadership Coordinator")

        # [Trust] Initialize agent profiles
        if self.teamwork_config.trust:
            trust_network = ctx.session.state.get('trust_network')
            recruited_agents = ctx.session.state.get('recruited_agents', [])
            if trust_network and recruited_agents:
                agent_ids = [a['agent_id'] for a in recruited_agents]
                trust_network.initialize_agents(agent_ids)

        # [MM] Create coordinator
        if self.teamwork_config.mutual_monitoring:
            leader_agent = ctx.session.state.get('leader_agent')
            if leader_agent:
                from teamwork_components import MutualMonitoringCoordinator
                mm_coordinator = MutualMonitoringCoordinator(
                    leader_agent=leader_agent,
                    config=self.teamwork_config
                )
                ctx.session.state['mm_coordinator'] = mm_coordinator
                logging.info("[MM] Created Mutual Monitoring Coordinator")

    async def _store_teamwork_metrics(self, ctx: InvocationContext) -> None:
        """
        Store comprehensive teamwork interactions and metrics in session.state.

        Captures ALL interactions including:
        - SMM content (question analysis, verified facts, debated points)
        - Leadership mediations (full text for each turn)
        - Mutual monitoring exchanges (concern, response, evaluation)
        - Trust score updates (history of changes)
        - Post-R2 processing details
        """
        teamwork_metrics = {}
        teamwork_interactions = {}  # NEW: Detailed interactions storage

        # ===== SMM: Store complete content =====
        smm = ctx.session.state.get('smm')
        if smm:
            teamwork_metrics['smm'] = smm.get_summary()
            teamwork_interactions['smm'] = {
                'question_analysis': smm.question_analysis if hasattr(smm, 'question_analysis') else None,
                'verified_facts': list(smm.verified_facts) if hasattr(smm, 'verified_facts') else [],
                'debated_points': list(smm.debated_points) if hasattr(smm, 'debated_points') else []
            }

        # ===== Trust: Store complete history =====
        trust_network = ctx.session.state.get('trust_network')
        if trust_network:
            teamwork_metrics['trust'] = trust_network.get_summary()
            # Store trust score updates history
            trust_updates = []
            for key, value in ctx.session.state.items():
                if 'trust_update' in key:
                    trust_updates.append(value)
            if trust_updates:
                teamwork_interactions['trust'] = {
                    'updates': trust_updates,
                    'final_scores': trust_network.get_trust_scores() if hasattr(trust_network, 'get_trust_scores') else {}
                }

        # ===== Leadership: Store ALL mediations with full text =====
        if self.teamwork_config and self.teamwork_config.leadership:
            mediations = []
            for key in sorted(ctx.session.state.keys()):
                if 'mediation_turn' in key:
                    turn_num = key.replace('mediation_turn_', '')
                    mediations.append({
                        'turn': int(turn_num) if turn_num.isdigit() else turn_num,
                        'mediation_text': ctx.session.state[key]
                    })

            teamwork_metrics['leadership'] = {
                'enabled': True,
                'mediations': len(mediations)
            }

            if mediations:
                teamwork_interactions['leadership'] = {
                    'mediations': mediations,
                    'post_r2_processing': ctx.session.state.get('leadership_post_r2', None)
                }

        # ===== Mutual Monitoring: Store complete exchanges =====
        mm_results = []
        mm_interactions = []
        for key in sorted(ctx.session.state.keys()):
            if 'mm_result_turn' in key:
                mm_result = ctx.session.state[key]
                mm_results.append(mm_result)
                # Store detailed exchange
                mm_interactions.append({
                    'turn': mm_result.get('turn', 'unknown'),
                    'challenged_agent': mm_result.get('challenged_agent', 'unknown'),
                    'concern': mm_result.get('concern', ''),
                    'agent_response': mm_result.get('agent_response', ''),
                    'response_quality': mm_result.get('response_quality', ''),
                    'trust_impact': mm_result.get('trust_impact', {})
                })

        if mm_results:
            teamwork_metrics['mutual_monitoring'] = {
                'challenges': len(mm_results),
                'results': mm_results
            }
            teamwork_interactions['mutual_monitoring'] = {
                'exchanges': mm_interactions
            }

        # ===== Team Orientation: Store roles and weights =====
        hierarchical_weights = ctx.session.state.get('hierarchical_weights')
        if hierarchical_weights:
            teamwork_metrics['team_orientation'] = {
                'hierarchical_weights': hierarchical_weights
            }
            # Store complete role assignments
            recruited_agents = ctx.session.state.get('recruited_agents', [])
            teamwork_interactions['team_orientation'] = {
                'role_assignments': [
                    {
                        'agent_id': agent.get('agent_id'),
                        'role': agent.get('role'),
                        'expertise': agent.get('expertise'),
                        'weight': hierarchical_weights.get(agent.get('agent_id')) if isinstance(hierarchical_weights, dict) else None
                    }
                    for agent in recruited_agents
                ],
                'formal_report': ctx.session.state.get('formal_medical_report', None)
            }

        # ===== Store Post-R2 Processing Details =====
        post_r2_details = {}
        if ctx.session.state.get('post_r2_facts_extracted'):
            post_r2_details['facts_extraction'] = ctx.session.state.get('post_r2_facts_extracted')
        if ctx.session.state.get('post_r2_trust_updated'):
            post_r2_details['trust_evaluation'] = ctx.session.state.get('post_r2_trust_updated')
        if ctx.session.state.get('formal_medical_report'):
            post_r2_details['formal_report_created'] = True

        if post_r2_details:
            teamwork_interactions['post_r2_processing'] = post_r2_details

        # Store both metrics and detailed interactions
        ctx.session.state['teamwork_metrics'] = teamwork_metrics
        ctx.session.state['teamwork_interactions'] = teamwork_interactions

        logging.info(f"Stored teamwork metrics: {list(teamwork_metrics.keys())}")
        logging.info(f"Stored teamwork interactions: {list(teamwork_interactions.keys())}")

    async def _aggregate_decisions(self, ctx: InvocationContext) -> tuple:
        """
        Aggregate agent rankings using configured method with teamwork integration.

        Supports:
        - Trust-weighted voting (if Trust enabled)
        - Hierarchical-weighted voting (if TeamO enabled)
        - Leadership tie-breaking (if Leadership enabled)

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

        # Determine aggregation method based on teamwork config
        method = self.aggregation_method
        trust_network = ctx.session.state.get('trust_network')
        hierarchical_weights = ctx.session.state.get('hierarchical_weights')

        # Override method if teamwork components are enabled
        if self.teamwork_config:
            if self.teamwork_config.trust and trust_network:
                method = 'trust_weighted'
                logging.info("[Trust] Using trust-weighted voting")
            elif self.teamwork_config.team_orientation and hierarchical_weights:
                method = 'hierarchical_weighted'
                logging.info("[TeamO] Using hierarchical-weighted voting")

        # Aggregate with appropriate method
        from .decision_aggregator_adk import detect_tie

        aggregation_result = aggregate_rankings(
            rankings=rankings,
            confidences=confidences,
            method=method,
            trust_network=trust_network if method == 'trust_weighted' else None,
            hierarchical_weights=hierarchical_weights if method == 'hierarchical_weighted' else None
        )

        final_answer = aggregation_result['winner']
        scores = aggregation_result.get('scores', {})

        # [Leadership] Tie-breaking
        is_tie, tied_options = detect_tie(scores)

        if is_tie and self.teamwork_config and self.teamwork_config.leadership:
            leadership_coord = ctx.session.state.get('leadership_coordinator')
            smm = ctx.session.state.get('smm')

            if leadership_coord:
                logging.info(f"[Leadership] Tie detected ({len(tied_options)} options), invoking Leader to break tie...")

                final_answer = await leadership_coord.resolve_tie(
                    ctx, tied_options, rankings, trust_network.get_all_trust_scores() if trust_network else None, smm
                )

                logging.info(f"[Leadership] Leader resolved tie: {final_answer}")
                aggregation_result['tie_broken_by_leader'] = True
                aggregation_result['tied_options'] = tied_options
            else:
                logging.warning("[Leadership] Tie detected but no Leadership coordinator available")

        logging.info(f"Aggregation complete: {final_answer} (method={method})")

        return final_answer, aggregation_result


__all__ = ['MultiAgentSystemADK']
