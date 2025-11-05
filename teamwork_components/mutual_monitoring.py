"""
Mutual Monitoring Component

Inter-round validation mechanism for multi-agent reasoning.
Leader identifies weakest reasoning and challenges agent to defend or correct.

Protocol:
1. Leader selects weakest-reasoning agent
2. Leader raises specific concern
3. Challenged agent responds (accept/justify)
4. Leader evaluates response quality
5. Update Trust scores and SMM based on outcome

Placement:
- Between R3 turns only (not after final turn)
- After R3.1 if n_turns=2
- After R3.1 and R3.2 if n_turns=3

Design:
- 3-4 API calls per MM phase
- Updates both Trust Network and SMM
- Requires Leadership to be enabled

Usage:
    mm = MutualMonitoringCoordinator(
        leader_agent=leader,
        config=teamwork_config
    )

    # Between R3 turns
    mm_result = await mm.execute_monitoring(
        ctx=ctx,
        turn_number=1,
        agent_discourses=discourses,
        trust_network=trust,
        smm=shared_mental_model
    )
"""

import logging
import re
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from google.adk.agents.invocation_context import InvocationContext
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False


@dataclass
class MutualMonitoringResult:
    """Result of a mutual monitoring challenge."""
    challenged_agent_id: str
    concern: str
    response: str
    response_quality: str  # 'strong', 'weak', or 'disputed'
    trust_adjustment: float
    smm_update: Optional[str] = None


class MutualMonitoringCoordinator:
    """
    Coordinates mutual monitoring challenges between R3 turns.

    Requires Leadership to be enabled.
    """

    def __init__(self, leader_agent: Any, config: Any):
        """
        Initialize Mutual Monitoring coordinator.

        Args:
            leader_agent: Leader agent instance
            config: TeamworkConfig instance
        """
        self.leader_agent = leader_agent
        self.config = config

        if not config.leadership:
            logging.warning("[MM] Mutual Monitoring requires Leadership, may not function properly")

        if not config.mutual_monitoring:
            logging.warning("[MM] Mutual Monitoring is disabled in config")

    async def execute_monitoring(
        self,
        ctx: InvocationContext,
        turn_number: int,
        agent_discourses: Dict[str, str],
        recruited_agents: list,
        trust_network: Optional[Any] = None,
        smm: Optional[Any] = None
    ) -> Optional[MutualMonitoringResult]:
        """
        Execute mutual monitoring protocol between R3 turns.

        Args:
            ctx: Invocation context
            turn_number: Current turn number
            agent_discourses: Dict mapping agent_id to discourse text
            recruited_agents: List of agent dicts
            trust_network: TrustNetwork instance (optional)
            smm: SharedMentalModel instance (optional)

        Returns:
            MutualMonitoringResult or None if error
        """
        if not self.config.mutual_monitoring:
            return None

        logging.info(f"[MM] Executing monitoring after Turn {turn_number}")

        # STEP 1: Leader selects weakest reasoning agent
        target_agent_id = await self._select_weakest_agent(ctx, agent_discourses)

        if not target_agent_id:
            logging.warning("[MM] Could not identify agent to challenge")
            return None

        logging.info(f"[MM] Selected {target_agent_id} for challenge")

        # STEP 2: Leader raises concern
        concern = await self._raise_concern(ctx, target_agent_id, agent_discourses[target_agent_id])

        if not concern:
            logging.error("[MM] Failed to generate concern")
            return None

        logging.info(f"[MM] Concern raised: {concern[:100]}...")

        # STEP 3: Challenged agent responds
        target_agent = self._get_agent_by_id(recruited_agents, target_agent_id)
        if not target_agent:
            logging.error(f"[MM] Could not find agent {target_agent_id}")
            return None

        response = await self._get_agent_response(ctx, target_agent, concern)

        if not response:
            logging.error("[MM] Failed to get agent response")
            return None

        logging.info(f"[MM] Agent response: {response[:100]}...")

        # STEP 4: Leader evaluates response quality
        response_quality = await self._evaluate_response(ctx, concern, response)

        logging.info(f"[MM] Response quality: {response_quality}")

        # STEP 5: Update Trust and SMM
        trust_adjustment = 0.0
        if trust_network:
            if response_quality == 'strong':
                trust_adjustment = 0.05
            elif response_quality == 'weak':
                trust_adjustment = -0.05
            else:  # disputed
                trust_adjustment = -0.02

            trust_network.update_after_mutual_monitoring(
                target_agent_id,
                response_quality,
                abs(trust_adjustment)
            )

        smm_update = None
        if smm and hasattr(smm, 'add_debated_point'):
            smm_update = f"MM Turn {turn_number}: {target_agent_id} challenged on reasoning, {response_quality} defense"
            smm.add_debated_point(smm_update)

        result = MutualMonitoringResult(
            challenged_agent_id=target_agent_id,
            concern=concern,
            response=response,
            response_quality=response_quality,
            trust_adjustment=trust_adjustment,
            smm_update=smm_update
        )

        logging.info(f"[MM] Monitoring complete for Turn {turn_number}")
        return result

    async def _select_weakest_agent(
        self,
        ctx: InvocationContext,
        agent_discourses: Dict[str, str]
    ) -> Optional[str]:
        """
        Leader selects agent with weakest reasoning to challenge.

        Args:
            ctx: Invocation context
            agent_discourses: Dict mapping agent_id to discourse

        Returns:
            Agent ID to challenge
        """
        # Build prompt for leader
        discourse_summary = []
        for agent_id, discourse in agent_discourses.items():
            discourse_summary.append(f"[{agent_id}]\n{discourse[:200]}...\n")

        prompt = f"""As team leader, identify which agent's reasoning in this turn is weakest or most questionable.

Agent discourses:
{chr(10).join(discourse_summary)}

Select ONE agent whose reasoning needs clarification or correction. Consider:
- Logical gaps
- Unsupported claims
- Contradictions with established facts

Respond with:
AGENT: [agent_id]
REASON: [1 sentence why]

AGENT:"""

        # Execute leader
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[MM] Error selecting agent: {e}")
            return None
        finally:
            self.leader_agent.instruction = original_instruction

        # Parse agent ID
        agent_match = re.search(r'AGENT:\s*(agent_\d+)', response_text, re.IGNORECASE)
        if agent_match:
            return agent_match.group(1)

        # Fallback: first agent in discourses
        return list(agent_discourses.keys())[0] if agent_discourses else None

    async def _raise_concern(
        self,
        ctx: InvocationContext,
        target_agent_id: str,
        discourse: str
    ) -> Optional[str]:
        """
        Leader raises specific concern about agent's reasoning.

        Args:
            ctx: Invocation context
            target_agent_id: ID of challenged agent
            discourse: Agent's discourse text

        Returns:
            Concern statement
        """
        prompt = f"""As team leader, raise a specific concern about {target_agent_id}'s reasoning.

{target_agent_id}'s discourse:
{discourse}

Identify ONE specific issue:
- Logical gap
- Unsupported claim
- Contradiction with established facts

Format:
CONCERN: [2-3 sentences stating the specific issue and asking for clarification/correction]

CONCERN:"""

        # Execute leader
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[MM] Error raising concern: {e}")
            return None
        finally:
            self.leader_agent.instruction = original_instruction

        # Parse concern
        concern_match = re.search(r'CONCERN:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
        if concern_match:
            return concern_match.group(1).strip()

        return response_text.strip() if response_text else None

    async def _get_agent_response(
        self,
        ctx: InvocationContext,
        agent: Any,
        concern: str
    ) -> Optional[str]:
        """
        Get challenged agent's response to concern.

        Args:
            ctx: Invocation context
            agent: Agent instance
            concern: Concern statement from leader

        Returns:
            Agent's response
        """
        prompt = f"""You have been challenged by the team leader on your reasoning.

Leader's concern:
{concern}

Respond by either:
1. ACCEPTING the concern and revising your position
2. JUSTIFYING your reasoning with additional evidence

Be concise (2-3 sentences).

Response:"""

        # Execute agent with retry logic
        original_instruction = agent.instruction
        agent.instruction = prompt

        response_text = ""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                async for event in agent.run_async(ctx):
                    if hasattr(event, 'content') and event.content:
                        from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                        response_text += extract_text_from_content(event.content)
                break  # Success

            except Exception as e:
                error_msg = str(e).lower()
                is_rate_limit = any(term in error_msg for term in ['429', 'resource_exhausted', 'quota', 'rate limit'])

                if is_rate_limit and attempt < max_retries - 1:
                    delay = 60 * (2 ** attempt)
                    logging.warning(f"[MM] Rate limit in agent response (attempt {attempt+1}), waiting {delay}s")
                    await asyncio.sleep(delay)
                else:
                    logging.error(f"[MM] Error getting agent response: {e}")
                    break

        agent.instruction = original_instruction
        return response_text.strip() if response_text else None

    async def _evaluate_response(
        self,
        ctx: InvocationContext,
        concern: str,
        response: str
    ) -> str:
        """
        Leader evaluates agent's response quality.

        Args:
            ctx: Invocation context
            concern: Original concern
            response: Agent's response

        Returns:
            Quality rating: 'strong', 'weak', or 'disputed'
        """
        prompt = f"""As team leader, evaluate the agent's response to your concern.

Your concern:
{concern}

Agent's response:
{response}

Evaluate the response quality:
- STRONG: Well-justified with evidence, reasoning is sound
- WEAK: Failed to address concern, reasoning still flawed
- DISPUTED: Partially addresses concern but still questionable

Respond with ONLY ONE WORD: STRONG, WEAK, or DISPUTED

Evaluation:"""

        # Execute leader
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[MM] Error evaluating response: {e}")
            return 'disputed'
        finally:
            self.leader_agent.instruction = original_instruction

        # Parse evaluation
        response_text_upper = response_text.upper()
        if 'STRONG' in response_text_upper:
            return 'strong'
        elif 'WEAK' in response_text_upper:
            return 'weak'
        else:
            return 'disputed'

    def _get_agent_by_id(self, recruited_agents: list, agent_id: str) -> Optional[Any]:
        """Get agent instance by ID from recruited_agents list."""
        for agent_data in recruited_agents:
            if agent_data.get('agent_id') == agent_id:
                return agent_data.get('agent')
        return None


__all__ = ['MutualMonitoringCoordinator', 'MutualMonitoringResult']
