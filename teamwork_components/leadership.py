"""
Leadership Component

Active orchestration for multi-agent medical reasoning.
The Recruiter agent takes on dual role as Leader when Leadership is enabled.

Powers:
- Update Shared Mental Model
- Create formal medical reports (TeamO)
- Mediate R3 discussions
- Resolve ties with correction authority
- Coordinate Mutual Monitoring

Pattern:
When OFF: Recruiter is passive coordinator, rule-based updates
When ON: Recruiter becomes active Leader with LLM-powered orchestration

Usage:
    leader = LeadershipCoordinator(
        leader_agent=recruiter_agent,
        config=teamwork_config
    )

    # During R3
    mediation = await leader.mediate_discussion(agent_responses, smm)

    # During aggregation
    if tie_detected:
        final_answer = await leader.resolve_tie(rankings, trust_scores, smm)
"""

import logging
import re
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass

try:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events import Event
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    logging.error("Google ADK not installed")


@dataclass
class LeadershipCoordinator:
    """
    Leadership coordinator for multi-agent system.

    Wraps a Leader agent (typically the Recruiter) with orchestration logic.
    """

    leader_agent: Any                    # ADK Agent instance (Recruiter in dual role)
    config: Any                          # TeamworkConfig instance

    def __post_init__(self):
        """Validate initialization."""
        if not self.config.leadership:
            logging.warning("LeadershipCoordinator created but Leadership is disabled in config")

    async def extract_verified_facts_from_responses(
        self,
        ctx: InvocationContext,
        agent_responses: Dict[str, Any],
        smm: Optional[Any] = None
    ) -> List[str]:
        """
        Use Leader to extract consensus facts from R2 agent responses.

        Leader analyzes all agent responses and identifies verified facts
        with medical grounding.

        Args:
            ctx: Invocation context
            agent_responses: Dict mapping agent_id to response dict with 'facts'
            smm: Shared Mental Model (optional, for additional context)

        Returns:
            List of verified facts
        """
        if not agent_responses:
            return []

        # Collect all facts from agents
        all_facts = []
        for agent_id, response in agent_responses.items():
            facts = response.get('facts', [])
            if isinstance(facts, list):
                for fact in facts:
                    all_facts.append(f"[{agent_id}] {fact}")

        if not all_facts:
            logging.info("[Leadership] No facts to extract")
            return []

        # Build prompt for Leader to extract consensus facts
        prompt = f"""As the team leader, review all agent-reported facts and extract VERIFIED consensus facts.

Facts reported by agents:
{chr(10).join(all_facts)}

Extract 3-7 key facts that are:
1. Mentioned by multiple agents OR medically sound
2. Relevant to answering the question
3. Free from speculation

Respond in this format:
VERIFIED FACTS:
1. [fact]
2. [fact]
...

VERIFIED FACTS:"""

        # Execute Leader agent
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[Leadership] Error extracting facts: {e}")
            return []
        finally:
            self.leader_agent.instruction = original_instruction

        # Parse verified facts
        facts = self._parse_numbered_list(response_text)
        logging.info(f"[Leadership] Extracted {len(facts)} verified facts")
        return facts

    async def create_formal_medical_report(
        self,
        ctx: InvocationContext,
        agent_responses: Dict[str, Any],
        smm: Optional[Any] = None
    ) -> str:
        """
        Leader creates a formal medical report summarizing R2 consensus.

        Report format:
        - Neutral, structured tone
        - Consensus findings
        - Conflicting viewpoints noted
        - Point-by-point summary

        Args:
            ctx: Invocation context
            agent_responses: R2 responses from all agents
            smm: Shared Mental Model (optional)

        Returns:
            Formal medical report string
        """
        if not agent_responses:
            return "No agent responses to summarize."

        # Build context
        agent_summaries = []
        for agent_id, response in agent_responses.items():
            agent_summaries.append(f"[{agent_id}] {response.get('justification', '')[:200]}")

        smm_context = ""
        if smm and hasattr(smm, 'get_context_string'):
            smm_context = f"\n\nShared Mental Model:\n{smm.get_context_string()}"

        prompt = f"""As the team leader, create a formal medical report summarizing the team's R2 analysis.

Agent analyses:
{chr(10).join(agent_summaries)}
{smm_context}

Complete the following steps:
1. Take careful and comprehensive consideration of the provided reports
2. Extract key medical knowledge from the reports
3. Derive a comprehensive and summarized analysis based on the extracted knowledge
4. Generate a refined and synthesized report based on your analysis

Create a structured report with:
1. CONSENSUS FINDINGS: Points all agents agree on
2. DIVERGENT VIEWS: Where agents disagree and why
3. KEY EVIDENCE: Critical facts supporting conclusions

Use neutral, professional medical language. Keep it concise (4-6 points total).

Format:
=== FORMAL MEDICAL REPORT ===

CONSENSUS FINDINGS:
- [point]

DIVERGENT VIEWS:
- [point]

KEY EVIDENCE:
- [point]

=== END REPORT ===

Report:"""

        # Execute Leader
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[Leadership] Error creating report: {e}")
            return "Error creating formal report."
        finally:
            self.leader_agent.instruction = original_instruction

        logging.info(f"[Leadership] Created formal medical report ({len(response_text)} chars)")
        return response_text

    async def mediate_discussion(
        self,
        ctx: InvocationContext,
        turn_number: int,
        agent_discourses: Dict[str, str],
        smm: Optional[Any] = None
    ) -> str:
        """
        Leader mediates after each R3 discussion turn.

        Identifies key controversies and provides guidance for next turn.

        Args:
            ctx: Invocation context
            turn_number: Current turn number
            agent_discourses: Dict mapping agent_id to discourse text
            smm: Shared Mental Model (optional)

        Returns:
            Mediation message
        """
        if not agent_discourses:
            return "No discussion to mediate."

        # Build context
        discourse_summary = []
        for agent_id, discourse in agent_discourses.items():
            discourse_summary.append(f"[{agent_id}] {discourse[:150]}")

        prompt = f"""As team leader, mediate Turn {turn_number} discussion.

Agent discourses:
{chr(10).join(discourse_summary)}

Provide a brief mediation (2-3 sentences):
1. Identify key controversy or convergence point
2. Guide agents on what to focus on next
3. Maintain neutral, facilitative tone

Mediation:"""

        # Execute Leader
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[Leadership] Error mediating: {e}")
            return "Mediation unavailable."
        finally:
            self.leader_agent.instruction = original_instruction

        logging.info(f"[Leadership] Mediation complete for Turn {turn_number}")
        return response_text.strip()

    async def resolve_tie(
        self,
        ctx: InvocationContext,
        tied_options: List[str],
        agent_rankings: Dict[str, List[str]],
        trust_scores: Optional[Dict[str, float]] = None,
        smm: Optional[Any] = None
    ) -> str:
        """
        Leader resolves tie with correction authority.

        Leader can override consensus if reasoning is flawed.

        Args:
            ctx: Invocation context
            tied_options: Options that are tied
            agent_rankings: Final rankings from all agents
            trust_scores: Trust scores (optional)
            smm: Shared Mental Model (optional)

        Returns:
            Final answer (single option)
        """
        # Build context
        rankings_summary = []
        for agent_id, ranking in agent_rankings.items():
            trust_info = f" (trust: {trust_scores.get(agent_id, 0.8):.2f})" if trust_scores else ""
            rankings_summary.append(f"[{agent_id}]{trust_info} {', '.join(ranking)}")

        smm_context = ""
        if smm and hasattr(smm, 'get_context_string'):
            smm_context = f"\n\n{smm.get_context_string()}"

        prompt = f"""As team leader, resolve this tie using your medical expertise and correction authority.

Tied options: {', '.join(tied_options)}

Agent rankings:
{chr(10).join(rankings_summary)}
{smm_context}

Analyze the reasoning and select the SINGLE BEST answer. You have authority to override consensus if reasoning is flawed.

Respond with:
DECISION: [single letter A/B/C/D]
RATIONALE: [2-3 sentences explaining why]

DECISION:"""

        # Execute Leader
        original_instruction = self.leader_agent.instruction
        self.leader_agent.instruction = prompt

        response_text = ""
        try:
            async for event in self.leader_agent.run_async(ctx):
                if hasattr(event, 'content') and event.content:
                    from adk_agents.dynamic_recruiter_adk import extract_text_from_content
                    response_text += extract_text_from_content(event.content)
        except Exception as e:
            logging.error(f"[Leadership] Error resolving tie: {e}")
            # Fallback: return first tied option
            return tied_options[0] if tied_options else 'A'
        finally:
            self.leader_agent.instruction = original_instruction

        # Parse decision
        decision_match = re.search(r'DECISION:\s*([A-Z])', response_text, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).upper()
            logging.info(f"[Leadership] Tie resolved: {decision}")
            return decision

        # Fallback
        logging.warning(f"[Leadership] Could not parse tie resolution, defaulting to {tied_options[0]}")
        return tied_options[0] if tied_options else 'A'

    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse numbered list from LLM response."""
        items = []
        lines = text.split('\n')
        for line in lines:
            # Match patterns like "1. fact" or "- fact"
            match = re.match(r'^\s*(?:\d+\.|-|\*)\s*(.+)', line.strip())
            if match:
                item = match.group(1).strip()
                if item and len(item) > 10:  # Filter out noise
                    items.append(item)
        return items


__all__ = ['LeadershipCoordinator']
