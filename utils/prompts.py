"""
Centralized Prompt Templates for Multi-Agent Medical Reasoning System

Enhanced with dynamic recruitment, teamwork components, and optimized context management.
All prompts designed for efficient token usage while maintaining collaborative effectiveness.

Format: Use {variable} for template substitution
"""

from typing import Dict, List, Optional
import re

# Import few-shot CoT prompts
try:
    from .few_shot_cot_prompts import get_few_shot_prompt, get_zero_shot_cot_prompt
    FEW_SHOT_AVAILABLE = True
except ImportError:
    FEW_SHOT_AVAILABLE = False

# ============================================================================
# AGENT SYSTEM PROMPTS
# ============================================================================

AGENT_SYSTEM_PROMPTS = {
    "base": """You are a {role} with expertise in {expertise}.

Task: {task_description}
Format: {expected_output_format}

Guidelines:
- Be precise, concise, evidence-based
- Focus on medical/clinical content
- Avoid unnecessary preamble
- Use professional medical communication""",

    "medical_expert": """You are a {role} with expertise in {expertise}. Your job is to collaborate with other medical experts in a team.

Core Principles:
- Base reasoning on established medical knowledge
- Be precise with medical terminology
- Acknowledge uncertainty appropriately
- Consider patient safety and best practices
- Collaborate constructively with other specialists
- Actively engage with peer opinions and reasoning
- Explain reasoning clearly and concisely to convince other experts""",
}

# ============================================================================
# RECRUITMENT PROMPTS - Dynamic Agent Selection
# ============================================================================

RECRUITMENT_PROMPTS = {
    # Determines how many agents (2-4) are needed
    "complexity_analysis": """You are a medical expert who conducts initial assessment. Analyze this medical question and determine the difficulty/complexity and optimal agent count (2-4).

Question: {question}
Options: {options}

Complexity levels:
1. LOW: A PCP or general physician can answer this simple medical knowledge question without consulting specialists (→ 2 agents)
2. MODERATE: A PCP or general physician needs consultation with specialists in a team (→ 3 agents)
3. HIGH: Multi-departmental specialists required with significant team effort and cross-consultation (→ 4 agents)

Consider:
- Complexity of medical reasoning required
- Distinct medical domains involved
- Need for diverse specialist perspectives

Respond with ONLY a number: 2, 3, or 4

Number of agents:""",

    # Creates specialized roles
    "role_assignment": """You are an experienced medical expert who recruits a group of experts with diverse identities for this medical query.

Question: {question}
Options: {options}

Create {n_agents} specialized medical expert roles with complementary expertise.

Also specify the communication structure between experts using:
- "==" for equal collaboration (e.g., A == B)
- ">" for hierarchical consultation (e.g., A > B means A leads, B consults)

Format EXACTLY as:
AGENT_1: [Role] - [One-line expertise]
AGENT_2: [Role] - [One-line expertise]
(Include only {n_agents} agents)
COMMUNICATION: [Structure, e.g., "Agent_1 == Agent_2 > Agent_3" or "Independent"]

Example:
AGENT_1: Cardiologist - Expert in cardiovascular disease and ECG interpretation
AGENT_2: Emergency Physician - Specialist in acute care and rapid decision-making
COMMUNICATION: Agent_1 == Agent_2

Create {n_agents} roles:""",

    # Enhanced team selection with weights
    "team_selection": """You recruit medical experts to solve this query.

IMPORTANT: Select experts with DISTINCT, NON-OVERLAPPING specialties.

Question: {question}

Recruit {num_agents} experts. For each:
- Assign weight (0.0-1.0, total=1.0) reflecting importance
- Specify communication structure (e.g., "A > B" or "Independent")

Example for 3 experts:
1. Pediatrician - Specializes in child/adolescent care - Hierarchy: Independent - Weight: 0.35
2. Cardiologist - Focuses on heart conditions - Hierarchy: Pediatrician > Cardiologist - Weight: 0.35
3. Pulmonologist - Specializes in respiratory disorders - Hierarchy: Independent - Weight: 0.30

Format your answer exactly as above. No additional explanation.""",
}

# ============================================================================
# ROUND 1 PROMPTS - INDEPENDENT ANALYSIS (Optimized)
# ============================================================================

ROUND1_PROMPTS = {
    "independent_analysis_mcq": """You are a {role} with expertise in {expertise}.

Analyze the following medical question and provide your initial prediction.

Question: {question}

Options:
{options}

**ALGO R2 - Initial Prediction Phase**

Provide your response in the following FORMAT (REQUIRED):

RANKING: [List ALL options in order of likelihood, e.g., D, A, B, C]
JUSTIFICATION: [In 2-3 sentences, explain your reasoning. Focus on: (1) Classic clinical presentation recognition, (2) Key clinical features (age, ethnicity, symptoms, family history), (3) Evidence-based reasoning]
FACTS: [List 2-5 key medical facts that support your reasoning, e.g., "Sickle cell disease is most common in African Americans", "Hand pain suggests vaso-occlusive crisis"]

Your response:""",

    "independent_analysis_yes_no_maybe": """You are a {role} with expertise in {expertise}.

Analyze the following medical question using your specialized knowledge.

Question: {question}

Instructions:
1. Consider only the information explicitly provided
2. Provide your answer (yes, no, or maybe) followed by brief reasoning

Your analysis:""",

    "independent_analysis_open": """You are a {role} with expertise in {expertise}.

Analyze the following medical question using your specialized knowledge.

Question: {question}

Instructions:
1. Consider only the information explicitly provided
2. Provide comprehensive answer from your specialty's perspective

Your analysis:""",
}

# ============================================================================
# ROUND 2 PROMPTS - COLLABORATIVE DISCUSSION (Token-Optimized)
# ============================================================================

ROUND2_PROMPTS = {
    "collaborative_discussion": """You are a {role} with expertise in {expertise}. You are collaborating with other medical experts in a team.

Question: {question}
Options: {options}

**Your R1 view**: {your_round1_summary}

**Teammates' R1 views**:
{other_agents_summaries}

**Round 2 - Collaborative Discussion**: After reviewing opinions from other medical agents in your team, provide your ranked prediction. Indicate whether you agree/disagree with other experts and deliver your opinion in a way to convince them with clear reasoning.

FORMAT (REQUIRED):
RANKING: [List all options in order of likelihood, e.g., A, B, C, D]
JUSTIFICATION: [In 2-3 sentences, state if you agree/disagree with consensus, have new insights, or maintain your position. Explain why with evidence to convince other experts.]

Your response:""",

    "respond_to_agent": """Team member ({agent_role}) said:
"{agent_message}"

As {role}, respond with your specialized perspective. Note agreements/disagreements and add insights.

Your response:""",
}

# ============================================================================
# ROUND 3 PROMPTS - FINAL RANKING (Token-Optimized)
# ============================================================================

ROUND3_PROMPTS = {
    "final_ranking_mcq": """You are a {role} and final medical decision maker who reviews all opinions from different medical experts.

Question: {question}
Options: {options}

**Team consensus**: {team_consensus}

**Round 3 - Final Decision**: Review all team opinions and make your FINAL independent ranking of ALL options. You have the authority to make the final decision based on your medical expertise and the team's collective reasoning.

**CRITICAL REMINDERS**:
- Consider classic disease presentations FIRST (e.g., African-American child + hand pain = sickle cell)
- Match clinical features to well-known syndromes
- Rank based on clinical likelihood, not abstract genetic reasoning alone
- Synthesize the best insights from team discussion

Format EXACTLY as:
RANKING:
1. [Option Letter] - [Brief clinical justification]
2. [Option Letter] - [Brief clinical justification]
3. [Option Letter] - [Brief clinical justification]
4. [Option Letter] - [Brief clinical justification]

CONFIDENCE: [High/Medium/Low]

EXPLANATION: [1-2 sentences - Why is #1 the most likely answer clinically?]

Your ranking:""",

    "final_decision_yes_no_maybe": """You are a {role}.

Question: {question}

**Team consensus**: {team_consensus}

**Round 3**: Provide final answer (yes/no/maybe).

Format:
FINAL ANSWER: [yes/no/maybe]
CONFIDENCE: [High/Medium/Low]
EXPLANATION: [Your reasoning]

Your decision:""",

    "mcq_final": """Based on initial analysis and team discussion, provide FINAL answer.

Your initial: {initial_analysis}
Team insights: {discussion_summary}

You MUST begin: "ANSWER: X" (replace X with option letter).
Then explain final reasoning and how discussion influenced your decision.

Your final answer:""",
}

# ============================================================================
# LEADERSHIP PROMPTS - Team Coordination
# ============================================================================

LEADERSHIP_PROMPTS = {
    "team_leadership": """As team leader:
1. Facilitate problem solving
2. Set performance expectations
3. Synchronize contributions
4. Evaluate team functioning
5. Clarify roles
6. Provide feedback
Be precise, concise, to the point.""",

    "synthesize": """As leader, synthesize team's perspectives into consensus.

Context: {context}

Create solution that:
1. Incorporates key insights
2. Balances perspectives
3. Provides clear reasoning

You MUST begin: "ANSWER: X" (option letter).

Final solution:""",

    "synthesize_multi_choice": """As leader, synthesize for multi-choice question (multiple correct answers).

Context: {context}

You MUST begin: "ANSWERS: X,Y,Z" (e.g., "ANSWERS: A,C").

Final solution:""",

    "synthesize_yes_no_maybe": """As leader, synthesize for research question.

Context: {context}

You MUST begin: "ANSWER: X" (yes/no/maybe).

Final answer:""",
}

# ============================================================================
# CLOSED-LOOP COMMUNICATION PROMPTS
# ============================================================================

COMMUNICATION_PROMPTS = {
    "closed_loop": """Use clear, specific communication. Acknowledge receipt and confirm understanding.""",

    "receiver_acknowledgment": """Message from {sender_role}: "{sender_message}"

Acknowledge:
1. "Understood: [key point]"
2. Your response

Be precise, concise.""",
}

# ============================================================================
# MUTUAL MONITORING PROMPTS
# ============================================================================

MONITORING_PROMPTS = {
    "mutual_monitoring": """Engage in mutual monitoring:
1. Track teammate performance
2. Check for errors/omissions
3. Provide constructive feedback
4. Ensure quality
5. Be precise, concise

Do this respectfully to improve team decisions.""",
}

# ============================================================================
# SHARED MENTAL MODEL PROMPTS
# ============================================================================

MENTAL_MODEL_PROMPTS = {
    "shared_mental_model": """Contribute to shared mental model:
1. State your understanding explicitly
2. Check alignment on task interpretation
3. Establish shared terminology
4. Clarify reasoning
5. Be precise, concise

Ensure team alignment.""",

    "team_mental_model": """Collaboratively answer: {task_description}

Team-related models:
1. Understand each other's specialty, defer appropriately
2. Use clear, structured communication with confirmation
3. Agree on approach for each question
4. Psychological safety - open to being wrong
5. Efficient division of labor

Task-specific:
Objective: {objective}
Criteria: {criteria}""",
}

# ============================================================================
# TEAM ORIENTATION PROMPTS
# ============================================================================

ORIENTATION_PROMPTS = {
    "team_orientation": """Demonstrate team orientation:
1. Consider alternative solutions from teammates
2. Value teammates' perspectives even when different
3. Prioritize team goals over individual achievement
4. Engage in information sharing and goal setting
5. Enhance performance through coordination

Value and incorporate diverse perspectives.""",
}

# ============================================================================
# MUTUAL TRUST PROMPTS
# ============================================================================

TRUST_PROMPTS = {
    "mutual_trust_base": """Foster mutual trust:
1. Share information openly
2. Admit mistakes, accept feedback
3. Assume positive intentions
4. Respect expertise and rights
5. Ask for help when needed
Be precise, concise.""",

    "high_trust": """HIGH TRUST environment:
- Share information with confidence
- Rely on teammates without excessive verification
- Express uncertainty safely
- Expect protection of contributions
- Focus on task, not monitoring""",

    "low_trust": """LOW TRUST environment:
- Be careful with sensitive info
- Verify information carefully
- Be explicit about reasoning
- Demonstrate reliability consistently
- Build trust gradually""",
}

# ============================================================================
# TASK ANALYSIS PROMPTS (Round 1)
# ============================================================================

TASK_ANALYSIS_PROMPTS = {
    "mcq_task": """As {role} with {expertise}, analyze INDEPENDENTLY:

{task_description}

Options: {options}

IMPORTANT: Independent Round 1 - no knowledge of other members.

1. Analyze each option systematically
2. Apply your specialized principles
3. Consider strengths/weaknesses
4. Provide reasoning

End with: "ANSWER: X" (option letter).

Your analysis:""",

    "multi_choice_mcq_task": """As {role} with {expertise}, analyze INDEPENDENTLY:

{task_description}

Options: {options}

IMPORTANT: MULTI-CHOICE - MORE THAN ONE answer may be correct. Select ALL correct.

1. Analyze each option for correctness
2. Apply specialized principles
3. Identify ALL correct options
4. Provide reasoning

End with: "ANSWERS: X,Y,Z" (ALL correct letters).

Your analysis:""",

    "yes_no_maybe_task": """As {role} with {expertise}, analyze INDEPENDENTLY:

{task_description}

IMPORTANT: Independent Round 1 - no knowledge of other members.

1. Analyze scientific evidence
2. Apply specialized principles
3. Determine if evidence supports/refutes/inconclusive
4. Provide reasoning

End with: "ANSWER: X" (yes/no/maybe).

Your analysis:""",
}

# ============================================================================
# FINAL DECISION PROMPTS (Round 3)
# ============================================================================

FINAL_DECISION_PROMPTS = {
    "mcq_final": """Based on initial analysis and team discussion, provide FINAL answer.

Initial: {initial_analysis}
Team: {discussion_summary}

IMPORTANT: Final independent decision. Consider discussion but make your own judgment.

You MUST begin: "ANSWER: X" (option letter).
Explain final reasoning and how discussion influenced you.

Your final answer:""",

    "multi_choice_final": """Based on initial analysis and team discussion, provide FINAL answer.

Initial: {initial_analysis}
Team: {discussion_summary}

IMPORTANT: MULTI-CHOICE - select ALL correct options. Make your own judgment.

You MUST begin: "ANSWERS: X,Y,Z" (ALL correct letters).
Explain final reasoning.

Your final answer:""",

    "yes_no_maybe_final": """Based on initial analysis and team discussion, provide FINAL answer.

Initial: {initial_analysis}
Team: {discussion_summary}

IMPORTANT: Final independent decision. Make your own judgment.

You MUST begin: "ANSWER: X" (yes/no/maybe).
Explain final reasoning.

Your final answer:""",
}

# ============================================================================
# DYNAMIC RECRUITMENT PROMPTS
# ============================================================================

DYNAMIC_RECRUITMENT_PROMPTS = {
    "team_size_determination": """Analyze this question and determine optimal agents (2-5).

Question: {question}
Complexity: {complexity}
Max Allowed: {max_agents}

Consider:
1. Question scope and expertise areas
2. Complexity and perspective needs
3. Collaboration benefits vs overhead
4. Decision quality balance

Guidelines:
- 2: Simple, limited diversity
- 3: Moderate, some specialization
- 4: Complex, multiple perspectives
- 5: Highly complex, interdisciplinary

Conclude with: TEAM_SIZE: X

Your analysis:""",

    "teamwork_config_selection": """Select teamwork components for this question:

Question: {question}
Team Size: {team_size}
Complexity: {complexity}

Available: leadership, monitoring, mental_model, orientation, trust, closed_loop

For medical diagnosis, recommend: leadership, monitoring

SELECTED_COMPONENTS: leadership, monitoring""",
}

# ============================================================================
# ANSWER EXTRACTION PROMPTS
# ============================================================================

EXTRACTION_PROMPTS = {
    "clarify_mcq_answer": """Your response didn't clearly indicate answer.

Question: {question}
Options: {options}
Your response: {previous_response}

Provide ONLY letter (A, B, C, D):
Answer:""",

    "clarify_ranking": """Your response didn't follow ranking format.

Provide ranking in EXACT format:

RANKING:
1. [Option]
2. [Option]
3. [Option]
4. [Option]

Your ranking:""",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_options(options: list) -> str:
    """Format options list as multi-line string."""
    if isinstance(options, list):
        return "\n".join(options)
    return str(options)

def format_agent_analyses(analyses: dict, exclude_agent_id: str = None) -> str:
    """Format agent analyses - OPTIMIZED (first 200 chars only)."""
    formatted = []
    for agent_id, analysis in analyses.items():
        if agent_id != exclude_agent_id:
            summary = analysis[:200] + "..." if len(analysis) > 200 else analysis
            formatted.append(f"{agent_id}: {summary}")
    return "\n".join(formatted)

def get_round1_prompt(task_type: str, role: str, expertise: str, question: str, options: list = None,
                     has_image: bool = False, image_mentioned_but_missing: bool = False, dataset: str = None,
                     use_few_shot: bool = True, n_few_shot_examples: int = 2) -> str:
    """
    Get Round 1 prompt based on task type with image handling and optional few-shot examples.

    Args:
        task_type: Type of task (mcq, yes_no_maybe, etc.)
        role: Agent's role
        expertise: Agent's expertise
        question: The question text
        options: Answer options
        has_image: Whether image is provided
        image_mentioned_but_missing: Whether image is mentioned but missing
        dataset: Dataset name (for few-shot selection)
        use_few_shot: Whether to include few-shot CoT examples
        n_few_shot_examples: Number of few-shot examples to include

    Returns:
        Formatted prompt string
    """
    options_str = format_options(options) if options else ""

    # Build few-shot prefix if enabled
    few_shot_prefix = ""
    if use_few_shot and FEW_SHOT_AVAILABLE and dataset:
        few_shot_text = get_few_shot_prompt(dataset, n_examples=n_few_shot_examples)
        if few_shot_text:
            few_shot_prefix = few_shot_text + "\n\n"
    elif use_few_shot and FEW_SHOT_AVAILABLE and not dataset:
        # Use zero-shot CoT if dataset not specified
        few_shot_prefix = get_zero_shot_cot_prompt() + "\n\n"

    # Add image constraint ONLY for datasets that actually have images (path-vqa, pmc-vqa)
    # For text-only datasets (medqa, medmcqa, pubmedqa), don't add constraint
    image_constraint = ""
    if dataset and dataset.lower() in ['path-vqa', 'pmc-vqa', 'pathvqa', 'pmcvqa']:
        # Only these datasets have images
        if image_mentioned_but_missing:
            image_constraint = """**CRITICAL CONSTRAINT**: The question references an image, but NO IMAGE IS PROVIDED.
DO NOT fabricate, hallucinate, or make assumptions about what the image shows.
Base your analysis ONLY on the textual information provided. If the question cannot be answered without the image, state this clearly.

"""
        elif has_image:
            image_constraint = """**IMAGE ANALYSIS**: An image is provided. Analyze it carefully and integrate your observations with clinical reasoning.

"""

    if task_type == "mcq":
        base_prompt = ROUND1_PROMPTS["independent_analysis_mcq"].format(
            role=role, expertise=expertise, question=question, options=options_str
        )
        full_prompt = few_shot_prefix + image_constraint + base_prompt if image_constraint else few_shot_prefix + base_prompt
        return full_prompt
    elif task_type == "yes_no_maybe":
        base_prompt = ROUND1_PROMPTS["independent_analysis_yes_no_maybe"].format(
            role=role, expertise=expertise, question=question
        )
        full_prompt = few_shot_prefix + image_constraint + base_prompt if image_constraint else few_shot_prefix + base_prompt
        return full_prompt
    else:
        base_prompt = ROUND1_PROMPTS["independent_analysis_open"].format(
            role=role, expertise=expertise, question=question
        )
        full_prompt = few_shot_prefix + image_constraint + base_prompt if image_constraint else few_shot_prefix + base_prompt
        return full_prompt

def get_round2_prompt(role: str, expertise: str, question: str, options: list,
                     your_round1: str, other_analyses: dict) -> str:
    """Get Round 2 collaborative discussion prompt - OPTIMIZED."""
    your_summary = your_round1[:150] + "..." if len(your_round1) > 150 else your_round1
    return ROUND2_PROMPTS["collaborative_discussion"].format(
        role=role, expertise=expertise, question=question,
        options=format_options(options),
        your_round1_summary=your_summary,
        other_agents_summaries=format_agent_analyses(other_analyses)
    )

def get_round3_prompt(task_type: str, role: str, expertise: str, question: str, options: list,
                     your_round1: str, round2_discussion: str,
                     has_image: bool = False, image_mentioned_but_missing: bool = False, dataset: str = None) -> str:
    """Get Round 3 final ranking prompt - OPTIMIZED with image handling."""
    team_consensus = round2_discussion[:250] + "..." if len(round2_discussion) > 250 else round2_discussion

    # Add image constraint ONLY for datasets that actually have images (path-vqa, pmc-vqa)
    # For text-only datasets (medqa, medmcqa, pubmedqa), don't add constraint
    image_constraint = ""
    if dataset and dataset.lower() in ['path-vqa', 'pmc-vqa', 'pathvqa', 'pmcvqa']:
        # Only these datasets have images
        if image_mentioned_but_missing:
            image_constraint = """**REMINDER**: The question references an image, but NO IMAGE WAS PROVIDED.
DO NOT make assumptions about what the image shows. Base your final decision ONLY on the textual information.

"""

    if task_type == "mcq":
        base_prompt = ROUND3_PROMPTS["final_ranking_mcq"].format(
            role=role, question=question, options=format_options(options),
            team_consensus=team_consensus
        )
        return image_constraint + base_prompt if image_constraint else base_prompt
    elif task_type == "yes_no_maybe":
        base_prompt = ROUND3_PROMPTS["final_decision_yes_no_maybe"].format(
            role=role, question=question, team_consensus=team_consensus
        )
        return image_constraint + base_prompt if image_constraint else base_prompt
    else:
        base_prompt = ROUND3_PROMPTS["final_ranking_mcq"].format(
            role=role, question=question, options=format_options(options),
            team_consensus=team_consensus
        )
        return image_constraint + base_prompt if image_constraint else base_prompt

def get_adaptive_prompt(base_key: str, task_type: str, **kwargs) -> str:
    """
    Get adaptive prompt based on task type with enhanced round support.

    Args:
        base_key: Base prompt key ("task_analysis", "final_decision", "leadership_synthesis")
        task_type: Task type ("mcq", "multi_choice_mcq", "yes_no_maybe", etc.)
        **kwargs: Additional formatting arguments

    Returns:
        Formatted prompt appropriate for task type and round
    """
    if base_key == "task_analysis":
        if task_type == "mcq":
            return TASK_ANALYSIS_PROMPTS["mcq_task"].format(**kwargs)
        elif task_type == "multi_choice_mcq":
            return TASK_ANALYSIS_PROMPTS["multi_choice_mcq_task"].format(**kwargs)
        elif task_type == "yes_no_maybe":
            return TASK_ANALYSIS_PROMPTS["yes_no_maybe_task"].format(**kwargs)
        else:
            return TASK_ANALYSIS_PROMPTS["mcq_task"].format(**kwargs)

    elif base_key == "final_decision":
        if task_type == "mcq":
            return FINAL_DECISION_PROMPTS["mcq_final"].format(**kwargs)
        elif task_type == "multi_choice_mcq":
            return FINAL_DECISION_PROMPTS["multi_choice_final"].format(**kwargs)
        elif task_type == "yes_no_maybe":
            return FINAL_DECISION_PROMPTS["yes_no_maybe_final"].format(**kwargs)
        else:
            return FINAL_DECISION_PROMPTS["mcq_final"].format(**kwargs)

    elif base_key == "leadership_synthesis":
        if task_type == "multi_choice_mcq":
            return LEADERSHIP_PROMPTS["synthesize_multi_choice"].format(**kwargs)
        elif task_type == "yes_no_maybe":
            return LEADERSHIP_PROMPTS["synthesize_yes_no_maybe"].format(**kwargs)
        else:
            return LEADERSHIP_PROMPTS["synthesize"].format(**kwargs)

    else:
        return f"Unknown prompt key: {base_key}"

def get_dynamic_recruitment_prompt(prompt_type: str, **kwargs) -> str:
    """Get dynamic recruitment prompt with proper formatting."""
    if prompt_type in DYNAMIC_RECRUITMENT_PROMPTS:
        return DYNAMIC_RECRUITMENT_PROMPTS[prompt_type].format(**kwargs)
    elif prompt_type in RECRUITMENT_PROMPTS:
        return RECRUITMENT_PROMPTS[prompt_type].format(**kwargs)
    else:
        raise ValueError(f"Unknown dynamic recruitment prompt type: {prompt_type}")

def get_teamwork_guidance(enabled_components: List[str]) -> str:
    """Generate teamwork guidance text based on enabled components."""
    guidance_map = {
        "use_team_leadership": "- Designate clear leader roles and hierarchy",
        "use_closed_loop_comm": "- Structure communication for acknowledgment",
        "use_mutual_monitoring": "- Enable cross-monitoring and feedback",
        "use_shared_mental_model": "- Foster shared understanding",
        "use_team_orientation": "- Emphasize collaborative integration",
        "use_mutual_trust": "- Encourage open information sharing"
    }

    guidance_lines = [guidance_map[c] for c in enabled_components if c in guidance_map]
    return "\n".join(guidance_lines) if guidance_lines else "- Focus on individual expertise"

def get_all_available_prompts() -> Dict[str, List[str]]:
    """Get dictionary of all available prompt categories and keys."""
    return {
        "agent_system": list(AGENT_SYSTEM_PROMPTS.keys()),
        "leadership": list(LEADERSHIP_PROMPTS.keys()),
        "communication": list(COMMUNICATION_PROMPTS.keys()),
        "monitoring": list(MONITORING_PROMPTS.keys()),
        "mental_model": list(MENTAL_MODEL_PROMPTS.keys()),
        "orientation": list(ORIENTATION_PROMPTS.keys()),
        "trust": list(TRUST_PROMPTS.keys()),
        "recruitment": list(RECRUITMENT_PROMPTS.keys()),
        "dynamic_recruitment": list(DYNAMIC_RECRUITMENT_PROMPTS.keys()),
        "task_analysis": list(TASK_ANALYSIS_PROMPTS.keys()),
        "final_decision": list(FINAL_DECISION_PROMPTS.keys()),
        "round1": list(ROUND1_PROMPTS.keys()),
        "round2": list(ROUND2_PROMPTS.keys()),
        "round3": list(ROUND3_PROMPTS.keys()),
        "extraction": list(EXTRACTION_PROMPTS.keys()),
    }

def validate_prompt_parameters(prompt_template: str, **kwargs) -> bool:
    """Validate that all required parameters are provided for a prompt."""
    required_params = set(re.findall(r'\{(\w+)\}', prompt_template))
    provided_params = set(kwargs.keys())
    missing_params = required_params - provided_params

    if missing_params:
        print(f"Missing parameters: {missing_params}")
        return False
    return True

# ============================================================================
# EXPORTS & BACKWARD COMPATIBILITY
# ============================================================================

# Backward compatibility alias
SYSTEM_PROMPTS = AGENT_SYSTEM_PROMPTS

__all__ = [
    "AGENT_SYSTEM_PROMPTS",
    "SYSTEM_PROMPTS",  # Backward compatibility
    "RECRUITMENT_PROMPTS",
    "ROUND1_PROMPTS",
    "ROUND2_PROMPTS",
    "ROUND3_PROMPTS",
    "LEADERSHIP_PROMPTS",
    "COMMUNICATION_PROMPTS",
    "MONITORING_PROMPTS",
    "MENTAL_MODEL_PROMPTS",
    "ORIENTATION_PROMPTS",
    "TRUST_PROMPTS",
    "TASK_ANALYSIS_PROMPTS",
    "FINAL_DECISION_PROMPTS",
    "DYNAMIC_RECRUITMENT_PROMPTS",
    "EXTRACTION_PROMPTS",
    "format_options",
    "format_agent_analyses",
    "get_round1_prompt",
    "get_round2_prompt",
    "get_round3_prompt",
    "get_adaptive_prompt",
    "get_dynamic_recruitment_prompt",
    "get_teamwork_guidance",
    "get_all_available_prompts",
    "validate_prompt_parameters",
]
