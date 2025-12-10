# Prompt Templates for Multi-Agent Medical Reasoning System

This document provides a comprehensive reference for all prompt templates used in our modular multi-agent collaboration framework. The system employs specialized prompts for agent recruitment, independent analysis, collaborative discussion, and five teamwork components.

---

## Table of Contents

1. [Single-Agent Baseline Prompts](#single-agent-baseline-prompts)
2. [Multi-Agent System Prompts](#multi-agent-system-prompts)
   - [Recruitment Phase](#recruitment-phase)
   - [Round 2: Initial Prediction](#round-2-initial-prediction)
   - [Round 2: Collaborative Discussion](#round-2-collaborative-discussion)
   - [Round 3: Final Ranking](#round-3-final-ranking)
3. [Teamwork Component Prompts](#teamwork-component-prompts)
   - [Shared Mental Model (SMM)](#1-shared-mental-model-smm)
   - [Leadership (L)](#2-leadership-l)
   - [Team Orientation (TO)](#3-team-orientation-to)
   - [Trust Network (T)](#4-trust-network-t)
   - [Mutual Monitoring (MM)](#5-mutual-monitoring-mm)
4. [Utility Prompts](#utility-prompts)

---

## Single-Agent Baseline Prompts

These prompts are used for single-agent baselines to compare against multi-agent performance.

### Few-Shot Multiple Choice

```
{{instruction}}
The following are multiple choice questions (with answers) about medical knowledge.

{{few_shot_examples}}

{{context}} **Question:** {{question}} {{answer_choices}}

**Answer:**
```

**Purpose:** Provides 2-5 example questions with solutions before the target question.

**Used in:** Zero-shot, Few-shot baseline evaluations

---

### Chain-of-Thought (CoT) Multiple Choice

```
{{instruction}}
The following are multiple choice questions (with answers) about medical knowledge.

{{few_shot_examples_with_CoT_reasoning}}

{{context}} **Question:** {{question}} {{answer_choices}}

**Answer:**
```

**Purpose:** Includes step-by-step reasoning in examples to elicit reasoning from the model.

**Used in:** CoT baseline evaluation

---

### Ensemble Refinement Multiple Choice

```
{{instruction}}
The following are multiple choice questions (with answers) about medical knowledge.

{{few_shot_examples_with_CoT_reasoning}}

{{context}} **Question:** {{question}} {{answer_choices}}

{{reasoning_paths_from_previous_attempts}}

**Answer:**
```

**Purpose:** Iterative refinement approach where model sees previous reasoning attempts.

**Used in:** Advanced single-agent baseline

---

## Multi-Agent System Prompts

### Recruitment Phase

#### 1. Complexity Analysis (Determine Agent Count)

```
You are a medical expert who conducts initial assessment. Analyze this medical
question and determine the difficulty/complexity and optimal agent count (2-4).

Question: {{question}}
Options: {{options}}

Complexity levels:
1. LOW: A PCP or general physician can answer this simple medical knowledge
   question without consulting specialists (→ 2 agents)
2. MODERATE: A PCP or general physician needs consultation with specialists
   in a team (→ 3 agents)
3. HIGH: Multi-departmental specialists required with significant team effort
   and cross-consultation (→ 4 agents)

Consider:
- Complexity of medical reasoning required
- Distinct medical domains involved
- Need for diverse specialist perspectives

Respond with ONLY a number: 2, 3, or 4

Number of agents:
```

**API Calls:** 1 call
**Output:** Integer (2, 3, or 4)
**Purpose:** Dynamic agent recruitment based on question complexity

---

#### 2. Role Assignment (Specialist Selection)

```
You are an experienced medical expert who recruits a group of experts with
diverse identities for this medical query.

Question: {{question}}
Options: {{options}}

Create {{n_agents}} specialized medical expert roles with complementary expertise.

Also specify the communication structure between experts using:
- "==" for equal collaboration (e.g., A == B)
- ">" for hierarchical consultation (e.g., A > B means A leads, B consults)

Format EXACTLY as:
AGENT_1: [Role] - [One-line expertise]
AGENT_2: [Role] - [One-line expertise]
(Include only {{n_agents}} agents)
COMMUNICATION: [Structure, e.g., "Agent_1 == Agent_2 > Agent_3" or "Independent"]

Example:
AGENT_1: Cardiologist - Expert in cardiovascular disease and ECG interpretation
AGENT_2: Emergency Physician - Specialist in acute care and rapid decision-making
COMMUNICATION: Agent_1 == Agent_2

Create {{n_agents}} roles:
```

**API Calls:** 1 call
**Output:** Agent roles, expertise descriptions, communication structure
**Purpose:** Assigns specialized medical roles with diversity and complementarity

---

### Round 2: Initial Prediction

#### Independent Analysis (MCQ)

```
You are a {{role}} with expertise in {{expertise}}.

Analyze the following medical question and provide your initial prediction.

Question: {{question}}

Options:
{{options}}

**ALGO R2 - Initial Prediction Phase**

Provide your response in the following FORMAT (REQUIRED):

RANKING: [List ALL options in order of likelihood, e.g., D, A, B, C]

JUSTIFICATION: [In 2-3 sentences, explain your reasoning. Focus on:
(1) Classic clinical presentation recognition,
(2) Key clinical features (age, ethnicity, symptoms, family history),
(3) Evidence-based reasoning]

FACTS: [List 2-5 key medical facts that support your reasoning, e.g.,
"Sickle cell disease is most common in African Americans",
"Hand pain suggests vaso-occlusive crisis"]

Your response:
```

**API Calls:** N calls (one per agent, run in parallel)
**Output:** Ranked options, justification, supporting facts
**Purpose:** Independent initial analysis by each specialist without team influence

---

#### Independent Analysis (Yes/No/Maybe)

```
You are a {{role}} with expertise in {{expertise}}.

Analyze the following medical question using your specialized knowledge.

Question: {{question}}

Instructions:
1. Consider only the information explicitly provided
2. Provide your answer (yes, no, or maybe) followed by brief reasoning

Your analysis:
```

**API Calls:** N calls (parallel)
**Output:** Yes/No/Maybe + reasoning
**Purpose:** Biomedical research question answering (PubMedQA format)

---

### Round 2: Collaborative Discussion

```
You are a {{role}} with expertise in {{expertise}}. You are collaborating with
other medical experts in a team.

Question: {{question}}
Options: {{options}}

**Your R1 view**: {{your_round1_summary}}

**Teammates' R1 views**:
{{other_agents_summaries}}

**Round 2 - Collaborative Discussion**: After reviewing opinions from other
medical agents in your team, provide your ranked prediction. Indicate whether
you agree/disagree with other experts and deliver your opinion in a way to
convince them with clear reasoning.

FORMAT (REQUIRED):
RANKING: [List all options in order of likelihood, e.g., A, B, C, D]

JUSTIFICATION: [In 2-3 sentences, state if you agree/disagree with consensus,
have new insights, or maintain your position. Explain why with evidence to
convince other experts.]

Your response:
```

**API Calls:** N calls (sequential, each agent sees previous round)
**Output:** Revised ranking, justification with consensus/disagreement
**Purpose:** Collaborative deliberation where agents consider peer opinions

**Token Optimization:**
- Your R1 summary: Truncated to 150 characters
- Other agents' summaries: Truncated to 200 characters each

---

### Round 3: Final Ranking

#### Final Decision (MCQ)

```
You are a {{role}} and final medical decision maker who reviews all opinions
from different medical experts.

Question: {{question}}
Options: {{options}}

**Team consensus**: {{team_consensus}}

**Round 3 - Final Decision**: Review all team opinions and make your FINAL
independent ranking of ALL options. You have the authority to make the final
decision based on your medical expertise and the team's collective reasoning.

**CRITICAL REMINDERS**:
- Consider classic disease presentations FIRST (e.g., African-American child +
  hand pain = sickle cell)
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

Your ranking:
```

**API Calls:** N calls (parallel)
**Output:** Final ranked list, confidence, explanation
**Purpose:** Each agent makes independent final decision after full discussion

**Token Optimization:**
- Team consensus: Truncated to 250 characters

---

#### Final Decision (Yes/No/Maybe)

```
You are a {{role}}.

Question: {{question}}

**Team consensus**: {{team_consensus}}

**Round 3**: Provide final answer (yes/no/maybe).

Format:
FINAL ANSWER: [yes/no/maybe]
CONFIDENCE: [High/Medium/Low]
EXPLANATION: [Your reasoning]

Your decision:
```

**API Calls:** N calls (parallel)
**Output:** Final answer, confidence, explanation
**Purpose:** Final independent decision for research questions

---

## Teamwork Component Prompts

These modular prompts are injected based on the teamwork configuration `C = {SMM, L, TO, T, MM}`.

### 1. Shared Mental Model (SMM)

**Component Symbol:** SMM
**State:** OFF = Independent agents | ON = Shared knowledge context
**Dependencies:** None

#### Shared Mental Model Contribution

```
Contribute to shared mental model:
1. State your understanding explicitly
2. Check alignment on task interpretation
3. Establish shared terminology
4. Clarify reasoning
5. Be precise, concise

Ensure team alignment.
```

**Injection Point:** Prepended to Round 2 and Round 3 prompts when SMM is enabled
**Purpose:** Encourages agents to align on terminology and interpretation

---

#### Team Mental Model (Extended)

```
Collaboratively answer: {{task_description}}

Team-related models:
1. Understand each other's specialty, defer appropriately
2. Use clear, structured communication with confirmation
3. Agree on approach for each question
4. Psychological safety - open to being wrong
5. Efficient division of labor

Task-specific:
Objective: {{objective}}
Criteria: {{criteria}}
```

**Data Structure Updated:**
```python
SMM = {
  question_analysis: String,      # Trick detection (1-2 sentences)
  verified_facts: Set[String],    # Consensus facts from R2
  debated_points: List[String]    # Controversies from R3+MM
}
```

**Update Points:** After R1 (question analysis), Post-R2 (facts), Post-MM (debates)

---

### 2. Leadership (L)

**Component Symbol:** L
**State:** OFF = Passive coordinator | ON = Active orchestrator
**Dependencies:** Enhances SMM, TO, T

#### Team Leadership

```
As team leader:
1. Facilitate problem solving
2. Set performance expectations
3. Synchronize contributions
4. Evaluate team functioning
5. Clarify roles
6. Provide feedback

Be precise, concise, to the point.
```

**Injection Point:** Designated leader agent receives this in system prompt
**Powers:**
1. Extract consensus facts from agent responses
2. Create formal medical reports
3. Mediate discussions after each turn
4. Resolve ties with override capability

---

#### Leadership Synthesis (MCQ)

```
As leader, synthesize team's perspectives into consensus.

Context: {{context}}

Create solution that:
1. Incorporates key insights
2. Balances perspectives
3. Provides clear reasoning

You MUST begin: "ANSWER: X" (option letter).

Final solution:
```

**API Calls:** 1 call per mediation (after each discussion round)
**Output:** Synthesized answer or mediation guidance
**Purpose:** Leader integrates team perspectives and resolves conflicts

---

#### Leadership Synthesis (Yes/No/Maybe)

```
As leader, synthesize for research question.

Context: {{context}}

You MUST begin: "ANSWER: X" (yes/no/maybe).

Final answer:
```

**Correction Authority:** Leader can override consensus if reasoning is flawed (MMedAgent-RL pattern)

---

### 3. Team Orientation (TO)

**Component Symbol:** TO
**State:** OFF = Generic roles, equal weights | ON = Specialist roles, hierarchical weights
**Dependencies:** Best with L

#### Team Orientation Guidance

```
Demonstrate team orientation:
1. Consider alternative solutions from teammates
2. Value teammates' perspectives even when different
3. Prioritize team goals over individual achievement
4. Engage in information sharing and goal setting
5. Enhance performance through coordination

Value and incorporate diverse perspectives.
```

**Injection Point:** Prepended to Round 2 prompts when TO is enabled
**Purpose:** Encourages collaborative mindset and integration of diverse views

---

#### Hierarchical Weights

**Features:**
- Assigns specific medical specialties (Cardiologist, Radiologist, etc.)
- Hierarchical weights:
  - N=2: {0.6, 0.4}
  - N=3: {0.5, 0.3, 0.2}
  - N=4: {0.4, 0.3, 0.2, 0.1}
- Weights hidden from agents but used in aggregation
- Formal medical report generation

**Example:**
```
AGENT_1: Cardiologist (Weight: 0.5) - Expert in cardiovascular disease
AGENT_2: Pulmonologist (Weight: 0.3) - Specialist in respiratory conditions
AGENT_3: Internal Medicine (Weight: 0.2) - General medical expertise
```

---

### 4. Trust Network (T)

**Component Symbol:** T
**State:** OFF = Equal weights (0.8) | ON = Dynamic scores (0.4-1.0)
**Dependencies:** L can evaluate

#### High Trust Environment

```
HIGH TRUST environment:
- Share information with confidence
- Rely on teammates without excessive verification
- Express uncertainty safely
- Expect protection of contributions
- Focus on task, not monitoring
```

**Injection Point:** System prompt when T is enabled and trust score > 0.7
**Trust Score Model:**
```
T(aᵢ) ∈ [0.4, 1.0]  # Default: 0.8
Update: T_new = α · T_old + (1 - α) · Q(aᵢ)  where α = 0.7
```

**Evaluation Criteria:** Fact accuracy, reasoning completeness, confidence alignment

---

#### Low Trust Environment

```
LOW TRUST environment:
- Be careful with sensitive info
- Verify information carefully
- Be explicit about reasoning
- Demonstrate reliability consistently
- Build trust gradually
```

**Injection Point:** System prompt when T is enabled and trust score ≤ 0.7
**Trust-Weighted Voting:**
```
Score(Aⱼ) = Σᵢ T(aᵢ) · BordaPoints(aᵢ, Aⱼ)
```

---

#### Mutual Trust Base

```
Foster mutual trust:
1. Share information openly
2. Admit mistakes, accept feedback
3. Assume positive intentions
4. Respect expertise and rights
5. Ask for help when needed

Be precise, concise.
```

**Purpose:** Baseline trust-building guidance

---

### 5. Mutual Monitoring (MM)

**Component Symbol:** MM
**State:** OFF = No inter-round validation | ON = Pairwise challenges
**Dependencies:** Requires L

#### Mutual Monitoring Guidance

```
Engage in mutual monitoring:
1. Track teammate performance
2. Check for errors/omissions
3. Provide constructive feedback
4. Ensure quality
5. Be precise, concise

Do this respectfully to improve team decisions.
```

**Injection Point:** Prepended to Round 3 prompts when MM is enabled
**Purpose:** Enables peer review and quality control

---

#### Mutual Monitoring Protocol (Leader-Initiated)

**Step 1: Leader Identifies Weakest Reasoning**
```
As leader, review agent responses and identify the weakest reasoning.

Agent responses:
{{agent_responses}}

Identify which agent has the weakest or most questionable reasoning.

Response format:
TARGET_AGENT: {{agent_id}}
CONCERN: [Specific issue with their reasoning]
```

**Step 2: Leader Raises Concern**
```
{{target_agent}}, I have a concern about your reasoning:

{{concern_description}}

Please respond:
1. Do you accept this concern and revise your answer?
2. Or justify why your reasoning is sound?
```

**Step 3: Agent Responds to Concern**
```
Responding to leader's concern: {{concern}}

Your response:
1. [ACCEPT] - I acknowledge the issue and revise my answer to: {{revised_answer}}
2. [JUSTIFY] - I maintain my position because: {{justification}}

Your response:
```

**Step 4: Leader Evaluates Response**
```
Evaluate the agent's response to the concern.

Original concern: {{concern}}
Agent's response: {{agent_response}}

Provide:
QUALITY_SCORE: [0.0-1.0, where 1.0 = excellent response]
TRUST_UPDATE: [Should trust increase, decrease, or stay same?]
RATIONALE: [Brief explanation]
```

**Placement:** Between R3 turns only (not after final turn)
**API Calls:** 3-4 calls if MM is enabled

---

## Utility Prompts

### Closed-Loop Communication

```
Use clear, specific communication. Acknowledge receipt and confirm understanding.
```

**Purpose:** Ensures message receipt and comprehension

---

### Receiver Acknowledgment

```
Message from {{sender_role}}: "{{sender_message}}"

Acknowledge:
1. "Understood: [key point]"
2. Your response

Be precise, concise.
```

**Purpose:** Structured communication with confirmation loop

---

### Answer Clarification (MCQ)

```
Your response didn't clearly indicate answer.

Question: {{question}}
Options: {{options}}
Your response: {{previous_response}}

Provide ONLY letter (A, B, C, D):
Answer:
```

**API Calls:** 1 additional call if answer extraction fails
**Purpose:** Fallback to ensure valid answer format

---

### Ranking Clarification

```
Your response didn't follow ranking format.

Provide ranking in EXACT format:

RANKING:
1. [Option]
2. [Option]
3. [Option]
4. [Option]

Your ranking:
```

**API Calls:** 1 additional call if ranking extraction fails
**Purpose:** Ensures consistent ranking format for Borda count

---

## Prompt Selection Logic

### Task Type Detection

```python
if "yes/no/maybe" in question_format:
    task_type = "yes_no_maybe"
    prompts = {
        "round1": "independent_analysis_yes_no_maybe",
        "round3": "final_decision_yes_no_maybe"
    }
elif "multiple answers" in question_format:
    task_type = "multi_choice_mcq"
    prompts = {
        "round1": "independent_analysis_mcq",
        "round3": "final_ranking_mcq"
    }
else:
    task_type = "mcq"
    prompts = {
        "round1": "independent_analysis_mcq",
        "round3": "final_ranking_mcq"
    }
```

---

### Teamwork Component Injection

```python
# Build system prompt with enabled components
system_prompt = AGENT_SYSTEM_PROMPTS["medical_expert"]

if config.use_shared_mental_model:
    system_prompt += "\n\n" + MENTAL_MODEL_PROMPTS["shared_mental_model"]

if config.use_team_orientation:
    system_prompt += "\n\n" + ORIENTATION_PROMPTS["team_orientation"]

if config.use_mutual_trust:
    trust_level = "high_trust" if agent.trust_score > 0.7 else "low_trust"
    system_prompt += "\n\n" + TRUST_PROMPTS[trust_level]

if config.use_mutual_monitoring:
    system_prompt += "\n\n" + MONITORING_PROMPTS["mutual_monitoring"]

if config.use_team_leadership and agent.is_leader:
    system_prompt += "\n\n" + LEADERSHIP_PROMPTS["team_leadership"]
```

---

## Token Optimization Strategies

### 1. Truncation in Round 2
- **Your R1 summary:** 150 characters max
- **Other agents' summaries:** 200 characters each
- **Rationale:** Reduces context size while preserving key information

### 2. Truncation in Round 3
- **Team consensus:** 250 characters max
- **Rationale:** Focuses on main points from discussion

### 3. Parallel API Calls
- **Round 2 (R2):** N agents in parallel → O(1) time
- **Round 3 (R3):** N agents in parallel → O(1) time
- **Rationale:** Minimizes latency despite multiple agents

### 4. Conditional Component Injection
- Only inject teamwork guidance for enabled components
- **Example:** If SMM=OFF, skip mental model prompts entirely
- **Rationale:** Reduces prompt length by ~30% when components are disabled

---

## API Call Complexity Summary

For N agents, n_turns discussion rounds:

| Configuration | R1 | R2 | R3 (2 turns) | R3 (3 turns) | Total (2T) | Total (3T) |
|---------------|----|----|--------------|--------------|------------|-----------|
| **Base** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+SMM** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+Leadership** | 2 | N+1 | 2N+2 | 3N+3 | 2N+5 | 3N+6 |
| **+TeamO** | 2 | N+1 | 2N+2 | 3N+3 | 2N+5 | 3N+6 |
| **+Trust** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+MM** | 2 | N+1 | 2N+3 | 3N+6 | 2N+6 | 3N+9 |
| **ALL ON** | 2 | N+1 | 2N+5 | 3N+9 | 2N+8 | 3N+12 |

**Example** (N=3, n_turns=2):
- Base: 2(3) + 3 = **9 calls**
- ALL ON: 2(3) + 8 = **14 calls** (+56% overhead for full teamwork)

---

## References

- **System Architecture:** See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for algorithm specification
- **Implementation:** See `utils/prompts.py` for actual prompt code
- **Teamwork Components:** See `teamwork_components/` for component implementations

---

*This document is automatically generated from the codebase. Last updated: 2025-01-16*
