
  Implementation Architecture

  Directory Structure

  E:\SLM-TeamMedAgents\
  ├── teamwork_components/
  │   ├── __init__.py                    # Component exports
  │   ├── config.py                      # TeamworkConfig class
  │   ├── shared_mental_model.py         # SMM implementation
  │   ├── leadership.py                  # Leadership coordinator
  │   ├── team_orientation.py            # Role specialization & weights
  │   ├── trust_network.py               # Trust scoring system
  │   └── mutual_monitoring.py           # Inter-round validation
  │
  ├── adk_agents/
  │   ├── multi_agent_system_adk.py      # [UPDATE] Add teamwork orchestration
  │   ├── dynamic_recruiter_adk.py       # [UPDATE] Add Leadership + SMM
  │   ├── three_round_debate_adk.py      # [UPDATE] Add all component support
  │   └── decision_aggregator_adk.py     # [UPDATE] Add Trust-weighted voting
  │
  └── run_simulation_adk.py              # [UPDATE] Add CLI flags for components

  Component Design Summary

  1. Shared Mental Model (SMM) - shared_mental_model.py
  - Passive knowledge repository
  - Stores: question_analysis, verified_facts, debated_points
  - Thread-safe updates
  - Serializable for logging

  2. Leadership - leadership.py
  - Extends recruiter agent with active orchestration
  - Powers: mediate discussions, break ties, update SMM
  - Delegates to leader agent for LLM-powered decisions

  3. Team Orientation - team_orientation.py
  - Specialist role assignment with medical domains
  - Hierarchical weights (0.5, 0.3, 0.2) hidden from agents
  - Formal medical report generation

  4. Trust Network - trust_network.py
  - Dynamic trust scores per agent (0.4-1.0, default 0.8)
  - Update triggers: post-R2, post-MM, post-R3
  - Evaluation criteria: fact accuracy, reasoning quality

  5. Mutual Monitoring - mutual_monitoring.py
  - Inter-turn challenge protocol
  - Leader identifies weakest reasoning → challenge → response → update
  - Updates both Trust and SMM

  Key Integration Points

  Phase 1: Recruitment (R1)
  - Recruiter analyzes question → N agents
  - [SMM ON] Detect tricks, store in SMM
  - [Leadership ON] Recruiter becomes Leader
  - [TeamO ON] Leader defines specialist roles using SMM

  Phase 2: Initial Prediction (R2)
  - N parallel agent predictions
  - [SMM ON] Agents receive SMM context
  - Post-R2 processing (1 combined call):
    - [SMM] Extract verified facts
    - [TeamO] Leader creates formal report
    - [Trust] Evaluate agent responses

  Phase 3: Collaborative Discussion (R3)
  - Multiple turns with round-robin discourse
  - [Leadership ON] Leader mediates each turn
  - [MM ON] Between-turn challenges
  - [Trust] Update after MM
  - [SMM] Update debated points

  Phase 4: Aggregation
  - [Trust ON] Weighted voting by trust scores
  - [TeamO ON] Hierarchical weights if Trust OFF
  - [Leadership ON] Tie-breaking with correction power



# API Call Plan

=== ROUND 1: RECRUIT (2 API calls) ===

API Call 1: Recruiter Agent
├─ Input: Question, enabled components config
├─ Analyzes complexity → determines N agents (2-4)
├─ [SMM] Adds question_analysis (trick detection, 1-line)
├─ [Leadership] Self-designates as Leader
└─ Output: Agent count, SMM entry, roles outline

API Call 2: Agent Initialization
├─ [TeamO + Leadership] Define N specialist agents with:
│  ├─ Specific medical roles (cardiology, radiology, etc.)
│  ├─ Hierarchical weights {0.5, 0.3, 0.2} (hidden from agents)
│  └─ Enhanced using SMM info if Leadership enabled
├─ [No TeamO] Generic medical agents, equal weights
└─ Output: Initialized agent profiles

=== ROUND 2: INITIAL PREDICTION (N+1 API calls) ===

API Calls 3-(N+2): Parallel Agent Predictions
├─ Each agent independently receives:
│  ├─ Question + options
│  ├─ [SMM] Current SMM content (if enabled)
│  └─ [TeamO] Role-specific instructions (weights hidden)
└─ Each outputs:
   ├─ Ranked list {B, C, A, D}
   ├─ Justification (2 sentences max)
   └─ FACTS list (2-5 items)

API Call (N+3): Post-R2 Processing (Combined)
├─ [SMM] Extract verified_facts (intersection logic)
│  └─ Leadership: Leader updates | Else: Recruiter/automated
├─ [TeamO] Leader creates formal medical report:
│  ├─ Neutral tone summary
│  ├─ Consensus facts
│  ├─ Conflicting arguments noted
│  └─ Concise, point-by-point format
├─ [Trust] Evaluate R2 quality → update trust_scores
│  └─ Leadership: Leader evaluates | Else: Judge agent
└─ Output: Updated SMM, formal report, trust scores

=== ROUND 3: COLLABORATIVE DISCUSSION ===

Round 3.1: First Discussion Turn
├─ All agents receive:
│  ├─ All R2 outputs
│  ├─ [SMM] Updated SMM
│  ├─ [Trust] Trust scores as context hints ("0.8 - trusted")
│  ├─ [TeamO] Formal medical report
│  └─ [Leadership] Explicit summary: "Resolve: X vs Y controversy"
├─ Round-robin discourse (n API calls for n agents)
└─ [Leadership] Leader mediates at turn end (1 API call)

=== ROUND 3.5: MUTUAL MONITORING (if enabled, after each non-final turn) ===
├─ Triggered: After R3.1 if n_turns=2, after R3.1 and R3.2 if n_turns=3
├─ Leader selects weakest-reasoning agent to challenge
├─ API Call A: Leader raises specific concern
├─ API Call B: Challenged agent responds (accept/justify)
├─ [Optional] API Call C: Leader acknowledges if justification disputed
└─ API Call D: Combined update:
   ├─ [Trust] Update trust_scores based on MM outcome
   └─ [SMM] Add debated_points from discussion

Repeat for remaining turns (if n_turns = 3, do R3.2 → MM → R3.3)

Round 3.Final: Last Discussion Turn
├─ Round-robin discourse (n API calls)
├─ [Leadership] Leader mediates (1 API call)
├─ No MM after final turn
└─ Each agent outputs: Final ranked list + concise reasoning

=== AGGREGATION ===

Step 1: Vote Calculation
├─ [Trust] Weighted vote using trust_scores
└─ [No Trust] Simple majority OR hierarchical weights from R1

Step 2: Conflict Resolution (Optional)
├─ If tie detected AND Leadership enabled:
│  └─ API Call: Leader breaks tie with correction power
└─ Output: Final answer {A/B/C/D} + decision rationale
```

---

## API Call Count Summary

| Configuration | R1 | R2 | R3 (2 turns) | R3 (3 turns) | Total (2T) | Total (3T) |
|---------------|----|----|-------------|-------------|-----------|-----------|
| **Base** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+SMM** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+Leadership** | 2 | N+1 | 2N+2 | 3N+3 | 2N+5 | 3N+6 |
| **+TeamO** | 2 | N+1 | 2N+2 | 3N+3 | 2N+5 | 3N+6 |
| **+Trust** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+MM** | 2 | N+1 | 2N+3 | 3N+6 | 2N+6 | 3N+9 |
| **ALL ON** | 2 | N+1 | 2N+5 | 3N+9 | 2N+8 | 3N+12 |

*N = number of agents (2-4)*  
*Tie-breaking (Leadership) adds +1 if triggered*

---

## Modularity Matrix

| Component | OFF State | ON State | Impact | Dependencies |
|-----------|-----------|----------|--------|--------------|
| **SMM** | Agents work independently | Shared KB context in prompts | Passive info | None |
| **Leadership** | Recruiter = generic coordinator | Recruiter = active Leader with correction power | Active intervention | Enhances SMM/TeamO/Trust |
| **Team Orientation** | Generic "medical expert" agents, equal weights | Specialized roles + hierarchical weights + formal report | Structured roles | Best with Leadership |
| **Trust Network** | All agents = 0.8 (equal), simple majority | Dynamic scores (0.4-1.0), weighted aggregation | Aggregation weights | Leadership can evaluate |
| **Mutual Monitoring** | Skip R3.5 entirely | Pairwise challenges between R3 turns | Fact-checking phase | Updates Trust/SMM |

---

## Key Design Principles

### 1. SMM as Shared Context Layer
- **Nature**: Passive knowledge repository, not a decision-maker
- **Content**: Question tricks, verified facts, debated points
- **Access**: Read-only for agents, write-only by Leader/Recruiter
- **Benefit**: Reduces redundant reasoning, grounds discussion

### 2. Leadership as Active Orchestrator
- **Dual Role**: Recruiter + Leader (same agent)
- **Powers**: Update SMM, create formal reports, mediate, resolve ties
- **Correction Authority**: Can override consensus if reasoning flawed (MMedAgent-RL pattern)
- **When OFF**: Automated rule-based updates, no mediation

### 3. Mutual Monitoring as Inter-Round Validation
- **Placement**: Between R3 turns only (not after final turn)
- **Trigger**: Leader identifies weakest reasoning
- **Protocol**: Closed-loop challenge → response → acknowledgment
- **Output**: Trust score adjustments, SMM updates
- **Efficiency**: 3-4 API calls per MM phase

### 4. Trust Network as Dynamic Weighting
- **Representation**: Metadata scores (0.4-1.0) per agent
- **Update Points**: After R2, after MM, after R3
- **Evaluation Criteria**: Fact accuracy, reasoning completeness, response quality
- **Impact**: Only affects final aggregation weights, not agent behavior
- **Default**: 0.8 (equal) when disabled

### 5. Team Orientation as Structural Specialization
- **Activation**: Coupled with Leadership for best results
- **Components**: 
  - Explicit medical specialties (not generic "expert")
  - Hierarchical weights {0.5, 0.3, 0.2} (hidden from agents)
  - Formal medical report (neutral, structured summary)
- **When OFF**: Generic agents, equal weights, no formal report

### 6. Efficiency Philosophy
- **Base System**: Minimal calls (2N+3 for 2 turns)
- **Optional Components**: Each adds specific value with bounded cost
- **Parallel Execution**: R2 predictions run simultaneously
- **Batched Operations**: Post-R2 processing in single call
- **Selective Activation**: Only enable components when needed

---

## Algorithm (Pseudocode Format)
```
ALGORITHM: Modular Multi-Agent Medical Decision System

INPUT: 
  - question: Medical query (MCQ or diagnosis)
  - config: {smm, leadership, team_orientation, trust, mutual_monitoring}
  - n_turns: Discussion rounds (2 or 3)

OUTPUT:
  - final_answer: Ranked list {A/B/C/D} with rationale

INITIALIZE:
  - smm ← SharedMentalModel() if config.smm else None
  - trust_scores ← {agent_i: 0.8 for all agents}
  - leader ← None

─────────────────────────────────────────────────────────
ROUND 1: RECRUIT (2 API calls)
─────────────────────────────────────────────────────────

CALL Recruiter(question, config):
  complexity ← analyze_complexity(question)
  N ← determine_agent_count(complexity)  // 2-4 agents
  
  IF config.smm:
    smm.question_analysis ← detect_tricks(question)  // 1-liner
  
  IF config.leadership:
    leader ← self  // Recruiter becomes Leader
  
  RETURN N, smm, leader

CALL InitializeAgents(N, question, config, smm, leader):
  agents ← []
  
  IF config.team_orientation AND config.leadership:
    roles ← leader.define_specialist_roles(question, N, smm)  // Use SMM context
    weights ← assign_hierarchical_weights(N)  // {0.5, 0.3, 0.2}
  ELSE:
    roles ← create_generic_roles(N)
    weights ← equal_weights(N)  // All 0.8
  
  FOR i = 1 to N:
    agents[i] ← Agent(role=roles[i], weight=weights[i])
  
  RETURN agents, weights

─────────────────────────────────────────────────────────
ROUND 2: INITIAL PREDICTION (N+1 API calls)
─────────────────────────────────────────────────────────

PARALLEL FOR each agent in agents:
  context ← {question, options}
  
  IF config.smm:
    context.add(smm)
  
  IF config.team_orientation:
    context.add(role_instructions[agent])
  
  response[agent] ← CALL agent.predict(context)
    // Returns: {ranked_list, justification[2 sentences], facts[2-5]}

CALL PostR2Processing(responses, config, leader):
  
  IF config.smm:
    IF config.leadership:
      smm.verified_facts ← leader.extract_consensus_facts(responses)
    ELSE:
      smm.verified_facts ← automated_fact_intersection(responses)
  
  IF config.team_orientation:
    formal_report ← leader.create_medical_report(responses, smm)
      // Neutral tone, consensus + conflicts, concise
  
  IF config.trust:
    IF config.leadership:
      trust_scores ← leader.evaluate_r2_quality(responses)
    ELSE:
      trust_scores ← judge_agent.evaluate(responses)
  
  RETURN smm, formal_report, trust_scores

─────────────────────────────────────────────────────────
ROUND 3: COLLABORATIVE DISCUSSION (2N+5 or 3N+9 calls)
─────────────────────────────────────────────────────────

FOR turn = 1 to n_turns:
  
  // Discussion Turn
  context ← {all_r2_outputs}
  
  IF config.smm:
    context.add(smm)
  
  IF config.trust:
    context.add(trust_hints)  // "Agent X - 0.8 trusted"
  
  IF config.team_orientation:
    context.add(formal_report)
  
  IF config.leadership:
    context.add(leader.generate_summary())  // "Resolve: controversy X"
  
  FOR each agent in round_robin_order(agents):
    discourse[agent][turn] ← CALL agent.discuss(context, previous_discourse)
  
  IF config.leadership:
    mediation ← CALL leader.mediate(discourse[turn])
    context.add(mediation)
  
  // Mutual Monitoring (only between turns, not after last)
  IF config.mutual_monitoring AND turn < n_turns:
    
    target ← leader.select_weakest_agent(discourse[turn])
    
    concern ← CALL leader.raise_concern(target, discourse[target][turn])
    response ← CALL target.respond_to_concern(concern)
    
    IF response.type == "dispute":
      ack ← CALL leader.evaluate_justification(response)
    
    CALL UpdateTrustAndSMM(concern, response, config):
      IF config.trust:
        trust_scores ← update_based_on_mm(concern, response)
      IF config.smm:
        smm.debated_points.add(extract_key_points(concern, response))

// Final Predictions
FOR each agent in agents:
  final_response[agent] ← CALL agent.final_prediction(all_context)
    // Returns: {ranked_list, reasoning}

─────────────────────────────────────────────────────────
AGGREGATION
─────────────────────────────────────────────────────────

IF config.trust:
  votes ← weighted_vote(final_responses, trust_scores)
ELSE:
  IF config.team_orientation:
    votes ← hierarchical_vote(final_responses, weights)
  ELSE:
    votes ← simple_majority(final_responses)

// Tie-breaking
IF is_tie(votes) AND config.leadership:
  final_answer ← CALL leader.resolve_conflict(final_responses, trust_scores)
    // Correction power: Can override consensus
ELSE:
  final_answer ← top_vote(votes)

RETURN final_answer with rationale

END ALGORITHM
```

---

## Component Interaction Flow
```
┌──────────────┐
│  Recruiter   │──┬──[SMM ON]──> Detect tricks
└──────────────┘  │
                  └──[Leadership ON]──> Self-designate as Leader
                          │
                          ▼
┌──────────────────────────────────────┐
│   Agent Initialization               │
│  [TeamO ON]: Specialist roles        │
│  [TeamO OFF]: Generic experts        │
└──────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────┐
│   Round 2: Predictions (Parallel)   │
│  [SMM ON]: Include shared KB         │
└──────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────┐
│   Post-R2 Processing                 │
│  [SMM]: Update facts                 │
│  [TeamO]: Create formal report       │
│  [Trust]: Evaluate quality           │
│  [Leader]: Orchestrates all updates  │
└──────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────┐
│   Round 3: Discussion Loop           │
│  [Leadership]: Mediate each turn     │
│  [MM]: Challenge between turns       │
│  [Trust]: Update after MM            │
│  [SMM]: Update debated points        │
└──────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────┐
│   Aggregation                        │
│  [Trust]: Weighted vote              │
│  [Leadership]: Tie-breaking          │
└──────────────────────────────────────┘