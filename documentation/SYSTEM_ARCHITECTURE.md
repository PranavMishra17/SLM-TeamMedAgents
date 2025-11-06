# System Architecture & Algorithm Specification

## Overview

A modular multi-agent framework for collaborative medical reasoning featuring five independently toggleable teamwork components. The system supports 2-4 dynamically recruited specialist agents engaging in structured deliberation over multiple rounds with final aggregation via weighted voting schemes.

---

## 1. Input-Output Specification

**Input:**
- Medical question `Q` (multiple-choice or open-ended)
- Answer options `{A₁, A₂, ..., Aₙ}` (if applicable)
- Teamwork configuration `C = {SMM, L, TO, T, MM}` (binary flags)
- Discussion parameters: `n_turns ∈ {2, 3}`

**Output:**
- Final answer `Â ∈ {A₁, A₂, ..., Aₙ}`
- Decision rationale with supporting evidence
- Teamwork metrics: SMM content, trust scores, convergence data

---

## 2. Modular Components

| Component | Symbol | OFF State | ON State | Dependencies |
|-----------|--------|-----------|----------|--------------|
| **Shared Mental Model** | SMM | Independent agents | Shared knowledge context | None |
| **Leadership** | L | Passive coordinator | Active orchestrator | Enhances SMM, TO, T |
| **Team Orientation** | TO | Generic roles, equal weights | Specialist roles, hierarchical weights | Best with L |
| **Trust Network** | T | Equal weights (0.8) | Dynamic scores (0.4-1.0) | L can evaluate |
| **Mutual Monitoring** | MM | No inter-round validation | Pairwise challenges | Requires L |

### 2.1 Shared Mental Model (SMM)
**Purpose:** Passive knowledge repository for shared context

**Data Structure:**
```python
SMM = {
  question_analysis: String,      # Trick detection (1-2 sentences)
  verified_facts: Set[String],    # Consensus facts from R2
  debated_points: List[String]    # Controversies from R3+MM
}
```

**Update Points:** After R1 (question analysis), Post-R2 (facts), Post-MM (debates)

### 2.2 Leadership
**Purpose:** Active orchestration with correction authority

**Powers:**
1. Extract consensus facts from agent responses
2. Create formal medical reports
3. Mediate discussions after each turn
4. Resolve ties with override capability

**Correction Authority:** Leader can override consensus if reasoning is flawed (MMedAgent-RL pattern)

### 2.3 Team Orientation
**Purpose:** Medical role specialization with hierarchical weighting

**Features:**
- Assigns specific medical specialties (Cardiologist, Radiologist, etc.)
- Hierarchical weights: N=2: {0.6, 0.4}, N=3: {0.5, 0.3, 0.2}, N=4: {0.4, 0.3, 0.2, 0.1}
- Weights hidden from agents but used in aggregation
- Formal medical report generation

### 2.4 Trust Network
**Purpose:** Dynamic agent reliability scoring

**Trust Score Model:**
```
T(aᵢ) ∈ [0.4, 1.0]  # Default: 0.8
Update: T_new = α · T_old + (1 - α) · Q(aᵢ)  where α = 0.7
```

**Evaluation Criteria:** Fact accuracy, reasoning completeness, confidence alignment

**Trust-Weighted Voting:**
```
Score(Aⱼ) = Σᵢ T(aᵢ) · BordaPoints(aᵢ, Aⱼ)
```

### 2.5 Mutual Monitoring
**Purpose:** Inter-round validation and quality control

**Protocol:**
1. Leader identifies weakest reasoning
2. Raises specific concern
3. Agent responds (accept/justify)
4. Evaluate response quality
5. Update trust scores and SMM

**Placement:** Between R3 turns only (not after final turn)

---

## 3. Algorithm

```
ALGORITHM: ModularMultiAgentReasoning(Q, Options, C, n_turns)

INPUT:
  Q          : Medical question text
  Options    : Set of answer choices {A₁, ..., Aₙ}
  C          : Teamwork configuration {SMM, L, TO, T, MM}
  n_turns    : Number of R3 discussion turns (default: 2)

OUTPUT:
  Â          : Final answer
  Evidence   : Supporting rationale

INITIALIZE:
  M_smm ← SharedMentalModel() if C.SMM else NULL
  T_scores ← {aᵢ: 0.8 for all agents}
  Leader ← NULL

─────────────────────────────────────────────
PHASE 1: AGENT RECRUITMENT (2 API calls)
─────────────────────────────────────────────

N ← DetermineAgentCount(Q)  # Returns N ∈ {2, 3, 4}

IF C.SMM:
  M_smm.question_analysis ← DetectQuestionTricks(Q, Options)

IF C.L:
  Leader ← Recruiter  # Recruiter assumes dual role

IF C.TO AND C.L:
  Roles ← AssignMedicalSpecialties(Q, N, M_smm)
  W ← AssignHierarchicalWeights(N)
ELSE:
  Roles ← {Expert₁, Expert₂, ..., Expertₙ}
  W ← {1/N, 1/N, ..., 1/N}

Agents ← {(aᵢ, Roles[i], W[i]) for i = 1 to N}

─────────────────────────────────────────────
PHASE 2: INITIAL PREDICTION - ROUND 2 (N+1 API calls)
─────────────────────────────────────────────

# Parallel agent predictions
FOR each aᵢ ∈ Agents IN PARALLEL:
  Context_i ← {Q, Options}

  IF C.SMM:
    Context_i ← Context_i ∪ M_smm.GetContextString()

  IF C.TO:
    Context_i ← Context_i ∪ RoleInstructions(aᵢ)

  R₂[aᵢ] ← aᵢ.Predict(Context_i)

# Post-R2 processing (1 combined call)
IF C.SMM:
  IF C.L:
    M_smm.verified_facts ← Leader.ExtractConsensusFacts(R₂)
  ELSE:
    M_smm.verified_facts ← AutomatedFactIntersection(R₂)

IF C.TO AND C.L:
  FormalReport ← Leader.CreateMedicalReport(R₂, M_smm)

IF C.T:
  IF C.L:
    T_scores ← Leader.EvaluateR2Quality(R₂)
  ELSE:
    T_scores ← AutomatedQualityScore(R₂)

─────────────────────────────────────────────
PHASE 3: COLLABORATIVE DISCUSSION - ROUND 3
─────────────────────────────────────────────

FOR turn ← 1 TO n_turns:

  # Build shared context
  Context_shared ← BuildR3Context(R₂, M_smm, T_scores, FormalReport)

  # Round-robin discourse (N API calls)
  FOR each aᵢ ∈ Agents:
    IF turn < n_turns:
      D[aᵢ][turn] ← aᵢ.Discuss(Context_shared, History[1:turn-1])
    ELSE:
      D[aᵢ][turn] ← aᵢ.FinalRanking(Context_shared, History[1:turn-1])

  # Leadership mediation (1 API call if C.L)
  IF C.L:
    Mediation[turn] ← Leader.Mediate(D[·][turn], M_smm)
    Context_shared ← Context_shared ∪ Mediation[turn]

  # Mutual Monitoring (3-4 API calls if C.MM and turn < n_turns)
  IF C.MM AND turn < n_turns:
    a_target ← Leader.SelectWeakestAgent(D[·][turn])
    Concern ← Leader.RaiseConcern(a_target, D[a_target][turn])
    Response ← a_target.RespondToConcern(Concern)
    Quality ← Leader.EvaluateResponse(Response)

    IF C.T:
      T_scores[a_target] ← UpdateTrustScore(T_scores[a_target], Quality)

    IF C.SMM:
      M_smm.debated_points.Add(ExtractDebatePoints(Concern, Response))

─────────────────────────────────────────────
PHASE 4: DECISION AGGREGATION
─────────────────────────────────────────────

# Extract final rankings
R₃ ← {aᵢ.ranking from D[aᵢ][n_turns] for all aᵢ}

# Aggregate rankings
IF C.T:
  Scores ← TrustWeightedBordaCount(R₃, T_scores)
ELSE IF C.TO:
  Scores ← HierarchicalWeightedBordaCount(R₃, W)
ELSE:
  Scores ← StandardBordaCount(R₃)

Â ← argmax(Scores)

# Tie-breaking
IF IsTie(Scores) AND C.L:
  Tied ← {Aᵢ : Scores[Aᵢ] = max(Scores)}
  Â ← Leader.ResolveTie(Tied, R₃, T_scores, M_smm)

RETURN Â, GenerateRationale(R₃, Scores, M_smm)
```

---

## 4. Complexity Analysis

### 4.1 API Call Count

For N agents, n_turns discussion rounds:

| Configuration | R1 | R2 | R3 (2 turns) | R3 (3 turns) | Total (2T) | Total (3T) |
|---------------|----|----|-------------|-------------|-----------|-----------|
| **Base** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+SMM** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+Leadership** | 2 | N+1 | 2N+2 | 3N+3 | 2N+5 | 3N+6 |
| **+TeamO** | 2 | N+1 | 2N+2 | 3N+3 | 2N+5 | 3N+6 |
| **+Trust** | 2 | N+1 | 2N | 3N | 2N+3 | 3N+3 |
| **+MM** | 2 | N+1 | 2N+3 | 3N+6 | 2N+6 | 3N+9 |
| **ALL ON** | 2 | N+1 | 2N+5 | 3N+9 | 2N+8 | 3N+12 |

**Example** (N=3, n_turns=2):
- Base: 2(3)(2) + 3 = **15 calls**
- ALL ON: 2(3)(2) + 4(2) - 1 = **19 calls**

### 4.2 Time Complexity
- **Recruitment:** O(1) LLM calls
- **R2:** O(N) parallel calls
- **R3:** O(N · n_turns) sequential calls
- **Aggregation:** O(N · |Options|)

**Total:** O(N · n_turns) dominated by R3

---

## 5. Implementation Architecture

```
SLM-TeamMedAgents/
├── adk_agents/                          # Core ADK agents
│   ├── multi_agent_system_adk.py        # Root coordinator
│   ├── dynamic_recruiter_adk.py         # Recruitment
│   ├── three_round_debate_adk.py        # 3-round orchestration
│   ├── decision_aggregator_adk.py       # Voting methods
│   └── gemma_agent_adk.py               # Agent factory
│
├── teamwork_components/                 # Modular components
│   ├── config.py                        # TeamworkConfig
│   ├── shared_mental_model.py           # SMM
│   ├── leadership.py                    # Leadership
│   ├── team_orientation.py              # Role specialization
│   ├── trust_network.py                 # Trust scoring
│   └── mutual_monitoring.py             # Validation
│
├── utils/                               # Utilities
│   ├── prompts.py                       # Prompt templates
│   ├── simulation_logger.py             # Logging
│   ├── results_storage.py               # Results management
│   └── metrics_calculator.py            # Performance metrics
│
└── run_simulation_adk.py                # Main entry point
```

---

## 6. Experimental Configurations

### 6.1 Base System (Control)
```
C = {SMM: OFF, L: OFF, TO: OFF, T: OFF, MM: OFF}
```
Minimal API calls: 2N·n_turns + 3

### 6.2 Ablation Study Configurations

**Individual Components:**
1. `C₁ = {SMM: ON, others: OFF}` - Test passive knowledge sharing
2. `C₂ = {L: ON, others: OFF}` - Test active orchestration
3. `C₃ = {TO: ON, L: ON, others: OFF}` - Test role specialization
4. `C₄ = {T: ON, others: OFF}` - Test dynamic weighting
5. `C₅ = {MM: ON, L: ON, T: ON, others: OFF}` - Test validation

**Combined Configurations:**
6. `C₆ = {SMM: ON, L: ON, TO: ON}` - Knowledge + Structure
7. `C₇ = {L: ON, T: ON, MM: ON}` - Full coordination
8. `C_full = {ALL: ON}` - Complete system

---

## 7. Evaluation Metrics

### Primary Metrics
- **Accuracy:** Fraction of correct answers
- **Convergence:** Agreement rate on first choice
- **Confidence Calibration:** Correlation between confidence and correctness

### Teamwork Metrics
- **SMM Usage:** Average facts extracted per question
- **Trust Divergence:** Std. dev. of final trust scores
- **MM Impact:** Accuracy delta after challenges
- **Leadership Interventions:** Tie-breaking frequency

### Efficiency Metrics
- **API Calls:** Total count per question
- **Latency:** End-to-end execution time
- **Token Usage:** Input/output token counts

---

## 8. Key Design Principles

1. **Modularity:** Each component operates independently with clear interfaces
2. **Composability:** Components can be combined without conflicts
3. **Efficiency:** Batched operations minimize API calls
4. **Robustness:** Graceful degradation when components disabled
5. **Explainability:** All decisions logged with rationales

---

## 9. Related Work

- **Debate-based Systems:** Multi-agent debate for improved reasoning
- **MMedAgent:** Medical multi-agent framework with retrieval
- **MMedAgent-RL:** Leadership with correction authority
- **ReConcile:** Multi-model reconciliation for medical QA
- **Med-PaLM:** Large language models for medical question answering

---

## References

See `ALGO.md` and `Teamwork_ALGO.md` for detailed pseudocode and additional specifications.
