# Algorithm Specification: Modular Multi-Agent Medical Reasoning System

## Abstract

We present a modular multi-agent framework for collaborative medical reasoning, featuring five pluggable teamwork mechanisms: Shared Mental Model (SMM), Leadership, Team Orientation, Trust Network, and Mutual Monitoring. Each component operates independently and can be selectively enabled for ablation studies. The system supports 2-4 dynamically recruited specialist agents engaging in structured deliberation over multiple rounds, with final aggregation via weighted voting schemes.

---

## 1. System Overview

### 1.1 Input-Output Specification

**Input**:
- Medical question `Q` (multiple-choice or open-ended)
- Answer options `{A₁, A₂, ..., Aₙ}` (if applicable)
- Teamwork configuration `C = {SMM, L, TO, T, MM}` (binary flags)
- Discussion parameters: `n_turns ∈ {2, 3}`

**Output**:
- Final answer `Â ∈ {A₁, A₂, ..., Aₙ}`
- Decision rationale with supporting evidence
- Teamwork metrics (optional): SMM content, trust scores, MM results

### 1.2 Component Modularity Matrix

| Component | Symbol | OFF State | ON State | Dependencies |
|-----------|--------|-----------|----------|--------------|
| **Shared Mental Model** | SMM | Independent agents | Shared knowledge context | None |
| **Leadership** | L | Passive coordinator | Active orchestrator | Enhances SMM, TO, T |
| **Team Orientation** | TO | Generic roles, equal weights | Specialist roles, hierarchical weights | Best with L |
| **Trust Network** | T | Equal weights (0.8) | Dynamic scores (0.4-1.0) | L can evaluate |
| **Mutual Monitoring** | MM | No inter-round validation | Pairwise challenges | Requires L |

---

## 2. Algorithm Specification

### Algorithm 1: Modular Multi-Agent Medical Reasoning
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
  T_scores ← {aᵢ: 0.8 for all agents}  // Default trust
  Leader ← NULL

─────────────────────────────────────────────
PHASE 1: AGENT RECRUITMENT (2 API calls)
─────────────────────────────────────────────

// Step 1.1: Analyze question and determine agent count
N ← DetermineAgentCount(Q)  // Returns N ∈ {2, 3, 4}

IF C.SMM:
  M_smm.question_analysis ← DetectQuestionTricks(Q, Options)

IF C.L:
  Leader ← Recruiter  // Recruiter assumes dual role

// Step 1.2: Initialize agents
IF C.TO AND C.L:
  // Specialized medical roles with hierarchical weights
  Roles ← AssignMedicalSpecialties(Q, N, M_smm)
  W ← AssignHierarchicalWeights(N)  // e.g., {0.5, 0.3, 0.2}
ELSE:
  // Generic expert roles with equal weights
  Roles ← {Expert₁, Expert₂, ..., Expertₙ}
  W ← {1/N, 1/N, ..., 1/N}

Agents ← {(aᵢ, Roles[i], W[i]) for i = 1 to N}

─────────────────────────────────────────────
PHASE 2: INITIAL PREDICTION - ROUND 2 (N+1 API calls)
─────────────────────────────────────────────

// Step 2.1: Parallel agent predictions
FOR each aᵢ ∈ Agents IN PARALLEL:
  Context_i ← {Q, Options}

  IF C.SMM:
    Context_i ← Context_i ∪ M_smm.GetContextString()

  IF C.TO:
    Context_i ← Context_i ∪ RoleInstructions(aᵢ)

  R₂[aᵢ] ← aᵢ.Predict(Context_i)
  // Returns: {ranking, justification, facts[]}

// Step 2.2: Post-R2 processing (1 combined call)
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

  // Step 3.1: Build shared context for this turn
  Context_shared ← BuildR3Context(R₂, M_smm, T_scores, FormalReport)

  // Step 3.2: Round-robin discourse (N API calls)
  FOR each aᵢ ∈ Agents:
    IF turn < n_turns:
      // Intermediate turn: Discussion only
      D[aᵢ][turn] ← aᵢ.Discuss(Context_shared, History[1:turn-1])
    ELSE:
      // Final turn: Ranking required
      D[aᵢ][turn] ← aᵢ.FinalRanking(Context_shared, History[1:turn-1])

  // Step 3.3: Leadership mediation (1 API call if C.L)
  IF C.L:
    Mediation[turn] ← Leader.Mediate(D[·][turn], M_smm)
    Context_shared ← Context_shared ∪ Mediation[turn]

  // Step 3.4: Mutual Monitoring (3-4 API calls if C.MM and turn < n_turns)
  IF C.MM AND turn < n_turns:
    // Select agent with weakest reasoning
    a_target ← Leader.SelectWeakestAgent(D[·][turn])

    // Challenge-response protocol
    Concern ← Leader.RaiseConcern(a_target, D[a_target][turn])
    Response ← a_target.RespondToConcern(Concern)
    Quality ← Leader.EvaluateResponse(Response)

    // Update trust and SMM
    IF C.T:
      T_scores[a_target] ← UpdateTrustScore(T_scores[a_target], Quality)

    IF C.SMM:
      M_smm.debated_points.Add(ExtractDebatePoints(Concern, Response))

─────────────────────────────────────────────
PHASE 4: DECISION AGGREGATION
─────────────────────────────────────────────

// Extract final rankings from last turn
R₃ ← {aᵢ.ranking from D[aᵢ][n_turns] for all aᵢ}

// Step 4.1: Aggregate rankings
IF C.T:
  Scores ← TrustWeightedBordaCount(R₃, T_scores)
ELSE IF C.TO:
  Scores ← HierarchicalWeightedBordaCount(R₃, W)
ELSE:
  Scores ← StandardBordaCount(R₃)

Â ← argmax(Scores)

// Step 4.2: Tie-breaking (if needed)
IF IsTie(Scores) AND C.L:
  Tied ← {Aᵢ : Scores[Aᵢ] = max(Scores)}
  Â ← Leader.ResolveTie(Tied, R₃, T_scores, M_smm)

RETURN Â, GenerateRationale(R₃, Scores, M_smm)

END ALGORITHM
```

---

## 3. Component Specifications

### 3.1 Shared Mental Model (SMM)

**Data Structure**:
```
SMM = {
  question_analysis: String,        // Trick detection (1-2 sentences)
  verified_facts: Set[String],      // Consensus facts from R2
  debated_points: List[String]      // Controversies from R3+MM
}
```

**Operations**:
- `DetectQuestionTricks(Q, Options)`: Rule-based or LLM-powered analysis
- `ExtractConsensusFacts(R₂)`: Intersection logic or Leader extraction
- `GetContextString()`: Format SMM for prompt injection

**Update Points**: After R1 (question analysis), Post-R2 (facts), Post-MM (debates)

### 3.2 Leadership Coordinator

**Powers**:
1. **Extract Facts**: `Leader.ExtractConsensusFacts(R₂) → List[String]`
2. **Create Report**: `Leader.CreateMedicalReport(R₂, M_smm) → String`
3. **Mediate**: `Leader.Mediate(D[·][turn], M_smm) → String`
4. **Resolve Ties**: `Leader.ResolveTie(Tied, R₃, T_scores, M_smm) → Answer`

**Correction Authority**: Leader can override consensus if reasoning is flawed

### 3.3 Team Orientation

**Role Assignment**:
```
AssignMedicalSpecialties(Q, N, M_smm):
  // Extract medical domain from question
  Domain ← IdentifyMedicalDomain(Q)

  // Define N specialist roles
  Specialties ← {
    Cardiologist, Pulmonologist, Neurologist,
    Radiologist, Pathologist, Internist, ...
  }

  // Select top N most relevant to Domain
  Roles ← TopK(Specialties, Domain, N)

  RETURN Roles
```

**Hierarchical Weights**:
- N=2: `{0.6, 0.4}`
- N=3: `{0.5, 0.3, 0.2}`
- N=4: `{0.4, 0.3, 0.2, 0.1}`

Weights are **hidden from agents** but used in aggregation.

### 3.4 Trust Network

**Trust Score Model**:
```
T(aᵢ) ∈ [0.4, 1.0]  // Default: 0.8
```

**Update Formula** (exponential moving average):
```
T_new(aᵢ) = α · T_old(aᵢ) + (1 - α) · Q(aᵢ)
where α = 0.7, Q(aᵢ) = QualityScore(response)
```

**Quality Evaluation Criteria**:
- Fact accuracy (if ground truth available)
- Reasoning completeness (length, structure)
- Confidence alignment with correctness

**Trust-Weighted Borda Count**:
```
Score(Aⱼ) = Σᵢ T(aᵢ) · BordaPoints(aᵢ, Aⱼ)
```

### 3.5 Mutual Monitoring

**Protocol**:
```
MutualMonitoring(turn, D[·][turn], T_scores, M_smm):

  // Step 1: Identify weakest reasoning
  a_target ← argmin_{aᵢ} ReasoningQuality(D[aᵢ][turn])

  // Step 2: Leader raises concern
  Concern ← Leader.RaiseConcern(a_target, D[a_target][turn])

  // Step 3: Agent responds
  Response ← a_target.RespondToConcern(Concern)

  // Step 4: Evaluate response quality
  Quality ← Leader.EvaluateResponse(Response) ∈ {strong, weak, disputed}

  // Step 5: Update trust
  IF Quality = strong:
    T_scores[a_target] += 0.05
  ELSE IF Quality = weak:
    T_scores[a_target] -= 0.05
  ELSE:  // disputed
    T_scores[a_target] -= 0.02

  // Step 6: Update SMM
  M_smm.debated_points.Add({
    turn: turn,
    agent: a_target,
    concern: Concern,
    response: Response,
    outcome: Quality
  })

  RETURN T_scores, M_smm
```

**Placement**: Between R3 turns only (not after final turn)

---

## 4. Complexity Analysis

### 4.1 API Call Count

For N agents, n_turns discussion rounds:

| Configuration | R1 | R2 | R3 | Total |
|---------------|----|----|----|----|
| Base | 2 | N+1 | N·n_turns | 2N·n_turns + 3 |
| +SMM | 2 | N+1 | N·n_turns | 2N·n_turns + 3 |
| +Leadership | 2 | N+1 | N·n_turns + n_turns | (2N+1)·n_turns + 3 |
| +TeamO | 2 | N+1 | N·n_turns + n_turns | (2N+1)·n_turns + 3 |
| +Trust | 2 | N+1 | N·n_turns | 2N·n_turns + 3 |
| +MM | 2 | N+1 | N·n_turns + 3(n_turns-1) | 2N·n_turns + 3n_turns |
| **ALL ON** | 2 | N+1 | N·n_turns + 4(n_turns-1) | 2N·n_turns + 4n_turns - 1 |

**Example** (N=3, n_turns=2):
- Base: 2(3)(2) + 3 = **15 calls**
- ALL ON: 2(3)(2) + 4(2) - 1 = **19 calls**

### 4.2 Time Complexity

- **Recruitment**: O(1) LLM calls
- **R2**: O(N) parallel calls
- **R3**: O(N · n_turns) sequential calls
- **Aggregation**: O(N · |Options|)

**Total**: O(N · n_turns) dominated by R3 sequential execution

### 4.3 Space Complexity

- SMM: O(|Q| + F · |fact|) where F = number of facts
- Trust scores: O(N)
- Debate history: O(N · n_turns · |response|)

**Total**: O(N · n_turns · |response|)

---

## 5. Experimental Configurations

### 5.1 Base System (Control)
```
C = {SMM: OFF, L: OFF, TO: OFF, T: OFF, MM: OFF}
```
- Generic expert agents with equal weights
- Simple Borda count aggregation
- No shared context or mediation
- Minimal API calls: 2N·n_turns + 3

### 5.2 Ablation Study Configurations

**Individual Components**:
1. `C₁ = {SMM: ON, others: OFF}` - Test passive knowledge sharing
2. `C₂ = {L: ON, others: OFF}` - Test active orchestration
3. `C₃ = {TO: ON, L: ON, others: OFF}` - Test role specialization
4. `C₄ = {T: ON, others: OFF}` - Test dynamic weighting
5. `C₅ = {MM: ON, L: ON, T: ON, others: OFF}` - Test inter-round validation

**Combined Configurations**:
6. `C₆ = {SMM: ON, L: ON, TO: ON}` - Knowledge + Structure
7. `C₇ = {L: ON, T: ON, MM: ON}` - Full coordination
8. `C_full = {ALL: ON}` - Complete system

### 5.3 Evaluation Metrics

**Primary Metrics**:
- **Accuracy**: Fraction of correct answers
- **Convergence**: Agreement rate on first choice
- **Confidence Calibration**: Correlation between confidence and correctness

**Teamwork Metrics**:
- **SMM Usage**: Average facts extracted per question
- **Trust Divergence**: Std. dev. of final trust scores
- **MM Impact**: Accuracy delta after challenges
- **Leadership Interventions**: Tie-breaking frequency

**Efficiency Metrics**:
- **API Calls**: Total count per question
- **Latency**: End-to-end execution time
- **Token Usage**: Input/output token counts

---

## 6. Implementation Notes

### 6.1 Error Handling

All API calls include exponential backoff retry:
```python
for attempt in range(max_retries):
  try:
    response = await agent.run_async(ctx)
    break
  except RateLimitError:
    delay = min(60 * (2 ** attempt), 300) * random.uniform(0.8, 1.2)
    await asyncio.sleep(delay)
```

### 6.2 Prompt Engineering

**R2 Prompt Structure** (with SMM):
```
[SMM Context]
{M_smm.GetContextString()}

[Your Role]
You are a {role} with expertise in {expertise}.

[Question]
{Q}

[Task]
Provide:
1. Ranked list of answers
2. Justification (2 sentences)
3. FACTS: 2-5 key medical facts
```

**R3 Final Turn Prompt** (with all components):
```
[SMM Context]
{M_smm.GetContextString()}

[Trust Context]
{T_scores formatted as hints}

[Formal Medical Report]
{FormalReport}

[Previous Discussion]
{Summary of turns 1 to n_turns-1}

[Your Final Ranking]
Provide RANKING: 1. A, 2. B, 3. C, 4. D
CONFIDENCE: High/Medium/Low
```

### 6.3 Backward Compatibility

System maintains full backward compatibility:
```python
# Base system (all components OFF)
config = TeamworkConfig()  # Default: all False

# Existing code works without changes
system = MultiAgentSystemADK(model_name='gemma3_4b', teamwork_config=config)
```

---

## 7. Theoretical Foundation

### 7.1 Design Principles

1. **Modularity**: Each component operates independently with clear interfaces
2. **Composability**: Components can be combined without conflicts
3. **Efficiency**: Batched operations minimize API calls
4. **Robustness**: Graceful degradation when components disabled
5. **Explainability**: All decisions logged with rationales

### 7.2 Related Work

- **Debate-based Systems**: Multi-agent debate for improved reasoning
- **MMedAgent**: Medical multi-agent framework with retrieval
- **MMedAgent-RL**: Leadership with correction authority
- **ReConcile**: Multi-model reconciliation for medical QA
- **Med-PaLM**: Large language models for medical question answering

### 7.3 Novel Contributions

1. **Modular Teamwork Architecture**: First system with independently toggleable components
2. **Hierarchical Weighting with Role Specialization**: Medical domain-specific role assignment
3. **Trust-based Weighted Aggregation**: Dynamic reliability scoring for voting
4. **Inter-round Mutual Monitoring**: Validation protocol between discussion turns
5. **Comprehensive Ablation Support**: Built-in experimental framework

---

## 8. Conclusion

This algorithm specification defines a modular multi-agent system for medical reasoning with five pluggable teamwork components. The system supports rigorous ablation studies through independent component toggles, maintains backward compatibility with base multi-agent systems, and provides comprehensive logging for analysis. The architecture balances effectiveness (through structured coordination) with efficiency (through batched operations and parallel execution).

---

## Appendix A: Notation Reference

| Symbol | Description |
|--------|-------------|
| Q | Medical question |
| {A₁, ..., Aₙ} | Answer options |
| N | Number of agents (2-4) |
| aᵢ | Agent i |
| C | Teamwork configuration |
| M_smm | Shared Mental Model |
| T_scores | Trust score dictionary |
| W | Hierarchical weight vector |
| R₂ | Round 2 responses |
| D[aᵢ][turn] | Discourse of agent i in turn |
| Â | Final answer |

---

## Appendix B: Pseudocode Functions

### B.1 Borda Count Variants

**Standard Borda Count**:
```
BordaCount(Rankings):
  Scores ← {}
  FOR each (aᵢ, ranking) in Rankings:
    FOR position p, option A in ranking:
      Scores[A] += (|Options| - p - 1)
  RETURN argmax(Scores)
```

**Trust-Weighted Borda**:
```
TrustWeightedBordaCount(Rankings, T_scores):
  Scores ← {}
  FOR each (aᵢ, ranking) in Rankings:
    FOR position p, option A in ranking:
      Scores[A] += T_scores[aᵢ] · (|Options| - p - 1)
  RETURN argmax(Scores)
```

**Hierarchical-Weighted Borda**:
```
HierarchicalWeightedBordaCount(Rankings, W):
  Scores ← {}
  FOR each (aᵢ, ranking) in Rankings:
    FOR position p, option A in ranking:
      Scores[A] += W[aᵢ] · (|Options| - p - 1)
  RETURN argmax(Scores)
```

### B.2 Context Building

```
BuildR3Context(R₂, M_smm, T_scores, FormalReport):
  Context ← {}

  IF M_smm ≠ NULL:
    Context['smm'] = M_smm.GetContextString()

  IF T_scores ≠ NULL:
    Context['trust'] = FormatTrustHints(T_scores)

  IF FormalReport ≠ NULL:
    Context['report'] = FormalReport

  Context['r2_summary'] = Summarize(R₂)

  RETURN Context
```

