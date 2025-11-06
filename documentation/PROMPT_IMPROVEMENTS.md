# Prompt Improvements - Multi-Paper Integration

## Summary

Integrated best practices from multiple medical reasoning papers into our prompt system while maintaining backward compatibility.

## Changes Made

### 1. Enhanced Prompt Templates (`utils/prompts.py`)

#### Complexity Analysis (Recruitment)
**Before**: Generic complexity assessment
**After**: Explicit PCP vs specialist framework
- LOW: PCP can answer without specialists (2 agents)
- MODERATE: PCP needs specialist consultation (3 agents)
- HIGH: Multi-departmental specialists required (4 agents)

#### Agent Initialization
**Before**: Individual agent setup
**After**: Team collaboration framing
- "Your job is to collaborate with other medical experts in a team"
- "Deliver your opinions in a way to convince other experts with clear reasoning"

#### Role Assignment (Recruitment)
**Before**: Simple role list
**After**: Communication structure specification
- Added hierarchy markers: `==` for equal collaboration, `>` for hierarchical
- Example: "Agent_1 == Agent_2 > Agent_3"

#### Round 2 (Collaborative Discussion)
**Before**: Basic peer review
**After**: Explicit conviction framing
- "Indicate whether you agree/disagree with other experts"
- "Deliver your opinion in a way to convince them with clear reasoning"

#### Round 3 (Final Decision)
**Before**: Agent provides ranking
**After**: Final decision maker authority
- "You are a final medical decision maker who reviews all opinions"
- "You have the authority to make the final decision"
- Emphasizes synthesizing best insights from team discussion

#### Leadership Report Generation
**Before**: Basic report synthesis
**After**: 4-step structured process
1. Take careful and comprehensive consideration of provided reports
2. Extract key medical knowledge from reports
3. Derive comprehensive and summarized analysis
4. Generate refined and synthesized report

### 2. Few-Shot Chain-of-Thought System (`utils/few_shot_cot_prompts.py`)

Created dataset-specific few-shot examples with detailed CoT reasoning for all 8 datasets:

#### MedQA (USMLE-style)
- Step 1: Identify key clinical features
- Step 2: Consider differential diagnosis
- Step 3: Evaluate treatment options
- Step 4: Select best answer with clinical justification

#### MedMCQA (Indian Medical Exam)
- Step 1: Recall relevant medical knowledge
- Step 2: Analyze each option systematically
- Step 3: Apply clinical reasoning with literature support

#### MMLU-Pro Health
- Step 1: Understand the question
- Step 2: Systematic elimination
- Step 3: Evidence-based selection

#### PubMedQA (Research Questions)
- Step 1: Analyze the research question
- Step 2: Consider study design and evidence
- Step 3: Make evidence-based determination (YES/NO/MAYBE)

#### MedBullets (Clinical Cases)
- Step 1: Extract case details
- Step 2: Generate differential diagnosis
- Step 3: Apply clinical decision-making
- Step 4: Validate against options

#### DDXPlus (Differential Diagnosis)
- Step 1: Systematic symptom analysis
- Step 2: Generate differential
- Step 3: Narrow differential with Occam's razor
- Step 4: Select most likely diagnosis

#### PMC-VQA (Medical Images)
- Step 1: Visual examination of medical image
- Step 2: Correlate visual with clinical context
- Step 3: Apply medical knowledge to pathology
- Step 4: Match to options

#### Path-VQA (Pathology Images)
- Step 1: Microscopic examination (staining, magnification, tissue)
- Step 2: Identify pathological features
- Step 3: Pathological diagnosis
- Step 4: Yes/No determination based on visual evidence

### 3. Integration Architecture

**Backward Compatible Design**:
```python
get_round1_prompt(
    task_type="mcq",
    role="Cardiologist",
    expertise="Cardiovascular disease",
    question="...",
    options=[...],
    dataset="medqa",           # NEW: enables dataset-specific examples
    use_few_shot=True,         # NEW: toggle few-shot (default: True)
    n_few_shot_examples=2      # NEW: number of examples (default: 2)
)
```

**Features**:
- Automatic few-shot prefix injection when dataset specified
- Zero-shot CoT fallback when dataset not specified
- Graceful degradation if few-shot system unavailable
- No breaking changes to existing code

### 4. Few-Shot Example Extraction

**Extracted Examples**:
- MedQA: 3 examples
- MedMCQA: 3 examples
- MMLU-Pro: 3 examples
- PubMedQA: 3 examples
- MedBullets: 3 examples
- DDXPlus: 2 examples
- PMC-VQA: 2 examples (with images)
- Path-VQA: 2 examples (with images)

**Storage**: `utils/few_shot_examples.json`

## Key Improvements from Other Papers

### 1. Explicit Complexity Definitions
**Source**: Multi-agent medical reasoning papers
**Impact**: Better team sizing decisions based on clinical complexity levels

### 2. Communication Hierarchy
**Source**: Agent communication structure papers
**Impact**: Explicit collaboration patterns (equal vs hierarchical)

### 3. Conviction-Based Discussion
**Source**: Collaborative deliberation papers
**Impact**: Agents explicitly try to convince peers with reasoning

### 4. Decision Maker Authority
**Source**: Final decision frameworks
**Impact**: Clear authority in R3 to make final determination

### 5. Structured Report Synthesis
**Source**: Medical report generation papers
**Impact**: 4-step systematic report creation process

### 6. Dataset-Specific CoT
**Source**: Few-shot learning papers
**Impact**: Tailored reasoning patterns for each dataset type

## Files Modified

1. `utils/prompts.py` - Core prompt templates updated
2. `adk_agents/gemma_agent_adk.py` - Agent initialization prompts
3. `adk_agents/dynamic_recruiter_adk.py` - Recruitment prompts
4. `teamwork_components/leadership.py` - Leadership report generation
5. `utils/few_shot_cot_prompts.py` - NEW: Few-shot CoT system

## Files Created

1. `utils/few_shot_examples.json` - Dataset examples
2. `utils/few_shot_cot_prompts.py` - Few-shot CoT templates
3. `extract_few_shot_examples.py` - Example extraction script
4. `run_baseline_benchmarks.bat` - Baseline benchmark runner

## Baseline Benchmark Suite

Created comprehensive baseline evaluation system:

**Script**: `run_baseline_benchmarks.bat`

**Configuration**:
- Datasets: 8 (all medical datasets)
- Methods: 3 (zero_shot, few_shot, cot)
- Seeds: 3 (1, 2, 3)
- Questions per run: 50
- Total runs: 72

**Usage**:
```bash
# Windows
run_baseline_benchmarks.bat

# Manual run example
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --num_questions 50 --random_seed 1
python slm_runner.py --dataset medqa --method few_shot --model gemma3_4b --num_questions 50 --random_seed 2
python slm_runner.py --dataset medqa --method cot --model gemma3_4b --num_questions 50 --random_seed 3
```

## Testing & Validation

**Backward Compatibility**: All existing code continues to work
- Default behavior: Few-shot enabled with 2 examples
- Can disable: `use_few_shot=False` in `get_round1_prompt()`
- Graceful fallback if few-shot system unavailable

**Variable Preservation**: All prompt variables maintained
- `{role}`, `{expertise}`, `{question}`, `{options}` unchanged
- All format requirements (RANKING:, JUSTIFICATION:, etc.) intact
- No breaking changes to parsing logic

## Impact

### Prompt Quality
- More explicit clinical reasoning frameworks
- Better team coordination through communication structures
- Dataset-specific reasoning patterns via few-shot examples

### System Performance (Expected)
- Improved accuracy from dataset-specific CoT examples
- Better team coordination from explicit collaboration framing
- More accurate complexity assessment for team sizing

### Research Value
- Comprehensive baseline benchmarks (zero-shot, few-shot, CoT)
- 72 baseline runs across all datasets and methods
- Direct comparison with multi-agent teamwork system

## Future Enhancements

1. **Dynamic Few-Shot Selection**: Select most relevant examples per question
2. **Cross-Dataset Transfer**: Use examples from similar datasets
3. **Adaptive CoT Depth**: Vary reasoning depth based on question complexity
4. **Performance-Based Selection**: Use examples from highest-accuracy questions
