"""
Few-Shot Chain-of-Thought (CoT) Prompts for Medical Reasoning

Dataset-specific few-shot examples with detailed medical reasoning.
Each dataset has custom CoT reasoning patterns tailored to its question type.

Usage:
    from utils.few_shot_cot_prompts import get_few_shot_prompt

    few_shot_text = get_few_shot_prompt('medqa', n_examples=2)
    full_prompt = few_shot_text + "\n\n" + current_question
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

# Load examples from JSON
EXAMPLES_FILE = Path(__file__).parent / 'few_shot_examples.json'

def load_examples() -> Dict:
    """Load few-shot examples from JSON file."""
    if not EXAMPLES_FILE.exists():
        return {}
    with open(EXAMPLES_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

EXAMPLES = load_examples()


# ============================================================================
# CoT REASONING TEMPLATES PER DATASET
# ============================================================================

def format_medqa_example_with_cot(example: Dict, index: int) -> str:
    """Format MedQA example with clinical reasoning CoT."""
    return f"""**Example {index}:**

**Question:** {example['question']}

**Options:**
{chr(10).join(example['options'])}

**Chain-of-Thought Reasoning:**
Step 1 - Identify key clinical features:
- Patient demographics and presentation
- Critical symptoms and signs
- Relevant lab/diagnostic findings

Step 2 - Consider differential diagnosis:
- What are the most likely conditions?
- What pathophysiology explains the presentation?

Step 3 - Evaluate treatment options:
- Match each option to clinical guidelines
- Consider patient-specific factors (age, comorbidities)
- Prioritize evidence-based interventions

Step 4 - Select best answer:
- Which option best addresses the underlying condition?
- What provides best long-term outcomes?

**Final Answer: {example['answer']}**

**Justification:** This answer is supported by established clinical guidelines and addresses the underlying pathophysiology. The key clinical features point directly to this intervention as the most appropriate for preventing future morbidity and mortality.
"""

def format_medmcqa_example_with_cot(example: Dict, index: int) -> str:
    """Format MedMCQA example with Indian medical exam style CoT."""
    return f"""**Example {index}:**

**Question:** {example['question']}

**Options:**
{chr(10).join(example['options'])}

**Chain-of-Thought Reasoning:**
Step 1 - Recall relevant medical knowledge:
- Key anatomical/physiological concepts
- Classic disease presentations or associations
- Standard diagnostic/treatment protocols

Step 2 - Analyze each option systematically:
- Is this option medically accurate?
- Does it match the question context?
- Are there any classic "traps" or distractors?

Step 3 - Apply clinical reasoning:
- What does current medical literature say?
- What would be standard practice?

**Final Answer: {example['answer']}**

**Explanation:** {example.get('explanation', 'Standard medical knowledge application')}
"""

def format_mmlu_pro_example_with_cot(example: Dict, index: int) -> str:
    """Format MMLU-Pro Health example with systematic reasoning."""
    return f"""**Example {index}:**

**Question:** {example['question']}

**Options:**
{chr(10).join(example['options'])}

**Chain-of-Thought Reasoning:**
Step 1 - Understand the question:
- What health/medical concept is being tested?
- What level of knowledge is required?

Step 2 - Systematic elimination:
- Which options are clearly incorrect?
- Which options contradict established health science?

Step 3 - Evidence-based selection:
- What does scientific literature support?
- What aligns with current health guidelines?

**Final Answer: {example['answer']}**

**Rationale:** This answer is supported by current health science evidence and established medical knowledge.
"""

def format_pubmedqa_example_with_cot(example: Dict, index: int) -> str:
    """Format PubMedQA example with research evidence CoT."""
    return f"""**Example {index}:**

**Research Question:** {example['question']}

**Options:** {', '.join(example['options'])}

**Chain-of-Thought Reasoning:**
Step 1 - Analyze the research question:
- What is the specific claim or relationship being tested?
- What type of evidence would support/refute this?

Step 2 - Consider study design and evidence:
- What does the abstract/context suggest?
- Are there methodological limitations?
- What is the strength of the evidence?

Step 3 - Make evidence-based determination:
- Does evidence support the claim? → YES
- Does evidence refute the claim? → NO
- Is evidence inconclusive or mixed? → MAYBE

**Final Answer: {example['answer']}**

**Justification:** Based on the research context, the evidence {'supports' if example['answer'] == 'yes' else 'refutes' if example['answer'] == 'no' else 'is inconclusive regarding'} the research question.
"""

def format_medbullets_example_with_cot(example: Dict, index: int) -> str:
    """Format MedBullets example with case-based CoT."""
    return f"""**Example {index}:**

**Clinical Case:** {example['question']}

**Options:**
{chr(10).join(example['options'])}

**Chain-of-Thought Reasoning:**
Step 1 - Extract case details:
- Patient demographics (age, sex, relevant history)
- Chief complaint and presenting symptoms
- Physical exam findings
- Laboratory/imaging results

Step 2 - Generate differential diagnosis:
- What are the most likely diagnoses?
- What key features support each?

Step 3 - Apply clinical decision-making:
- What is the most likely diagnosis?
- What is the next best step?
- What treatment is most appropriate?

Step 4 - Validate against options:
- Which option best matches clinical reasoning?
- Which follows evidence-based guidelines?

**Final Answer: {example['answer']}**

**Clinical Reasoning:** The case presentation with specific clinical features points to this diagnosis/management approach as the most appropriate based on standard medical practice.
"""

def format_ddxplus_example_with_cot(example: Dict, index: int) -> str:
    """Format DDXPlus example with differential diagnosis CoT."""
    return f"""**Example {index}:**

**Patient Presentation:** {example['question'][:500]}...

**Differential Diagnosis Options:**
{chr(10).join(example['options'][:5])}  # Show first 5 options

**Chain-of-Thought Reasoning:**
Step 1 - Systematic symptom analysis:
- What are the key presenting symptoms?
- What is the temporal pattern?
- Are there associated symptoms?

Step 2 - Generate differential:
- Which conditions commonly present this way?
- What is the epidemiology (age, sex, demographics)?

Step 3 - Narrow differential:
- Which symptoms are specific vs non-specific?
- What is the most parsimonious explanation?
- Apply Occam's razor

Step 4 - Select most likely diagnosis:
- Which diagnosis best explains ALL symptoms?
- What is most probable given demographics?

**Final Answer: {example['answer']}**

**Diagnostic Reasoning:** The constellation of symptoms, patient demographics, and clinical pattern most strongly suggests this diagnosis.
"""

def format_pmc_vqa_example_with_cot(example: Dict, index: int) -> str:
    """Format PMC-VQA example with visual medical analysis CoT."""
    return f"""**Example {index} (WITH MEDICAL IMAGE):**

**Question:** {example['question']}

**Options:**
{chr(10).join(example['options'])}

**Chain-of-Thought Reasoning (with Image Analysis):**
Step 1 - Visual examination:
- What type of medical image is this? (histology, radiology, etc.)
- What anatomical structures are visible?
- What abnormalities or features stand out?

Step 2 - Correlate visual with clinical:
- What clinical context does the image suggest?
- What pathological changes are evident?

Step 3 - Apply medical knowledge:
- What diagnosis do these visual features suggest?
- What treatment/findings correlate with this image?

Step 4 - Match to options:
- Which option best describes what is seen?
- Which aligns with visual pathology?

**Final Answer: {example['answer']}**

**Visual-Clinical Integration:** The image features demonstrate specific pathological changes that directly support this answer.
"""

def format_path_vqa_example_with_cot(example: Dict, index: int) -> str:
    """Format Path-VQA example with pathology image CoT."""
    return f"""**Example {index} (WITH PATHOLOGY IMAGE):**

**Question:** {example['question']}

**Options:** {', '.join(example['options'])}

**Chain-of-Thought Reasoning (Pathology Analysis):**
Step 1 - Microscopic examination:
- What staining method is used?
- What magnification level?
- What tissue type is visible?

Step 2 - Identify pathological features:
- Are there inflammatory changes?
- Is there cellular atypia?
- What structural abnormalities are present?

Step 3 - Pathological diagnosis:
- What condition does this represent?
- Is the feature present or absent?

Step 4 - Yes/No determination:
- Based on visual evidence, is the question true?
- YES if feature is clearly present
- NO if feature is absent or contradicted

**Final Answer: {example['answer']}**

**Pathological Evidence:** Microscopic examination {'confirms' if example['answer'] == 'yes' else 'refutes'} the presence of the queried feature.
"""


# ============================================================================
# FEW-SHOT PROMPT GENERATION
# ============================================================================

FORMATTER_MAP = {
    'medqa': format_medqa_example_with_cot,
    'medmcqa': format_medmcqa_example_with_cot,
    'mmlu_pro': format_mmlu_pro_example_with_cot,
    'pubmedqa': format_pubmedqa_example_with_cot,
    'medbullets': format_medbullets_example_with_cot,
    'ddxplus': format_ddxplus_example_with_cot,
    'pmc_vqa': format_pmc_vqa_example_with_cot,
    'path_vqa': format_path_vqa_example_with_cot,
}

def get_few_shot_prompt(dataset: str, n_examples: Optional[int] = None) -> str:
    """
    Get few-shot CoT prompt for a specific dataset.

    Args:
        dataset: Dataset name (medqa, medmcqa, etc.)
        n_examples: Number of examples to include (None = all available)

    Returns:
        Formatted few-shot prompt with Chain-of-Thought reasoning
    """
    dataset = dataset.lower().replace('-', '_').replace('_', '')

    # Normalize dataset names
    dataset_map = {
        'medqa': 'medqa',
        'medmcqa': 'medmcqa',
        'mmlupro': 'mmlu_pro',
        'mmluprohealth': 'mmlu_pro',
        'pubmedqa': 'pubmedqa',
        'medbullets': 'medbullets',
        'ddxplus': 'ddxplus',
        'pmcvqa': 'pmc_vqa',
        'pathvqa': 'path_vqa',
    }

    normalized_dataset = dataset_map.get(dataset, dataset)

    if normalized_dataset not in EXAMPLES:
        return ""

    examples = EXAMPLES[normalized_dataset]
    if n_examples:
        examples = examples[:n_examples]

    formatter = FORMATTER_MAP.get(normalized_dataset)
    if not formatter:
        return ""

    prompt_parts = [
        "The following are examples of medical questions with Chain-of-Thought reasoning:\n"
    ]

    for i, example in enumerate(examples, 1):
        prompt_parts.append(formatter(example, i))
        prompt_parts.append("\n" + "="*80 + "\n")

    prompt_parts.append("Now, apply the same systematic reasoning approach to the following NEW question:\n")

    return "\n".join(prompt_parts)


def get_zero_shot_cot_prompt() -> str:
    """Get zero-shot CoT prompt (no examples, just instruction)."""
    return """**Instructions:** Think step-by-step through your medical reasoning:

Step 1 - Analyze the question and clinical context
Step 2 - Consider relevant medical knowledge and evidence
Step 3 - Systematically evaluate each option
Step 4 - Select the best answer with clear justification

Now answer the following question:
"""


__all__ = [
    'get_few_shot_prompt',
    'get_zero_shot_cot_prompt',
    'EXAMPLES',
]
