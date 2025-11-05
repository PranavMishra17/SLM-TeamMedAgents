"""
Extract few-shot examples from each dataset for CoT prompting.

This script loads 2-3 examples from each of the 8 datasets and formats them
with Chain-of-Thought reasoning for few-shot prompting.
"""

import json
import logging
from pathlib import Path
from medical_datasets.dataset_loader import DatasetLoader, VisionDatasetLoader
from medical_datasets.dataset_formatters import (
    MedQAFormatter, MedMCQAFormatter, MMLUProMedFormatter,
    PubMedQAFormatter, DDXPlusFormatter, MedBulletsFormatter
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

def extract_medqa_examples(n=3):
    """Extract MedQA examples."""
    logging.info("\n=== EXTRACTING MEDQA EXAMPLES ===")
    questions = DatasetLoader.load_medqa(num_questions=n, random_seed=123)

    examples = []
    for i, q in enumerate(questions[:n]):
        agent_task, eval_data = MedQAFormatter.format(q)
        examples.append({
            'question': agent_task['description'],
            'options': agent_task['options'],
            'answer': eval_data['ground_truth'],
            'explanation': q.get('metamap_phrases', 'Clinical reasoning required')
        })
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Q: {agent_task['description'][:100]}...")
        logging.info(f"Answer: {eval_data['ground_truth']}")

    return examples

def extract_medmcqa_examples(n=3):
    """Extract MedMCQA examples."""
    logging.info("\n=== EXTRACTING MEDMCQA EXAMPLES ===")
    questions, errors = DatasetLoader.load_medmcqa(num_questions=n, random_seed=123)

    examples = []
    for i, q in enumerate(questions[:n]):
        agent_task, eval_data, is_valid = MedMCQAFormatter.format(q)
        if is_valid:
            examples.append({
                'question': agent_task['description'],
                'options': agent_task['options'],
                'answer': eval_data['ground_truth'],
                'explanation': q.get('exp', 'Medical knowledge application required')
            })
            logging.info(f"\nExample {i+1}:")
            logging.info(f"Q: {agent_task['description'][:100]}...")
            logging.info(f"Answer: {eval_data['ground_truth']}")

    return examples

def extract_mmlu_pro_examples(n=3):
    """Extract MMLU-Pro Health examples."""
    logging.info("\n=== EXTRACTING MMLU-PRO HEALTH EXAMPLES ===")
    questions = DatasetLoader.load_mmlupro_med(num_questions=n, random_seed=123)

    examples = []
    for i, q in enumerate(questions[:n]):
        agent_task, eval_data = MMLUProMedFormatter.format(q)
        examples.append({
            'question': agent_task['description'],
            'options': agent_task['options'],
            'answer': eval_data['ground_truth'],
            'explanation': 'Systematic health science reasoning'
        })
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Q: {agent_task['description'][:100]}...")
        logging.info(f"Answer: {eval_data['ground_truth']}")

    return examples

def extract_pubmedqa_examples(n=3):
    """Extract PubMedQA examples."""
    logging.info("\n=== EXTRACTING PUBMEDQA EXAMPLES ===")
    questions = DatasetLoader.load_pubmedqa(num_questions=n, random_seed=123)

    examples = []
    for i, q in enumerate(questions[:n]):
        agent_task, eval_data = PubMedQAFormatter.format(q)
        # Get context for reasoning
        context = q.get('CONTEXTS', [])
        reasoning = ' '.join(context[:2]) if context else 'Evidence-based analysis required'

        examples.append({
            'question': agent_task['description'],
            'options': ['yes', 'no', 'maybe'],
            'answer': eval_data['ground_truth'],
            'explanation': reasoning[:200]
        })
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Q: {agent_task['description'][:100]}...")
        logging.info(f"Answer: {eval_data['ground_truth']}")

    return examples

def extract_medbullets_examples(n=3):
    """Extract MedBullets examples."""
    logging.info("\n=== EXTRACTING MEDBULLETS EXAMPLES ===")
    questions = DatasetLoader.load_medbullets(num_questions=n, random_seed=123)

    examples = []
    for i, q in enumerate(questions[:n]):
        agent_task, eval_data = MedBulletsFormatter.format(q)
        examples.append({
            'question': agent_task['description'],
            'options': agent_task['options'],
            'answer': eval_data['ground_truth'],
            'explanation': 'Clinical case-based reasoning'
        })
        logging.info(f"\nExample {i+1}:")
        logging.info(f"Q: {agent_task['description'][:100]}...")
        logging.info(f"Answer: {eval_data['ground_truth']}")

    return examples

def extract_ddxplus_examples(n=2):
    """Extract DDXPlus examples - skip if dataset not available."""
    logging.info("\n=== EXTRACTING DDXPLUS EXAMPLES ===")
    try:
        questions = DatasetLoader.load_ddxplus(num_questions=n, random_seed=123)
        if not questions:
            logging.warning("DDXPlus dataset not available, skipping...")
            return []

        examples = []
        for i, q in enumerate(questions[:n]):
            agent_task, eval_data = DDXPlusFormatter.format(q)
            examples.append({
                'question': agent_task['description'],
                'options': agent_task['options'],
                'answer': eval_data['ground_truth'],
                'explanation': 'Differential diagnosis based on symptoms'
            })
            logging.info(f"\nExample {i+1}:")
            logging.info(f"Q: {agent_task['description'][:100]}...")
            logging.info(f"Answer: {eval_data['ground_truth']}")

        return examples
    except Exception as e:
        logging.warning(f"DDXPlus extraction failed: {e}, skipping...")
        return []

def extract_pmc_vqa_examples(n=2):
    """Extract PMC-VQA examples (vision) - skip if takes too long."""
    logging.info("\n=== EXTRACTING PMC-VQA EXAMPLES (VISION) ===")
    try:
        questions = VisionDatasetLoader.load_pmc_vqa(num_questions=n, random_seed=123)
        if not questions:
            logging.warning("PMC-VQA dataset returned no questions, skipping...")
            return []

        examples = []
        for i, q in enumerate(questions[:n]):
            # Direct formatting without formatter
            question_text = q.get('Question', '').strip()
            options = []
            for j in range(4):
                opt = q.get(f'Choice {chr(65+j)}', '').strip()
                if opt:
                    options.append(f"{chr(65+j)}. {opt}")

            answer = q.get('Answer', 'A')

            examples.append({
                'question': question_text,
                'options': options,
                'answer': answer,
                'explanation': 'Visual medical analysis required',
                'has_image': True
            })
            logging.info(f"\nExample {i+1} (with image):")
            logging.info(f"Q: {question_text[:100]}...")
            logging.info(f"Answer: {answer}")

        return examples
    except Exception as e:
        logging.warning(f"PMC-VQA extraction failed/timeout: {e}, skipping...")
        return []

def extract_path_vqa_examples(n=2):
    """Extract Path-VQA examples (vision) - skip if takes too long."""
    logging.info("\n=== EXTRACTING PATH-VQA EXAMPLES (VISION) ===")
    try:
        questions = VisionDatasetLoader.load_path_vqa(num_questions=n, random_seed=123)
        if not questions:
            logging.warning("Path-VQA dataset returned no questions, skipping...")
            return []

        examples = []
        for i, q in enumerate(questions[:n]):
            # Direct formatting without formatter
            question_text = q.get('question', '').strip()
            answer = q.get('answer', '').lower()

            examples.append({
                'question': question_text,
                'options': ['yes', 'no'],
                'answer': answer,
                'explanation': 'Pathology image interpretation',
                'has_image': True
            })
            logging.info(f"\nExample {i+1} (with image):")
            logging.info(f"Q: {question_text[:100]}...")
            logging.info(f"Answer: {answer}")

        return examples
    except Exception as e:
        logging.warning(f"Path-VQA extraction failed/timeout: {e}, skipping...")
        return []

def main():
    """Extract all few-shot examples."""
    all_examples = {
        'medqa': extract_medqa_examples(3),
        'medmcqa': extract_medmcqa_examples(3),
        'mmlu_pro': extract_mmlu_pro_examples(3),
        'pubmedqa': extract_pubmedqa_examples(3),
        'medbullets': extract_medbullets_examples(3),
        'ddxplus': extract_ddxplus_examples(2),
        'pmc_vqa': extract_pmc_vqa_examples(2),
        'path_vqa': extract_path_vqa_examples(2)
    }

    # Save to JSON
    output_file = Path('utils/few_shot_examples.json')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)

    logging.info(f"\n\n✅ Saved few-shot examples to {output_file}")

    # Print summary
    logging.info("\n=== SUMMARY ===")
    for dataset, examples in all_examples.items():
        logging.info(f"{dataset}: {len(examples)} examples")

if __name__ == '__main__':
    main()
