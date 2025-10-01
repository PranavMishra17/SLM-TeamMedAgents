"""
Script to fix path-VQA results by re-extracting answers with correct pattern.
The issue: task_type was 'yes_no_maybe' but models responded with A/B format.
This script re-extracts A/B answers from existing result files.
"""

import json
import re
import os
from pathlib import Path
from typing import Optional


def extract_answer_mcq(response: str) -> Optional[str]:
    """Extract MCQ answer (A/B) from model response."""
    patterns = [
        r"(?:Final\s+)?Answer:\s*([A-J])\b",
        r"(?:Final\s+)?Answer:\s*\(?([A-J])\)",
        r"^([A-J])\.",
        r"\b([A-J])\b(?=\s*[-\.\)]|\s*$)"
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    return None


def fix_result_file(file_path: str, dry_run: bool = False) -> dict:
    """Fix a single result file by re-extracting answers."""
    print(f"\nProcessing: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'results' not in data:
        print(f"  Skipping - not a results file")
        return {"skipped": 1}

    summary = data.get('summary', {})
    if summary.get('dataset') != 'path_vqa':
        print(f"  Skipping - not path_vqa dataset")
        return {"skipped": 1}

    results = data['results']
    stats = {
        "total": len(results),
        "fixed": 0,
        "already_correct": 0,
        "still_null": 0,
        "changed_from_wrong_to_correct": 0,
        "changed_from_wrong_to_wrong": 0
    }

    for result in results:
        old_extracted = result.get('extracted_answer')
        full_response = result.get('full_response', '')
        ground_truth = result.get('ground_truth')
        old_is_correct = result.get('is_correct', False)

        # Re-extract with MCQ pattern
        new_extracted = extract_answer_mcq(full_response)

        if new_extracted != old_extracted:
            stats['fixed'] += 1
            result['extracted_answer'] = new_extracted

            # Re-evaluate correctness
            new_is_correct = (new_extracted == ground_truth) if new_extracted else False
            result['is_correct'] = new_is_correct

            if new_is_correct and not old_is_correct:
                stats['changed_from_wrong_to_correct'] += 1
                print(f"  Q{result['question_index']}: {old_extracted} -> {new_extracted} (now CORRECT, GT: {ground_truth})")
            elif not new_is_correct and old_is_correct:
                print(f"  Q{result['question_index']}: {old_extracted} -> {new_extracted} (now WRONG, GT: {ground_truth})")
            else:
                stats['changed_from_wrong_to_wrong'] += 1
                print(f"  Q{result['question_index']}: {old_extracted} -> {new_extracted} (still wrong, GT: {ground_truth})")
        else:
            if old_is_correct:
                stats['already_correct'] += 1

        if new_extracted is None:
            stats['still_null'] += 1

    # Recalculate summary statistics
    correct_answers = sum(1 for r in results if r.get('is_correct', False))
    total_questions = len(results)
    accuracy = (correct_answers / total_questions) if total_questions > 0 else 0.0

    old_accuracy = summary.get('accuracy', 0.0)
    old_correct = summary.get('correct_answers', 0)

    data['summary']['correct_answers'] = correct_answers
    data['summary']['accuracy'] = accuracy

    print(f"  Accuracy: {old_correct}/{total_questions} ({old_accuracy:.1%}) -> {correct_answers}/{total_questions} ({accuracy:.1%})")
    print(f"  Fixed: {stats['fixed']}, Correct now: {stats['changed_from_wrong_to_correct']}")

    # Save fixed file
    if not dry_run:
        # Backup original
        backup_path = file_path + '.backup'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  Backup saved: {backup_path}")

        # Save fixed version
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Fixed file saved")
    else:
        print(f"  DRY RUN - no files modified")

    return stats


def find_path_vqa_result_files(base_dir: str) -> list:
    """Find all path_vqa result files."""
    result_files = []
    base_path = Path(base_dir)

    # Search for path_vqa result files
    for json_file in base_path.rglob('*path_vqa*.json'):
        if json_file.name.startswith('results_'):
            result_files.append(str(json_file))

    return sorted(result_files)


def main():
    """Main function to fix all path-VQA result files."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix path-VQA results by re-extracting A/B answers')
    parser.add_argument('--results_dir', default='SLM_Results', help='Base results directory')
    parser.add_argument('--dry_run', action='store_true', help='Dry run - do not modify files')

    args = parser.parse_args()

    print("="*60)
    print("Path-VQA Results Fixer")
    print("="*60)

    result_files = find_path_vqa_result_files(args.results_dir)

    if not result_files:
        print(f"\nNo path_vqa result files found in {args.results_dir}")
        return

    print(f"\nFound {len(result_files)} path_vqa result file(s)")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")

    total_stats = {
        "files_processed": 0,
        "total_fixed": 0,
        "total_changed_to_correct": 0
    }

    for file_path in result_files:
        stats = fix_result_file(file_path, args.dry_run)

        if 'fixed' in stats:
            total_stats['files_processed'] += 1
            total_stats['total_fixed'] += stats['fixed']
            total_stats['total_changed_to_correct'] += stats['changed_from_wrong_to_correct']

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total answers fixed: {total_stats['total_fixed']}")
    print(f"Total changed to correct: {total_stats['total_changed_to_correct']}")

    if args.dry_run:
        print("\n*** This was a DRY RUN - run without --dry_run to apply fixes ***")
    else:
        print("\n*** Fixes applied! Original files backed up with .backup extension ***")


if __name__ == '__main__':
    main()
