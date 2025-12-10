"""
Calculate Accuracy from Partial Results

For incomplete runs (e.g., interrupted before completion), this script calculates
accuracy from individual question result files.

Usage:
    python calculate_partial_accuracy.py <results_directory>

Example:
    python calculate_partial_accuracy.py multi-agent-gemma/full/ddxplus_500q_seed42_run1
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def calculate_accuracy_from_questions(questions_dir):
    """
    Calculate accuracy from individual question result files.

    Args:
        questions_dir: Path to the questions directory containing q*_results.json files

    Returns:
        dict: Accuracy metrics including correct count, total, and percentage
    """
    questions_path = Path(questions_dir)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions directory not found: {questions_dir}")

    # Find all question result files
    question_files = sorted(questions_path.glob("q*_results.json"))

    if not question_files:
        raise ValueError(f"No question result files found in {questions_dir}")

    # Track results
    correct = 0
    total = 0
    errors = 0

    # Detailed breakdown
    results_by_agent = defaultdict(lambda: {"correct": 0, "total": 0})
    convergence_data = []
    question_details = []

    print(f"Processing {len(question_files)} question files...")
    print()

    for q_file in question_files:
        try:
            with open(q_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract key information
            question_id = data.get("question_id", "unknown")
            ground_truth = data.get("ground_truth")
            is_correct = data.get("is_correct")

            # Get final answer
            final_decision = data.get("final_decision", {})
            primary_answer = final_decision.get("primary_answer")

            # Get convergence info
            convergence = final_decision.get("convergence", {})
            converged = convergence.get("converged", False)
            agreement_rate = convergence.get("first_choice_agreement", 0)

            # Track overall accuracy
            if is_correct is not None:
                total += 1
                if is_correct:
                    correct += 1

                # Store details
                question_details.append({
                    "id": question_id,
                    "correct": is_correct,
                    "predicted": primary_answer,
                    "actual": ground_truth,
                    "converged": converged,
                    "agreement": agreement_rate
                })

            convergence_data.append({
                "converged": converged,
                "agreement": agreement_rate
            })

            # Track per-agent accuracy (from round3 results)
            round3 = data.get("round3_results", {})
            for agent_id, agent_data in round3.items():
                if isinstance(agent_data, dict):
                    agent_answer = agent_data.get("answer")
                    if agent_answer and ground_truth:
                        results_by_agent[agent_id]["total"] += 1
                        if agent_answer == ground_truth:
                            results_by_agent[agent_id]["correct"] += 1

        except Exception as e:
            errors += 1
            print(f"Warning: Error processing {q_file.name}: {e}")

    # Calculate statistics
    accuracy = (correct / total * 100) if total > 0 else 0
    convergence_rate = (sum(1 for x in convergence_data if x["converged"]) / len(convergence_data) * 100) if convergence_data else 0
    avg_agreement = (sum(x["agreement"] for x in convergence_data) / len(convergence_data) * 100) if convergence_data else 0

    results = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "errors": errors,
        "convergence_rate": convergence_rate,
        "avg_agreement_rate": avg_agreement,
        "per_agent": dict(results_by_agent),
        "questions": question_details
    }

    return results

def print_report(results, run_dir):
    """Print a formatted accuracy report."""
    print("=" * 80)
    print("PARTIAL ACCURACY REPORT")
    print("=" * 80)
    print()
    print(f"Results Directory: {run_dir}")
    print()
    print("-" * 80)
    print("OVERALL ACCURACY")
    print("-" * 80)
    print(f"Correct:           {results['correct']}")
    print(f"Total Questions:   {results['total']}")
    print(f"Accuracy:          {results['accuracy']:.2f}%")
    print(f"Errors/Skipped:    {results['errors']}")
    print()
    print("-" * 80)
    print("CONVERGENCE METRICS")
    print("-" * 80)
    print(f"Convergence Rate:  {results['convergence_rate']:.2f}%")
    print(f"Avg Agreement:     {results['avg_agreement_rate']:.2f}%")
    print()

    # Per-agent accuracy
    if results['per_agent']:
        print("-" * 80)
        print("PER-AGENT ACCURACY")
        print("-" * 80)
        for agent_id, stats in sorted(results['per_agent'].items()):
            agent_acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{agent_id:10s}: {stats['correct']:3d}/{stats['total']:3d} = {agent_acc:6.2f}%")
        print()

    print("=" * 80)
    print()

def save_report(results, run_dir):
    """Save a JSON report of the results."""
    report_path = Path(run_dir) / "partial_accuracy_report.json"

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Detailed report saved to: {report_path}")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_partial_accuracy.py <results_directory>")
        print()
        print("Example:")
        print("  python calculate_partial_accuracy.py multi-agent-gemma/full/ddxplus_500q_seed42_run1")
        sys.exit(1)

    run_dir = sys.argv[1]
    questions_dir = Path(run_dir) / "questions"

    try:
        # Calculate accuracy
        results = calculate_accuracy_from_questions(questions_dir)

        # Print report
        print_report(results, run_dir)

        # Save detailed JSON report
        save_report(results, run_dir)

        # Print summary for easy copying
        print("SUMMARY FOR RECORDS:")
        print(f"  Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")
        print(f"  Convergence: {results['convergence_rate']:.2f}%")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
