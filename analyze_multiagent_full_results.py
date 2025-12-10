import json
import os
import pandas as pd
from pathlib import Path

def extract_metrics_from_summary(file_path):
    """Extract all relevant metrics from a summary_report.json or partial_accuracy_report.json file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if this is a partial report (simpler structure)
        # Partial reports have 'correct' key at root and 'accuracy' as a float, not nested
        is_partial = ('correct' in data and 'accuracy' in data and
                     isinstance(data.get('accuracy'), (int, float)) and
                     'metadata' not in data)

        if is_partial:
            # Handle partial_accuracy_report.json format
            # Convert accuracy from percentage to decimal if needed
            accuracy_val = data.get('accuracy', 0)
            if isinstance(accuracy_val, (int, float)) and accuracy_val > 1:
                accuracy_val = accuracy_val / 100.0

            # Convert convergence rate
            convergence_val = data.get('convergence_rate', 0)
            if isinstance(convergence_val, (int, float)) and convergence_val > 1:
                convergence_val = convergence_val / 100.0

            # Convert agreement rate
            agreement_val = data.get('avg_agreement_rate', 0)
            if isinstance(agreement_val, (int, float)) and agreement_val > 1:
                agreement_val = agreement_val / 100.0

            metrics = {
                # Accuracy metrics (limited)
                'overall_accuracy': accuracy_val,
                'borda_accuracy': accuracy_val,
                'majority_accuracy': 0,
                'weighted_accuracy': 0,
                'correct_count': data.get('correct', 0),
                'total_count': data.get('total', 0),

                # Metadata (limited)
                'total_questions': data.get('total', 0),
                'avg_agents_per_question': 0,
                'questions_with_ground_truth': data.get('total', 0),

                # Token usage (not available)
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'questions_processed': data.get('total', 0),
                'avg_input_tokens_per_question': 0,
                'avg_output_tokens_per_question': 0,
                'avg_total_tokens_per_question': 0,

                # API calls (not available)
                'total_api_calls': 0,
                'avg_calls_per_question': 0,

                # Timing (not available)
                'total_time': 0,
                'avg_time_per_question': 0,
                'avg_recruit_time': 0,
                'avg_round1_time': 0,
                'avg_round2_time': 0,
                'avg_round3_time': 0,
                'avg_aggregation_time': 0,

                # Convergence (limited)
                'overall_convergence_rate': convergence_val,
                'round1_convergence': convergence_val,
                'round3_convergence': convergence_val,
                'convergence_increase': 0,

                # Agreement metrics (limited)
                'most_agreeable_pair': 'N/A',
                'most_agreeable_rate': 0,
                'most_disagreeable_pair': 'N/A',
                'most_disagreeable_rate': 0,
                'average_pairwise_agreement': agreement_val,

                # Opinion changes (not available)
                'overall_change_rate': 0,
            }
        else:
            # Handle full summary_report.json format
            metrics = {
                # Accuracy metrics
                'overall_accuracy': data.get('accuracy', {}).get('overall_accuracy', 0),
                'borda_accuracy': data.get('accuracy', {}).get('borda_accuracy', 0),
                'majority_accuracy': data.get('accuracy', {}).get('majority_accuracy', 0),
                'weighted_accuracy': data.get('accuracy', {}).get('weighted_accuracy', 0),
                'correct_count': data.get('accuracy', {}).get('correct_count', 0),
                'total_count': data.get('accuracy', {}).get('total_count', 0),

                # Metadata
                'total_questions': data.get('metadata', {}).get('total_questions', 0),
                'avg_agents_per_question': data.get('metadata', {}).get('avg_agents_per_question', 0),
                'questions_with_ground_truth': data.get('metadata', {}).get('questions_with_ground_truth', 0),

                # Token usage
                'total_input_tokens': data.get('token_usage', {}).get('total_input_tokens', 0),
                'total_output_tokens': data.get('token_usage', {}).get('total_output_tokens', 0),
                'total_tokens': data.get('token_usage', {}).get('total_tokens', 0),
                'questions_processed': data.get('token_usage', {}).get('questions_processed', 0),
                'avg_input_tokens_per_question': data.get('token_usage', {}).get('avg_input_tokens_per_question', 0),
                'avg_output_tokens_per_question': data.get('token_usage', {}).get('avg_output_tokens_per_question', 0),
                'avg_total_tokens_per_question': data.get('token_usage', {}).get('avg_total_tokens_per_question', 0),

                # API calls
                'total_api_calls': data.get('api_calls', {}).get('total_calls', 0),
                'avg_calls_per_question': data.get('api_calls', {}).get('avg_calls_per_question', 0),

                # Timing
                'total_time': data.get('timing', {}).get('total_time', 0),
                'avg_time_per_question': data.get('timing', {}).get('avg_time_per_question', 0),
                'avg_recruit_time': data.get('timing', {}).get('avg_recruit_time', 0),
                'avg_round1_time': data.get('timing', {}).get('avg_round1_time', 0),
                'avg_round2_time': data.get('timing', {}).get('avg_round2_time', 0),
                'avg_round3_time': data.get('timing', {}).get('avg_round3_time', 0),
                'avg_aggregation_time': data.get('timing', {}).get('avg_aggregation_time', 0),

                # Convergence
                'overall_convergence_rate': data.get('convergence', {}).get('overall_convergence_rate', 0),
                'round1_convergence': data.get('convergence', {}).get('round1_convergence', 0),
                'round3_convergence': data.get('convergence', {}).get('round3_convergence', 0),
                'convergence_increase': data.get('convergence', {}).get('convergence_increase', 0),

                # Agreement metrics
                'most_agreeable_pair': data.get('convergence', {}).get('most_agreeable_pair', 'N/A'),
                'most_agreeable_rate': data.get('convergence', {}).get('most_agreeable_rate', 0),
                'most_disagreeable_pair': data.get('convergence', {}).get('most_disagreeable_pair', 'N/A'),
                'most_disagreeable_rate': data.get('convergence', {}).get('most_disagreeable_rate', 0),
                'average_pairwise_agreement': data.get('convergence', {}).get('average_pairwise_agreement', 0),

                # Opinion changes
                'overall_change_rate': data.get('opinion_changes', {}).get('overall_change_rate', 0),
            }

        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    # Base path for multi-agent full results
    base_path = r"E:\SLM-TeamMedAgents\multi-agent-gemma\full"

    # Dataset configurations
    datasets = [
        'ddxplus',
        'medbullets',
        'medmcqa',
        'medqa',
        'mmlupro',
        'path_vqa',
        'pmc_vqa',
        'pubmedqa'
    ]

    # Collect all results
    all_results = {}

    print("="*80)
    print("EXTRACTING MULTI-AGENT FULL DATASET RESULTS")
    print("="*80)

    for dataset in datasets:
        # Construct path to summary report
        run_dir = f"{dataset}_500q_seed42_run1"
        summary_path = os.path.join(base_path, run_dir, "summary_report.json")
        partial_path = os.path.join(base_path, run_dir, "partial_accuracy_report.json")

        print(f"\nProcessing {dataset}...")

        # Try summary_report.json first, then partial_accuracy_report.json
        if os.path.exists(summary_path):
            print(f"  Found: summary_report.json")
            report_path = summary_path
        elif os.path.exists(partial_path):
            print(f"  Found: partial_accuracy_report.json (limited metrics)")
            report_path = partial_path
        else:
            print(f"  WARNING: No report file found!")
            all_results[dataset] = None
            continue

        # Extract metrics
        metrics = extract_metrics_from_summary(report_path)

        if metrics:
            print(f"  [OK] Successfully extracted metrics")
            print(f"    - Total questions: {metrics['total_questions']}")
            print(f"    - Borda accuracy: {metrics['borda_accuracy']*100:.2f}%")
            all_results[dataset] = metrics
        else:
            print(f"  [FAIL] Failed to extract metrics")
            all_results[dataset] = None

    # === TABLE 1: ACCURACY ONLY ===
    print("\n" + "="*80)
    print("TABLE 1: ACCURACY SUMMARY")
    print("="*80)

    accuracy_data = []
    for dataset in datasets:
        if all_results.get(dataset):
            metrics = all_results[dataset]
            row = {
                'Dataset': dataset,
                'Total Questions': metrics['total_questions'],
                'Borda Accuracy (%)': f"{metrics['borda_accuracy']*100:.2f}",
                'Overall Accuracy (%)': f"{metrics['overall_accuracy']*100:.2f}",
                'Majority Vote (%)': f"{metrics['majority_accuracy']*100:.2f}",
                'Weighted (%)': f"{metrics['weighted_accuracy']*100:.2f}",
                'Correct Count': metrics['correct_count']
            }
            accuracy_data.append(row)
        else:
            row = {
                'Dataset': dataset,
                'Total Questions': 'N/A',
                'Borda Accuracy (%)': 'N/A',
                'Overall Accuracy (%)': 'N/A',
                'Majority Vote (%)': 'N/A',
                'Weighted (%)': 'N/A',
                'Correct Count': 'N/A'
            }
            accuracy_data.append(row)

    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_csv_path = r"E:\SLM-TeamMedAgents\multiagent_full_accuracy_summary.csv"
    accuracy_df.to_csv(accuracy_csv_path, index=False)

    print(f"\nSaved to: {accuracy_csv_path}")
    print("\nPreview:")
    print(accuracy_df.to_string(index=False))

    # === TABLE 2: TOKEN USAGE AND INFERENCE ===
    print("\n" + "="*80)
    print("TABLE 2: TOKEN USAGE AND INFERENCE SUMMARY")
    print("="*80)

    token_data = []
    for dataset in datasets:
        if all_results.get(dataset):
            metrics = all_results[dataset]
            row = {
                'Dataset': dataset,
                'Total Questions': metrics['total_questions'],
                'Avg Agents/Question': f"{metrics['avg_agents_per_question']:.2f}",
                'Avg Input Tokens': f"{metrics['avg_input_tokens_per_question']:.1f}",
                'Avg Output Tokens': f"{metrics['avg_output_tokens_per_question']:.1f}",
                'Avg Total Tokens': f"{metrics['avg_total_tokens_per_question']:.1f}",
                'Total API Calls': metrics['total_api_calls'],
                'Avg API Calls/Q': f"{metrics['avg_calls_per_question']:.2f}",
                'Avg Time/Question (s)': f"{metrics['avg_time_per_question']:.2f}",
                'Total Time (s)': f"{metrics['total_time']:.2f}",
            }
            token_data.append(row)
        else:
            row = {
                'Dataset': dataset,
                'Total Questions': 'N/A',
                'Avg Agents/Question': 'N/A',
                'Avg Input Tokens': 'N/A',
                'Avg Output Tokens': 'N/A',
                'Avg Total Tokens': 'N/A',
                'Total API Calls': 'N/A',
                'Avg API Calls/Q': 'N/A',
                'Avg Time/Question (s)': 'N/A',
                'Total Time (s)': 'N/A',
            }
            token_data.append(row)

    token_df = pd.DataFrame(token_data)
    token_csv_path = r"E:\SLM-TeamMedAgents\multiagent_full_token_inference_summary.csv"
    token_df.to_csv(token_csv_path, index=False)

    print(f"\nSaved to: {token_csv_path}")
    print("\nPreview:")
    print(token_df.to_string(index=False))

    # === TABLE 3: CONVERGENCE AND DISAGREEMENT ===
    print("\n" + "="*80)
    print("TABLE 3: CONVERGENCE AND AGREEMENT ANALYSIS")
    print("="*80)

    convergence_data = []
    for dataset in datasets:
        if all_results.get(dataset):
            metrics = all_results[dataset]
            row = {
                'Dataset': dataset,
                'Overall Convergence (%)': f"{metrics['overall_convergence_rate']*100:.2f}",
                'Round 1 Convergence (%)': f"{metrics['round1_convergence']*100:.2f}",
                'Round 3 Convergence (%)': f"{metrics['round3_convergence']*100:.2f}",
                'Convergence Increase (%)': f"{metrics['convergence_increase']*100:.2f}",
                'Avg Pairwise Agreement (%)': f"{metrics['average_pairwise_agreement']*100:.2f}",
                'Most Agreeable Pair': metrics['most_agreeable_pair'],
                'Most Agreeable Rate (%)': f"{metrics['most_agreeable_rate']*100:.2f}",
                'Most Disagreeable Pair': metrics['most_disagreeable_pair'],
                'Most Disagreeable Rate (%)': f"{metrics['most_disagreeable_rate']*100:.2f}",
                'Opinion Change Rate (%)': f"{metrics['overall_change_rate']*100:.2f}",
            }
            convergence_data.append(row)
        else:
            row = {
                'Dataset': dataset,
                'Overall Convergence (%)': 'N/A',
                'Round 1 Convergence (%)': 'N/A',
                'Round 3 Convergence (%)': 'N/A',
                'Convergence Increase (%)': 'N/A',
                'Avg Pairwise Agreement (%)': 'N/A',
                'Most Agreeable Pair': 'N/A',
                'Most Agreeable Rate (%)': 'N/A',
                'Most Disagreeable Pair': 'N/A',
                'Most Disagreeable Rate (%)': 'N/A',
                'Opinion Change Rate (%)': 'N/A',
            }
            convergence_data.append(row)

    convergence_df = pd.DataFrame(convergence_data)
    convergence_csv_path = r"E:\SLM-TeamMedAgents\multiagent_full_convergence_summary.csv"
    convergence_df.to_csv(convergence_csv_path, index=False)

    print(f"\nSaved to: {convergence_csv_path}")
    print("\nPreview:")
    print(convergence_df.to_string(index=False))

    # === DETAILED TIMING BREAKDOWN ===
    print("\n" + "="*80)
    print("TABLE 4: DETAILED TIMING BREAKDOWN")
    print("="*80)

    timing_data = []
    for dataset in datasets:
        if all_results.get(dataset):
            metrics = all_results[dataset]
            row = {
                'Dataset': dataset,
                'Total Time (s)': f"{metrics['total_time']:.2f}",
                'Avg Time/Question (s)': f"{metrics['avg_time_per_question']:.2f}",
                'Avg Recruit Time (s)': f"{metrics['avg_recruit_time']:.2f}",
                'Avg Round 1 Time (s)': f"{metrics['avg_round1_time']:.2f}",
                'Avg Round 2 Time (s)': f"{metrics['avg_round2_time']:.2f}",
                'Avg Round 3 Time (s)': f"{metrics['avg_round3_time']:.2f}",
                'Avg Aggregation Time (s)': f"{metrics['avg_aggregation_time']:.4f}",
            }
            timing_data.append(row)
        else:
            row = {
                'Dataset': dataset,
                'Total Time (s)': 'N/A',
                'Avg Time/Question (s)': 'N/A',
                'Avg Recruit Time (s)': 'N/A',
                'Avg Round 1 Time (s)': 'N/A',
                'Avg Round 2 Time (s)': 'N/A',
                'Avg Round 3 Time (s)': 'N/A',
                'Avg Aggregation Time (s)': 'N/A',
            }
            timing_data.append(row)

    timing_df = pd.DataFrame(timing_data)
    timing_csv_path = r"E:\SLM-TeamMedAgents\multiagent_full_timing_breakdown.csv"
    timing_df.to_csv(timing_csv_path, index=False)

    print(f"\nSaved to: {timing_csv_path}")
    print("\nPreview:")
    print(timing_df.to_string(index=False))

    # === SUMMARY STATISTICS ===
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    successful = sum(1 for dataset in datasets if all_results.get(dataset) is not None)
    print(f"\nTotal datasets: {len(datasets)}")
    print(f"Successfully processed: {successful}")
    print(f"Missing/Failed: {len(datasets) - successful}")

    if successful > 0:
        # Calculate aggregate statistics
        valid_results = [metrics for metrics in all_results.values() if metrics is not None]

        avg_borda_accuracy = sum(m['borda_accuracy'] for m in valid_results) / len(valid_results) * 100
        avg_convergence = sum(m['overall_convergence_rate'] for m in valid_results) / len(valid_results) * 100
        avg_tokens_per_q = sum(m['avg_total_tokens_per_question'] for m in valid_results) / len(valid_results)
        avg_time_per_q = sum(m['avg_time_per_question'] for m in valid_results) / len(valid_results)

        print(f"\nAggregate Statistics (across {successful} datasets):")
        print(f"  Average Borda Accuracy: {avg_borda_accuracy:.2f}%")
        print(f"  Average Convergence Rate: {avg_convergence:.2f}%")
        print(f"  Average Tokens per Question: {avg_tokens_per_q:.1f}")
        print(f"  Average Time per Question: {avg_time_per_q:.2f}s")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {accuracy_csv_path}")
    print(f"  2. {token_csv_path}")
    print(f"  3. {convergence_csv_path}")
    print(f"  4. {timing_csv_path}")

if __name__ == "__main__":
    main()
