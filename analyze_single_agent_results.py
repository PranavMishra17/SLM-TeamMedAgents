import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def find_results_files(base_path, dataset, method):
    """Find all results JSON files for a given dataset and method"""
    method_path = os.path.join(base_path, dataset, method)

    # Pattern to match results files
    pattern = f"results_{dataset}_{method}_*.json"
    results_files = glob.glob(os.path.join(method_path, pattern))

    return results_files

def extract_metrics_from_file(file_path):
    """Extract metrics from a single results JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        summary = data.get('summary', {})

        metrics = {
            'accuracy': summary.get('accuracy', 0),
            'total_questions': summary.get('total_questions', 0),
            'correct_answers': summary.get('correct_answers', 0),
            'total_time': summary.get('total_time', 0),
            'avg_time_per_question': summary.get('avg_time_per_question', 0),
            'total_input_tokens': summary.get('token_usage', {}).get('total_input_tokens', 0),
            'total_output_tokens': summary.get('token_usage', {}).get('total_output_tokens', 0),
            'total_tokens': summary.get('token_usage', {}).get('total_tokens', 0),
            'avg_input_tokens_per_question': summary.get('token_usage', {}).get('avg_input_tokens_per_question', 0),
            'avg_output_tokens_per_question': summary.get('token_usage', {}).get('avg_output_tokens_per_question', 0),
            'avg_total_tokens_per_question': summary.get('token_usage', {}).get('avg_total_tokens_per_question', 0)
        }

        return metrics
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def calculate_stats(values):
    """Calculate mean ± std for a list of values"""
    if not values or len(values) == 0:
        return "N/A", "N/A"

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0

    return mean, std

def format_mean_std(mean, std, decimals=2):
    """Format mean ± std as string"""
    if isinstance(mean, str):
        return mean
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def main():
    # Base path for results
    base_path = r"E:\SLM-TeamMedAgents\SLM_Results\gemma3_4b"

    # All datasets
    datasets = [
        'ddxplus',
        'medbullets',
        'medmcqa',
        'medqa',
        'mmlupro-med',
        'path_vqa',
        'pmc_vqa',
        'pubmedqa'
    ]

    # All methods
    methods = ['zero_shot', 'few_shot', 'cot']

    # Method display names
    method_names = {
        'zero_shot': 'Zero-Shot',
        'few_shot': 'Few-Shot',
        'cot': 'CoT'
    }

    # Collect all results
    all_results = {}

    for dataset in datasets:
        all_results[dataset] = {}

        for method in methods:
            print(f"Processing {dataset} - {method}...")

            # Find all results files
            results_files = find_results_files(base_path, dataset, method)

            if not results_files:
                print(f"  Warning: No results files found for {dataset}/{method}")
                all_results[dataset][method] = None
                continue

            print(f"  Found {len(results_files)} results files")

            # Extract metrics from each file
            metrics_list = []
            for file_path in results_files:
                metrics = extract_metrics_from_file(file_path)
                if metrics:
                    metrics_list.append(metrics)

            if not metrics_list:
                print(f"  Warning: Could not extract metrics for {dataset}/{method}")
                all_results[dataset][method] = None
                continue

            # Calculate statistics for each metric
            stats = {}
            for metric_name in metrics_list[0].keys():
                values = [m[metric_name] for m in metrics_list]
                mean, std = calculate_stats(values)
                stats[metric_name] = {
                    'mean': mean,
                    'std': std,
                    'values': values
                }

            all_results[dataset][method] = stats

    # === TABLE 1: ACCURACY ===
    print("\n" + "="*80)
    print("Creating Table 1: Accuracy")
    print("="*80)

    accuracy_data = []

    for method in methods:
        row = {'Method': method_names[method]}

        for dataset in datasets:
            if all_results[dataset].get(method):
                stats = all_results[dataset][method]['accuracy']
                # Convert to percentage
                mean_pct = stats['mean'] * 100
                std_pct = stats['std'] * 100
                row[dataset] = format_mean_std(mean_pct, std_pct, decimals=2)
            else:
                row[dataset] = "N/A"

        accuracy_data.append(row)

    # Create DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)

    # Reorder columns
    columns_order = ['Method'] + datasets
    accuracy_df = accuracy_df[columns_order]

    # Save to CSV
    accuracy_csv_path = r"E:\SLM-TeamMedAgents\single_agent_accuracy_summary.csv"
    accuracy_df.to_csv(accuracy_csv_path, index=False)
    print(f"\nAccuracy table saved to: {accuracy_csv_path}")
    print("\nAccuracy Table Preview:")
    print(accuracy_df.to_string(index=False))

    # === TABLE 2: TOKEN USAGE AND INFERENCE SUMMARY ===
    print("\n" + "="*80)
    print("Creating Table 2: Token Usage and Inference Summary")
    print("="*80)

    token_data = []

    for method in methods:
        for dataset in datasets:
            if all_results[dataset].get(method):
                stats = all_results[dataset][method]

                row = {
                    'Dataset': dataset,
                    'Method': method_names[method],
                    'Accuracy (%)': format_mean_std(stats['accuracy']['mean'] * 100,
                                                    stats['accuracy']['std'] * 100,
                                                    decimals=2),
                    'Avg Input Tokens': format_mean_std(stats['avg_input_tokens_per_question']['mean'],
                                                        stats['avg_input_tokens_per_question']['std'],
                                                        decimals=1),
                    'Avg Output Tokens': format_mean_std(stats['avg_output_tokens_per_question']['mean'],
                                                         stats['avg_output_tokens_per_question']['std'],
                                                         decimals=1),
                    'Avg Total Tokens': format_mean_std(stats['avg_total_tokens_per_question']['mean'],
                                                        stats['avg_total_tokens_per_question']['std'],
                                                        decimals=1),
                    'Avg Time/Question (s)': format_mean_std(stats['avg_time_per_question']['mean'],
                                                             stats['avg_time_per_question']['std'],
                                                             decimals=2),
                    'Total Questions': int(stats['total_questions']['mean'])
                }

                token_data.append(row)
            else:
                row = {
                    'Dataset': dataset,
                    'Method': method_names[method],
                    'Accuracy (%)': "N/A",
                    'Avg Input Tokens': "N/A",
                    'Avg Output Tokens': "N/A",
                    'Avg Total Tokens': "N/A",
                    'Avg Time/Question (s)': "N/A",
                    'Total Questions': "N/A"
                }
                token_data.append(row)

    # Create DataFrame
    token_df = pd.DataFrame(token_data)

    # Save to CSV
    token_csv_path = r"E:\SLM-TeamMedAgents\single_agent_token_inference_summary.csv"
    token_df.to_csv(token_csv_path, index=False)
    print(f"\nToken and Inference table saved to: {token_csv_path}")
    print("\nToken and Inference Table Preview (first 15 rows):")
    print(token_df.head(15).to_string(index=False))

    # === SUMMARY STATISTICS ===
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Count successful extractions
    total_combinations = len(datasets) * len(methods)
    successful = sum(1 for dataset in datasets for method in methods
                    if all_results[dataset].get(method) is not None)

    print(f"\nTotal dataset-method combinations: {total_combinations}")
    print(f"Successfully processed: {successful}")
    print(f"Missing/Failed: {total_combinations - successful}")

    # Print detailed breakdown
    print("\nDetailed breakdown:")
    for dataset in datasets:
        methods_found = [method for method in methods if all_results[dataset].get(method)]
        methods_missing = [method for method in methods if not all_results[dataset].get(method)]

        status = "[OK]" if len(methods_found) == 3 else "[!]"
        print(f"  {status} {dataset}: {len(methods_found)}/3 methods found", end="")
        if methods_missing:
            print(f" (missing: {', '.join(methods_missing)})", end="")
        print()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
