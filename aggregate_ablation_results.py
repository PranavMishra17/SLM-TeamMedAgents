"""
Aggregate Teamwork Ablation Results

Reads all results from multi-agent-gemma/ablation/ and generates a comprehensive table.

For each dataset, reads 4 configurations × 3 seeds and calculates mean ± std dev.

Configuration mapping (based on run number):
- run1: Team Orientation + Mutual Monitoring
- run2: SMM + Trust
- run3: Team Orientation + SMM + Leadership
- run4: All Teamwork Components
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# Configuration mapping
CONFIG_MAPPING = {
    "run1": "TO+MM",
    "run2": "SMM+Trust",
    "run3": "TO+SMM+L",
    "run4": "All"
}

CONFIG_FULL_NAMES = {
    "run1": "Team Orientation + Mutual Monitoring",
    "run2": "SMM + Trust",
    "run3": "Team Orientation + SMM + Leadership",
    "run4": "All Teamwork Components"
}

# Dataset names
DATASETS = [
    "ddxplus",
    "medbullets",
    "medmcqa",
    "medqa",
    "path_vqa",
    "pmc_vqa",
    "pubmedqa",
    "mmlupro"
]

def read_summary_report(file_path):
    """Read summary_report.json and extract key metrics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract overall accuracy (borda_count)
    accuracy = data['accuracy']['overall_accuracy'] * 100  # Convert to percentage

    # Extract convergence rate
    convergence = data['convergence']['overall_convergence_rate'] * 100

    # Extract token usage
    total_tokens = data['token_usage']['total_tokens']

    # Extract timing
    total_time = data['timing']['total_time']

    return {
        'accuracy': accuracy,
        'convergence': convergence,
        'total_tokens': total_tokens,
        'total_time': total_time
    }

def parse_directory_name(dir_name):
    """
    Parse directory name to extract dataset, seed, and run number.

    Example: ddxplus_50q_seed1_run1 -> (ddxplus, 1, run1)
    """
    parts = dir_name.split('_')

    # Find seed and run indices
    seed_idx = None
    run_idx = None

    for i, part in enumerate(parts):
        if part.startswith('seed'):
            seed_idx = i
        elif part.startswith('run'):
            run_idx = i

    if seed_idx is None or run_idx is None:
        return None, None, None

    # Extract dataset name (everything before seed)
    dataset = '_'.join(parts[:seed_idx-1])  # -1 to exclude '50q'

    # Extract seed number
    seed_str = parts[seed_idx]
    seed_num = int(seed_str.replace('seed', ''))

    # Extract run identifier
    run_str = parts[run_idx]

    return dataset, seed_num, run_str

def aggregate_results(ablation_dir):
    """
    Aggregate all results from ablation directory.

    Returns:
        dict: Nested dictionary structure:
            {dataset: {config: {metric: [values_for_each_seed]}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    ablation_path = Path(ablation_dir)

    # Iterate through all directories
    for dir_path in ablation_path.iterdir():
        if not dir_path.is_dir():
            continue

        # Parse directory name
        dataset, seed, run = parse_directory_name(dir_path.name)

        if dataset is None or dataset not in DATASETS:
            continue

        # Read summary report
        summary_path = dir_path / "summary_report.json"
        if not summary_path.exists():
            print(f"Warning: No summary_report.json in {dir_path.name}")
            continue

        try:
            metrics = read_summary_report(summary_path)

            # Store results
            results[dataset][run]['accuracy'].append(metrics['accuracy'])
            results[dataset][run]['convergence'].append(metrics['convergence'])
            results[dataset][run]['total_tokens'].append(metrics['total_tokens'])
            results[dataset][run]['total_time'].append(metrics['total_time'])
            results[dataset][run]['seeds'].append(seed)

        except Exception as e:
            print(f"Error reading {dir_path.name}: {e}")

    return results

def calculate_stats(values):
    """Calculate mean and standard deviation."""
    if not values:
        return None, None

    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0

    return mean, std

def format_result(mean, std):
    """Format result as 'mean ± std'."""
    if mean is None:
        return "N/A"
    return f"{mean:.2f} ± {std:.2f}"

def create_accuracy_table(results):
    """Create accuracy table with configurations as rows and datasets as columns."""

    # Create DataFrame with configurations as rows
    rows = []

    for run in ['run1', 'run2', 'run3', 'run4']:
        config_name = CONFIG_MAPPING[run]
        config_full = CONFIG_FULL_NAMES[run]

        row = {
            'Configuration': config_name,
            'Full_Name': config_full
        }

        for dataset in DATASETS:
            if run in results[dataset]:
                accuracies = results[dataset][run]['accuracy']
                mean, std = calculate_stats(accuracies)
                row[dataset.upper()] = format_result(mean, std)
            else:
                row[dataset.upper()] = "N/A"

        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def create_full_stats_table(results):
    """Create comprehensive table with all metrics."""

    rows = []

    for dataset in DATASETS:
        for run in ['run1', 'run2', 'run3', 'run4']:
            config_name = CONFIG_MAPPING[run]
            config_full = CONFIG_FULL_NAMES[run]

            if run not in results[dataset]:
                continue

            # Calculate statistics for each metric
            acc_mean, acc_std = calculate_stats(results[dataset][run]['accuracy'])
            conv_mean, conv_std = calculate_stats(results[dataset][run]['convergence'])
            tok_mean, tok_std = calculate_stats(results[dataset][run]['total_tokens'])
            time_mean, time_std = calculate_stats(results[dataset][run]['total_time'])

            n_seeds = len(results[dataset][run]['seeds'])

            row = {
                'Dataset': dataset.upper(),
                'Configuration': config_name,
                'Config_Full': config_full,
                'Accuracy (%)': format_result(acc_mean, acc_std),
                'Convergence (%)': format_result(conv_mean, conv_std),
                'Total Tokens': format_result(tok_mean, tok_std),
                'Total Time (s)': format_result(time_mean, time_std),
                'N_Seeds': n_seeds,
                'Accuracy_Mean': acc_mean,
                'Accuracy_Std': acc_std
            }

            rows.append(row)

    df = pd.DataFrame(rows)
    return df

def save_results(ablation_dir, output_dir="."):
    """Main function to aggregate and save results."""

    print("=" * 80)
    print("AGGREGATING TEAMWORK ABLATION RESULTS")
    print("=" * 80)
    print()

    # Aggregate results
    print("Reading results from:", ablation_dir)
    results = aggregate_results(ablation_dir)

    # Count total runs found
    total_runs = sum(
        sum(len(results[ds][run]['accuracy'])
            for run in results[ds])
        for ds in results
    )
    print(f"Found {total_runs} runs across {len(results)} datasets")
    print()

    # Create accuracy table
    print("Creating accuracy table...")
    accuracy_df = create_accuracy_table(results)

    # Create full statistics table
    print("Creating comprehensive statistics table...")
    full_df = create_full_stats_table(results)

    # Save to CSV
    accuracy_csv = Path(output_dir) / "ablation_accuracy_summary.csv"
    full_csv = Path(output_dir) / "ablation_full_stats.csv"

    accuracy_df.to_csv(accuracy_csv, index=False)
    full_df.to_csv(full_csv, index=False)

    print(f"\nSaved accuracy table to: {accuracy_csv}")
    print(f"Saved full stats table to: {full_csv}")
    print()

    # Print accuracy table
    print("=" * 80)
    print("ACCURACY SUMMARY (Mean ± Std Dev)")
    print("=" * 80)
    print()
    print(accuracy_df.to_string(index=False))
    print()

    # Print best configurations per dataset
    print("=" * 80)
    print("BEST CONFIGURATION PER DATASET")
    print("=" * 80)
    print()

    for dataset in DATASETS:
        best_config = None
        best_acc = -1

        for run in ['run1', 'run2', 'run3', 'run4']:
            if run in results[dataset]:
                accuracies = results[dataset][run]['accuracy']
                mean, _ = calculate_stats(accuracies)
                if mean is not None and mean > best_acc:
                    best_acc = mean
                    best_config = CONFIG_MAPPING[run]

        if best_config is not None:
            print(f"{dataset.upper():15s}: {best_config:15s} ({best_acc:.2f}%)")
        else:
            print(f"{dataset.upper():15s}: No results found")

    print()
    print("=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)

    return accuracy_df, full_df

if __name__ == "__main__":
    # Set paths
    ablation_dir = "multi-agent-gemma/ablation"
    output_dir = "."

    # Check if directory exists
    if not os.path.exists(ablation_dir):
        print(f"Error: Ablation directory not found: {ablation_dir}")
        print("Please ensure the path is correct.")
        exit(1)

    # Run aggregation
    accuracy_df, full_df = save_results(ablation_dir, output_dir)
