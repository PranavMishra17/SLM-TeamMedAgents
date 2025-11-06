"""
Results Storage - Systematic Results Organization

Manages the storage of simulation results in an organized directory structure.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class ResultsStorage:
    """
    Systematic storage of simulation results.

    Directory structure:
    results/
      └── run_YYYYMMDD_HHMMSS/
          ├── config.json
          ├── questions/
          │   ├── q001_results.json
          │   └── ...
          ├── metrics/
          │   ├── accuracy_summary.csv
          │   ├── convergence_analysis.json
          │   └── ...
          ├── logs/  # Managed by SimulationLogger
          └── summary_report.json
    """

    def __init__(self, output_dir: str, run_id: str = None, dataset_name: str = None, n_questions: int = None, random_seed: int = None):
        """
        Initialize results storage.

        Args:
            output_dir: Base output directory
            run_id: Unique run identifier (auto-generated if None)
            dataset_name: Dataset name for run_id generation
            n_questions: Number of questions for run_id generation
            random_seed: Random seed for unique run identification (prevents overwrites)
        """
        if run_id:
            self.run_id = run_id
        elif dataset_name and n_questions:
            # Create descriptive run_id: medqa_50q_seed1_run1, medqa_50q_seed2_run1, etc.
            self.run_id = self._generate_run_id(output_dir, dataset_name, n_questions, random_seed)
        else:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.base_dir = Path(output_dir) / self.run_id

        # Create directory structure
        self.questions_dir = self.base_dir / "questions"
        self.metrics_dir = self.base_dir / "metrics"

        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _generate_run_id(self, output_dir: str, dataset_name: str, n_questions: int, random_seed: int = None) -> str:
        """
        Generate descriptive run_id like medqa_50q_seed1_run1, medqa_50q_seed2_run1, etc.

        Args:
            output_dir: Base output directory
            dataset_name: Dataset name
            n_questions: Number of questions
            random_seed: Random seed for unique identification

        Returns:
            Descriptive run_id with seed and incremental counter
        """
        # Include seed in base_id to prevent overwrites between different seeds
        if random_seed is not None:
            base_id = f"{dataset_name}_{n_questions}q_seed{random_seed}"
        else:
            base_id = f"{dataset_name}_{n_questions}q"

        base_path = Path(output_dir)

        # Find existing runs with this pattern
        existing_runs = []
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and item.name.startswith(base_id):
                    existing_runs.append(item.name)

        # Determine next run number
        if not existing_runs:
            return f"{base_id}_run1"

        # Extract run numbers
        run_numbers = []
        for run_name in existing_runs:
            if "_run" in run_name:
                try:
                    num = int(run_name.split("_run")[-1])
                    run_numbers.append(num)
                except ValueError:
                    continue

        next_num = max(run_numbers, default=0) + 1
        return f"{base_id}_run{next_num}"

    def save_config(self, config: Dict):
        """Save run configuration to config.json."""
        config_path = self.base_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def save_question_result(self, question_id: str, result: Dict):
        """Save individual question result."""
        result_path = self.questions_dir / f"{question_id}_results.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    def save_accuracy_summary(self, accuracy_metrics: Dict):
        """Save accuracy metrics to CSV."""
        csv_path = self.metrics_dir / "accuracy_summary.csv"

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Metric', 'Value', 'Correct', 'Total'])

            # Overall accuracy
            writer.writerow([
                'Overall Accuracy',
                f"{accuracy_metrics.get('overall_accuracy', 0):.2%}",
                accuracy_metrics.get('correct_count', 0),
                accuracy_metrics.get('total_count', 0)
            ])

            # By method
            writer.writerow([])
            writer.writerow(['Method Comparison', '', '', ''])

            method_comparison = accuracy_metrics.get('method_comparison', {})
            for method, metrics in method_comparison.items():
                writer.writerow([
                    method,
                    f"{metrics.get('accuracy', 0):.2%}",
                    metrics.get('correct', 0),
                    accuracy_metrics.get('total_count', 0)
                ])

            # By task type
            by_task_type = accuracy_metrics.get('by_task_type', {})
            if by_task_type:
                writer.writerow([])
                writer.writerow(['By Task Type', '', '', ''])

                for task_type, metrics in by_task_type.items():
                    writer.writerow([
                        task_type,
                        f"{metrics.get('accuracy', 0):.2%}",
                        metrics.get('correct', 0),
                        metrics.get('total', 0)
                    ])

    def save_convergence_analysis(self, convergence_metrics: Dict):
        """Save convergence analysis to JSON."""
        json_path = self.metrics_dir / "convergence_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convergence_metrics, f, indent=2, ensure_ascii=False)

    def save_disagreement_matrix(self, disagreement_data: Dict):
        """Save agent disagreement matrix to CSV."""
        csv_path = self.metrics_dir / "disagreement_matrix.csv"

        agent_pairs = disagreement_data.get('agent_pairs', {})

        if not agent_pairs:
            return

        # Extract unique agent IDs from pair keys
        agent_ids = set()
        for pair_key in agent_pairs.keys():
            agents = pair_key.split('-')
            agent_ids.update(agents)

        agent_ids = sorted(list(agent_ids))

        # Create agreement matrix
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([''] + agent_ids)

            # Matrix rows
            for agent1 in agent_ids:
                row = [agent1]
                for agent2 in agent_ids:
                    if agent1 == agent2:
                        row.append('1.00')  # Perfect self-agreement
                    else:
                        # Find pair (order-independent)
                        pair_key1 = f"{agent1}-{agent2}"
                        pair_key2 = f"{agent2}-{agent1}"

                        pair_data = agent_pairs.get(pair_key1) or agent_pairs.get(pair_key2)

                        if pair_data:
                            agreement_rate = pair_data.get('agreement_rate', 0.0)
                            row.append(f"{agreement_rate:.2f}")
                        else:
                            row.append('N/A')

                writer.writerow(row)

        # Also save summary statistics
        summary_path = self.metrics_dir / "disagreement_summary.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Pair', 'Agreement Rate', 'Agreement Count', 'Disagreement Count', 'Total'])

            for pair_key, data in sorted(agent_pairs.items(), key=lambda x: x[1]['agreement_rate'], reverse=True):
                writer.writerow([
                    pair_key,
                    f"{data['agreement_rate']:.2%}",
                    data['agreement_count'],
                    data['disagreement_count'],
                    data['total_questions']
                ])

    def save_agent_performance(self, agent_performance: Dict):
        """Save individual agent performance to JSON."""
        json_path = self.metrics_dir / "agent_performance.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(agent_performance, f, indent=2, ensure_ascii=False)

    def save_summary_report(self, summary: Dict):
        """Save comprehensive summary report."""
        summary_path = self.base_dir / "summary_report.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def get_results_path(self) -> str:
        """Return the path to this run's results directory."""
        return str(self.base_dir)


__all__ = ["ResultsStorage"]
