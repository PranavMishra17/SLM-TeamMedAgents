"""
Main Entry Point - Multi-Agent Simulation Runner

Supports both single question and batch dataset processing with comprehensive
logging, metrics calculation, and results storage.

Usage:
    # Single question
    python run_simulation.py --question "..." --options "A,B,C,D"

    # Batch dataset
    python run_simulation.py --dataset medqa --n_questions 50

    # Fixed agents
    python run_simulation.py --dataset medqa --n_questions 50 --n_agents 3
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime

# Add parent directory to path
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available - install with 'pip install tqdm' for progress bars")

# Import multi-agent components
from components import MultiAgentSystem, MultiAgentRateLimiter
from utils import SimulationLogger, ResultsStorage
from utils.metrics_calculator import MetricsCalculator
import config as multi_agent_config

# Import existing dataset loaders and formatters
try:
    from medical_datasets.dataset_loader import DatasetLoader
    from medical_datasets.dataset_formatters import MedQAFormatter, MedMCQAFormatter, PubMedQAFormatter
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.warning("Dataset loaders not available - single question mode only")


class BatchSimulationRunner:
    """
    Runs multi-agent simulations on batches of questions from medical datasets.

    Features:
    - Progress tracking with tqdm
    - Rate limiting coordination
    - Time estimation
    - Comprehensive metrics calculation
    - Results storage and export
    """

    def __init__(self,
                 model_name: str = "gemma3_4b",
                 n_agents: int = None,
                 output_dir: str = "multi-agent-gemma/results",
                 dataset_name: str = None,
                 n_questions: int = 10):
        """
        Initialize batch simulation runner.

        Args:
            model_name: Gemma model to use
            n_agents: Fixed agent count or None for dynamic
            output_dir: Base directory for results
            dataset_name: Dataset to load (medqa, medmcqa, etc.)
            n_questions: Number of questions to process
        """
        self.model_name = model_name
        self.n_agents_config = n_agents
        self.dataset_name = dataset_name
        self.n_questions = n_questions

        # Initialize multi-agent system
        self.system = MultiAgentSystem(
            model_name=model_name,
            n_agents=n_agents,
            enable_dynamic_recruitment=(n_agents is None)
        )

        # Create run ID and setup storage
        self.storage = ResultsStorage(
            output_dir=output_dir,
            dataset_name=dataset_name,
            n_questions=n_questions
        )
        self.logger = SimulationLogger(output_dir=output_dir, run_id=self.storage.run_id)
        self.metrics = MetricsCalculator()

        # Rate limiter (will be initialized after knowing actual n_agents)
        self.rate_limiter = None

        # Configuration
        self.config = {
            "model_name": model_name,
            "n_agents": n_agents,
            "dataset": dataset_name,
            "n_questions": n_questions,
            "run_id": self.storage.run_id,
            "timestamp": datetime.now().isoformat(),
            "enable_dynamic_recruitment": (n_agents is None)
        }

        # Save config
        self.storage.save_config(self.config)

        # Load dataset
        self.questions = self._load_dataset(dataset_name, n_questions)
        self.ground_truth = self._extract_ground_truth(self.questions)

        logging.info(f"Initialized BatchSimulationRunner: {dataset_name}, {n_questions} questions")

    def _load_dataset(self, dataset_name: str, n_questions: int) -> List[Dict]:
        """Load questions from specified dataset."""
        if not DATASETS_AVAILABLE:
            raise ImportError("Dataset loaders not available. Install required packages.")

        logging.info(f"Loading {n_questions} questions from {dataset_name}...")

        try:
            if dataset_name == "medqa":
                questions = DatasetLoader.load_medqa(n_questions)
            elif dataset_name == "medmcqa":
                questions, errors = DatasetLoader.load_medmcqa(n_questions)
                if errors:
                    logging.warning(f"Dataset had {len(errors)} loading errors")
            elif dataset_name == "pubmedqa":
                questions = DatasetLoader.load_pubmedqa(n_questions)
            elif dataset_name == "ddxplus":
                questions = DatasetLoader.load_ddxplus(n_questions)
            elif dataset_name == "medbullets":
                questions = DatasetLoader.load_medbullets(n_questions)
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            logging.info(f"Loaded {len(questions)} questions successfully")
            return questions

        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    def _extract_ground_truth(self, questions: List[Dict]) -> Dict[str, str]:
        """Extract ground truth answers from questions."""
        ground_truth = {}

        for i, q in enumerate(questions):
            question_id = f"q{i+1:03d}"

            # Extract answer based on dataset format
            # Try different field names used by various datasets
            answer = None

            if 'expected_output' in q:
                # MedQA format
                answer = q['expected_output']
            elif 'answer' in q:
                answer = q['answer']
            elif 'answer_idx' in q and 'options' in q:
                # Convert index to letter
                idx = q['answer_idx']
                if isinstance(idx, int) and 0 <= idx < len(q['options']):
                    answer = chr(65 + idx)  # 0→A, 1→B, etc.
            elif 'cop' in q:
                # MedMCQA format - correct option (1-4)
                cop = q['cop']
                if isinstance(cop, int) and 1 <= cop <= 4:
                    answer = chr(64 + cop)  # 1→A, 2→B, etc.
                else:
                    answer = str(cop)

            if answer:
                ground_truth[question_id] = str(answer).strip().upper()
                logging.debug(f"Extracted ground truth for {question_id}: {answer}")

        logging.info(f"Extracted ground truth for {len(ground_truth)}/{len(questions)} questions")
        return ground_truth

    def _format_question_for_simulation(self, question_data: Dict, question_id: str) -> Dict:
        """Format dataset question for simulation using existing formatters."""
        # Use appropriate formatter based on dataset
        agent_task = None
        eval_data = None

        try:
            if self.dataset_name == "medqa":
                agent_task, eval_data = MedQAFormatter.format(question_data)
            elif self.dataset_name == "medmcqa":
                agent_task, eval_data, is_valid = MedMCQAFormatter.format(question_data)
                if not is_valid:
                    logging.warning(f"Invalid question {question_id}: {eval_data}")
                    return None
            elif self.dataset_name == "pubmedqa":
                agent_task, eval_data = PubMedQAFormatter.format(question_data)
            else:
                # Fallback for other datasets
                agent_task = {
                    'description': question_data.get('question', ''),
                    'options': question_data.get('options', []),
                    'type': 'mcq'
                }
                eval_data = {'ground_truth': question_data.get('answer', '')}

            # Extract fields for simulation
            question_text = agent_task.get('description', '')
            options = agent_task.get('options', [])
            task_type = agent_task.get('type', 'mcq')

            # Get ground truth from eval_data (kept separate from agent context)
            ground_truth = eval_data.get('ground_truth', '') if eval_data else None

            # Store ground truth
            if ground_truth:
                self.ground_truth[question_id] = str(ground_truth).strip().upper()

            return {
                'question': question_text,
                'options': options,
                'task_type': task_type,
                'ground_truth': ground_truth,
                'question_id': question_id
            }

        except Exception as e:
            logging.error(f"Error formatting question {question_id}: {e}")
            return None

    def run(self) -> Dict[str, Any]:
        """Execute batch simulation with full tracking."""
        self.logger.log_run_start(self.config)

        # Initialize rate limiter with estimated n_agents
        actual_n_agents = self.n_agents_config or 3  # Estimate for dynamic
        self.rate_limiter = MultiAgentRateLimiter(
            model_name=self.model_name,
            n_agents=actual_n_agents
        )

        # Print time estimate
        self.rate_limiter.print_time_estimate(len(self.questions))

        # Process questions with progress bar
        start_time = time.time()

        if TQDM_AVAILABLE:
            question_iterator = tqdm(enumerate(self.questions), total=len(self.questions),
                                   desc="Processing questions", unit="question")
        else:
            question_iterator = enumerate(self.questions)

        for i, question_data in question_iterator:
            question_id = f"q{i+1:03d}"

            try:
                # Format question
                sim_input = self._format_question_for_simulation(question_data, question_id)

                # Skip invalid questions
                if sim_input is None:
                    logging.warning(f"Skipping invalid question {question_id}")
                    continue

                # Log question start
                self.logger.log_question_start(question_id, sim_input['question'])

                # Run simulation
                q_start = time.time()

                result = self.system.run_simulation(
                    question=sim_input['question'],
                    options=sim_input['options'],
                    task_type=sim_input['task_type'],
                    ground_truth=sim_input['ground_truth']
                )

                q_time = time.time() - q_start

                # Add question ID to result
                result['metadata']['question_id'] = question_id

                # Update rate limiter if dynamic recruitment
                if self.n_agents_config is None and self.rate_limiter:
                    actual_n = len(result['recruited_agents'])
                    self.rate_limiter.n_agents = actual_n

                # Log agent recruitment
                agent_roles = [a['role'] for a in result['recruited_agents']]
                self.logger.log_agent_recruitment(len(agent_roles), agent_roles)

                # Extract results
                final_answer = result['final_decision']['primary_answer']
                is_correct = result.get('is_correct', False)

                # Log completion
                self.logger.log_question_complete(question_id, final_answer, is_correct, q_time)

                # Save question result
                self.storage.save_question_result(question_id, result)

                # Add to metrics
                self.metrics.add_question_result(result)

                # Rate limiting - just log status, don't block
                if multi_agent_config.ENABLE_RATE_LIMITING and self.rate_limiter:
                    status = self.rate_limiter.get_rate_status()
                    if status['rpm_utilization'] > 0.8 or status['tpm_utilization'] > 0.8:
                        logging.warning(f"Rate limit approaching: RPM={status['rpm_utilization']:.1%}, TPM={status['tpm_utilization']:.1%}")

            except Exception as e:
                self.logger.log_error(
                    error_type="QuestionProcessingError",
                    error_message=str(e),
                    context={"question_id": question_id}
                )
                logging.error(f"Error processing {question_id}: {e}")
                continue

        # Calculate final metrics
        total_time = time.time() - start_time
        summary = self._generate_final_summary(total_time)

        # Save all metrics
        self._save_all_metrics()

        # Log completion
        self.logger.log_run_complete(summary)

        return summary

    def _generate_final_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate final summary with all metrics."""
        summary_report = self.metrics.generate_summary_report(self.ground_truth)

        # Add timing information
        summary_report['timing'] = {
            'total_time': total_time,
            'avg_time_per_question': total_time / len(self.questions) if self.questions else 0,
            'questions_processed': len(self.questions)
        }

        # Add results path
        summary_report['results_path'] = self.storage.get_results_path()

        return summary_report

    def _save_all_metrics(self):
        """Save all calculated metrics to files."""
        logging.info("Saving metrics...")

        # Generate summary report
        summary_report = self.metrics.generate_summary_report(self.ground_truth)

        # Accuracy
        accuracy = summary_report['accuracy']
        self.storage.save_accuracy_summary(accuracy)

        # Convergence
        convergence = summary_report['convergence']
        self.storage.save_convergence_analysis(convergence)

        # Disagreement
        disagreement = summary_report['disagreement']
        self.storage.save_disagreement_matrix(disagreement)

        # Agent performance
        agent_perf = summary_report['agent_performance']
        self.storage.save_agent_performance(agent_perf)

        # Summary report
        self.storage.save_summary_report(summary_report)

        logging.info(f"Metrics saved to {self.storage.get_results_path()}")


def run_single_question(question: str,
                       options: List[str],
                       model_name: str = "gemma3_4b",
                       n_agents: int = None,
                       ground_truth: str = None) -> Dict[str, Any]:
    """
    Run simulation on a single question.

    Args:
        question: Question text
        options: List of answer options
        model_name: Model to use
        n_agents: Fixed agent count or None for dynamic
        ground_truth: Correct answer (optional)

    Returns:
        Simulation result dictionary
    """
    print(f"\n{'='*70}")
    print("SINGLE QUESTION SIMULATION")
    print(f"{'='*70}\n")

    # Initialize system
    system = MultiAgentSystem(
        model_name=model_name,
        n_agents=n_agents
    )

    # Run simulation
    result = system.run_simulation(
        question=question,
        options=options,
        ground_truth=ground_truth
    )

    # Display results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")

    print("Recruited Agents:")
    for agent in result['recruited_agents']:
        print(f"  • {agent['role']}: {agent['expertise']}")

    print(f"\nFinal Decision:")
    print(f"  Primary Answer: {result['final_decision']['primary_answer']}")
    print(f"  Borda Count: {result['final_decision']['borda_count']}")
    print(f"  Majority Vote: {result['final_decision']['majority_vote']}")

    if ground_truth:
        print(f"\nEvaluation:")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Correct: {'✅ YES' if result['is_correct'] else '❌ NO'}")

    if result.get('agreement_metrics'):
        print(f"\nAgreement:")
        print(f"  Full Agreement: {result['agreement_metrics']['full_agreement']}")
        print(f"  Partial Agreement: {result['agreement_metrics']['partial_agreement_rate']:.1%}")

    print(f"\nTiming:")
    print(f"  Total Time: {result['metadata']['total_time']:.2f}s")
    print(f"  Agents: {result['metadata']['n_agents']}")

    print(f"\n{'='*70}\n")

    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Gemma Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single question
  python run_simulation.py --question "What is the diagnosis?" --options "A,B,C,D"

  # Batch dataset
  python run_simulation.py --dataset medqa --n_questions 50

  # Fixed 3 agents
  python run_simulation.py --dataset medqa --n_questions 50 --n_agents 3

  # Dynamic recruitment
  python run_simulation.py --dataset medqa --n_questions 50
        """
    )

    # Question mode
    parser.add_argument("--question", type=str,
                       help="Single question text")
    parser.add_argument("--options", type=str,
                       help="Comma-separated options (e.g., 'A. Option,B. Option,C. Option')")
    parser.add_argument("--ground-truth", type=str,
                       help="Correct answer for evaluation")

    # Dataset batch mode
    parser.add_argument("--dataset", type=str,
                       choices=['medqa', 'medmcqa', 'pubmedqa', 'ddxplus', 'medbullets'],
                       help="Dataset to process")
    parser.add_argument("--n-questions", type=int, default=10,
                       help="Number of questions to process (default: 10)")

    # Model & agent configuration
    parser.add_argument("--model", type=str, default="gemma3_4b",
                       choices=['gemma3_4b', 'medgemma_4b'],
                       help="Gemma model to use (default: gemma3_4b)")
    parser.add_argument("--n-agents", type=int,
                       help="Fixed agent count (2-4) or omit for dynamic")

    # Output configuration
    parser.add_argument("--output-dir", type=str,
                       default="multi-agent-gemma/results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Validate arguments
    if args.question and args.dataset:
        parser.error("Cannot specify both --question and --dataset")

    if not args.question and not args.dataset:
        parser.error("Must specify either --question or --dataset")

    try:
        # Single question mode
        if args.question:
            if not args.options:
                parser.error("--options required when using --question")

            options = [opt.strip() for opt in args.options.split(',')]

            result = run_single_question(
                question=args.question,
                options=options,
                model_name=args.model,
                n_agents=args.n_agents,
                ground_truth=args.ground_truth
            )

            # Save result to JSON
            output_file = Path(args.output_dir) / f"single_question_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)

            print(f"Result saved to: {output_file}")

        # Batch dataset mode
        elif args.dataset:
            print(f"\n{'='*70}")
            print(f"BATCH SIMULATION: {args.dataset.upper()}")
            print(f"{'='*70}\n")
            print(f"Model: {args.model}")
            print(f"Questions: {args.n_questions}")
            print(f"Agents: {'Dynamic (2-4)' if args.n_agents is None else args.n_agents}")
            print(f"Output: {args.output_dir}")
            print(f"{'='*70}\n")

            # Create runner
            runner = BatchSimulationRunner(
                model_name=args.model,
                n_agents=args.n_agents,
                output_dir=args.output_dir,
                dataset_name=args.dataset,
                n_questions=args.n_questions
            )

            # Run batch simulation
            summary = runner.run()

            # Print summary
            print(f"\n{'='*70}")
            print("BATCH SIMULATION COMPLETE")
            print(f"{'='*70}")
            print(f"Questions Processed: {summary['metadata']['total_questions']}")
            print(f"Overall Accuracy: {summary['accuracy']['overall_accuracy']:.2%} "
                  f"({summary['accuracy']['correct_count']}/{summary['accuracy']['total_count']})")
            print(f"Convergence Rate: {summary['convergence']['overall_convergence_rate']:.2%}")
            print(f"Total Time: {summary['timing']['total_time']:.1f}s "
                  f"({summary['timing']['avg_time_per_question']:.1f}s/question)")
            print(f"\nResults saved to: {summary['results_path']}")
            print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
