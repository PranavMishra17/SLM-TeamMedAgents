"""
ADK-Based Multi-Agent Simulation Runner

Main entry point for running multi-agent medical reasoning simulations using
Google's Agent Development Kit (ADK) instead of custom infrastructure.

Maintains full compatibility with existing:
- Dataset loaders (medical_datasets/)
- Logging system (utils/SimulationLogger)
- Results storage (utils/ResultsStorage)
- Metrics calculation (utils/MetricsCalculator)

Usage:
    # Fixed agent count
    python run_simulation_adk.py --dataset medqa --n-questions 10 --n-agents 3 --model gemma3_4b

    # Dynamic agent count (2-4)
    python run_simulation_adk.py --dataset medqa --n-questions 10 --model gemma3_4b

    # Full run with all options
    python run_simulation_adk.py --dataset medmcqa --n-questions 50 --n-agents 3 --model gemma2_9b
"""

# Gemma image token constant
# According to Gemma documentation, images are encoded as 258 tokens
GEMMA_IMAGE_TOKENS = 258

import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List
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
    logging.warning("tqdm not available - progress bars disabled")

# Import ADK components
try:
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService
    from adk_agents import MultiAgentSystemADK
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("ERROR: Google ADK not installed. Run: pip install google-adk")
    sys.exit(1)

# Import existing infrastructure
from utils import SimulationLogger, ResultsStorage
from utils.metrics_calculator import MetricsCalculator
from utils.results_logger import TokenCounter

# Import dataset loaders
try:
    from medical_datasets.dataset_loader import DatasetLoader, VisionDatasetLoader
    from medical_datasets.dataset_formatters import (
        MedQAFormatter, MedMCQAFormatter, PubMedQAFormatter,
        MMLUProMedFormatter, DDXPlusFormatter, MedBulletsFormatter
    )
    from medical_datasets.vision_dataset_formatters import PMCVQAFormatter, PathVQAFormatter
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logging.error("Dataset loaders not available")


def extract_text_from_content(content) -> str:
    """Extract text from ADK Content object."""
    if isinstance(content, str):
        return content
    if hasattr(content, 'parts') and content.parts:
        text_parts = [part.text for part in content.parts if hasattr(part, 'text') and part.text]
        return ''.join(text_parts)
    return str(content)


class BatchSimulationRunnerADK:
    """
    Runs ADK-based multi-agent simulations on batches of medical questions.

    Features:
    - Full integration with existing logging/results/metrics infrastructure
    - Progress tracking with tqdm
    - Comprehensive metrics calculation
    - Same output format as original system for easy comparison
    """

    def __init__(
        self,
        model_name: str = "gemma3_4b",
        n_agents: int = None,
        output_dir: str = "multi-agent-gemma/results",
        dataset_name: str = None,
        n_questions: int = 10
    ):
        """
        Initialize ADK-based batch simulation runner.

        Args:
            model_name: Gemma model to use
            n_agents: Fixed agent count or None for dynamic (2-4)
            output_dir: Base directory for results
            dataset_name: Dataset to load (medqa, medmcqa, pubmedqa)
            n_questions: Number of questions to process
        """
        self.model_name = model_name
        self.n_agents_config = n_agents
        self.dataset_name = dataset_name
        self.n_questions = n_questions

        # Initialize ADK multi-agent system
        self.system = MultiAgentSystemADK(
            model_name=model_name,
            n_agents=n_agents
        )

        # Initialize ADK Runner
        self.runner = Runner(
            app_name="medical_reasoning_adk",
            agent=self.system,
            session_service=InMemorySessionService()
        )

        # Setup storage and logging (same as original system)
        self.storage = ResultsStorage(
            output_dir=output_dir,
            dataset_name=dataset_name,
            n_questions=n_questions
        )
        self.logger = SimulationLogger(
            output_dir=output_dir,
            run_id=self.storage.run_id
        )
        self.metrics = MetricsCalculator()
        self.token_counter = TokenCounter()

        # Configuration
        self.config = {
            "framework": "Google ADK",
            "model_name": model_name,
            "n_agents": n_agents if n_agents else "dynamic (2-4)",
            "dataset": dataset_name,
            "n_questions": n_questions,
            "timestamp": datetime.now().isoformat()
        }

        # Ground truth storage
        self.ground_truth = {}

        logging.info(f"Initialized ADK BatchSimulationRunner: {dataset_name}, {n_questions} questions")

    def load_dataset(self) -> List[Dict]:
        """Load dataset using existing loaders."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("Dataset loaders not available")

        logging.info(f"Loading {self.dataset_name} dataset ({self.n_questions} questions)...")

        if self.dataset_name == "medqa":
            questions = DatasetLoader.load_medqa(self.n_questions, random_seed=42)
        elif self.dataset_name == "medmcqa":
            questions, errors = DatasetLoader.load_medmcqa(self.n_questions, random_seed=42)
            if errors:
                logging.warning(f"Skipped {len(errors)} invalid MedMCQA questions")
        elif self.dataset_name == "pubmedqa":
            questions = DatasetLoader.load_pubmedqa(self.n_questions, random_seed=42)
        elif self.dataset_name == "mmlupro":
            questions = DatasetLoader.load_mmlupro_med(self.n_questions, random_seed=42)
        elif self.dataset_name == "ddxplus":
            questions = DatasetLoader.load_ddxplus(self.n_questions, random_seed=42)
        elif self.dataset_name == "medbullets":
            questions = DatasetLoader.load_medbullets(self.n_questions, random_seed=42)
        elif self.dataset_name == "pmc_vqa":
            questions = VisionDatasetLoader.load_pmc_vqa(self.n_questions, random_seed=42)
        elif self.dataset_name == "path_vqa":
            questions = VisionDatasetLoader.load_path_vqa(self.n_questions, random_seed=42)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        logging.info(f"Loaded {len(questions)} questions")
        return questions

    def format_question(self, question_data: Dict, question_id: str) -> Dict:
        """Format dataset question using existing formatters."""
        try:
            # Format based on dataset type
            if self.dataset_name == "medqa":
                agent_task, eval_data = MedQAFormatter.format(question_data)
            elif self.dataset_name == "medmcqa":
                agent_task, eval_data, is_valid = MedMCQAFormatter.format(question_data)
                if not is_valid:
                    logging.warning(f"Invalid question {question_id}")
                    return None
            elif self.dataset_name == "pubmedqa":
                agent_task, eval_data = PubMedQAFormatter.format(question_data)
            elif self.dataset_name == "mmlupro":
                agent_task, eval_data = MMLUProMedFormatter.format(question_data)
            elif self.dataset_name == "ddxplus":
                agent_task, eval_data = DDXPlusFormatter.format(question_data)
            elif self.dataset_name == "medbullets":
                agent_task, eval_data = MedBulletsFormatter.format(question_data)
            elif self.dataset_name == "pmc_vqa":
                agent_task, eval_data = PMCVQAFormatter.format(question_data)
            elif self.dataset_name == "path_vqa":
                agent_task, eval_data = PathVQAFormatter.format(question_data)
            else:
                logging.error(f"Unsupported dataset: {self.dataset_name}")
                return None

            # Extract fields
            question_text = agent_task.get('description', '')
            options = agent_task.get('options', [])
            task_type = agent_task.get('type', 'mcq')

            # Extract image data for vision datasets
            image_data = agent_task.get('image_data', {})
            image = image_data.get('image', None) if image_data else None

            # Get ground truth (kept separate from agent context)
            ground_truth = eval_data.get('ground_truth', '') if eval_data else None
            if ground_truth:
                self.ground_truth[question_id] = str(ground_truth).strip().upper()

            return {
                'question': question_text,
                'options': options,
                'task_type': task_type,
                'ground_truth': ground_truth,
                'question_id': question_id,
                'image': image  # Add PIL Image object for vision datasets
            }

        except Exception as e:
            logging.error(f"Error formatting question {question_id}: {e}")
            return None

    async def process_question(self, question_data: Dict, question_idx: int, max_retries: int = 5) -> Dict:
        """
        Process a single question through ADK multi-agent system with retry logic.

        Args:
            question_data: Question data from dataset
            question_idx: Question index
            max_retries: Maximum retry attempts (default: 5 for rate limits)

        Returns:
            Dict with complete results (same format as original system)
        """
        import time as time_module
        import random

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logging.warning(f"Retry attempt {attempt}/{max_retries} for q{question_idx+1:03d}")

            try:
                return await self._process_question_attempt(question_data, question_idx)
            except Exception as e:
                error_msg = str(e).lower()

                # Check error type
                is_rate_limit = any(term in error_msg for term in ['429', 'resource_exhausted', 'quota', 'rate limit'])
                is_timeout = 'timeout' in error_msg or 'winerror 121' in error_msg or 'semaphore' in error_msg

                # Extract retry delay from error message if available
                retry_delay = None
                if 'retry in' in error_msg:
                    import re
                    match = re.search(r'retry in ([\d.]+)s', error_msg)
                    if match:
                        retry_delay = float(match.group(1))

                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    # Calculate exponential backoff delay
                    if retry_delay:
                        # Use API-suggested delay
                        delay = retry_delay
                        logging.warning(f"Rate limit hit (attempt {attempt+1}), API suggested waiting {delay:.1f}s")
                    elif is_rate_limit:
                        # Exponential backoff for rate limits: 60s base + exponential
                        delay = min(60 * (2 ** attempt), 300)  # Cap at 5 minutes
                        # Add jitter to prevent thundering herd
                        jitter = random.uniform(0.8, 1.2)
                        delay *= jitter
                        logging.warning(f"Rate limit error on attempt {attempt+1}, waiting {delay:.1f}s: {type(e).__name__}")
                    else:
                        # Shorter backoff for timeouts
                        delay = min(3 * (2 ** attempt), 30)
                        logging.warning(f"Timeout on attempt {attempt+1}, waiting {delay:.1f}s")

                    await asyncio.sleep(delay)
                    continue
                elif is_rate_limit or is_timeout:
                    logging.error(f"Failed after {max_retries+1} attempts due to {'rate limit' if is_rate_limit else 'timeout'}")
                    return None
                else:
                    # Non-retryable error
                    logging.error(f"Non-retryable error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

        return None

    async def _process_question_attempt(self, question_data: Dict, question_idx: int) -> Dict:
        """
        Single attempt to process a question.

        Returns:
            Dict with complete results
        """
        question_id = f"q{question_idx+1:03d}"

        # Format question
        formatted = self.format_question(question_data, question_id)
        if not formatted:
            return None

        question = formatted['question']
        options = formatted['options']
        task_type = formatted['task_type']
        ground_truth = formatted['ground_truth']
        image = formatted.get('image', None)  # PIL Image object for vision datasets

        self.logger.log_question_start(question_id, question)

        # Prepare initial state for ADK session
        initial_state = {
            'question': question,
            'options': options,
            'task_type': task_type,
            'ground_truth': ground_truth,
            'image': image,  # PIL Image object for vision datasets
            'dataset': self.dataset_name  # Pass dataset name for prompt logic
        }

        # Create session first
        await self.runner.session_service.create_session(
            app_name="medical_reasoning_adk",
            user_id="simulation",
            session_id=question_id,
            state=initial_state
        )

        # Run ADK system using Runner
        try:
            # Create initial message to trigger agent execution
            from google.genai import types
            initial_message = types.Content(
                parts=[types.Part(text=f"Analyze this medical question: {question}")]
            )

            # Store results from events (workaround for session state not persisting)
            captured_state = None

            async for event in self.runner.run_async(
                user_id="simulation",
                session_id=question_id,
                new_message=initial_message
            ):
                # Log events (optional, can be verbose)
                if hasattr(event, 'content') and event.content:
                    content_text = extract_text_from_content(event.content)
                    logging.debug(f"[{event.author}] {content_text[:100]}")

                    # Capture state from MultiAgentSystemADK's final event
                    if event.author == "multi_agent_system" and "RESULTS:" in content_text:
                        import json
                        try:
                            # Extract JSON from event content
                            json_start = content_text.find("{")
                            if json_start >= 0:
                                captured_state = json.loads(content_text[json_start:])
                                logging.debug(f"Captured state from event: {list(captured_state.keys())}")
                        except json.JSONDecodeError:
                            logging.warning("Failed to parse state from event")

        except Exception as e:
            logging.error(f"Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Use captured state from events (workaround for session state not persisting)
        if captured_state:
            logging.info("Using captured state from event")
            final_answer = captured_state.get('final_answer', 'A')
            is_correct = captured_state.get('is_correct', False)
            timing = captured_state.get('timing', {})
            recruited_agents = captured_state.get('recruited_agents', [])
            round1_results = captured_state.get('round1_results', {})
            round2_results = captured_state.get('round2_results', {})
            round3_results = captured_state.get('round3_results', {})
            aggregation_result = captured_state.get('aggregation_result', {})
            convergence = captured_state.get('convergence', {})

            logging.debug(f"Extracted from event: agents={len(recruited_agents)}, r1={len(round1_results)}, r2={len(round2_results)}, r3={len(round3_results)}")
        else:
            # Fallback: try session state (won't work with current ADK but keep for reference)
            logging.warning("No captured state from events, trying session service (likely empty)")
            try:
                session = await self.runner.session_service.get_session(
                    app_name="medical_reasoning_adk",
                    user_id="simulation",
                    session_id=question_id
                )
                final_answer = session.state.get('final_answer', 'A')
                is_correct = session.state.get('is_correct', False)
                timing = session.state.get('timing', {})
                recruited_agents = session.state.get('recruited_agents', [])
                round1_results = session.state.get('round1_results', {})
                round2_results = session.state.get('round2_results', {})
                round3_results = session.state.get('round3_results', {})
                aggregation_result = session.state.get('aggregation_result', {})
                convergence = session.state.get('convergence', {})
            except Exception as e:
                logging.error(f"Error retrieving session {question_id}: {e}")
                return None

        # Calculate token usage for this question
        # Collect all input text (prompts) and output text (responses)
        total_input_text = question  # Initial question
        total_output_text = ""

        # Add all round results to token calculation
        for agent_id, response in round1_results.items():
            total_output_text += response
        for agent_id, response in round2_results.items():
            total_output_text += response
        for agent_id, result in round3_results.items():
            total_output_text += result.get('raw', '')

        # Track tokens for this question
        token_data = self.token_counter.log_question_tokens(
            input_text=total_input_text,
            output_text=total_output_text,
            question_index=question_idx,
            usage_metadata=None,  # ADK doesn't expose this
            model=f"gemma-3-{self.model_name.split('_')[-1]}-it" if 'gemma' in self.model_name else self.model_name
        )

        # Add image tokens if this is a vision dataset with image
        # Each image contributes GEMMA_IMAGE_TOKENS to the input token count
        # This happens once per agent per round (3 rounds total)
        if image is not None:
            n_agents = len(recruited_agents)
            n_rounds = 3
            image_tokens_total = GEMMA_IMAGE_TOKENS * n_agents * n_rounds

            # Add to token data
            token_data['input_tokens'] += image_tokens_total
            token_data['total_tokens'] += image_tokens_total
            token_data['image_tokens'] = image_tokens_total
            token_data['has_image'] = True

            # Update global counter
            self.token_counter.input_tokens += image_tokens_total
            self.token_counter.total_tokens += image_tokens_total

            logging.debug(f"Added {image_tokens_total} image tokens ({GEMMA_IMAGE_TOKENS} per agent per round × {n_agents} agents × {n_rounds} rounds)")
        else:
            token_data['image_tokens'] = 0
            token_data['has_image'] = False

        # Calculate API calls (1 per agent per round + 1 for recruitment if dynamic)
        n_agents = len(recruited_agents)
        api_calls = n_agents * 3  # 3 rounds
        if self.n_agents_config is None:  # Dynamic recruitment adds 1-2 calls
            api_calls += 2  # Count determination + role generation

        # Log completion
        self.logger.log_question_complete(
            question_id=question_id,
            final_answer=final_answer,
            is_correct=is_correct,
            time_taken=timing.get('total_time', 0)
        )

        # Package results (same format as original system)
        result = {
            'question_id': question_id,
            'question': question,
            'options': options,
            'task_type': task_type,
            'image_path': None if image is None else "PIL_Image_Object",  # Mark if image was used

            # Recruited agents
            'recruited_agents': [
                {
                    'agent_id': a['agent_id'],
                    'role': a['role'],
                    'expertise': a['expertise']
                }
                for a in recruited_agents
            ],

            # Round results
            'round1_results': round1_results,
            'round2_results': round2_results,
            'round3_results': {
                agent_id: {
                    'answer': r.get('answer'),
                    'ranking': r.get('ranking'),
                    'confidence': r.get('confidence'),
                    'raw': r.get('raw', '')
                }
                for agent_id, r in round3_results.items()
            },

            # Final decision
            'final_decision': {
                'primary_answer': final_answer,
                'borda_count': aggregation_result,
                'convergence': convergence
            },

            # Ground truth comparison
            'ground_truth': ground_truth,
            'is_correct': is_correct,

            # Metadata
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_time': timing.get('total_time', 0),
                'recruit_time': timing.get('recruit_time', 0),
                'round1_time': timing.get('round1_time', 0),
                'round2_time': timing.get('round2_time', 0),
                'round3_time': timing.get('round3_time', 0),
                'aggregation_time': timing.get('aggregation_time', 0),
                'api_calls': api_calls,
                'n_agents': n_agents,
                'framework': 'Google ADK'
            },

            # Token usage
            'token_usage': token_data
        }

        return result

    async def run(self) -> Dict[str, Any]:
        """Execute batch simulation with full tracking."""
        self.logger.log_run_start(self.config)
        logging.info(f"\n{'='*80}\nBATCH SIMULATION: {self.dataset_name.upper()} (ADK)\n{'='*80}")

        # Load dataset
        questions = self.load_dataset()

        # Save config
        self.storage.save_config(self.config)

        # Process questions
        results = []
        correct_count = 0

        if TQDM_AVAILABLE:
            progress = tqdm(questions, desc="Processing questions", unit="question")
        else:
            progress = questions

        for idx, question_data in enumerate(progress):
            result = await self.process_question(question_data, idx)

            if result:
                results.append(result)

                # Save individual result
                self.storage.save_question_result(
                    question_id=result['question_id'],
                    result=result
                )

                # Update metrics
                if result['is_correct']:
                    correct_count += 1

        # Calculate final metrics
        accuracy = correct_count / len(results) if results else 0

        # Generate comprehensive metrics
        logging.info("Generating comprehensive metrics...")
        metrics_summary = self._generate_metrics_summary(results)

        # Save metrics
        self.storage.save_accuracy_summary(metrics_summary.get('accuracy', {}))
        self.storage.save_convergence_analysis(metrics_summary.get('convergence', {}))
        self.storage.save_agent_performance(metrics_summary.get('agent_performance', {}))
        self.storage.save_summary_report(metrics_summary)

        # Log completion
        self.logger.log_run_complete(metrics_summary)

        logging.info(f"\n{'='*80}\nSIMULATION COMPLETE\n{'='*80}")
        logging.info(f"Questions Processed: {len(results)}")
        logging.info(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")

        # Log token usage
        token_summary = metrics_summary.get('token_usage', {})
        if token_summary:
            logging.info(f"\nToken Usage:")
            logging.info(f"  Total Tokens: {token_summary.get('total_tokens', 0):,}")
            logging.info(f"  Input Tokens: {token_summary.get('total_input_tokens', 0):,}")
            logging.info(f"  Output Tokens: {token_summary.get('total_output_tokens', 0):,}")
            logging.info(f"  Avg Tokens/Question: {token_summary.get('avg_total_tokens_per_question', 0):.1f}")

        # Log API calls
        api_calls = metrics_summary.get('api_calls', {})
        if api_calls:
            logging.info(f"\nAPI Calls:")
            logging.info(f"  Total Calls: {api_calls.get('total_calls', 0)}")
            logging.info(f"  Avg Calls/Question: {api_calls.get('avg_calls_per_question', 0):.1f}")

        # Log timing
        timing = metrics_summary.get('timing', {})
        if timing:
            logging.info(f"\nTiming:")
            logging.info(f"  Total Time: {timing.get('total_time', 0):.2f}s")
            logging.info(f"  Avg Time/Question: {timing.get('avg_time_per_question', 0):.2f}s")
            logging.info(f"  Avg Recruit: {timing.get('avg_recruit_time', 0):.2f}s")
            logging.info(f"  Avg Round1: {timing.get('avg_round1_time', 0):.2f}s")
            logging.info(f"  Avg Round2: {timing.get('avg_round2_time', 0):.2f}s")
            logging.info(f"  Avg Round3: {timing.get('avg_round3_time', 0):.2f}s")

        logging.info(f"\nResults saved to: {self.storage.get_results_path()}")

        return metrics_summary

    def _generate_metrics_summary(self, results: List[Dict]) -> Dict:
        """Generate comprehensive metrics (reusing existing calculator)."""
        if not results:
            return {}

        # Convert results to format expected by MetricsCalculator
        for result in results:
            # MetricsCalculator expects the full result dict
            self.metrics.add_question_result(result)

        # Calculate comprehensive metrics
        metrics_report = self.metrics.generate_summary_report()

        # Add token usage summary
        token_summary = self.token_counter.get_summary()
        metrics_report['token_usage'] = token_summary

        # Add per-question token breakdown
        metrics_report['token_usage']['per_question'] = self.token_counter.question_tokens

        # Calculate API call statistics
        total_api_calls = sum(r.get('metadata', {}).get('api_calls', 0) for r in results)
        avg_api_calls = total_api_calls / len(results) if results else 0
        metrics_report['api_calls'] = {
            'total_calls': total_api_calls,
            'avg_calls_per_question': avg_api_calls
        }

        # Add timing statistics
        total_time = sum(r.get('metadata', {}).get('total_time', 0) for r in results)
        avg_time = total_time / len(results) if results else 0
        metrics_report['timing'] = {
            'total_time': total_time,
            'avg_time_per_question': avg_time,
            'avg_recruit_time': sum(r.get('metadata', {}).get('recruit_time', 0) for r in results) / len(results) if results else 0,
            'avg_round1_time': sum(r.get('metadata', {}).get('round1_time', 0) for r in results) / len(results) if results else 0,
            'avg_round2_time': sum(r.get('metadata', {}).get('round2_time', 0) for r in results) / len(results) if results else 0,
            'avg_round3_time': sum(r.get('metadata', {}).get('round3_time', 0) for r in results) / len(results) if results else 0,
            'avg_aggregation_time': sum(r.get('metadata', {}).get('aggregation_time', 0) for r in results) / len(results) if results else 0
        }

        return metrics_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ADK-based multi-agent medical reasoning simulation"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['medqa', 'medmcqa', 'pubmedqa', 'mmlupro', 'ddxplus', 'medbullets', 'pmc_vqa', 'path_vqa'],
        help='Dataset to use'
    )

    parser.add_argument(
        '--n-questions',
        type=int,
        default=10,
        help='Number of questions to process'
    )

    parser.add_argument(
        '--n-agents',
        type=int,
        default=None,
        help='Fixed number of agents (default: dynamic 2-4)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gemma3_4b',
        choices=['gemma3_4b', 'gemma2_9b', 'gemma2_27b', 'medgemma_4b'],
        help='Gemma model to use'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='multi-agent-gemma/results',
        help='Output directory for results'
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create and run simulation
    runner = BatchSimulationRunnerADK(
        model_name=args.model,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        n_questions=args.n_questions
    )

    try:
        summary = await runner.run()
    finally:
        # Clean up any open sessions
        try:
            import gc
            import aiohttp

            # Force garbage collection to close any lingering connections
            gc.collect()

            # Give time for connections to close gracefully
            await asyncio.sleep(1.0)

            # Close any remaining aiohttp ClientSession objects
            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession):
                    if not obj.closed:
                        try:
                            await obj.close()
                        except:
                            pass

            # Final wait for cleanup
            await asyncio.sleep(0.5)
        except Exception as e:
            logging.debug(f"Cleanup warning: {e}")

    # Print final summary
    print(f"\n{'='*80}")
    print("BATCH SIMULATION COMPLETE (ADK)")
    print(f"{'='*80}")
    print(f"Questions Processed: {summary.get('timing', {}).get('questions_processed', 0)}")

    accuracy_data = summary.get('accuracy', {})
    if isinstance(accuracy_data, dict):
        overall_accuracy = accuracy_data.get('overall', {}).get('accuracy', 0)
    else:
        overall_accuracy = 0

    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Results saved to: {runner.storage.get_results_path()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
