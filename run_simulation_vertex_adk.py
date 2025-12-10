"""
Vertex AI Endpoint Multi-Agent Simulation Runner

Main entry point for running multi-agent medical reasoning simulations using
MedGemma deployed on Vertex AI endpoints via Google's Agent Development Kit (ADK).

This script is designed for use with MedGemma models deployed on Vertex AI,
providing production-grade scalability while maintaining full compatibility with
the existing multi-agent system architecture.

Prerequisites:
    1. Deploy MedGemma to Vertex AI Model Garden endpoint
    2. Set up authentication (gcloud auth application-default login)
    3. Configure environment variables (see below)

Environment Variables Required:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_CLOUD_LOCATION: Vertex AI region (e.g., us-central1)
    - VERTEX_AI_ENDPOINT_ID: Your MedGemma endpoint ID
    - GOOGLE_GENAI_USE_VERTEXAI: Set to "TRUE"

Maintains full compatibility with existing:
- Dataset loaders (medical_datasets/)
- Logging system (utils/SimulationLogger)
- Results storage (utils/ResultsStorage)
- Metrics calculation (utils/MetricsCalculator)
- All teamwork components

Usage:
    # Basic usage with Vertex AI endpoint
    python run_simulation_vertex_adk.py \\
        --dataset medqa \\
        --n-questions 10 \\
        --n-agents 3

    # With teamwork components
    python run_simulation_vertex_adk.py \\
        --dataset medqa \\
        --n-questions 50 \\
        --all-teamwork

    # Vision dataset with Vertex AI
    python run_simulation_vertex_adk.py \\
        --dataset pmc_vqa \\
        --n-questions 20 \\
        --all-teamwork
"""

# MedGemma image token constant (assuming same as Gemma)
# According to Gemma documentation, images are encoded as 258 tokens
MEDGEMMA_IMAGE_TOKENS = 258

import sys
import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass


def verify_vertex_config() -> Dict[str, str]:
    """
    Verify Vertex AI configuration from environment variables.

    Returns:
        Dict with verified configuration

    Raises:
        RuntimeError: If required environment variables are not set
    """
    config = {
        'project_id': os.environ.get('GOOGLE_CLOUD_PROJECT'),
        'location': os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
        'endpoint_id': os.environ.get('VERTEX_AI_ENDPOINT_ID'),
        'use_vertex': os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper()
    }

    errors = []

    if not config['project_id']:
        errors.append("GOOGLE_CLOUD_PROJECT not set (your GCP project ID)")

    if not config['endpoint_id']:
        errors.append("VERTEX_AI_ENDPOINT_ID not set (your MedGemma endpoint ID)")

    if config['use_vertex'] != 'TRUE':
        errors.append("GOOGLE_GENAI_USE_VERTEXAI not set to TRUE")

    if errors:
        error_msg = "Vertex AI configuration incomplete:\n"
        for error in errors:
            error_msg += f"  ✗ {error}\n"
        error_msg += "\nPlease set these environment variables:\n"
        error_msg += "  export GOOGLE_CLOUD_PROJECT='your-project-id'\n"
        error_msg += "  export GOOGLE_CLOUD_LOCATION='us-central1'\n"
        error_msg += "  export VERTEX_AI_ENDPOINT_ID='your-endpoint-id'\n"
        error_msg += "  export GOOGLE_GENAI_USE_VERTEXAI=TRUE\n"
        error_msg += "\nOr add them to your .env file."
        raise RuntimeError(error_msg)

    logging.info("✓ Vertex AI configuration verified:")
    logging.info(f"  Project: {config['project_id']}")
    logging.info(f"  Location: {config['location']}")
    logging.info(f"  Endpoint: {config['endpoint_id']}")

    return config


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
    # NOTE: We cannot directly import MultiAgentSystemADK as it uses GemmaAgentFactory
    # We will need to create agents differently for Vertex AI
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("ERROR: Google ADK not installed. Run: pip install google-adk")
    sys.exit(1)

# Import Vertex AI agent factory
try:
    from adk_agents.gemma_agent_vertex_adk import VertexAIAgentFactory
    VERTEX_AGENT_AVAILABLE = True
except ImportError:
    VERTEX_AGENT_AVAILABLE = False
    print("ERROR: Vertex AI agent factory not available. Check adk_agents/gemma_agent_vertex_adk.py")
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


class VertexAISimulationRunner:
    """
    Runs Vertex AI endpoint-based multi-agent simulations on medical questions.

    This runner uses MedGemma deployed on Vertex AI endpoints instead of
    Google AI Studio, providing production-grade scalability.

    Features:
    - Full integration with existing logging/results/metrics infrastructure
    - Progress tracking with tqdm
    - Comprehensive metrics calculation
    - Same output format as Google AI Studio version
    - Support for both text and multimodal (image) datasets
    """

    def __init__(
        self,
        endpoint_id: str,
        project_id: str,
        location: str = "us-central1",
        n_agents: int = None,
        output_dir: str = "multi-agent-gemma/results_vertex",
        dataset_name: str = None,
        n_questions: int = 10,
        teamwork_config=None,
        random_seed: int = 42
    ):
        """
        Initialize Vertex AI simulation runner.

        Args:
            endpoint_id: Vertex AI endpoint ID for MedGemma
            project_id: GCP project ID
            location: Vertex AI region (default: us-central1)
            n_agents: Fixed agent count or None for dynamic (2-4)
            output_dir: Base directory for results
            dataset_name: Dataset to load
            n_questions: Number of questions to process
            teamwork_config: TeamworkConfig instance for modular components
            random_seed: Random seed for dataset sampling
        """
        self.endpoint_id = endpoint_id
        self.project_id = project_id
        self.location = location
        self.n_agents_config = n_agents
        self.dataset_name = dataset_name
        self.n_questions = n_questions
        self.teamwork_config = teamwork_config
        self.random_seed = random_seed

        # Import and initialize multi-agent system
        # NOTE: The same MultiAgentSystemADK works for both Google AI Studio and Vertex AI
        # The key is that GOOGLE_GENAI_USE_VERTEXAI=TRUE is set, which causes
        # GemmaAgentFactory.create_agent() to automatically delegate to VertexAIAgentFactory
        from adk_agents import MultiAgentSystemADK

        # Verify Vertex AI env var is set (should be set in main() before creating runner)
        use_vertex = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper() == 'TRUE'
        if not use_vertex:
            logging.warning("GOOGLE_GENAI_USE_VERTEXAI not set to TRUE - agents may use Google AI Studio instead!")
            logging.warning("Setting GOOGLE_GENAI_USE_VERTEXAI=TRUE now...")
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'TRUE'

        # Create multi-agent system
        # model_name is not used by Vertex AI (endpoint comes from env vars),
        # but we pass it for logging/identification purposes
        self.system = MultiAgentSystemADK(
            model_name=f"medgemma-vertex-ai",  # Identifier for logging (not used by Vertex AI)
            n_agents=n_agents,
            teamwork_config=teamwork_config
        )

        # Initialize ADK Runner
        self.runner = Runner(
            app_name="medical_reasoning_vertex_adk",
            agent=self.system,
            session_service=InMemorySessionService()
        )

        # Setup storage and logging
        self.storage = ResultsStorage(
            output_dir=output_dir,
            dataset_name=dataset_name,
            n_questions=n_questions,
            random_seed=random_seed
        )
        self.logger = SimulationLogger(
            output_dir=output_dir,
            run_id=self.storage.run_id
        )
        self.metrics = MetricsCalculator()
        self.token_counter = TokenCounter()

        # Configuration
        self.config = {
            "framework": "Google ADK + Vertex AI",
            "endpoint_id": endpoint_id,
            "project_id": project_id,
            "location": location,
            "n_agents": n_agents if n_agents else "dynamic (2-4)",
            "dataset": dataset_name,
            "n_questions": n_questions,
            "random_seed": random_seed,
            "timestamp": datetime.now().isoformat()
        }

        # Add teamwork config to configuration
        if teamwork_config:
            self.config['teamwork'] = teamwork_config.to_dict()

        # Ground truth storage
        self.ground_truth = {}

        logging.info(f"Initialized Vertex AI SimulationRunner: {dataset_name}, {n_questions} questions")
        logging.info(f"Using Vertex AI endpoint: {endpoint_id}")

    def load_dataset(self) -> List[Dict]:
        """Load dataset using existing loaders."""
        if not DATASETS_AVAILABLE:
            raise RuntimeError("Dataset loaders not available")

        logging.info(f"Loading {self.dataset_name} dataset ({self.n_questions} questions, seed={self.random_seed})...")

        if self.dataset_name == "medqa":
            questions = DatasetLoader.load_medqa(self.n_questions, random_seed=self.random_seed)
        elif self.dataset_name == "medmcqa":
            questions, errors = DatasetLoader.load_medmcqa(self.n_questions, random_seed=self.random_seed)
            if errors:
                logging.warning(f"Skipped {len(errors)} invalid MedMCQA questions")
        elif self.dataset_name == "pubmedqa":
            questions = DatasetLoader.load_pubmedqa(self.n_questions, random_seed=self.random_seed)
        elif self.dataset_name == "mmlupro":
            questions = DatasetLoader.load_mmlupro_med(self.n_questions, random_seed=self.random_seed)
        elif self.dataset_name == "ddxplus":
            questions = DatasetLoader.load_ddxplus(self.n_questions, random_seed=self.random_seed)
        elif self.dataset_name == "medbullets":
            questions = DatasetLoader.load_medbullets(self.n_questions, random_seed=self.random_seed)
        elif self.dataset_name == "pmc_vqa":
            questions = VisionDatasetLoader.load_pmc_vqa(self.n_questions, random_seed=self.random_seed)
        elif self.dataset_name == "path_vqa":
            questions = VisionDatasetLoader.load_path_vqa(self.n_questions, random_seed=self.random_seed)
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

            # Get ground truth
            ground_truth = eval_data.get('ground_truth', '') if eval_data else None
            if ground_truth:
                self.ground_truth[question_id] = str(ground_truth).strip().upper()

            return {
                'question': question_text,
                'options': options,
                'task_type': task_type,
                'ground_truth': ground_truth,
                'question_id': question_id,
                'image': image  # PIL Image object for vision datasets
            }

        except Exception as e:
            logging.error(f"Error formatting question {question_id}: {e}")
            return None

    async def process_question(self, question_data: Dict, question_idx: int, max_retries: int = 5) -> Dict:
        """
        Process a single question through Vertex AI multi-agent system with retry logic.

        Args:
            question_data: Question data from dataset
            question_idx: Question index
            max_retries: Maximum retry attempts

        Returns:
            Dict with complete results
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

                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    if is_rate_limit:
                        delay = min(60 * (2 ** attempt), 300)
                        jitter = random.uniform(0.8, 1.2)
                        delay *= jitter
                        logging.warning(f"Rate limit error, waiting {delay:.1f}s")
                    else:
                        delay = min(3 * (2 ** attempt), 30)
                        logging.warning(f"Timeout, waiting {delay:.1f}s")

                    await asyncio.sleep(delay)
                    continue
                elif is_rate_limit or is_timeout:
                    logging.error(f"Failed after {max_retries+1} attempts")
                    return None
                else:
                    logging.error(f"Non-retryable error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

        return None

    async def _process_question_attempt(self, question_data: Dict, question_idx: int) -> Dict:
        """
        Single attempt to process a question via Vertex AI.

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
        image = formatted.get('image', None)

        self.logger.log_question_start(question_id, question)

        # Prepare initial state for ADK session
        initial_state = {
            'question': question,
            'options': options,
            'task_type': task_type,
            'ground_truth': ground_truth,
            'image': image,
            'dataset': self.dataset_name
        }

        # Create session
        await self.runner.session_service.create_session(
            app_name="medical_reasoning_vertex_adk",
            user_id="simulation",
            session_id=question_id,
            state=initial_state
        )

        # Run system
        try:
            from google.genai import types
            initial_message = types.Content(
                parts=[types.Part(text=f"Analyze this medical question: {question}")]
            )

            captured_state = None

            async for event in self.runner.run_async(
                user_id="simulation",
                session_id=question_id,
                new_message=initial_message
            ):
                if hasattr(event, 'content') and event.content:
                    content_text = extract_text_from_content(event.content)
                    logging.debug(f"[{event.author}] {content_text[:100]}")

                    if event.author == "multi_agent_system" and "RESULTS:" in content_text:
                        import json
                        try:
                            json_start = content_text.find("{")
                            if json_start >= 0:
                                captured_state = json.loads(content_text[json_start:])
                                logging.debug(f"Captured state from event")
                        except json.JSONDecodeError:
                            logging.warning("Failed to parse state from event")

        except Exception as e:
            logging.error(f"Error processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Extract results from captured state
        if captured_state:
            final_answer = captured_state.get('final_answer', 'A')
            is_correct = captured_state.get('is_correct', False)
            timing = captured_state.get('timing', {})
            recruited_agents = captured_state.get('recruited_agents', [])
            round1_results = captured_state.get('round1_results', {})
            round2_results = captured_state.get('round2_results', {})
            round3_results = captured_state.get('round3_results', {})
            aggregation_result = captured_state.get('aggregation_result', {})
            convergence = captured_state.get('convergence', {})
            teamwork_interactions = captured_state.get('teamwork_interactions', {})
            api_call_count = captured_state.get('api_call_count', 0)
        else:
            logging.error("No captured state from events")
            return None

        # Track token usage
        total_input_text = question
        total_output_text = ""

        for agent_id, response in round1_results.items():
            total_output_text += response
        for agent_id, response in round2_results.items():
            total_output_text += response
        for agent_id, result in round3_results.items():
            total_output_text += result.get('raw', '')

        token_data = self.token_counter.log_question_tokens(
            input_text=total_input_text,
            output_text=total_output_text,
            question_index=question_idx,
            usage_metadata=None,
            model=f"medgemma-endpoint-{self.endpoint_id}"
        )

        # Add image tokens if present
        if image is not None:
            n_agents = len(recruited_agents)
            n_rounds = 3
            image_tokens_total = MEDGEMMA_IMAGE_TOKENS * n_agents * n_rounds

            token_data['input_tokens'] += image_tokens_total
            token_data['total_tokens'] += image_tokens_total
            token_data['image_tokens'] = image_tokens_total
            token_data['has_image'] = True

            self.token_counter.input_tokens += image_tokens_total
            self.token_counter.total_tokens += image_tokens_total
        else:
            token_data['image_tokens'] = 0
            token_data['has_image'] = False

        # Log completion
        self.logger.log_question_complete(
            question_id=question_id,
            final_answer=final_answer,
            is_correct=is_correct,
            time_taken=timing.get('total_time', 0)
        )

        # Package results (same format as Google AI Studio version)
        result = {
            'question_id': question_id,
            'question': question,
            'options': options,
            'task_type': task_type,
            'image_path': None if image is None else "PIL_Image_Object",

            'recruited_agents': [
                {
                    'agent_id': a['agent_id'],
                    'role': a['role'],
                    'expertise': a['expertise']
                }
                for a in recruited_agents
            ],

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

            'final_decision': {
                'primary_answer': final_answer,
                'borda_count': aggregation_result,
                'convergence': convergence
            },

            'ground_truth': ground_truth,
            'is_correct': is_correct,

            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_time': timing.get('total_time', 0),
                'recruit_time': timing.get('recruit_time', 0),
                'round1_time': timing.get('round1_time', 0),
                'round2_time': timing.get('round2_time', 0),
                'round3_time': timing.get('round3_time', 0),
                'aggregation_time': timing.get('aggregation_time', 0),
                'api_calls': api_call_count if api_call_count > 0 else len(recruited_agents) * 3,
                'n_agents': len(recruited_agents),
                'framework': 'Google ADK + Vertex AI'
            },

            'token_usage': token_data
        }

        if teamwork_interactions:
            result['teamwork_interactions'] = teamwork_interactions

        return result

    async def run(self) -> Dict[str, Any]:
        """Execute batch simulation with full tracking."""
        self.logger.log_run_start(self.config)
        logging.info(f"\n{'='*80}\nVERTEX AI SIMULATION: {self.dataset_name.upper()}\n{'='*80}")

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

        logging.info(f"\n{'='*80}\nVERTEX AI SIMULATION COMPLETE\n{'='*80}")
        logging.info(f"Questions Processed: {len(results)}")
        logging.info(f"Overall Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        logging.info(f"Results saved to: {self.storage.get_results_path()}")

        return metrics_summary

    def _generate_metrics_summary(self, results: List[Dict]) -> Dict:
        """Generate comprehensive metrics."""
        if not results:
            return {}

        for result in results:
            self.metrics.add_question_result(result)

        metrics_report = self.metrics.generate_summary_report()
        token_summary = self.token_counter.get_summary()
        metrics_report['token_usage'] = token_summary
        metrics_report['token_usage']['per_question'] = self.token_counter.question_tokens

        # API call statistics
        total_api_calls = sum(r.get('metadata', {}).get('api_calls', 0) for r in results)
        metrics_report['api_calls'] = {
            'total_calls': total_api_calls,
            'avg_calls_per_question': total_api_calls / len(results) if results else 0
        }

        # Timing statistics
        total_time = sum(r.get('metadata', {}).get('total_time', 0) for r in results)
        metrics_report['timing'] = {
            'total_time': total_time,
            'avg_time_per_question': total_time / len(results) if results else 0,
            'avg_recruit_time': sum(r.get('metadata', {}).get('recruit_time', 0) for r in results) / len(results) if results else 0,
            'avg_round1_time': sum(r.get('metadata', {}).get('round1_time', 0) for r in results) / len(results) if results else 0,
            'avg_round2_time': sum(r.get('metadata', {}).get('round2_time', 0) for r in results) / len(results) if results else 0,
            'avg_round3_time': sum(r.get('metadata', {}).get('round3_time', 0) for r in results) / len(results) if results else 0
        }

        return metrics_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Vertex AI endpoint multi-agent medical reasoning simulation"
    )

    # Dataset arguments
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
        '--output-dir',
        type=str,
        default='multi-agent-gemma/results_vertex',
        help='Output directory for results'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for dataset sampling'
    )

    # Vertex AI configuration (optional if using env vars)
    parser.add_argument(
        '--endpoint-id',
        type=str,
        help='Vertex AI endpoint ID (or use VERTEX_AI_ENDPOINT_ID env var)'
    )

    parser.add_argument(
        '--project-id',
        type=str,
        help='GCP project ID (or use GOOGLE_CLOUD_PROJECT env var)'
    )

    parser.add_argument(
        '--location',
        type=str,
        default='us-central1',
        help='Vertex AI region (default: us-central1)'
    )

    # Teamwork component flags
    parser.add_argument('--smm', action='store_true', help='Enable Shared Mental Model')
    parser.add_argument('--leadership', action='store_true', help='Enable Leadership')
    parser.add_argument('--team-orientation', action='store_true', help='Enable Team Orientation')
    parser.add_argument('--trust', action='store_true', help='Enable Trust Network')
    parser.add_argument('--mutual-monitoring', action='store_true', help='Enable Mutual Monitoring')
    parser.add_argument('--all-teamwork', action='store_true', help='Enable ALL teamwork components')
    parser.add_argument('--n-turns', type=int, default=2, choices=[2, 3], help='Number of R3 discussion turns')

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

    # Verify Vertex AI configuration
    try:
        vertex_config = verify_vertex_config()

        # Use CLI args if provided, otherwise use env vars
        endpoint_id = args.endpoint_id or vertex_config['endpoint_id']
        project_id = args.project_id or vertex_config['project_id']
        location = args.location or vertex_config['location']

        # CRITICAL: Ensure GOOGLE_GENAI_USE_VERTEXAI is set to TRUE
        # This must be set BEFORE creating any agents, as GemmaAgentFactory checks this
        # during agent creation (in recruiter initialization)
        if os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', 'FALSE').upper() != 'TRUE':
            os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'TRUE'
            logging.info("Set GOOGLE_GENAI_USE_VERTEXAI=TRUE for agent creation")

    except RuntimeError as e:
        logging.error(f"Configuration Error: {e}")
        return

    # Create teamwork config
    from teamwork_components import TeamworkConfig

    if args.all_teamwork:
        teamwork_config = TeamworkConfig.all_enabled(n_turns=args.n_turns)
        logging.info("All teamwork components ENABLED")
    else:
        teamwork_config = TeamworkConfig(
            smm=args.smm,
            leadership=args.leadership,
            team_orientation=args.team_orientation,
            trust=args.trust,
            mutual_monitoring=args.mutual_monitoring,
            n_turns=args.n_turns
        )

        active = teamwork_config.get_active_components()
        if active:
            logging.info(f"Teamwork components ENABLED: {', '.join(active)}")
        else:
            logging.info("Base system mode (all teamwork components OFF)")

    # Create and run simulation
    runner = VertexAISimulationRunner(
        endpoint_id=endpoint_id,
        project_id=project_id,
        location=location,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        n_questions=args.n_questions,
        teamwork_config=teamwork_config,
        random_seed=args.seed
    )

    try:
        summary = await runner.run()
    finally:
        # Cleanup
        try:
            import gc
            import aiohttp
            gc.collect()
            await asyncio.sleep(1.0)

            for obj in gc.get_objects():
                if isinstance(obj, aiohttp.ClientSession):
                    if not obj.closed:
                        try:
                            await obj.close()
                        except:
                            pass

            await asyncio.sleep(0.5)
        except Exception as e:
            logging.debug(f"Cleanup warning: {e}")

    # Print final summary
    print(f"\n{'='*80}")
    print("VERTEX AI SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Endpoint: {endpoint_id}")
    print(f"Questions Processed: {len(summary.get('accuracy', {}).get('by_question', []))}")

    accuracy_data = summary.get('accuracy', {})
    if isinstance(accuracy_data, dict):
        overall_accuracy = accuracy_data.get('overall_accuracy', 0)
    else:
        overall_accuracy = 0

    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Results saved to: {runner.storage.get_results_path()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
