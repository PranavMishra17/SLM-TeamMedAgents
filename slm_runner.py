"""
Updated SLM Runner with modular chat instances.
Supports Google AI Studio, Hugging Face, and extensible architecture.
"""

import argparse
import logging
import os
import json
import random
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import traceback

# Import existing dataset functions
from medical_datasets import (
    load_medqa_dataset, format_medqa_for_task,
    load_medmcqa_dataset, format_medmcqa_for_task, 
    load_mmlupro_med_dataset, format_mmlupro_med_for_task,
    load_pubmedqa_dataset, format_pubmedqa_for_task,
    load_ddxplus_dataset, format_ddxplus_for_task,
    load_medbullets_dataset, format_medbullets_for_task,
    load_pmc_vqa_dataset, format_pmc_vqa_for_task,
    load_path_vqa_dataset, format_path_vqa_for_task
)

# Import updated configuration and chat instances
from slm_config import *
from chat_instances import ChatInstanceFactory, BaseChatInstance

class SLMAgent:
    """Updated SLM Agent using modular chat instances."""
    
    def __init__(self, model_config: Dict[str, Any], chat_instance_type: str = "google_ai_studio"):
        """Initialize SLM agent with specified chat instance type."""
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.display_name = model_config["display_name"]
        self.chat_instance_type = chat_instance_type
        
        # Create chat instance using factory
        try:
            self.chat_instance = ChatInstanceFactory.create_chat_instance(
                model_config, chat_instance_type
            )
            logging.info(f"Initialized {self.display_name} agent with {chat_instance_type} chat instance")
        except Exception as e:
            logging.error(f"Failed to initialize {self.display_name} with {chat_instance_type}: {e}")
            raise
    
    def simple_chat(self, message: str, image_path: str = None) -> str:
        """Simple single-turn chat with optional image support."""
        return self.chat_instance.simple_chat(message, image_path)
    
    def streaming_chat(self, message: str, image_path: str = None):
        """Streaming chat response generator."""
        return self.chat_instance.streaming_chat(message, image_path)
    
    def conversation_chat(self, messages: List[Dict[str, str]], image_paths: List[str] = None) -> str:
        """Multi-turn conversation chat."""
        return self.chat_instance.conversation_chat(messages, image_paths)

class SLMMethodRunner:
    """Updated SLM Method Runner with modular chat support."""
    
    def __init__(self, model_name: str = None, chat_instance_type: str = None, output_base_dir: str = None):
        """Initialize SLM method runner with specified chat instance type."""
        if chat_instance_type is None:
            chat_instance_type = DEFAULT_CHAT_INSTANCE
        
        self.chat_instance_type = chat_instance_type
        self.model_config = get_model_config(model_name, chat_instance_type)
        self.agent = SLMAgent(self.model_config, chat_instance_type)
        self.output_base_dir = output_base_dir or OUTPUT_BASE_DIR
        self.setup_logging()
        
        logging.info(f"Initialized SLM runner with {self.model_config['display_name']} using {chat_instance_type}")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        # Create logs directory
        log_dir = LOG_DIR
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        log_file = os.path.join(log_dir, f"slm_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG["level"]),
            format=LOGGING_CONFIG["format"],
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info(f"Logging initialized. Log file: {log_file}")
    
    def load_dataset(self, dataset_name: str, num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
        """Load dataset using existing dataset functions."""
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available: {get_supported_datasets()}")
        
        dataset_config = SUPPORTED_DATASETS[dataset_name]
        load_function_name = dataset_config["load_function"]
        
        logging.info(f"Loading {dataset_name} dataset with {num_questions} questions")
        
        # Call the appropriate load function
        if dataset_name == "medqa":
            questions = load_medqa_dataset(num_questions, random_seed)
        elif dataset_name == "medmcqa":
            questions, errors = load_medmcqa_dataset(num_questions, random_seed)
            if errors:
                logging.warning(f"Dataset loading had {len(errors)} errors")
        elif dataset_name == "mmlupro-med":
            questions = load_mmlupro_med_dataset(num_questions, random_seed)
        elif dataset_name == "pubmedqa":
            questions = load_pubmedqa_dataset(num_questions, random_seed)
        elif dataset_name == "ddxplus":
            questions = load_ddxplus_dataset(num_questions, random_seed)
        elif dataset_name == "medbullets":
            questions = load_medbullets_dataset(num_questions, random_seed)
        elif dataset_name == "pmc_vqa":
            questions = load_pmc_vqa_dataset(num_questions, random_seed)
        elif dataset_name == "path_vqa":
            questions = load_path_vqa_dataset(num_questions, random_seed)
        else:
            raise ValueError(f"Load function not implemented for {dataset_name}")
        
        logging.info(f"Successfully loaded {len(questions)} questions from {dataset_name}")
        return questions
    
    def format_question(self, dataset_name: str, question: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format question using existing format functions."""
        dataset_config = SUPPORTED_DATASETS[dataset_name]
        
        # Call the appropriate format function
        if dataset_name == "medqa":
            return format_medqa_for_task(question)
        elif dataset_name == "medmcqa":
            agent_task, eval_data, is_valid = format_medmcqa_for_task(question)
            if not is_valid:
                raise ValueError(f"Invalid question: {eval_data}")
            return agent_task, eval_data
        elif dataset_name == "mmlupro-med":
            return format_mmlupro_med_for_task(question)
        elif dataset_name == "pubmedqa":
            return format_pubmedqa_for_task(question)
        elif dataset_name == "ddxplus":
            return format_ddxplus_for_task(question)
        elif dataset_name == "medbullets":
            return format_medbullets_for_task(question)
        elif dataset_name == "pmc_vqa":
            return format_pmc_vqa_for_task(question)
        elif dataset_name == "path_vqa":
            return format_path_vqa_for_task(question)
        else:
            raise ValueError(f"Format function not implemented for {dataset_name}")
    
    def create_zero_shot_prompt(self, agent_task: Dict[str, Any], is_image_task: bool = False) -> str:
        """Create zero-shot prompt with optional image support."""
        question_text = agent_task["description"]
        options = agent_task.get("options", [])
        task_type = agent_task.get("type", "mcq")
        
        # Add image analysis instruction if this is an image task
        image_instruction = ""
        if is_image_task:
            image_instruction = "Analyze the provided medical image carefully and then "
        
        if task_type == "mcq":
            options_text = "\n".join(options)
            prompt = f"""{image_instruction}Answer the following multiple choice question. Provide your answer as a single letter (A, B, C, D, etc.) followed by a brief explanation.

Question:
{question_text}

Options:
{options_text}

Answer: """
        
        elif task_type == "yes_no_maybe":
            prompt = f"""{image_instruction}Answer the following question with yes, no, or maybe, followed by a brief explanation.

Question:
{question_text}

Answer: """
        
        else:
            prompt = f"""{image_instruction}Answer the following question:

{question_text}

Answer: """
        
        return prompt
    
    def create_few_shot_prompt(self, agent_task: Dict[str, Any], dataset_name: str, is_image_task: bool = False) -> str:
        """Create few-shot prompt with examples and optional image support."""
        # Get examples for this dataset or use general examples
        examples = FEW_SHOT_EXAMPLES.get(dataset_name, FEW_SHOT_EXAMPLES["general"])
        
        question_text = agent_task["description"]
        options = agent_task.get("options", [])
        task_type = agent_task.get("type", "mcq")
        
        # Add image analysis instruction if this is an image task
        image_instruction = ""
        if is_image_task:
            image_instruction = "For image-based questions, first analyze the provided medical image carefully, then "
        
        # Build examples section
        examples_text = f"{image_instruction}Here are some examples of how to answer similar questions:\n\n"
        
        for i, example in enumerate(examples):
            examples_text += f"Example {i+1}:\n"
            examples_text += f"Question: {example['question']}\n"
            if 'options' in example:
                examples_text += f"Options:\n"
                for opt in example['options']:
                    examples_text += f"{opt}\n"
            examples_text += f"Answer: {example['answer']}\n"
            examples_text += f"Reasoning: {example['reasoning']}\n\n"
        
        # Build main question
        analyze_instruction = "Analyze the provided medical image and then answer" if is_image_task else "Answer"
        
        if task_type == "mcq":
            options_text = "\n".join(options)
            main_question = f"""Now {analyze_instruction.lower()} this question in the same format:

Question:
{question_text}

Options:
{options_text}

Answer: """
        elif task_type == "yes_no_maybe":
            main_question = f"""Now {analyze_instruction.lower()} this question in the same format:

Question:
{question_text}

Answer: """
        else:
            main_question = f"""Now {analyze_instruction.lower()} this question:

{question_text}

Answer: """
        
        return examples_text + main_question
    
    def create_cot_prompt(self, agent_task: Dict[str, Any], is_image_task: bool = False) -> str:
        """Create Chain-of-Thought prompt with explicit reasoning demonstration."""
        question_text = agent_task["description"]
        options = agent_task.get("options", [])
        task_type = agent_task.get("type", "mcq")
        
        # Determine if this is medical or general
        is_medical = any(term in question_text.lower() for term in 
                        ['patient', 'diagnosis', 'treatment', 'symptom', 'medical', 'clinical'])
        
        cot_template = COT_TEMPLATES["medical_mcq" if is_medical else "general_mcq"]
        
        # Add image-specific reasoning steps if this is an image task
        if is_image_task:
            image_cot_addition = """
Before applying the reasoning framework, first analyze the image:
1. **Visual Analysis**: What structures, abnormalities, or features are visible?
2. **Clinical Correlation**: How do visual findings relate to the question?
3. **Integration**: Combine image findings with clinical reasoning below.

"""
            cot_template = image_cot_addition + cot_template
        
        if task_type == "mcq":
            options_text = "\n".join(options)
            instruction = "Analyze the provided medical image and answer" if is_image_task else "Answer"
            
            # Enhanced CoT prompt with explicit step-by-step requirement
            prompt = f"""{instruction} the following multiple choice question using step-by-step reasoning.

Question:
{question_text}

Options:
{options_text}

{cot_template}

Now work through THIS question step by step:

Step 1 - Clinical presentation analysis:
[Analyze the key symptoms, findings, and clinical context]

Step 2 - Pathophysiology consideration:
[Consider the underlying disease processes or mechanisms]

Step 3 - Option evaluation:
[Go through each option A, B, C, D and evaluate how well it fits]

Step 4 - Knowledge application:
[Apply relevant medical knowledge and eliminate incorrect options]

Step 5 - Final reasoning:
[Explain why the chosen answer is most appropriate]

Final Answer: """
        
        elif task_type == "yes_no_maybe":
            instruction = "Analyze the provided medical image and answer" if is_image_task else "Answer"
            prompt = f"""{instruction} the following question using step-by-step reasoning.

Question:
{question_text}

{cot_template}

Now work through THIS question step by step:

Step 1 - Question analysis:
[What exactly is being asked?]

Step 2 - Evidence consideration:
[What evidence supports yes, no, or maybe?]

Step 3 - Knowledge application:
[What medical/scientific knowledge applies?]

Step 4 - Final reasoning:
[Explain the logic leading to your answer]

Final Answer: """
        
        else:
            instruction = "Analyze the provided medical image and answer" if is_image_task else "Answer"
            prompt = f"""{instruction} the following question using step-by-step reasoning:

{question_text}

{cot_template}

Final Answer: """
        
        return prompt
    
    def extract_answer(self, response: str, task_type: str = "mcq") -> Optional[str]:
        """Extract answer from model response."""
        import re
        
        if task_type == "mcq":
            # Look for patterns like "Answer: A" or "Final Answer: B"
            patterns = [
                r"(?:Final\s+)?Answer:\s*([A-J])\b",
                r"(?:Final\s+)?Answer:\s*\(?([A-J])\)",
                r"^([A-J])\.",
                r"\b([A-J])\b(?=\s*[-\.\)]|\s*$)"
            ]
        elif task_type == "yes_no_maybe":
            patterns = [
                r"(?:Final\s+)?Answer:\s*(yes|no|maybe)",
                r"\b(yes|no|maybe)\b"
            ]
        else:
            return response.strip()
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper() if task_type == "mcq" else match.group(1).lower()
        
        return None
    
    def process_single_question(self, question: Dict[str, Any], dataset_name: str, 
                               method: str, question_index: int) -> Dict[str, Any]:
        """Process a single question with specified method."""
        try:
            # Format question
            agent_task, eval_data = self.format_question(dataset_name, question)
            
            # Check if this is an image-based task
            is_image_task = SUPPORTED_DATASETS.get(dataset_name, {}).get("requires_images", False)
            image_path = agent_task.get("image_path") if is_image_task else None
            
            # Create prompt based on method
            if method == "zero_shot":
                prompt = self.create_zero_shot_prompt(agent_task, is_image_task)
            elif method == "few_shot":
                prompt = self.create_few_shot_prompt(agent_task, dataset_name, is_image_task)
            elif method == "cot":
                prompt = self.create_cot_prompt(agent_task, is_image_task)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Get model response with optional image
            start_time = time.time()
            response = self.agent.simple_chat(prompt, image_path)
            end_time = time.time()
            
            # Extract answer
            task_type = agent_task.get("type", "mcq")
            extracted_answer = self.extract_answer(response, task_type)
            
            # Evaluate correctness
            ground_truth = eval_data["ground_truth"]
            is_correct = (extracted_answer == ground_truth) if extracted_answer else False
            
            result = {
                "question_index": question_index,
                "dataset": dataset_name,
                "method": method,
                "model": self.model_config["display_name"],
                "chat_instance_type": self.chat_instance_type,
                "question_text": agent_task["description"][:200] + "..." if len(agent_task["description"]) > 200 else agent_task["description"],
                "options": agent_task.get("options", []),
                "ground_truth": ground_truth,
                "extracted_answer": extracted_answer,
                "is_correct": is_correct,
                "full_response": response,
                "prompt": prompt,
                "response_time": end_time - start_time,
                "task_metadata": eval_data.get("metadata", {}),
                "timestamp": datetime.now().isoformat()
            }
            
            status_symbol = "CORRECT" if is_correct else "WRONG"
            logging.info(f"Q{question_index}: {dataset_name}/{method} - "
                        f"Answer: {extracted_answer} (GT: {ground_truth}) - "
                        f"{status_symbol} - {end_time - start_time:.2f}s")
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing Q{question_index}: {e}")
            logging.error(traceback.format_exc())
            return {
                "question_index": question_index,
                "dataset": dataset_name,
                "method": method,
                "model": self.model_config["display_name"],
                "chat_instance_type": self.chat_instance_type,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_dataset_evaluation(self, dataset_name: str, method: str, 
                             num_questions: int = 50, random_seed: int = 42) -> Dict[str, Any]:
        """Run evaluation on a dataset with specified method."""
        model_name = self.model_config["name"]
        logging.info(f"Starting {method} evaluation on {dataset_name} with {num_questions} questions using {self.model_config['display_name']} via {self.chat_instance_type}")
        
        # Create output directory with model-specific path
        output_dir = get_output_dir(method, dataset_name, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        questions = self.load_dataset(dataset_name, num_questions, random_seed)
        
        if not questions:
            raise ValueError(f"No questions loaded for {dataset_name}")
        
        # Process questions
        results = []
        errors = []
        start_time = time.time()
        
        for i, question in enumerate(questions):
            result = self.process_single_question(question, dataset_name, method, i)
            
            if "error" in result:
                errors.append(result)
            else:
                results.append(result)
            
            # Save intermediate results every 10 questions
            if (i + 1) % 10 == 0:
                intermediate_file = os.path.join(output_dir, f"intermediate_results_{i+1}.json")
                with open(intermediate_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        
        # Calculate summary statistics
        correct_answers = sum(1 for r in results if r.get("is_correct", False))
        total_questions = len(results)
        accuracy = (correct_answers / total_questions) if total_questions > 0 else 0.0
        
        summary = {
            "dataset": dataset_name,
            "method": method,
            "model": self.model_config["display_name"],
            "model_name": self.model_config["name"],
            "chat_instance_type": self.chat_instance_type,
            "model_family": self.model_config.get("model_family", "unknown"),
            "specialized_domain": self.model_config.get("specialized_domain", "general"),
            "supports_vision": self.model_config.get("supports_vision", False),
            "total_questions": total_questions,
            "correct_answers": correct_answers,
            "accuracy": accuracy,
            "total_errors": len(errors),
            "total_time": end_time - start_time,
            "avg_time_per_question": (end_time - start_time) / len(questions) if questions else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save detailed results
        detailed_results = {
            "summary": summary,
            "results": results,
            "errors": errors,
            "configuration": {
                "model_config": self.model_config,
                "method_config": PROMPTING_METHODS[method],
                "chat_instance_type": self.chat_instance_type,
                "random_seed": random_seed
            }
        }
        
        results_file = os.path.join(output_dir, f"results_{dataset_name}_{method}_{self.model_config['name']}_{self.chat_instance_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Save summary
        summary_file = os.path.join(output_dir, f"summary_{dataset_name}_{method}_{self.model_config['name']}_{self.chat_instance_type}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Completed {method} evaluation on {dataset_name} using {self.model_config['display_name']} via {self.chat_instance_type}")
        logging.info(f"Results: {correct_answers}/{total_questions} correct ({accuracy:.2%})")
        logging.info(f"Results saved to: {results_file}")
        
        return detailed_results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Updated SLM Runner with modular chat instances")
    
    parser.add_argument("--dataset", required=True, choices=get_supported_datasets(),
                       help="Dataset to evaluate on")
    parser.add_argument("--method", required=True, choices=list(PROMPTING_METHODS.keys()),
                       help="Prompting method to use")
    parser.add_argument("--model", choices=get_supported_models(), 
                       default=DEFAULT_MODEL,
                       help="Model to use")
    parser.add_argument("--chat_instance", choices=get_supported_chat_instances(),
                       default=DEFAULT_CHAT_INSTANCE,
                       help="Chat instance type to use")
    parser.add_argument("--num_questions", type=int, default=50,
                       help="Number of questions to evaluate")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str,
                       help="Custom output directory")
    
    args = parser.parse_args()
    
    try:
        # Validate configuration for selected chat instance
        if not validate_configuration(args.chat_instance):
            print(f"Configuration validation failed for {args.chat_instance}")
            sys.exit(1)
        
        # Initialize runner
        runner = SLMMethodRunner(
            model_name=args.model, 
            chat_instance_type=args.chat_instance,
            output_base_dir=args.output_dir
        )
        
        # Run evaluation
        results = runner.run_dataset_evaluation(
            dataset_name=args.dataset,
            method=args.method,
            num_questions=args.num_questions,
            random_seed=args.random_seed
        )
        
        # Print summary
        summary = results["summary"]
        print(f"\n{'='*50}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*50}")
        print(f"Dataset: {summary['dataset']}")
        print(f"Method: {summary['method']}")
        print(f"Model: {summary['model']}")
        print(f"Chat Instance: {summary['chat_instance_type']}")
        print(f"Accuracy: {summary['correct_answers']}/{summary['total_questions']} ({summary['accuracy']:.2%})")
        print(f"Total Time: {summary['total_time']:.2f}s")
        print(f"Avg Time/Question: {summary['avg_time_per_question']:.2f}s")
        if summary['total_errors'] > 0:
            print(f"Errors: {summary['total_errors']}")
        print(f"{'='*50}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

def test_sample_evaluation():
    """Test function to run a small sample evaluation."""
    print("Running sample test evaluation with modular chat instances...")
    
    # Test configuration
    dataset_name = "medqa"
    method = "zero_shot"
    num_questions = 2
    
    # Test both chat instance types
    chat_instances = get_supported_chat_instances()
    
    for chat_instance in chat_instances:
        print(f"\nTesting with {chat_instance} chat instance...")
        
        if not validate_configuration(chat_instance):
            print(f"Skipping {chat_instance} - configuration not valid")
            continue
        
        try:
            # Initialize runner
            runner = SLMMethodRunner(chat_instance_type=chat_instance)
            
            # Run evaluation
            results = runner.run_dataset_evaluation(
                dataset_name=dataset_name,
                method=method,
                num_questions=num_questions,
                random_seed=42
            )
            
            # Print results
            summary = results["summary"]
            print(f"[{chat_instance}] Results: {summary['correct_answers']}/{summary['total_questions']} correct ({summary['accuracy']:.2%})")
            
        except Exception as e:
            print(f"[{chat_instance}] Test failed: {e}")
            continue
    
    return True

if __name__ == "__main__":
    # Check if running as test
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == "test"):
        print("Running in test mode...")
        success = test_sample_evaluation()
        sys.exit(0 if success else 1)
    else:
        main()