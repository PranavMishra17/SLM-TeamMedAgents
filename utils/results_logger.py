"""
Results logging and aggregation utility for SLM TeamMedAgents.
Handles token counting, result aggregation, and summary generation.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging


class TokenCounter:
    """Token counter for tracking input/output tokens per question and run using Google's API."""
    
    def __init__(self):
        self.reset()
        self._setup_google_client()
    
    def _setup_google_client(self):
        """Setup Google AI client for token counting."""
        try:
            import google.genai as genai
            from google.genai.types import HttpOptions
            import os
            
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if api_key:
                # Use v1beta as some Gemma models (and token-counting endpoints) are under the beta surface
                # and may not be available on v1. Keep a tolerant setup and log which API version is used.
                self.client = genai.Client(
                    api_key=api_key,
                    http_options=HttpOptions(api_version="v1beta")
                )
                self.google_available = True
                logging.info("Google AI client initialized for token counting")
            else:
                self.google_available = False
                logging.warning("Google API key not found, falling back to approximation")
        except ImportError as e:
            self.google_available = False
            logging.warning(f"Google genai not available, falling back to approximation: {e}")
        except Exception as e:
            self.google_available = False
            logging.warning(f"Error setting up Google client: {e}")
    
    def reset(self):
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.question_tokens = []
    
    def count_tokens_google(self, text: str, model: str = "gemma-3-4b-it") -> int:
        """Count tokens using Google's API."""
        if not self.google_available or not text:
            return self.count_tokens_approximate(text)
        
        try:
            # Try the straightforward count_tokens call first. Some client versions expose
            # models.count_tokens while others require calling the REST endpoint under v1beta.
            # We'll try both patterns and fall back to approximation on any 404/unsupported errors.
            token_count = None
            try:
                response = self.client.models.count_tokens(
                    model=model,
                    contents=text
                )
                token_count = getattr(response, 'total_tokens', None)
            except Exception as inner_e:
                # If the client doesn't expose count_tokens or the model isn't found under v1,
                # try the v1beta REST-style helper if available on the client object.
                logging.debug(f"primary count_tokens failed: {inner_e}")
                try:
                    # Some clients expose a lower-level http client to call the beta endpoint.
                    # Attempt a fallback POST to the v1beta model countTokens method.
                    # Build a model path that works with both 'gemma-3-4b-it' and 'models/gemma-3-4b-it'
                    model_path = model
                    if not model.startswith('models/'):
                        model_path = f"models/{model}"
                    # call_rest is a conservative helper; if not present we'll raise and fallback
                    if hasattr(self.client, 'http') and hasattr(self.client.http, 'post'):
                        # low-level POST to the beta countTokens endpoint
                        url = f"/v1beta/{model_path}:countTokens"
                        http_resp = self.client.http.post(url, json={"content": text})
                        # try to extract total_tokens
                        token_count = http_resp.json().get('totalTokens') or http_resp.json().get('total_tokens')
                    else:
                        raise inner_e
                except Exception as inner_e2:
                    logging.debug(f"fallback v1beta countTokens attempt failed: {inner_e2}")

            if token_count is not None:
                return int(token_count)
            else:
                # If we reached here, token_count couldn't be determined by remote call
                logging.warning("Google token counting did not return total_tokens, using approximation")
                return self.count_tokens_approximate(text)
        except Exception as e:
            # If the error indicates the model/method isn't found (HTTP 404 from the service),
            # fall back to approximation but keep the logs informative for debugging.
            logging.warning(f"Google token counting failed, using approximation: {e}")
            return self.count_tokens_approximate(text)
    
    def count_tokens_approximate(self, text: str) -> int:
        """Fallback token counting approximation (words * 1.3)."""
        if not text:
            return 0
        # Rough approximation: 1 token ≈ 0.75 words for English
        word_count = len(text.split())
        return int(word_count * 1.3)
    
    def log_question_tokens(self, input_text: str, output_text: str, question_index: int, 
                          usage_metadata: dict = None, model: str = "gemma-3-4b-it"):
        """Log tokens for a single question with optional Google API usage metadata."""
        if usage_metadata:
            # Use actual token counts from Google API response, ensure they're not None
            input_count = usage_metadata.get("prompt_token_count") or 0
            output_count = usage_metadata.get("candidates_token_count") or 0
            total_count = usage_metadata.get("total_token_count") or (input_count + output_count)
        else:
            # Fallback to counting
            input_count = self.count_tokens_google(input_text, model) or 0
            output_count = self.count_tokens_google(output_text, model) or 0
            total_count = input_count + output_count
        
        # Ensure all counts are integers
        input_count = int(input_count) if input_count is not None else 0
        output_count = int(output_count) if output_count is not None else 0
        total_count = int(total_count) if total_count is not None else (input_count + output_count)
        
        question_token_data = {
            "question_index": question_index,
            "input_tokens": input_count,
            "output_tokens": output_count,
            "total_tokens": total_count,
            "method": "google_api" if usage_metadata else "google_count" if self.google_available else "approximation",
            "timestamp": datetime.now().isoformat()
        }
        
        self.question_tokens.append(question_token_data)
        self.input_tokens += input_count
        self.output_tokens += output_count
        self.total_tokens += total_count
        
        return question_token_data
    
    def get_summary(self) -> Dict[str, Any]:
        """Get token usage summary."""
        num_questions = len(self.question_tokens)
        return {
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "questions_processed": num_questions,
            "avg_input_tokens_per_question": self.input_tokens / num_questions if num_questions > 0 else 0,
            "avg_output_tokens_per_question": self.output_tokens / num_questions if num_questions > 0 else 0,
            "avg_total_tokens_per_question": self.total_tokens / num_questions if num_questions > 0 else 0
        }


class ResultsLogger:
    """Enhanced results logger with token tracking and aggregation."""
    
    def __init__(self, output_base_dir: str = "SLM_Results"):
        self.output_base_dir = output_base_dir
        self.token_counter = TokenCounter()
    
    def get_result_path(self, model_name: str, dataset: str, method: str) -> str:
        """Get result directory path: {output_base_dir}/model/dataset/method."""
        return os.path.join(self.output_base_dir, model_name, dataset, method)
    
    def ensure_directory(self, path: str):
        """Ensure directory exists."""
        os.makedirs(path, exist_ok=True)
    
    def log_question_result(self, result: Dict[str, Any], model_name: str, dataset: str, method: str):
        """Log a single question result with token tracking."""
        # Add token information to result
        if "prompt" in result and "full_response" in result:
            token_data = self.token_counter.log_question_tokens(
                result["prompt"],
                result["full_response"],
                result.get("question_index", 0)
            )
            result["token_usage"] = token_data
        
        # Get result directory
        result_dir = self.get_result_path(model_name, dataset, method)
        self.ensure_directory(result_dir)
        
        # Save individual question result
        question_file = os.path.join(result_dir, f"question_{result.get('question_index', 0)}.json")
        with open(question_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def save_run_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any],
                        model_name: str, dataset: str, method: str, chat_instance_type: str, random_seed: int = 42):
        """Save complete run results with enhanced summary - INCLUDES SEED IN FILENAME."""
        result_dir = self.get_result_path(model_name, dataset, method)
        self.ensure_directory(result_dir)

        # Add token summary to main summary
        token_summary = self.token_counter.get_summary()
        summary["token_usage"] = token_summary
        summary["random_seed"] = random_seed  # CRITICAL: Store seed in summary

        # Create detailed results
        detailed_results = {
            "summary": summary,
            "results": results,
            "token_details": self.token_counter.question_tokens,
            "configuration": {
                "model_name": model_name,
                "dataset": dataset,
                "method": method,
                "chat_instance_type": chat_instance_type,
                "random_seed": random_seed,
                "timestamp": datetime.now().isoformat()
            }
        }

        # Save detailed results with SEED in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(result_dir, f"seed_{random_seed}_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)

        # Save summary with SEED in filename
        summary_file = os.path.join(result_dir, f"seed_{random_seed}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Reset token counter for next run
        self.token_counter.reset()

        logging.info(f"Results saved to: {results_file}")
        logging.info(f"Summary saved to: {summary_file}")
        return results_file, summary_file
    
    def aggregate_method_results(self, model_name: str, dataset: str, method: str):
        """Aggregate all results for a specific method in a dataset across all seeds."""
        method_dir = self.get_result_path(model_name, dataset, method)

        if not os.path.exists(method_dir):
            return None

        # Find all seed summary files (seed_1_summary.json, seed_2_summary.json, etc.)
        summary_files = [f for f in os.listdir(method_dir) if f.startswith('seed_') and f.endswith('_summary.json')]
        
        if not summary_files:
            return None
        
        # Load and aggregate summaries
        aggregated_data = []
        total_questions = 0
        total_correct = 0
        total_time = 0
        total_tokens = 0
        
        for summary_file in summary_files:
            with open(os.path.join(method_dir, summary_file), 'r', encoding='utf-8') as f:
                summary = json.load(f)
                aggregated_data.append(summary)
                total_questions += summary.get('total_questions', 0)
                total_correct += summary.get('correct_answers', 0)
                total_time += summary.get('total_time', 0)
                if 'token_usage' in summary:
                    total_tokens += summary['token_usage'].get('total_tokens', 0)
        
        # Create method-level summary
        method_summary = {
            "model": model_name,
            "dataset": dataset,
            "method": method,
            "runs_aggregated": len(aggregated_data),
            "total_questions": total_questions,
            "total_correct_answers": total_correct,
            "overall_accuracy": total_correct / total_questions if total_questions > 0 else 0,
            "total_time": total_time,
            "avg_time_per_question": total_time / total_questions if total_questions > 0 else 0,
            "total_tokens": total_tokens,
            "avg_tokens_per_question": total_tokens / total_questions if total_questions > 0 else 0,
            "runs": aggregated_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save method summary
        method_summary_file = os.path.join(method_dir, f"method_summary_{method}.json")
        with open(method_summary_file, 'w', encoding='utf-8') as f:
            json.dump(method_summary, f, indent=2, ensure_ascii=False)
        
        return method_summary_file
    
    def aggregate_dataset_results(self, model_name: str, dataset: str):
        """Aggregate all results for all methods in a dataset."""
        dataset_dir = os.path.join(self.output_base_dir, model_name, dataset)
        
        if not os.path.exists(dataset_dir):
            return None
        
        methods = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        dataset_summary = {
            "model": model_name,
            "dataset": dataset,
            "methods": {},
            "overall_stats": {},
            "timestamp": datetime.now().isoformat()
        }
        
        total_questions_all_methods = 0
        total_correct_all_methods = 0
        total_time_all_methods = 0
        total_tokens_all_methods = 0
        
        for method in methods:
            method_summary_file = self.aggregate_method_results(model_name, dataset, method)
            if method_summary_file:
                with open(method_summary_file, 'r', encoding='utf-8') as f:
                    method_data = json.load(f)
                    dataset_summary["methods"][method] = {
                        "accuracy": method_data.get("overall_accuracy", 0),
                        "total_questions": method_data.get("total_questions", 0),
                        "correct_answers": method_data.get("total_correct_answers", 0),
                        "total_time": method_data.get("total_time", 0),
                        "total_tokens": method_data.get("total_tokens", 0),
                        "runs": method_data.get("runs_aggregated", 0)
                    }
                    
                    # Add to overall totals
                    total_questions_all_methods += method_data.get("total_questions", 0)
                    total_correct_all_methods += method_data.get("total_correct_answers", 0)
                    total_time_all_methods += method_data.get("total_time", 0)
                    total_tokens_all_methods += method_data.get("total_tokens", 0)
        
        dataset_summary["overall_stats"] = {
            "total_questions": total_questions_all_methods,
            "total_correct": total_correct_all_methods,
            "overall_accuracy": total_correct_all_methods / total_questions_all_methods if total_questions_all_methods > 0 else 0,
            "total_time": total_time_all_methods,
            "total_tokens": total_tokens_all_methods,
            "methods_tested": len(dataset_summary["methods"])
        }
        
        # Save dataset summary
        dataset_summary_file = os.path.join(dataset_dir, f"dataset_summary_{dataset}.json")
        with open(dataset_summary_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
        
        return dataset_summary_file
    
    def aggregate_model_results(self, model_name: str):
        """Aggregate all results for all datasets under a model."""
        model_dir = os.path.join(self.output_base_dir, model_name)
        
        if not os.path.exists(model_dir):
            return None
        
        datasets = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        model_summary = {
            "model": model_name,
            "datasets": {},
            "overall_stats": {},
            "timestamp": datetime.now().isoformat()
        }
        
        total_questions_all_datasets = 0
        total_correct_all_datasets = 0
        total_time_all_datasets = 0
        total_tokens_all_datasets = 0
        
        for dataset in datasets:
            dataset_summary_file = self.aggregate_dataset_results(model_name, dataset)
            if dataset_summary_file:
                with open(dataset_summary_file, 'r', encoding='utf-8') as f:
                    dataset_data = json.load(f)
                    model_summary["datasets"][dataset] = dataset_data["overall_stats"]
                    
                    # Add to overall totals
                    stats = dataset_data["overall_stats"]
                    total_questions_all_datasets += stats.get("total_questions", 0)
                    total_correct_all_datasets += stats.get("total_correct", 0)
                    total_time_all_datasets += stats.get("total_time", 0)
                    total_tokens_all_datasets += stats.get("total_tokens", 0)
        
        model_summary["overall_stats"] = {
            "total_questions": total_questions_all_datasets,
            "total_correct": total_correct_all_datasets,
            "overall_accuracy": total_correct_all_datasets / total_questions_all_datasets if total_questions_all_datasets > 0 else 0,
            "total_time": total_time_all_datasets,
            "total_tokens": total_tokens_all_datasets,
            "datasets_tested": len(model_summary["datasets"])
        }
        
        # Save model summary
        model_summary_file = os.path.join(model_dir, f"model_summary_{model_name}.json")
        with open(model_summary_file, 'w', encoding='utf-8') as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)
        
        return model_summary_file