"""
Medical Dataset Loader Module
Handles loading of various medical datasets for SLM evaluation.
"""

import logging
import random
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

try:
    from datasets import load_dataset
    import pandas as pd
    from PIL import Image
    DATASETS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dataset dependencies not available: {e}")
    DATASETS_AVAILABLE = False


class DatasetLoader:
    """Centralized dataset loading with validation."""
    
    @staticmethod
    def validate_dependencies():
        """Check if required dependencies are available."""
        if not DATASETS_AVAILABLE:
            raise ImportError("Required packages not installed. Run: pip install datasets pandas pillow")
    
    @staticmethod
    def load_medqa(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
        """Load MedQA dataset."""
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading MedQA dataset with {num_questions} random questions")
        
        try:
            ds = load_dataset("sickgpt/001_MedQA_raw")
            questions = list(ds["train"])
            
            random.seed(random_seed)
            if num_questions < len(questions):
                selected_questions = random.sample(questions, num_questions)
            else:
                selected_questions = questions
                logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
            
            logging.info(f"Successfully loaded {len(selected_questions)} questions from MedQA dataset")
            return selected_questions
        
        except Exception as e:
            logging.error(f"Error loading MedQA dataset: {str(e)}")
            return []

    @staticmethod
    def load_medmcqa(num_questions: int = 50, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load MedMCQA dataset with validation."""
        from .dataset_formatters import MedMCQAFormatter
        
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading MedMCQA dataset with {num_questions} random questions")
        
        valid_questions = []
        errors = []
        
        try:
            ds = load_dataset("openlifescienceai/medmcqa")
            questions = list(ds["train"])
            
            random.seed(random_seed)
            if num_questions < len(questions):
                selected_questions = random.sample(questions, num_questions * 2)  # Get extra for validation
            else:
                selected_questions = questions
            
            for question_data in selected_questions:
                try:
                    agent_task, eval_data, is_valid = MedMCQAFormatter.format(question_data)
                    
                    if is_valid:
                        valid_questions.append(question_data)
                        if len(valid_questions) >= num_questions:
                            break
                    else:
                        errors.append(eval_data)
                        logging.warning(f"Skipped question {eval_data.get('question_id', 'unknown')}")
                        
                except Exception as e:
                    error_info = {
                        "question_id": question_data.get("id", "unknown"),
                        "error_type": "formatting_error", 
                        "message": f"Error formatting question: {str(e)}"
                    }
                    errors.append(error_info)
            
            logging.info(f"Successfully loaded {len(valid_questions)} valid questions from MedMCQA dataset")
            logging.info(f"Skipped {len(errors)} questions due to validation errors")
            
            return valid_questions[:num_questions], errors
        
        except Exception as e:
            logging.error(f"Error loading MedMCQA dataset: {str(e)}")
            return [], [{"error_type": "dataset_loading_error", "message": str(e)}]

    @staticmethod
    def load_mmlupro_med(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
        """Load MMLU-Pro Health dataset."""
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading MMLU-Pro Health dataset with {num_questions} random questions")
        
        try:
            ds = load_dataset("TIGER-Lab/MMLU-Pro")
            available_splits = list(ds.keys())
            logging.info(f"Available splits: {available_splits}")
            
            health_questions = []
            for split in available_splits:
                logging.info(f"Processing split: {split} with {len(ds[split])} questions")
                
                for item in ds[split]:
                    category = item.get("category", "").lower()
                    if "health" in category:
                        health_questions.append(item)
                        if len(health_questions) % 50 == 0:
                            logging.info(f"Found {len(health_questions)} health questions so far...")
            
            logging.info(f"Found {len(health_questions)} health questions total")
            
            if not health_questions:
                logging.error("No health questions found in MMLU-Pro dataset")
                return []
            
            random.seed(random_seed)
            if num_questions < len(health_questions):
                selected_questions = random.sample(health_questions, num_questions)
            else:
                selected_questions = health_questions
                logging.warning(f"Requested {num_questions} questions but found only {len(health_questions)} health questions. Using all available.")
            
            logging.info(f"Successfully loaded {len(selected_questions)} health questions from MMLU-Pro dataset")
            return selected_questions
        
        except Exception as e:
            logging.error(f"Error loading MMLU-Pro Health dataset: {str(e)}")
            return []

    @staticmethod
    def load_pubmedqa(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
        """Load PubMedQA dataset."""
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading PubMedQA dataset with {num_questions} random questions")
        
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
            questions = list(ds["train"])
            
            random.seed(random_seed)
            if num_questions < len(questions):
                selected_questions = random.sample(questions, num_questions)
            else:
                selected_questions = questions
                logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
            
            logging.info(f"Successfully loaded {len(selected_questions)} questions from PubMedQA dataset")
            return selected_questions
        
        except Exception as e:
            logging.error(f"Error loading PubMedQA dataset: {str(e)}")
            return []

    @staticmethod
    def load_ddxplus(num_questions: int = 50, random_seed: int = 42, 
                     dataset_split: str = "train") -> List[Dict[str, Any]]:
        """Load DDXPlus dataset."""
        from .dataset_formatters import DDXPlusFormatter
        
        logging.info(f"Loading DDXPlus dataset with {num_questions} random questions from {dataset_split} split")
        
        try:
            dataset_dir = Path("dataset/ddx")
            if not dataset_dir.exists():
                logging.error(f"DDXPlus dataset directory not found: {dataset_dir}")
                return []
            
            # Load metadata
            evidences_file = dataset_dir / "release_evidences.json"
            conditions_file = dataset_dir / "release_conditions.json"
            
            if not evidences_file.exists() or not conditions_file.exists():
                logging.error(f"Missing DDXPlus metadata files")
                return []
            
            with open(evidences_file, 'r', encoding='utf-8') as f:
                evidences = json.load(f)
            with open(conditions_file, 'r', encoding='utf-8') as f:
                conditions = json.load(f)
                
            logging.info(f"Loaded {len(evidences)} evidences and {len(conditions)} conditions")
            
            # Load patient data
            csv_mapping = {
                "train": "release_train_patients.csv",
                "validate": "release_validate_patients.csv", 
                "test": "release_test_patients.csv"
            }
            
            if dataset_split not in csv_mapping:
                logging.error(f"Invalid dataset split: {dataset_split}")
                return []
            
            patients_file = dataset_dir / csv_mapping[dataset_split]
            if not patients_file.exists():
                logging.error(f"DDXPlus patients CSV file not found: {patients_file}")
                return []
            
            df = pd.read_csv(patients_file)
            patients = df.to_dict('records')
            logging.info(f"Successfully loaded CSV with {len(patients)} patients")
            
            random.seed(random_seed)
            if num_questions < len(patients):
                selected_patients = random.sample(patients, num_questions)
            else:
                selected_patients = patients
                logging.warning(f"Requested {num_questions} questions but dataset only has {len(patients)} patients. Using all available.")
            
            # Convert patients to questions
            questions = []
            for i, patient in enumerate(selected_patients):
                try:
                    question_data = DDXPlusFormatter.convert_patient_to_question(patient, evidences, conditions, i)
                    if question_data:
                        questions.append(question_data)
                except Exception as e:
                    logging.error(f"Error converting DDXPlus patient {i} to question: {str(e)}")
                    continue
            
            logging.info(f"Successfully converted {len(questions)} DDXPlus patients to questions")
            return questions
        
        except Exception as e:
            logging.error(f"Error loading DDXPlus dataset: {str(e)}")
            return []

    @staticmethod
    def load_medbullets(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
        """Load MedBullets dataset."""
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading MedBullets dataset with {num_questions} random questions")
        
        try:
            ds = load_dataset("JesseLiu/medbulltes5op")
            available_splits = list(ds.keys())
            logging.info(f"MedBullets available splits: {available_splits}")
            
            if "test" in available_splits:
                questions = list(ds["test"])
                logging.info(f"Using 'test' split with {len(questions)} questions")
            elif "validation" in available_splits:
                questions = list(ds["validation"])
                logging.info(f"Using 'validation' split with {len(questions)} questions")
            elif available_splits:
                split_name = available_splits[0]
                questions = list(ds[split_name])
                logging.info(f"Using '{split_name}' split with {len(questions)} questions")
            else:
                logging.error("No splits found in MedBullets dataset")
                return []
            
            random.seed(random_seed)
            if num_questions < len(questions):
                selected_questions = random.sample(questions, num_questions)
            else:
                selected_questions = questions
                logging.warning(f"Requested {num_questions} questions but dataset only has {len(questions)}. Using all available questions.")
            
            logging.info(f"Successfully loaded {len(selected_questions)} questions from MedBullets dataset")
            return selected_questions
        
        except Exception as e:
            logging.error(f"Error loading MedBullets dataset: {str(e)}")
            return []


class VisionDatasetLoader:
    """Specialized loader for vision-based medical datasets."""
    
    @staticmethod
    def validate_image(image) -> bool:
        """Validate image compatibility with vision API."""
        if image is None:
            return False
        
        try:
            if not hasattr(image, 'size') or not hasattr(image, 'mode'):
                return False
            
            width, height = image.size
            if width < 10 or height < 10 or width > 4096 or height > 4096:
                return False
            
            if image.mode not in ('RGB', 'L', 'RGBA'):
                try:
                    test_img = image.convert('RGB')
                except:
                    return False
            
            return True
        except Exception:
            return False

    @staticmethod
    def load_pmc_vqa(num_questions: int = 50, random_seed: int = 42, 
                     dataset_split: str = "test") -> List[Dict[str, Any]]:
        """Load PMC-VQA dataset with image validation."""
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading PMC-VQA dataset with {num_questions} questions from {dataset_split} split")
        
        try:
            ds = load_dataset("hamzamooraj99/PMC-VQA-1", streaming=True)
            available_splits = list(ds.keys())
            logging.info(f"PMC-VQA available splits: {available_splits}")
            
            if dataset_split not in available_splits:
                dataset_split = available_splits[0] if available_splits else "train"
                logging.warning(f"Requested split not found, using {dataset_split}")
            
            questions = []
            random.seed(random_seed)
            
            attempted = 0
            max_attempts = num_questions * 10
            
            for question in ds[dataset_split]:
                attempted += 1
                
                try:
                    img = question.get('image')
                    if not VisionDatasetLoader.validate_image(img):
                        continue
                    
                    question_text = question.get("Question", "").strip()
                    if not question_text:
                        continue
                    
                    has_choices = any(question.get(f"Choice {chr(65+i)}", "").strip() 
                                    for i in range(4))
                    if not has_choices:
                        continue
                    
                    questions.append(question)
                    
                    if len(questions) >= num_questions:
                        break
                        
                except Exception as e:
                    logging.debug(f"Skipped PMC-VQA question due to error: {e}")
                    continue
                
                if attempted >= max_attempts:
                    logging.warning(f"Reached max attempts ({max_attempts}), stopping with {len(questions)} questions")
                    break
            
            logging.info(f"Successfully loaded {len(questions)} PMC-VQA questions with valid images")
            return questions[:num_questions]
            
        except Exception as e:
            logging.error(f"Error loading PMC-VQA dataset: {str(e)}")
            return []

    @staticmethod
    def load_path_vqa(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
        """Load Path-VQA dataset with strict yes/no filtering."""
        DatasetLoader.validate_dependencies()
        logging.info(f"Loading Path-VQA dataset - requesting {num_questions} yes/no questions")
        
        try:
            ds = load_dataset("flaviagiammarino/path-vqa", streaming=True)
            split_name = list(ds.keys())[0]
            
            # Phase 1: Collect yes/no candidates
            candidate_pool = []
            pool_target = num_questions * 3
            attempted = 0
            max_search_limit = num_questions * 10
            
            logging.info(f"Phase 1: Collecting {pool_target} yes/no candidate questions...")
            
            for question in ds[split_name]:
                attempted += 1
                
                try:
                    answer = question.get('answer', '').lower().strip()
                    if answer not in ['yes', 'no']:
                        continue
                        
                    question_text = question.get('question', '').strip()
                    if not question_text:
                        continue
                        
                    img = question.get('image')
                    if img is None:
                        continue
                    
                    candidate_pool.append(question)
                    
                    if len(candidate_pool) >= pool_target:
                        break
                        
                except Exception as e:
                    logging.debug(f"Skipped candidate due to error: {e}")
                    continue
                
                if attempted >= max_search_limit:
                    break
            
            logging.info(f"Phase 1 complete: Found {len(candidate_pool)} yes/no candidates")
            
            if len(candidate_pool) < num_questions:
                logging.error(f"INSUFFICIENT YES/NO QUESTIONS: Only {len(candidate_pool)} found, need {num_questions}")
                return candidate_pool
            
            # Phase 2: Final validation
            random.seed(random_seed)
            random.shuffle(candidate_pool)
            
            final_questions = []
            for question in candidate_pool:
                try:
                    answer = question.get('answer', '').lower().strip()
                    if answer not in ['yes', 'no']:
                        continue
                    
                    img = question.get('image')
                    if not VisionDatasetLoader.validate_image(img):
                        continue
                    
                    question_text = question.get('question', '').strip()
                    if not question_text or len(question_text) < 10:
                        continue
                    
                    final_questions.append(question)
                    
                    if len(final_questions) >= num_questions:
                        break
                        
                except Exception as e:
                    logging.debug(f"Validation error: {e}")
                    continue
            
            # Final verification
            verified_questions = []
            for q in final_questions:
                ans = q.get('answer', '').lower().strip()
                if ans in ['yes', 'no']:
                    verified_questions.append(q)
            
            logging.info(f"SUCCESS: Loaded {len(verified_questions)} PURE yes/no Path-VQA questions")
            return verified_questions[:num_questions]
            
        except Exception as e:
            logging.error(f"Error loading Path-VQA dataset: {str(e)}")
            return []


# Legacy function wrappers for backward compatibility
def load_medqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    return DatasetLoader.load_medqa(num_questions, random_seed)

def load_medmcqa_dataset(num_questions: int = 50, random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return DatasetLoader.load_medmcqa(num_questions, random_seed)

def load_mmlupro_med_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    return DatasetLoader.load_mmlupro_med(num_questions, random_seed)

def load_pubmedqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    return DatasetLoader.load_pubmedqa(num_questions, random_seed)

def load_ddxplus_dataset(num_questions: int = 50, random_seed: int = 42, dataset_split: str = "train") -> List[Dict[str, Any]]:
    return DatasetLoader.load_ddxplus(num_questions, random_seed, dataset_split)

def load_medbullets_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    return DatasetLoader.load_medbullets(num_questions, random_seed)

def load_pmc_vqa_dataset(num_questions: int = 50, random_seed: int = 42, dataset_split: str = "test") -> List[Dict[str, Any]]:
    return VisionDatasetLoader.load_pmc_vqa(num_questions, random_seed, dataset_split)

def load_path_vqa_dataset(num_questions: int = 50, random_seed: int = 42) -> List[Dict[str, Any]]:
    return VisionDatasetLoader.load_path_vqa(num_questions, random_seed)