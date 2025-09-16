"""
Updated Dataset Runner Module
Integrates the modular dataset loading and formatting architecture.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

# Import modular dataset components
from .dataset_loader import (
    DatasetLoader, VisionDatasetLoader,
    # Legacy function wrappers for backward compatibility
    load_medqa_dataset, load_medmcqa_dataset, load_mmlupro_med_dataset,
    load_pubmedqa_dataset, load_ddxplus_dataset, load_medbullets_dataset,
    load_pmc_vqa_dataset, load_path_vqa_dataset
)

from .dataset_formatters import (
    MedQAFormatter, MedMCQAFormatter, MMLUProMedFormatter,
    PubMedQAFormatter, DDXPlusFormatter, MedBulletsFormatter,
    # Legacy function wrappers for backward compatibility
    format_medqa_for_task, format_medmcqa_for_task, format_mmlupro_med_for_task,
    format_pubmedqa_for_task, format_ddxplus_for_task, format_medbullets_for_task
)

from .vision_dataset_formatters import (
    PMCVQAFormatter, PathVQAFormatter, VisionDatasetValidator,
    VisionDatasetImageHandler,
    # Enhanced formatting functions with image handling
    format_pmc_vqa_for_task, format_path_vqa_for_task,
    validate_vision_dataset_compatibility
)


class DatasetManager:
    """Centralized dataset management with validation and statistics."""
    
    SUPPORTED_DATASETS = {
        # Text-based medical datasets
        "medqa": {
            "loader": DatasetLoader.load_medqa,
            "formatter": MedQAFormatter.format,
            "type": "text",
            "domain": "medical",
            "format": "mcq"
        },
        "medmcqa": {
            "loader": DatasetLoader.load_medmcqa,
            "formatter": MedMCQAFormatter.format,
            "type": "text", 
            "domain": "medical",
            "format": "mcq",
            "validation_required": True
        },
        "mmlupro-med": {
            "loader": DatasetLoader.load_mmlupro_med,
            "formatter": MMLUProMedFormatter.format,
            "type": "text",
            "domain": "medical", 
            "format": "mcq"
        },
        "pubmedqa": {
            "loader": DatasetLoader.load_pubmedqa,
            "formatter": PubMedQAFormatter.format,
            "type": "text",
            "domain": "medical",
            "format": "yes_no_maybe"
        },
        "ddxplus": {
            "loader": DatasetLoader.load_ddxplus,
            "formatter": DDXPlusFormatter.format,
            "type": "text",
            "domain": "medical",
            "format": "mcq",
            "clinical_context": True
        },
        "medbullets": {
            "loader": DatasetLoader.load_medbullets,
            "formatter": MedBulletsFormatter.format,
            "type": "text",
            "domain": "medical",
            "format": "mcq",
            "usmle_context": True
        },
        
        # Vision-based medical datasets
        "pmc_vqa": {
            "loader": VisionDatasetLoader.load_pmc_vqa,
            "formatter": PMCVQAFormatter.format,
            "type": "vision",
            "domain": "medical",
            "format": "mcq",
            "requires_images": True
        },
        "path_vqa": {
            "loader": VisionDatasetLoader.load_path_vqa,
            "formatter": PathVQAFormatter.format,
            "type": "vision",
            "domain": "medical",
            "format": "mcq",
            "requires_images": True,
            "specialized_domain": "pathology"
        }
    }
    
    @classmethod
    def get_supported_datasets(cls) -> List[str]:
        """Get list of supported dataset names."""
        return list(cls.SUPPORTED_DATASETS.keys())
    
    @classmethod
    def get_text_datasets(cls) -> List[str]:
        """Get list of text-only datasets."""
        return [name for name, config in cls.SUPPORTED_DATASETS.items() 
                if config["type"] == "text"]
    
    @classmethod
    def get_vision_datasets(cls) -> List[str]:
        """Get list of vision-based datasets.""" 
        return [name for name, config in cls.SUPPORTED_DATASETS.items()
                if config["type"] == "vision"]
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        return cls.SUPPORTED_DATASETS[dataset_name].copy()
    
    @classmethod
    def is_vision_dataset(cls, dataset_name: str) -> bool:
        """Check if dataset requires vision capabilities."""
        config = cls.SUPPORTED_DATASETS.get(dataset_name, {})
        return config.get("type") == "vision"
    
    @classmethod
    def is_medical_dataset(cls, dataset_name: str) -> bool:
        """Check if dataset is medical-focused."""
        config = cls.SUPPORTED_DATASETS.get(dataset_name, {})
        return config.get("domain") == "medical"
    
    @classmethod
    def load_dataset(cls, dataset_name: str, num_questions: int = 50, 
                    random_seed: int = 42, **kwargs) -> List[Dict[str, Any]]:
        """Load dataset using appropriate loader."""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        config = cls.SUPPORTED_DATASETS[dataset_name]
        loader_func = config["loader"]
        
        logging.info(f"Loading {dataset_name} dataset using {loader_func.__name__}")
        
        try:
            if config.get("validation_required"):
                # For datasets that return validation results (like MedMCQA)
                questions, errors = loader_func(num_questions, random_seed, **kwargs)
                if errors:
                    logging.info(f"Dataset loading completed with {len(errors)} validation errors")
                return questions
            else:
                return loader_func(num_questions, random_seed, **kwargs)
                
        except Exception as e:
            logging.error(f"Error loading {dataset_name}: {e}")
            raise
    
    @classmethod
    def format_question(cls, dataset_name: str, question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format question using appropriate formatter."""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        config = cls.SUPPORTED_DATASETS[dataset_name]
        formatter_func = config["formatter"]
        
        try:
            if config.get("validation_required"):
                # For formatters that return validation status (like MedMCQA)
                agent_task, eval_data, is_valid = formatter_func(question_data)
                if not is_valid:
                    raise ValueError(f"Question validation failed: {eval_data}")
                return agent_task, eval_data
            else:
                return formatter_func(question_data)
                
        except Exception as e:
            logging.error(f"Error formatting {dataset_name} question: {e}")
            raise
    
    @classmethod
    def validate_dataset_compatibility(cls, dataset_name: str, 
                                     questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate dataset compatibility."""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            return {"error": f"Dataset {dataset_name} not supported"}
        
        config = cls.SUPPORTED_DATASETS[dataset_name]
        
        validation_results = {
            "dataset": dataset_name,
            "type": config["type"],
            "domain": config["domain"],
            "format": config["format"],
            "total_questions": len(questions),
            "compatibility_score": 1.0
        }
        
        # Vision-specific validation
        if config["type"] == "vision":
            vision_validation = validate_vision_dataset_compatibility(dataset_name, questions)
            validation_results.update(vision_validation)
        
        # Format-specific validation
        valid_questions = 0
        format_errors = []
        
        for i, question in enumerate(questions[:10]):  # Sample validation
            try:
                agent_task, eval_data = cls.format_question(dataset_name, question)
                valid_questions += 1
            except Exception as e:
                format_errors.append(f"Question {i}: {str(e)}")
        
        validation_results.update({
            "sample_questions_valid": valid_questions,
            "sample_questions_tested": min(10, len(questions)),
            "format_errors": format_errors[:5]  # Limit error list
        })
        
        # Update compatibility score
        if len(questions) > 0:
            sample_score = valid_questions / min(10, len(questions))
            if config["type"] == "vision":
                vision_score = validation_results.get("compatibility_score", 0.0)
                validation_results["compatibility_score"] = (sample_score + vision_score) / 2
            else:
                validation_results["compatibility_score"] = sample_score
        
        return validation_results


class DatasetStatistics:
    """Generate statistics and analysis for datasets."""
    
    @staticmethod
    def analyze_dataset(dataset_name: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset characteristics and provide statistics."""
        if not questions:
            return {"error": "No questions provided"}
        
        config = DatasetManager.get_dataset_info(dataset_name)
        
        stats = {
            "dataset_name": dataset_name,
            "dataset_type": config["type"],
            "dataset_domain": config["domain"],
            "total_questions": len(questions),
            "format_type": config["format"],
            "analysis_timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Analyze question characteristics
        question_lengths = []
        choice_counts = []
        
        for question in questions:
            # Question text analysis
            if isinstance(question.get("question"), str):
                question_lengths.append(len(question["question"].split()))
            elif isinstance(question.get("Question"), str):  # PMC-VQA format
                question_lengths.append(len(question["Question"].split()))
            
            # Choice analysis
            if "choices" in question:
                choice_counts.append(len(question["choices"]))
            elif "choicesA" in question:  # MedBullets format
                choices = sum(1 for key in ["choicesA", "choicesB", "choicesC", "choicesD", "choicesE"]
                            if question.get(key, "").strip())
                choice_counts.append(choices)
        
        if question_lengths:
            stats["question_analysis"] = {
                "avg_words": sum(question_lengths) / len(question_lengths),
                "min_words": min(question_lengths),
                "max_words": max(question_lengths),
                "word_distribution": {
                    "short": sum(1 for length in question_lengths if length < 20),
                    "medium": sum(1 for length in question_lengths if 20 <= length < 100),
                    "long": sum(1 for length in question_lengths if length >= 100)
                }
            }
        
        if choice_counts:
            stats["choice_analysis"] = {
                "avg_choices": sum(choice_counts) / len(choice_counts),
                "choice_distribution": {
                    str(i): choice_counts.count(i) for i in set(choice_counts)
                }
            }
        
        # Vision-specific analysis
        if config["type"] == "vision":
            vision_stats = DatasetStatistics._analyze_vision_dataset(questions)
            stats.update(vision_stats)
        
        return stats
    
    @staticmethod
    def _analyze_vision_dataset(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze vision-specific dataset characteristics."""
        vision_stats = {
            "vision_analysis": {
                "images_present": 0,
                "images_valid": 0,
                "images_invalid": 0,
                "image_sizes": []
            }
        }
        
        for question in questions:
            image = question.get("image")
            if image is not None:
                vision_stats["vision_analysis"]["images_present"] += 1
                
                if VisionDatasetValidator.validate_image_for_vision_api(image):
                    vision_stats["vision_analysis"]["images_valid"] += 1
                    try:
                        width, height = image.size
                        vision_stats["vision_analysis"]["image_sizes"].append((width, height))
                    except:
                        pass
                else:
                    vision_stats["vision_analysis"]["images_invalid"] += 1
        
        # Calculate image size statistics
        if vision_stats["vision_analysis"]["image_sizes"]:
            sizes = vision_stats["vision_analysis"]["image_sizes"]
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            
            vision_stats["vision_analysis"]["size_statistics"] = {
                "avg_width": sum(widths) / len(widths),
                "avg_height": sum(heights) / len(heights),
                "min_width": min(widths),
                "max_width": max(widths),
                "min_height": min(heights),
                "max_height": max(heights)
            }
        
        return vision_stats


# Legacy compatibility layer - maintain existing function signatures
def get_supported_datasets():
    """Legacy function for backward compatibility."""
    return DatasetManager.get_supported_datasets()

def is_vision_dataset(dataset_name: str) -> bool:
    """Legacy function for backward compatibility."""
    return DatasetManager.is_vision_dataset(dataset_name)

def is_medical_dataset(dataset_name: str) -> bool:
    """Legacy function for backward compatibility."""
    return DatasetManager.is_medical_dataset(dataset_name)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Dataset Runner - Modular Architecture")
    print("=" * 50)
    
    print(f"Supported datasets: {DatasetManager.get_supported_datasets()}")
    print(f"Text datasets: {DatasetManager.get_text_datasets()}")
    print(f"Vision datasets: {DatasetManager.get_vision_datasets()}")
    
    # Example dataset info
    for dataset in DatasetManager.get_supported_datasets()[:3]:
        info = DatasetManager.get_dataset_info(dataset)
        print(f"\n{dataset}: {info}")
    
    print("\nModular dataset architecture ready!")