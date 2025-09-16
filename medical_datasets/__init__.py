"""
Medical Dataset Processing Module
Modular architecture for loading and formatting medical datasets.
"""

# Import legacy compatibility functions to maintain existing imports
from .dataset_runner import (
    # Legacy dataset loading functions
    load_medqa_dataset,
    load_medmcqa_dataset, 
    load_mmlupro_med_dataset,
    load_pubmedqa_dataset,
    load_ddxplus_dataset,
    load_medbullets_dataset,
    load_pmc_vqa_dataset,
    load_path_vqa_dataset,
    
    # Legacy dataset formatting functions
    format_medqa_for_task,
    format_medmcqa_for_task,
    format_mmlupro_med_for_task,
    format_pubmedqa_for_task,
    format_ddxplus_for_task,
    format_medbullets_for_task,
    format_pmc_vqa_for_task,
    format_path_vqa_for_task,
    
    # Legacy utility functions
    get_supported_datasets,
    is_vision_dataset,
    is_medical_dataset,
    
    # New modular architecture classes
    DatasetManager,
    DatasetStatistics
)

# Import new modular components for advanced usage
from .dataset_loader import DatasetLoader, VisionDatasetLoader
from .dataset_formatters import (
    MedQAFormatter, MedMCQAFormatter, MMLUProMedFormatter,
    PubMedQAFormatter, DDXPlusFormatter, MedBulletsFormatter
)
from .vision_dataset_formatters import (
    PMCVQAFormatter, PathVQAFormatter, VisionDatasetValidator,
    VisionDatasetImageHandler, validate_vision_dataset_compatibility
)

__version__ = "2.0.0"
__author__ = "Medical AI Research Team"

__all__ = [
    # Legacy functions for backward compatibility
    'load_medqa_dataset', 'load_medmcqa_dataset', 'load_mmlupro_med_dataset',
    'load_pubmedqa_dataset', 'load_ddxplus_dataset', 'load_medbullets_dataset',
    'load_pmc_vqa_dataset', 'load_path_vqa_dataset',
    'format_medqa_for_task', 'format_medmcqa_for_task', 'format_mmlupro_med_for_task',
    'format_pubmedqa_for_task', 'format_ddxplus_for_task', 'format_medbullets_for_task',
    'format_pmc_vqa_for_task', 'format_path_vqa_for_task',
    'get_supported_datasets', 'is_vision_dataset', 'is_medical_dataset',
    
    # New modular architecture
    'DatasetManager', 'DatasetStatistics',
    'DatasetLoader', 'VisionDatasetLoader',
    'MedQAFormatter', 'MedMCQAFormatter', 'MMLUProMedFormatter',
    'PubMedQAFormatter', 'DDXPlusFormatter', 'MedBulletsFormatter',
    'PMCVQAFormatter', 'PathVQAFormatter', 'VisionDatasetValidator',
    'VisionDatasetImageHandler', 'validate_vision_dataset_compatibility'
]

# Convenience imports for quick access
SUPPORTED_DATASETS = DatasetManager.get_supported_datasets()
TEXT_DATASETS = DatasetManager.get_text_datasets()
VISION_DATASETS = DatasetManager.get_vision_datasets()

def get_dataset_info(dataset_name: str):
    """Quick access to dataset information."""
    return DatasetManager.get_dataset_info(dataset_name)

def load_and_format_dataset(dataset_name: str, num_questions: int = 50, 
                           random_seed: int = 42, **kwargs):
    """Convenience function to load and get formatting info for a dataset."""
    questions = DatasetManager.load_dataset(dataset_name, num_questions, random_seed, **kwargs)
    dataset_info = DatasetManager.get_dataset_info(dataset_name)
    
    return {
        "questions": questions,
        "dataset_info": dataset_info,
        "total_loaded": len(questions),
        "formatter": dataset_info["formatter"]
    }