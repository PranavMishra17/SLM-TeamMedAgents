"""
Updated SLM Configuration with modular chat instance support.
Supports Google AI Studio, Hugging Face, and extensible architecture.
"""

import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()

# Default instance type for chat
DEFAULT_CHAT_INSTANCE = "google_ai_studio"  # Options: "google_ai_studio", "huggingface"

# SLM Model Configurations with multi-provider support
SLM_MODELS = {
    "gemma3_4b": {
        "name": "gemma3_4b",
        "display_name": "Gemma3-4B-IT",
        "folder_name": "gemma3_4b",
        "max_tokens": 8192,
        "temperature": 0.3,
        "supports_vision": True,
        "model_family": "gemma3",
        "specialized_domain": "general",
        
        # Provider-specific configurations
        "google_ai_studio": {
            "model": "gemma-3-4b-it",
            "type": "google_ai_studio"
        },
        "huggingface": {
            "model": "google/gemma-3-4b-it",
            "hf_model_name": "google/gemma-3-4b-it",
            "type": "huggingface"
        }
    },
    "medgemma_4b": {
        "name": "medgemma_4b",
        "display_name": "MedGemma-4B-IT",
        "folder_name": "medgemma_4b",
        "max_tokens": 8192,
        "temperature": 0.3,
        "supports_vision": True,
        "model_family": "medgemma",
        "specialized_domain": "medical",
        
        # Provider-specific configurations
        "google_ai_studio": {
            "model": "medgemma-4b-it",
            "type": "google_ai_studio"
        },
        "huggingface": {
            "model": "google/medgemma-4b-it",
            "hf_model_name": "google/medgemma-4b-it", 
            "type": "huggingface"
        }
    }
}

# Default model selection
DEFAULT_MODEL = "gemma3_4b"

# SLM Prompting Methods Configuration (unchanged)
PROMPTING_METHODS = {
    "zero_shot": {
        "name": "Zero-Shot",
        "description": "Direct question answering without examples",
        "requires_examples": False,
        "output_dir": "zero-shot"
    },
    "few_shot": {
        "name": "Few-Shot",
        "description": "Question answering with 2-3 examples", 
        "requires_examples": True,
        "num_examples": 3,
        "output_dir": "few-shot"
    },
    "cot": {
        "name": "Chain-of-Thought",
        "description": "Step-by-step reasoning approach",
        "requires_examples": False,
        "output_dir": "cot"
    }
}

# Output and logging configuration (unchanged)
OUTPUT_BASE_DIR = "SLM_Results"
LOG_DIR = "logs/slm"

# Dataset configurations (unchanged)
SUPPORTED_DATASETS = {
    "medqa": {
        "name": "MedQA",
        "format_function": "format_medqa_for_task",
        "load_function": "load_medqa_dataset",
        "supports_vision": False,
        "is_medical": True
    },
    "medmcqa": {
        "name": "MedMCQA", 
        "format_function": "format_medmcqa_for_task",
        "load_function": "load_medmcqa_dataset",
        "supports_vision": False,
        "is_medical": True
    },
    "mmlupro-med": {
        "name": "MMLU-Pro Medical",
        "format_function": "format_mmlupro_med_for_task", 
        "load_function": "load_mmlupro_med_dataset",
        "supports_vision": False,
        "is_medical": True
    },
    "pubmedqa": {
        "name": "PubMedQA",
        "format_function": "format_pubmedqa_for_task",
        "load_function": "load_pubmedqa_dataset", 
        "supports_vision": False,
        "is_medical": True
    },
    "ddxplus": {
        "name": "DDXPlus",
        "format_function": "format_ddxplus_for_task",
        "load_function": "load_ddxplus_dataset",
        "supports_vision": False,
        "is_medical": True
    },
    "medbullets": {
        "name": "MedBullets",
        "format_function": "format_medbullets_for_task",
        "load_function": "load_medbullets_dataset",
        "supports_vision": False,
        "is_medical": True
    },
    "pmc_vqa": {
        "name": "PMC-VQA",
        "format_function": "format_pmc_vqa_for_task", 
        "load_function": "load_pmc_vqa_dataset",
        "supports_vision": True,
        "is_medical": True,
        "requires_images": True
    },
    "path_vqa": {
        "name": "Path-VQA",
        "format_function": "format_path_vqa_for_task",
        "load_function": "load_path_vqa_dataset", 
        "supports_vision": True,
        "is_medical": True,
        "requires_images": True
    }
}

# Few-shot examples (unchanged from original)
FEW_SHOT_EXAMPLES = {
    "medqa": [
        {
            "question": "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?",
            "options": ["A. Left anterior descending", "B. Right coronary artery", "C. Left circumflex", "D. Left main"],
            "answer": "B",
            "reasoning": "ST elevation in leads II, III, and aVF indicates inferior wall MI, which is typically caused by right coronary artery occlusion."
        },
        {
            "question": "A 28-year-old woman presents with fatigue and pallor. Lab results show Hb 8.2 g/dL, MCV 68 fL, and low ferritin. What is the most likely diagnosis?",
            "options": ["A. Iron deficiency anemia", "B. Thalassemia", "C. Chronic disease anemia", "D. B12 deficiency"],
            "answer": "A", 
            "reasoning": "Low hemoglobin, low MCV (microcytic), and low ferritin are classic for iron deficiency anemia."
        }
    ],
    "pmc_vqa": [
        {
            "question": "What type of medical imaging technique is shown in this image?",
            "options": ["A. X-ray", "B. MRI", "C. CT scan", "D. Ultrasound"],
            "answer": "B",
            "reasoning": "The image characteristics including tissue contrast and cross-sectional view indicate this is an MRI scan."
        }
    ],
    "path_vqa": [
        {
            "question": "Are there signs of inflammation visible in this pathology image?",
            "answer": "yes",
            "reasoning": "The tissue shows characteristic inflammatory changes including increased cellularity and tissue architecture disruption typical of inflammatory processes."
        }
    ],
    "general": [
        {
            "question": "What is the capital of France?",
            "options": ["A. London", "B. Berlin", "C. Paris", "D. Madrid"],
            "answer": "C",
            "reasoning": "Paris is the capital and largest city of France."
        }
    ]
}

# Chain-of-Thought prompting templates with actual reasoning examples
COT_TEMPLATES = {
    "medical_mcq": """
Let me think through this step by step:

1. **Analyze the clinical presentation**: What are the key symptoms and findings?
2. **Consider the pathophysiology**: What underlying processes could cause these findings?
3. **Evaluate each option**: How well does each choice fit the clinical picture?
4. **Apply medical knowledge**: What does current medical understanding suggest?
5. **Make the diagnosis**: Which option is most consistent with the evidence?

Here's an example of how to work through a medical question systematically:

Example: "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?"

Step 1 - Clinical presentation: 65-year-old male, chest pain + shortness of breath = classic MI symptoms
Step 2 - ECG findings: ST elevation in leads II, III, aVF = these are inferior leads
Step 3 - Anatomical correlation: Inferior wall MI is typically supplied by the right coronary artery (RCA)
Step 4 - Rule out alternatives: LAD supplies anterior wall (V1-V6), LCX supplies lateral (I, aVL, V5-V6)
Step 5 - Conclusion: RCA occlusion is most likely

Now apply this same systematic approach to the current question:
""",
    "general_mcq": """
Let me approach this systematically:

1. **Understand the question**: What is being asked?
2. **Analyze the options**: What are the key differences between choices?
3. **Apply relevant knowledge**: What facts or principles apply here?
4. **Eliminate incorrect options**: Which choices can be ruled out and why?
5. **Select the best answer**: Which option is most accurate?

Here's an example of systematic reasoning:

Example: "Which gas makes up the largest percentage of Earth's atmosphere?"
A. Oxygen (21%)  B. Nitrogen (78%)  C. Carbon dioxide (0.04%)  D. Argon (0.93%)

Step 1 - Question asks: Largest percentage gas in atmosphere
Step 2 - Compare percentages: Nitrogen 78% > Oxygen 21% > Argon 0.93% > CO2 0.04%
Step 3 - Apply knowledge: Atmosphere composition is well-established scientific fact
Step 4 - Eliminate: A (oxygen is second), C (CO2 is trace), D (argon is small)
Step 5 - Select: B (nitrogen) has the highest percentage at 78%

Now apply this reasoning process to the current question:
"""
}

# Logging and request configurations (unchanged)
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "file_handler": True,
    "console_handler": True
}

REQUEST_SETTINGS = {
    "timeout": 60,
    "max_retries": 3,
    "retry_delay": 2,
    "backoff_factor": 1.5
}

TOKEN_LIMITS = {
    "max_input_tokens": 16000,
    "max_output_tokens": 8192,
    "budget_per_question": 25000
}

# Rate limits for different models
RATE_LIMITS = {
    "gemma3_4b": {
        "rpm": 30,      # Requests per minute
        "tpm": 15000,   # Tokens per minute
        "rpd": 14400,   # Requests per day
        "model_family": "gemma3"
    },
    "medgemma_4b": {
        "rpm": 30,      # Same limits as gemma3
        "tpm": 15000,   
        "rpd": 14400,
        "model_family": "gemma3"
    }
}

# Retry configuration
RETRY_CONFIG = {
    "max_retries": 5,
    "initial_delay": 1.0,        # Initial delay in seconds
    "max_delay": 60.0,          # Maximum delay in seconds
    "exponential_base": 2.0,     # Exponential backoff base
    "jitter": True,             # Add random jitter to prevent thundering herd
    "rate_limit_delay": 65.0    # Fixed delay for rate limit errors (65 seconds)
}

def get_model_config(model_name: str = None, chat_instance_type: str = None) -> Dict[str, Any]:
    """Get configuration for specified model and chat instance type."""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if chat_instance_type is None:
        chat_instance_type = DEFAULT_CHAT_INSTANCE
    
    if model_name not in SLM_MODELS:
        raise ValueError(f"Model {model_name} not found. Available: {list(SLM_MODELS.keys())}")
    
    base_config = SLM_MODELS[model_name].copy()
    
    # Get provider-specific config
    if chat_instance_type in base_config:
        provider_config = base_config[chat_instance_type]
        # Merge provider-specific config with base config
        base_config.update(provider_config)
    
    return base_config

def get_output_dir(method: str, dataset: str, model_name: str = None, output_base_dir: str = None) -> str:
    """Get output directory path for method, dataset, and model in format: {base_dir}/model/dataset/method."""
    if method not in PROMPTING_METHODS:
        raise ValueError(f"Method {method} not supported. Available: {list(PROMPTING_METHODS.keys())}")
    
    base_dir = output_base_dir or OUTPUT_BASE_DIR
    method_dir = PROMPTING_METHODS[method]["output_dir"]
    
    if model_name:
        # New structure: {base_dir}/model/dataset/method
        model_config = get_model_config(model_name)
        model_folder = model_config["folder_name"]
        return os.path.join(base_dir, model_folder, dataset, method_dir)
    else:
        return os.path.join(base_dir, dataset, method_dir)

def get_supported_datasets() -> List[str]:
    """Get list of supported dataset names."""
    return list(SUPPORTED_DATASETS.keys())

def get_supported_models() -> List[str]:
    """Get list of supported model names."""
    return list(SLM_MODELS.keys())

def get_supported_chat_instances() -> List[str]:
    """Get list of supported chat instance types."""
    return ["google_ai_studio", "huggingface"]

def is_medical_dataset(dataset_name: str) -> bool:
    """Check if dataset is medical-focused."""
    if dataset_name not in SUPPORTED_DATASETS:
        return False
    return SUPPORTED_DATASETS[dataset_name].get("is_medical", False)

def is_vision_model(model_name: str) -> bool:
    """Check if model supports vision."""
    if model_name not in SLM_MODELS:
        return False
    return SLM_MODELS[model_name].get("supports_vision", False)

def get_all_configurations_for_model(model_name: str) -> List[Dict[str, str]]:
    """Get all dataset-method combinations for a specific model."""
    if model_name not in SLM_MODELS:
        raise ValueError(f"Model {model_name} not supported. Available: {get_supported_models()}")
    
    configurations = []
    for dataset in get_supported_datasets():
        for method in PROMPTING_METHODS.keys():
            configurations.append({
                "model": model_name,
                "dataset": dataset,
                "method": method
            })
    
    return configurations

def validate_configuration(chat_instance_type: str = None) -> bool:
    """Validate that required configurations are present."""
    if chat_instance_type is None:
        chat_instance_type = DEFAULT_CHAT_INSTANCE
    
    missing_vars = []
    
    if chat_instance_type == "google_ai_studio":
        if not (os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')):
            missing_vars.append('GEMINI_API_KEY or GOOGLE_API_KEY')
    
    elif chat_instance_type == "huggingface":
        if not os.environ.get('HUGGINGFACE_TOKEN'):
            missing_vars.append('HUGGINGFACE_TOKEN')
    
    if missing_vars:
        print(f"Warning: Missing environment variables for {chat_instance_type}: {missing_vars}")
        return False
    
    # Validate model configurations
    for model_name, config in SLM_MODELS.items():
        if chat_instance_type not in config:
            print(f"Warning: Model {model_name} missing {chat_instance_type} configuration")
            return False
    
    return True

# Create required directories
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Print configuration on import
if __name__ == "__main__":
    print("\n=== SLM Configuration (Updated) ===")
    print(f"Available Models: {get_supported_models()}")
    for model_name in get_supported_models():
        model_config = get_model_config(model_name)
        print(f"  - {model_config['display_name']} ({model_name}): Vision={model_config['supports_vision']}, Domain={model_config['specialized_domain']}")
    print(f"Default Model: {DEFAULT_MODEL}")
    print(f"Default Chat Instance: {DEFAULT_CHAT_INSTANCE}")
    print(f"Supported Chat Instances: {get_supported_chat_instances()}")
    print(f"Supported Methods: {list(PROMPTING_METHODS.keys())}")
    print(f"Supported Datasets: {get_supported_datasets()}")
    print(f"Output Base Directory: {OUTPUT_BASE_DIR}")
    print(f"Configuration Valid (Google AI): {validate_configuration('google_ai_studio')}")
    print(f"Configuration Valid (HuggingFace): {validate_configuration('huggingface')}")
    print("=================================\n")