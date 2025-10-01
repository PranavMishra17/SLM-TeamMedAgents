# SLM-TeamMedAgents
A flexible multi-agent system using SLMs with modular teamwork components based on the Big Five teamwork model. Agents collaborate on various tasks with configurable teamwork behaviors.

## Overview

This is a refactored version of the SLM (Small Language Model) TeamMedAgents that implements a modular architecture for different chat instances. The system now supports multiple model providers and can be easily extended for future providers.

## Key Features

### Modular Chat Instances
- **Google AI Studio**: Uses `google-genai` library for managed API access
- **Hugging Face**: Uses `transformers` library for local model execution
- **Extensible**: Easy to add new providers (Vertex AI, Azure OpenAI, etc.)

### Supported Models
- **Gemma3-4B-IT**: General-purpose model with vision capabilities
- **MedGemma-4B-IT**: Medical-specialized model with vision capabilities

### Prompting Methods
- **Zero-shot**: Direct question answering
- **Few-shot**: Learning from examples
- **Chain-of-Thought**: Step-by-step reasoning

## Architecture

```
slm_runner.py (Main Runner)
├── slm_config.py (Configuration)
├── chat_instances.py (Modular Chat Providers)
│   ├── BaseChatInstance (Abstract Base)
│   ├── GoogleAIStudioChatInstance
│   ├── HuggingFaceChatInstance
│   └── ChatInstanceFactory
├── utils/
│   └── results_logger.py (Enhanced Results & Token Tracking)
└── medical_datasets/ (Dataset Loading)
```

## Environment Setup

### For Google AI Studio
```bash
# Set API key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Install dependencies
pip install google-genai
```

### For Hugging Face
```bash
# Set HF token
export HUGGINGFACE_TOKEN="your_hf_token_here"

# Install dependencies
pip install transformers torch pillow

# Login to HuggingFace (for gated models)
huggingface-cli login
```

## Usage Examples

### Basic Usage
```bash
# Using Google AI Studio (default)
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b

# Using Hugging Face
python slm_runner.py --dataset medqa --method zero_shot --model gemma3_4b --chat_instance huggingface

# Using MedGemma with few-shot
python slm_runner.py --dataset medqa --method few_shot --model medgemma_4b --chat_instance google_ai_studio
```

### Comprehensive Evaluation
```bash
# Run all configurations for specific model
python slm_runner.py --model gemma3_4b --all --num_questions 20
python slm_runner.py --model medgemma_4b --all --num_questions 20

# Or use batch files
run_all_gemma3.bat
run_all_medgemma.bat
```

### Testing Chat Instances
```bash
# Test all available chat instances
python test_chat_instances.py
```

## Configuration

### Model Configuration
Models are configured in `slm_config.py` with provider-specific settings:

```python
"gemma3_4b": {
    "name": "gemma3_4b",
    "display_name": "Gemma3-4B-IT",
    "supports_vision": True,
    "google_ai_studio": {
        "model": "gemma-3-4b-it"
    },
    "huggingface": {
        "model": "google/gemma-3-4b-it"
    }
}
```

### Chat Instance Selection
Default chat instance is set in configuration:
```python
DEFAULT_CHAT_INSTANCE = "google_ai_studio"
```

Can be overridden at runtime:
```bash
--chat_instance huggingface
```

## Vision Support

Both models support multimodal inputs for image-based datasets:

### Image Datasets
- **PMC-VQA**: Medical visual question answering
- **Path-VQA**: Pathology image analysis

### Usage with Images
```python
# Automatic image handling
runner = SLMMethodRunner(model_name="gemma3_4b", chat_instance_type="google_ai_studio")
response = runner.agent.simple_chat("Analyze this medical image", image_path="path/to/image.jpg")
```

## Performance Comparison

### Google AI Studio
- **Pros**: Managed service, consistent performance, no local setup
- **Cons**: API rate limits, requires internet, usage costs

### Hugging Face Local
- **Pros**: Local control, no API limits, privacy
- **Cons**: Requires GPU/CPU resources, setup complexity

## Directory Structure

```
SLM_Results/
├── gemma3_4b/
│   ├── medqa/
│   │   ├── zero-shot/ (results, summaries, token tracking)
│   │   ├── few-shot/
│   │   └── cot/
│   ├── medmcqa/
│   └── comprehensive_results_gemma3_4b_*.json
└── medgemma_4b/
    ├── medqa/
    ├── medmcqa/
    └── comprehensive_results_medgemma_4b_*.json
```

Results include detailed token usage, aggregated summaries, and comprehensive evaluation reports.

## Extension Guide

### Adding New Chat Instances

1. **Create New Instance Class**:
```python
class NewProviderChatInstance(BaseChatInstance):
    def __init__(self, model_config):
        # Initialize provider
        pass
    
    def simple_chat(self, message, image_path=None):
        # Implement chat logic
        pass
```

2. **Register in Factory**:
```python
ChatInstanceFactory.INSTANCE_TYPES["new_provider"] = NewProviderChatInstance
```

3. **Update Configuration**:
```python
"model_name": {
    "new_provider": {
        "model": "provider_specific_model_id"
    }
}
```

### Adding New Models

1. **Add to Model Configuration**:
```python
"new_model": {
    "name": "new_model",
    "display_name": "New Model",
    "google_ai_studio": {"model": "provider-model-id"},
    "huggingface": {"model": "hf-model-id"}
}
```

2. **Test with Existing Chat Instances**:
```bash
python test_chat_instances.py
```

## Testing

### Quick Test
```bash
python slm_runner.py test
```

### Comprehensive Testing
```bash
python test_chat_instances.py
```

### Run All Configurations
```bash
# For Gemma3-4B
python slm_runner.py --model gemma3_4b --all

# For MedGemma-4B
python slm_runner.py --model medgemma_4b --all
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**:
   - Google AI Studio: Set `GEMINI_API_KEY`
   - Hugging Face: Set `HUGGINGFACE_TOKEN`

2. **Import Errors**:
   - Google AI Studio: `pip install google-genai`
   - Hugging Face: `pip install transformers torch`

3. **Model Access**:
   - Ensure API keys have proper permissions
   - For HF gated models: `huggingface-cli login`

4. **GPU Issues**:
   - Check CUDA availability for HF local inference
   - Adjust model precision if memory limited

### Validation
```bash
python -c "from slm_config import validate_configuration; print(validate_configuration('google_ai_studio'))"
python -c "from slm_config import validate_configuration; print(validate_configuration('huggingface'))"
```

## Future Extensions

The modular architecture supports easy addition of:
- **Vertex AI SDK**: Direct Google Cloud integration
- **Azure OpenAI**: Microsoft's hosted models
- **AWS Bedrock**: Amazon's model service
- **Custom Endpoints**: Self-hosted models
- **Multi-agent Systems**: Complex reasoning workflows

## Benefits of Modular Design

1. **Flexibility**: Switch between providers without code changes
2. **Extensibility**: Easy to add new providers
3. **Testing**: Compare performance across providers
4. **Future-proofing**: Adapt to new model hosting options
5. **Development**: Prototype with managed APIs, deploy with local models