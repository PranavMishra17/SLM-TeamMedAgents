# SLM-TeamMedAgents Knowledge Base

## Project Overview
Multi-agent medical evaluation system using Small Language Models (Gemma3-4B, MedGemma-4B) with modular chat instances, comprehensive rate limiting, and enhanced results tracking.

## Critical Architecture Principles

### 1. Modular Chat Instances
- **GoogleAIStudioChatInstance**: Primary provider with vision support
- **HuggingFaceChatInstance**: Local inference alternative
- **Extensible**: Easy addition of new providers (Vertex AI, Azure OpenAI, etc.)

### 2. Directory Structure
```
SLM_Results/{output_base_dir}/
└── {model_name}/           # gemma3_4b, medgemma_4b
    └── {dataset_name}/     # medqa, medmcqa, etc.
        └── {method}/       # zero-shot, few-shot, cot
```
**Critical**: Never change this structure - results aggregation depends on it.

## Rate Limiting & API Management

### Rate Limits (Google AI Studio)
```python
"gemma3_4b": {
    "rpm": 30,      # Requests per minute
    "tpm": 15000,   # Tokens per minute  
    "rpd": 14400,   # Requests per day
}
```

### Retry Logic
- **Exponential backoff**: Base 2.0, max 60s delay
- **Jitter**: Prevents thundering herd
- **Rate limit errors**: Fixed 65s delay
- **Max retries**: 5 attempts
- **Error differentiation**: Rate limits vs network vs other errors

### Image Handling Constraints
- **Max size**: 3072x3072 pixels (auto-resize)
- **Max file size**: 4MB (auto-compress to JPEG)
- **Formats**: Convert to RGB/RGBA, save as PNG/JPEG
- **Fallback**: Continue without image if processing fails

## Dataset-Specific Requirements

### DDXPlus
- **Local only**: Load from `dataset/ddx/` directory
- **Never use Hugging Face**: Too slow for repeated runs
- **Required files**:
  ```
  dataset/ddx/
  ├── release_evidences.json
  ├── release_conditions.json
  ├── release_train_patients.csv
  ├── release_validate_patients.csv
  └── release_test_patients.csv
  ```

### Path-VQA
- **Yes/No only**: Filter questions with answer in ["yes", "no"]
- **MCQ conversion**: Convert to "A. Yes", "B. No" options
- **Task type**: `yes_no_maybe` for proper answer extraction
- **Answer mapping**: A → yes, B → no

### Vision Datasets (PMC-VQA, Path-VQA)
- **Image validation**: Check size, format, accessibility
- **Fallback handling**: Retry without image on API errors
- **Temporary files**: Auto-cleanup image paths

## Token Counting System

### Google API Integration
- **Primary**: Use `usage_metadata` from API responses
- **Fallback**: Google's `count_tokens` API
- **Final fallback**: Word-based approximation (words × 1.3)
- **Null handling**: All token counts must handle None/null values

### Token Counter Edge Cases
```python
# Handle None values from API
input_count = usage_metadata.get("prompt_token_count") or 0
output_count = usage_metadata.get("candidates_token_count") or 0
total_count = int(total_count) if total_count is not None else 0
```

## Command Line Interface

### --all Parameter
- **Scope**: Single model only (`--model gemma3_4b --all`)
- **Default questions**: 20 for --all, 50 for single runs
- **Configurable**: `--num_questions 10` or `--num_questions 100`
- **Never mix**: --all cannot be used with --dataset or --method

### Output Directory
- **Custom base**: `--output_dir SLM_Results/LOG`
- **Maintains structure**: Still creates model/dataset/method subdirs
- **Path handling**: Use forward slashes, auto-create directories

## Error Handling Patterns

### Question Processing
- **Individual failures**: Log error, continue with remaining questions
- **Token counting failures**: Fallback to approximation, never crash
- **Image processing failures**: Continue without image
- **API failures**: Exponential backoff retry, then fail gracefully

### Results Aggregation
- **Auto-generated**: Method, dataset, and model-level summaries
- **Failure handling**: Partial results still saved and aggregated
- **File conflicts**: Timestamp-based naming prevents overwrites

## Configuration Management

### Environment Variables
- **Primary**: `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- **Secondary**: `HUGGINGFACE_TOKEN`
- **Validation**: Check before initialization, fail fast

### Model Configuration
- **Provider-specific**: Each model has settings for all chat instances
- **Rate limits**: Per-model configuration
- **Vision support**: Model capability flags

## Critical Code Patterns

### Null-Safe Token Handling
```python
input_count = usage_metadata.get("prompt_token_count") or 0
input_count = int(input_count) if input_count is not None else 0
```

### Rate-Limited API Calls
```python
def make_api_call():
    return self.client.models.generate_content(...)

if self.rate_limiter:
    response = self.rate_limiter.exponential_backoff_retry(make_api_call)
else:
    response = make_api_call()
```

### Results Structure
```python
return {
    "summary": summary,
    "results": results, 
    "errors": errors,
    "configuration": {...}
}
```

## Testing Principles

### Dataset Loading
- **Test empty results**: Handle gracefully
- **Test malformed data**: Skip invalid entries
- **Test missing files**: Fallback or clear error message

### API Integration  
- **Mock responses**: Test with None/null values
- **Network failures**: Test retry logic
- **Rate limits**: Test backoff behavior

### Results Validation
- **Token counts**: Never negative, handle None
- **File paths**: Auto-create directories
- **JSON serialization**: Handle datetime, special characters

## Performance Considerations

### Dataset Loading
- **Cache locally**: Avoid repeated Hugging Face downloads
- **Batch processing**: Load all questions at once
- **Memory management**: Clear large datasets after use

### API Efficiency
- **Rate limit awareness**: Stay within quotas
- **Batch similar requests**: Minimize API calls
- **Retry intelligently**: Don't retry permanent failures

### File I/O
- **Streaming**: For large result files
- **Atomic writes**: Prevent corrupted files
- **Directory pre-creation**: Avoid repeated os.makedirs calls

## Common Pitfalls

1. **Token counting None errors**: Always handle null API responses
2. **DDXPlus slow loading**: Use local files, not Hugging Face
3. **Image size errors**: Validate and resize before API calls
4. **Path-VQA wrong format**: Filter yes/no only, convert to MCQ
5. **Rate limit violations**: Implement proper backoff
6. **Results directory chaos**: Maintain strict model/dataset/method structure
7. **Mixed --all usage**: Never combine --all with specific dataset/method
8. **Configuration validation**: Check API keys before starting long runs

## Extension Guidelines

### Adding New Models
1. Add to `SLM_MODELS` in config
2. Add rate limits to `RATE_LIMITS`  
3. Test with all chat instances
4. Update model selection in CLI

### Adding New Datasets
1. Create loader in `dataset_loader.py`
2. Create formatter in appropriate formatter file
3. Add to `SUPPORTED_DATASETS` config
4. Handle vision requirements if needed

### Adding New Chat Instances
1. Inherit from `BaseChatInstance`
2. Implement required methods
3. Add to factory registration
4. Update configuration validation