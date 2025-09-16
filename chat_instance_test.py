"""
Test script for comparing different chat instances and models.
"""

import os
import sys
import logging
from datetime import datetime
from slm_config import *
from slm_runner import SLMMethodRunner

def test_chat_instance_initialization():
    """Test that all chat instances can be initialized properly."""
    
    print("="*60)
    print("CHAT INSTANCE INITIALIZATION TEST")
    print("="*60)
    
    models = get_supported_models()
    chat_instances = get_supported_chat_instances()
    
    initialization_results = {}
    
    for model in models:
        initialization_results[model] = {}
        print(f"\nTesting Model: {model}")
        print("-" * 40)
        
        for chat_instance in chat_instances:
            print(f"  Chat Instance: {chat_instance}")
            
            try:
                # Check configuration first
                if not validate_configuration(chat_instance):
                    print(f"    [SKIP] Configuration not valid for {chat_instance}")
                    initialization_results[model][chat_instance] = "CONFIG_INVALID"
                    continue
                
                # Try to initialize runner
                runner = SLMMethodRunner(
                    model_name=model, 
                    chat_instance_type=chat_instance
                )
                
                print(f"    [SUCCESS] {runner.model_config['display_name']} via {chat_instance}")
                initialization_results[model][chat_instance] = "SUCCESS"
                
            except Exception as e:
                print(f"    [ERROR] Failed: {e}")
                initialization_results[model][chat_instance] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("INITIALIZATION SUMMARY")
    print("="*60)
    
    for model in models:
        print(f"\n{model}:")
        for chat_instance in chat_instances:
            status = initialization_results[model][chat_instance]
            print(f"  {chat_instance}: {status}")
    
    return initialization_results

def test_simple_inference():
    """Test simple inference with available chat instances."""
    
    print("="*60)
    print("SIMPLE INFERENCE TEST")
    print("="*60)
    
    # Simple test question
    test_prompt = "What is the capital of France? Answer with just the city name."
    
    models = get_supported_models()
    chat_instances = get_supported_chat_instances()
    
    inference_results = {}
    
    for model in models:
        inference_results[model] = {}
        print(f"\nTesting Model: {model}")
        print("-" * 40)
        
        for chat_instance in chat_instances:
            print(f"  Chat Instance: {chat_instance}")
            
            try:
                # Check if configuration is valid
                if not validate_configuration(chat_instance):
                    print(f"    [SKIP] Configuration not valid")
                    inference_results[model][chat_instance] = "SKIPPED"
                    continue
                
                # Initialize runner
                runner = SLMMethodRunner(
                    model_name=model, 
                    chat_instance_type=chat_instance
                )
                
                # Test simple chat
                response = runner.agent.simple_chat(test_prompt)
                
                print(f"    [SUCCESS] Response: {response[:50]}...")
                inference_results[model][chat_instance] = response.strip()
                
            except Exception as e:
                print(f"    [ERROR] Failed: {e}")
                inference_results[model][chat_instance] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    
    for model in models:
        print(f"\n{model}:")
        for chat_instance in chat_instances:
            result = inference_results[model][chat_instance]
            if result.startswith("ERROR"):
                print(f"  {chat_instance}: {result}")
            elif result == "SKIPPED":
                print(f"  {chat_instance}: SKIPPED")
            else:
                print(f"  {chat_instance}: {result}")
    
    return inference_results

def test_medical_inference():
    """Test medical inference capabilities."""
    
    print("="*60)
    print("MEDICAL INFERENCE TEST")
    print("="*60)
    
    # Medical test question
    medical_prompt = """A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. Which coronary artery is most likely occluded?

A. Left anterior descending
B. Right coronary artery
C. Left circumflex
D. Left main

Answer with just the letter (A, B, C, or D)."""
    
    models = get_supported_models()
    chat_instances = get_supported_chat_instances()
    
    medical_results = {}
    
    for model in models:
        medical_results[model] = {}
        print(f"\nTesting Model: {model}")
        print("-" * 40)
        
        for chat_instance in chat_instances:
            print(f"  Chat Instance: {chat_instance}")
            
            try:
                # Check if configuration is valid
                if not validate_configuration(chat_instance):
                    print(f"    [SKIP] Configuration not valid")
                    medical_results[model][chat_instance] = "SKIPPED"
                    continue
                
                # Initialize runner
                runner = SLMMethodRunner(
                    model_name=model, 
                    chat_instance_type=chat_instance
                )
                
                # Test medical inference
                response = runner.agent.simple_chat(medical_prompt)
                
                print(f"    [SUCCESS] Response: {response[:100]}...")
                medical_results[model][chat_instance] = response.strip()
                
            except Exception as e:
                print(f"    [ERROR] Failed: {e}")
                medical_results[model][chat_instance] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "="*60)
    print("MEDICAL INFERENCE SUMMARY")
    print("="*60)
    print("Expected Answer: B (Right coronary artery)")
    print()
    
    for model in models:
        print(f"\n{model}:")
        for chat_instance in chat_instances:
            result = medical_results[model][chat_instance]
            if result.startswith("ERROR"):
                print(f"  {chat_instance}: {result}")
            elif result == "SKIPPED":
                print(f"  {chat_instance}: SKIPPED")
            else:
                # Try to extract answer
                answer = "Unknown"
                if "B" in result.upper()[:10]:
                    answer = "B ✓"
                elif any(letter in result.upper()[:10] for letter in ["A", "C", "D"]):
                    answer = result.upper()[:10].strip()
                
                print(f"  {chat_instance}: {answer}")
    
    return medical_results

def generate_comparison_report():
    """Generate a comprehensive comparison report."""
    
    print("="*60)
    print("GENERATING COMPREHENSIVE COMPARISON REPORT")
    print("="*60)
    
    # Run all tests
    init_results = test_chat_instance_initialization()
    simple_results = test_simple_inference()
    medical_results = test_medical_inference()
    
    # Generate report
    report_content = f"""# SLM Chat Instance Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration Status

"""
    
    models = get_supported_models()
    chat_instances = get_supported_chat_instances()
    
    for model in models:
        report_content += f"### {model}\n\n"
        for chat_instance in chat_instances:
            status = init_results[model][chat_instance]
            report_content += f"- **{chat_instance}**: {status}\n"
        report_content += "\n"
    
    report_content += """## Performance Comparison

### Simple Inference Test
Test: "What is the capital of France?"

"""
    
    for model in models:
        report_content += f"#### {model}\n\n"
        for chat_instance in chat_instances:
            result = simple_results[model][chat_instance]
            report_content += f"- **{chat_instance}**: {result}\n"
        report_content += "\n"
    
    report_content += """### Medical Inference Test
Test: Medical MCQ about coronary artery occlusion
Expected Answer: B (Right coronary artery)

"""
    
    for model in models:
        report_content += f"#### {model}\n\n"
        for chat_instance in chat_instances:
            result = medical_results[model][chat_instance]
            report_content += f"- **{chat_instance}**: {result[:100]}...\n"
        report_content += "\n"
    
    report_content += """## Recommendations

### For Production Use:
- **Google AI Studio**: Reliable, managed service, good for consistent performance
- **Hugging Face**: Local control, potentially faster for batch processing, requires more setup

### For Development:
- Use Google AI Studio for initial prototyping and testing
- Switch to Hugging Face for fine-tuning and custom deployments

### Model Selection:
- **Gemma3-4B**: General purpose, good baseline performance
- **MedGemma-4B**: Optimized for medical tasks, potentially better domain-specific performance
"""
    
    # Save report
    report_file = f"chat_instance_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report saved to: {report_file}")
    return report_file

def main():
    """Main test function."""
    
    print("Starting comprehensive chat instance testing...")
    print()
    
    # Run individual tests
    test_chat_instance_initialization()
    print("\n" + "="*60 + "\n")
    
    test_simple_inference()
    print("\n" + "="*60 + "\n")
    
    test_medical_inference()
    print("\n" + "="*60 + "\n")
    
    # Generate comprehensive report
    report_file = generate_comparison_report()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"Comprehensive report saved to: {report_file}")
    print()
    print("Summary:")
    print("- Initialization tests completed")
    print("- Simple inference tests completed")
    print("- Medical inference tests completed")
    print("- Comparison report generated")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
