"""
Vision Dataset Formatters Module
Handles formatting of medical vision datasets (PMC-VQA, Path-VQA) for agent tasks.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from .dataset_formatters import BaseFormatter


class VisionDatasetValidator:
    """Validator for vision dataset images."""
    
    @staticmethod
    def validate_image_for_vision_api(image) -> bool:
        """Validate image compatibility with vision API."""
        if image is None:
            return False
        
        try:
            if not hasattr(image, 'size') or not hasattr(image, 'mode'):
                return False
            
            width, height = image.size
            if width < 10 or height < 10:
                return False
            
            if width > 4096 or height > 4096:
                return False
            
            if image.mode not in ('RGB', 'L', 'RGBA'):
                try:
                    test_img = image.convert('RGB')
                except:
                    return False
            
            return True
        except Exception:
            return False


class PMCVQAFormatter(BaseFormatter):
    """Formatter for PMC-VQA medical image questions."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format PMC-VQA with enhanced medical image analysis context."""
        question_text = question_data.get("Question", "")
        answer = question_data.get("Answer", "")
        answer_label = question_data.get("Answer_label", "")
        image = question_data.get("image")
        
        # Validate image
        image_valid = VisionDatasetValidator.validate_image_for_vision_api(image)
        if not image_valid:
            logging.warning("PMC-VQA question has invalid image, setting image to None")
            image = None
        
        # Extract choices
        choices = []
        for i, key in enumerate(["Choice A", "Choice B", "Choice C", "Choice D"]):
            choice_text = question_data.get(key, "")
            if choice_text and choice_text.strip():
                choices.append(f"{chr(65+i)}. {choice_text}")
        
        # Use Answer_label directly as ground truth
        correct_letter = answer_label.strip().upper() if answer_label else "A"
        
        # Validate it's a proper option
        if correct_letter not in "ABCD":
            logging.warning(f"Invalid answer_label '{answer_label}', defaulting to A")
            correct_letter = "A"
        
        # Enhanced description for medical image analysis
        enhanced_description = f"""MEDICAL IMAGE ANALYSIS QUESTION

Question: {question_text}

Instructions: Carefully examine the provided medical image and use your visual analysis to answer this question. Consider:
- Anatomical structures visible in the image
- Any pathological changes or abnormalities  
- Relevant clinical features
- Integration of visual findings with medical knowledge

Medical Image Analysis Context:
- Image Type: Medical/Clinical imaging
- Analysis Required: Visual pattern recognition, anatomical identification, pathological assessment
- Clinical Reasoning: Integration of image findings with medical knowledge
- Diagnostic Skills: Visual interpretation of medical imaging modalities

Provide your analysis and select the most appropriate answer based on the visual evidence in the image."""
        
        agent_task = PMCVQAFormatter.create_agent_task(
            name="PMC-VQA Medical Image Question",
            description=enhanced_description,
            task_type="mcq",
            options=choices,
            expected_format=f"Single letter (A-{chr(64+len(choices))}) with detailed image analysis and medical reasoning",
            image_data={
                "image": image,
                "image_available": image is not None,
                "requires_visual_analysis": True,
                "image_type": "medical_image",
                "image_valid": image_valid
            },
            vision_context={
                "analysis_type": "medical_imaging",
                "required_skills": [
                    "visual_pattern_recognition",
                    "anatomical_identification", 
                    "pathological_assessment",
                    "clinical_correlation"
                ],
                "image_modality": "clinical_medical",
                "visual_reasoning_required": True
            }
        )
        
        eval_data = PMCVQAFormatter.create_eval_data(
            ground_truth=correct_letter,
            rationale={correct_letter: answer},
            metadata={
                "dataset": "pmc_vqa", 
                "has_image": image is not None,
                "original_answer": answer,
                "original_answer_label": answer_label,
                "image_validated": image_valid,
                "num_choices": len(choices),
                "requires_vision": True,
                "image_analysis_type": "medical_clinical"
            }
        )
        
        return agent_task, eval_data


class PathVQAFormatter(BaseFormatter):
    """Formatter for Path-VQA pathology image questions."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format Path-VQA as binary MCQ with enhanced pathology context."""
        question_text = question_data.get("question", "")
        answer = question_data.get("answer", "").lower().strip()
        image = question_data.get("image")
        
        # Validate image
        image_valid = VisionDatasetValidator.validate_image_for_vision_api(image)
        if not image_valid:
            logging.warning("Path-VQA question has invalid image, setting image to None")
            image = None
        
        choices = ["A. Yes", "B. No"]
        correct_letter = "A" if answer == "yes" else "B"
        
        # Create enhanced pathology context
        enhanced_description = f"""PATHOLOGY IMAGE ANALYSIS QUESTION

Question: {question_text}

Instructions: Examine the provided pathology/histology image carefully. This is a microscopic image that requires detailed visual analysis. Consider:
- Cellular morphology and architecture
- Tissue patterns and organization  
- Pathological changes or features
- Staining patterns and characteristics
- Integration with pathological knowledge

Pathological Analysis Context:
- Image Type: Histopathology/Microscopic tissue section
- Analysis Required: Cellular morphology assessment, tissue architecture evaluation
- Pathological Skills: Histological pattern recognition, cellular abnormality detection
- Microscopic Features: Cell shape, size, organization, staining characteristics
- Diagnostic Correlation: Integration of microscopic findings with pathological knowledge

Based on your visual examination, answer: You must respond with option A (Yes) or B (No), not yes/no directly.

Visual Analysis Framework:
1. Examine overall tissue architecture
2. Assess cellular morphology and distribution
3. Identify any pathological changes
4. Consider staining patterns and cellular features
5. Correlate findings with the specific question asked"""
        
        agent_task = PathVQAFormatter.create_agent_task(
            name="Path-VQA Pathology Question",
            description=enhanced_description,
            task_type="mcq",
            options=choices,
            expected_format="A for Yes, B for No with detailed pathological analysis and reasoning",
            image_data={
                "image": image,
                "image_available": image is not None,
                "is_pathology_image": True,
                "requires_visual_analysis": True,
                "image_type": "pathology_slide",
                "image_valid": image_valid
            },
            vision_context={
                "analysis_type": "histopathology",
                "required_skills": [
                    "cellular_morphology_assessment",
                    "tissue_architecture_evaluation",
                    "pathological_pattern_recognition", 
                    "microscopic_feature_analysis"
                ],
                "image_modality": "microscopic_pathology",
                "visual_reasoning_required": True,
                "specialized_domain": "pathology"
            },
            pathology_context={
                "image_type": "histological_section",
                "analysis_level": "cellular_and_tissue",
                "pathology_skills_required": [
                    "histological_interpretation",
                    "cellular_abnormality_detection",
                    "tissue_pattern_recognition"
                ]
            }
        )
        
        eval_data = PathVQAFormatter.create_eval_data(
            ground_truth=correct_letter,
            rationale={correct_letter: f"Pathology analysis: {answer.title()}"},
            metadata={
                "dataset": "path_vqa", 
                "original_answer": answer, 
                "has_image": image is not None,
                "image_validated": image_valid,
                "requires_vision": True,
                "image_analysis_type": "histopathology",
                "answer_type": "binary_yes_no"
            }
        )
        
        return agent_task, eval_data


class VisionDatasetImageHandler:
    """Utility class for handling images in vision datasets."""
    
    @staticmethod
    def save_image_temporarily(image, question_id: str, dataset_name: str) -> Optional[str]:
        """Save image temporarily for processing."""
        try:
            import tempfile
            import os
            from PIL import Image as PILImage
            
            if not VisionDatasetValidator.validate_image_for_vision_api(image):
                return None
            
            # Create temporary file
            temp_dir = tempfile.mkdtemp(prefix=f"{dataset_name}_")
            image_path = os.path.join(temp_dir, f"{question_id}.png")
            
            # Convert and save image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(image_path, 'PNG')
            return image_path
            
        except Exception as e:
            logging.error(f"Error saving image temporarily: {e}")
            return None
    
    @staticmethod
    def get_image_info(image) -> Dict[str, Any]:
        """Get comprehensive image information."""
        if not image:
            return {"available": False}
        
        try:
            width, height = image.size
            return {
                "available": True,
                "valid": VisionDatasetValidator.validate_image_for_vision_api(image),
                "width": width,
                "height": height,
                "mode": image.mode,
                "format": getattr(image, 'format', 'Unknown'),
                "size_category": VisionDatasetImageHandler._categorize_image_size(width, height)
            }
        except Exception as e:
            return {"available": True, "valid": False, "error": str(e)}
    
    @staticmethod
    def _categorize_image_size(width: int, height: int) -> str:
        """Categorize image size for analysis."""
        total_pixels = width * height
        
        if total_pixels < 100000:  # Less than 100K pixels
            return "small"
        elif total_pixels < 1000000:  # Less than 1M pixels
            return "medium" 
        elif total_pixels < 4000000:  # Less than 4M pixels
            return "large"
        else:
            return "very_large"


# Enhanced formatting functions with image handling
def format_pmc_vqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format PMC-VQA with comprehensive image handling."""
    agent_task, eval_data = PMCVQAFormatter.format(question_data)
    
    # Add image handling information
    image = question_data.get("image")
    image_info = VisionDatasetImageHandler.get_image_info(image)
    
    # Update agent task with image information
    agent_task.setdefault("image_processing", {}).update({
        "image_info": image_info,
        "preprocessing_required": image_info.get("valid", False),
        "vision_api_compatible": image_info.get("valid", False)
    })
    
    # Store image path in agent task for chat instances
    if image and image_info.get("valid", False):
        question_id = eval_data["metadata"].get("original_answer_label", "unknown")
        temp_path = VisionDatasetImageHandler.save_image_temporarily(image, question_id, "pmc_vqa")
        if temp_path:
            agent_task["image_path"] = temp_path
    
    return agent_task, eval_data


def format_path_vqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Format Path-VQA with comprehensive image handling.""" 
    agent_task, eval_data = PathVQAFormatter.format(question_data)
    
    # Add image handling information
    image = question_data.get("image")
    image_info = VisionDatasetImageHandler.get_image_info(image)
    
    # Update agent task with image information
    agent_task.setdefault("image_processing", {}).update({
        "image_info": image_info,
        "preprocessing_required": image_info.get("valid", False),
        "vision_api_compatible": image_info.get("valid", False)
    })
    
    # Store image path in agent task for chat instances
    if image and image_info.get("valid", False):
        question_id = f"pathvqa_{eval_data['metadata'].get('original_answer', 'unknown')}"
        temp_path = VisionDatasetImageHandler.save_image_temporarily(image, question_id, "path_vqa")
        if temp_path:
            agent_task["image_path"] = temp_path
    
    return agent_task, eval_data


# Vision dataset validation utilities
def validate_vision_dataset_compatibility(dataset_name: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate vision dataset compatibility."""
    validation_results = {
        "dataset": dataset_name,
        "total_questions": len(questions),
        "valid_images": 0,
        "invalid_images": 0,
        "missing_images": 0,
        "image_size_distribution": {"small": 0, "medium": 0, "large": 0, "very_large": 0},
        "compatibility_score": 0.0
    }
    
    for question in questions:
        image = question.get("image")
        
        if image is None:
            validation_results["missing_images"] += 1
        elif VisionDatasetValidator.validate_image_for_vision_api(image):
            validation_results["valid_images"] += 1
            
            # Categorize image size
            try:
                width, height = image.size
                size_cat = VisionDatasetImageHandler._categorize_image_size(width, height)
                validation_results["image_size_distribution"][size_cat] += 1
            except:
                pass
        else:
            validation_results["invalid_images"] += 1
    
    # Calculate compatibility score
    if validation_results["total_questions"] > 0:
        validation_results["compatibility_score"] = validation_results["valid_images"] / validation_results["total_questions"]
    
    return validation_results