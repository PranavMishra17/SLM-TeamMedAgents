"""
Medical Dataset Formatters Module
Handles formatting of various medical datasets for agent tasks and evaluation.
"""

import logging
import ast
from typing import Dict, Any, List, Optional, Tuple


class BaseFormatter:
    """Base class for dataset formatters."""
    
    @staticmethod
    def create_agent_task(name: str, description: str, task_type: str, options: List[str], 
                         expected_format: str, **kwargs) -> Dict[str, Any]:
        """Create standardized agent task structure."""
        return {
            "name": name,
            "description": description,
            "type": task_type,
            "options": options,
            "expected_output_format": expected_format,
            **kwargs
        }
    
    @staticmethod
    def create_eval_data(ground_truth: str, rationale: Dict[str, str], 
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized evaluation data structure."""
        return {
            "ground_truth": ground_truth,
            "rationale": rationale,
            "metadata": metadata
        }


class MedQAFormatter(BaseFormatter):
    """Formatter for MedQA dataset."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format MedQA question into agent task and evaluation data."""
        question_text = question_data.get("question", "")
        choices = question_data.get("choices", [])
        
        if not isinstance(choices, list):
            choices = []
        
        options = []
        for i, choice in enumerate(choices):
            if isinstance(choice, str):
                options.append(f"{chr(65+i)}. {choice}")
        
        expected_output = question_data.get("expected_output", "")
        
        agent_task = MedQAFormatter.create_agent_task(
            name="MedQA Question",
            description=question_text,
            task_type="mcq",
            options=options,
            expected_format="Single letter selection with rationale"
        )
        
        eval_data = MedQAFormatter.create_eval_data(
            ground_truth=expected_output,
            rationale={},
            metadata={
                "dataset": "MedQA",
                "question_id": question_data.get("id", ""),
                "metamap": question_data.get("metamap", ""),
                "answer_idx": question_data.get("answer_idx", "")
            }
        )
        
        return agent_task, eval_data


class MedMCQAFormatter(BaseFormatter):
    """Formatter for MedMCQA dataset."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Format MedMCQA question with validation."""
        question_text = question_data.get("question", "")
        opa, opb, opc, opd = (question_data.get("opa", ""), question_data.get("opb", ""), 
                             question_data.get("opc", ""), question_data.get("opd", ""))
        
        option_letters = ['A', 'B', 'C', 'D']
        option_values = [opa, opb, opc, opd]
        options = [f"{letter}. {value}" for letter, value in zip(option_letters, option_values) if value.strip()]
        
        # Parse and validate cop field
        cop = question_data.get("cop", "")
        ground_truth, is_valid_cop = MedMCQAFormatter._parse_cop_field(cop)
        
        if not is_valid_cop:
            error_info = {
                "question_id": question_data.get("id", "unknown"),
                "invalid_cop": cop,
                "error_type": "invalid_ground_truth",
                "message": f"Invalid cop value '{cop}' - question skipped"
            }
            return {}, error_info, False
        
        # Enhanced description for context
        enhanced_description = question_text
        if len(question_text.split()) < 10:
            options_text = " | ".join([f"{letter}: {value}" for letter, value in zip(option_letters, option_values) if value.strip()])
            enhanced_description = f"{question_text}\n\nAnswer options provide context: {options_text}"
        
        explanation = question_data.get("exp", "")
        subject_name = question_data.get("subject_name", "")
        topic_name = question_data.get("topic_name", "")
        
        agent_task = MedMCQAFormatter.create_agent_task(
            name="MedMCQA Question",
            description=enhanced_description,
            task_type="mcq",
            options=options,
            expected_format="Single letter selection with rationale",
            subject_context=subject_name,
            topic_context=topic_name
        )
        
        eval_data = MedMCQAFormatter.create_eval_data(
            ground_truth=ground_truth,
            rationale={ground_truth: explanation} if explanation else {},
            metadata={
                "subject": subject_name,
                "topic": topic_name,
                "question_id": question_data.get("id", ""),
                "original_choice_type": question_data.get("choice_type", "single"),
                "cop_original": question_data.get("cop", ""),
                "original_question": question_text,
                "enhanced_for_recruitment": len(question_text.split()) < 10
            }
        )
        
        return agent_task, eval_data, True
    
    @staticmethod
    def _parse_cop_field(cop) -> Tuple[str, bool]:
        """Parse the cop (correct option) field with 0-based indexing."""
        if isinstance(cop, int):
            if 0 <= cop <= 3:
                return chr(65 + cop), True  # 0→A, 1→B, 2→C, 3→D
            else:
                return "A", False
        
        elif isinstance(cop, str):
            cop = cop.strip()
            
            if cop.isdigit():
                cop_int = int(cop)
                if 0 <= cop_int <= 3:
                    return chr(65 + cop_int), True
                else:
                    return "A", False
            
            elif cop.upper() in ['A', 'B', 'C', 'D']:
                return cop.upper(), True
            
            elif ',' in cop:
                first_part = cop.split(',')[0].strip()
                return MedMCQAFormatter._parse_cop_field(first_part)
            
            elif not cop:
                return "A", False
            
            else:
                import re
                match = re.search(r'\d+', cop)
                if match:
                    cop_int = int(match.group())
                    if 0 <= cop_int <= 3:
                        return chr(65 + cop_int), True
                    else:
                        return "A", False
                else:
                    return "A", False
        
        elif isinstance(cop, (list, tuple)) and len(cop) > 0:
            return MedMCQAFormatter._parse_cop_field(cop[0])
        
        else:
            try:
                return MedMCQAFormatter._parse_cop_field(str(cop))
            except:
                return "A", False


class MMLUProMedFormatter(BaseFormatter):
    """Formatter for MMLU-Pro medical questions."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format MMLU-Pro medical question with up to 10 options."""
        question_text = question_data.get("question", "")
        options_data = question_data.get("options", [])
        options = []
        
        if isinstance(options_data, list):
            for i, option in enumerate(options_data):
                if i < 10:  # Support up to 10 options (A-J)
                    letter = chr(65 + i)
                    options.append(f"{letter}. {option}")
                else:
                    break
        elif isinstance(options_data, dict):
            for key in sorted(options_data.keys()):
                options.append(f"{key}. {options_data[key]}")
        
        ground_truth = question_data.get("answer", "")
        if not ground_truth and "answer_idx" in question_data:
            idx = question_data["answer_idx"]
            if isinstance(idx, int) and 0 <= idx < len(options_data):
                ground_truth = chr(65 + idx)
        
        num_options = len(options)
        if num_options <= 4:
            expected_format = "Single letter selection (A-D) with rationale"
        elif num_options <= 10:
            last_letter = chr(64 + num_options)
            expected_format = f"Single letter selection (A-{last_letter}) with rationale"
        else:
            expected_format = "Single letter selection with rationale"
        
        agent_task = MMLUProMedFormatter.create_agent_task(
            name="MMLU-Pro Health Question",
            description=question_text,
            task_type="mcq",
            options=options,
            expected_format=expected_format,
            num_options=num_options
        )
        
        eval_data = MMLUProMedFormatter.create_eval_data(
            ground_truth=ground_truth,
            rationale={},
            metadata={
                "category": question_data.get("category", ""),
                "answer_idx": question_data.get("answer_idx", ""),
                "num_options": num_options
            }
        )
        
        return agent_task, eval_data


class PubMedQAFormatter(BaseFormatter):
    """Formatter for PubMedQA questions."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format PubMedQA question into agent task and evaluation data."""
        question_text = question_data.get("question", "")
        context = question_data.get("context", "")
        context_text = ""
        
        if isinstance(context, dict):
            for section, text in context.items():
                context_text += f"{section}: {text}\n\n"
        elif isinstance(context, list):
            context_text = "\n\n".join(context)
        elif isinstance(context, str):
            context_text = context
        
        expected_output = question_data.get("final_decision", "").lower()
        
        agent_task = PubMedQAFormatter.create_agent_task(
            name="PubMedQA Question",
            description=f"Research Question: {question_text}\n\nAbstract Context:\n{context_text}",
            task_type="yes_no_maybe",
            options=["yes", "no", "maybe"],
            expected_format="Answer (yes/no/maybe) with detailed scientific justification"
        )
        
        eval_data = PubMedQAFormatter.create_eval_data(
            ground_truth=expected_output,
            rationale={"long_answer": question_data.get("long_answer", "")},
            metadata={
                "pubid": question_data.get("pubid", "")
            }
        )
        
        return agent_task, eval_data


class DDXPlusFormatter(BaseFormatter):
    """Formatter for DDXPlus dataset."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format DDXPlus question with comprehensive medical context."""
        question_text = question_data.get("question", "")
        choices = question_data.get("choices", [])
        
        options = []
        for i, choice in enumerate(choices):
            options.append(f"{chr(65+i)}. {choice}")
        
        correct_letter = question_data.get("correct_letter", "A")
        correct_answer = question_data.get("correct_answer", "")
        
        age = question_data.get("age", "Unknown")
        sex = question_data.get("sex", "Unknown")
        evidences_list = question_data.get("evidences", [])
        initial_evidence = question_data.get("initial_evidence", "")
        differential_diagnosis = question_data.get("differential_diagnosis", [])
        
        medical_context = DDXPlusFormatter._create_medical_context(
            age, sex, evidences_list, initial_evidence, differential_diagnosis
        )
        
        enhanced_description = f"""{question_text}

COMPREHENSIVE MEDICAL CONTEXT:

Patient Demographics:
- Age: {age}
- Sex: {sex}

Chief Complaint & Initial Presentation:
- Initial Evidence: {initial_evidence if initial_evidence else "Not specified"}

Evidence Analysis:
{medical_context['evidence_analysis']}

Symptom Categories:
{medical_context['symptom_categories']}

Differential Diagnosis Considerations:
{medical_context['differential_context']}

Clinical Decision Factors:
- Case Complexity: {medical_context['complexity_level']}
- Number of Evidence Points: {len(evidences_list)}
- Diagnostic Certainty: {medical_context['diagnostic_certainty']}
"""
        
        agent_task = DDXPlusFormatter.create_agent_task(
            name="DDXPlus Clinical Diagnosis Case",
            description=enhanced_description,
            task_type="mcq",
            options=options,
            expected_format="Single letter selection with comprehensive clinical reasoning including differential diagnosis analysis",
            clinical_context={
                "patient_demographics": {
                    "age": age,
                    "sex": sex,
                    "age_category": DDXPlusFormatter._categorize_age(age)
                },
                "case_characteristics": {
                    "complexity_level": medical_context['complexity_level'],
                    "evidence_count": len(evidences_list),
                    "has_initial_evidence": bool(initial_evidence),
                    "has_differential": bool(differential_diagnosis),
                    "diagnostic_certainty": medical_context['diagnostic_certainty']
                },
                "medical_specialties_needed": medical_context['specialties_needed'],
                "evidence_types": medical_context['evidence_types'],
                "symptom_systems": medical_context['symptom_systems']
            },
            raw_medical_data={
                "evidence_codes": evidences_list,
                "initial_presenting_evidence": initial_evidence,
                "differential_probabilities": medical_context['differential_probs'],
                "symptom_categories": medical_context['raw_symptom_categories']
            }
        )
        
        eval_data = DDXPlusFormatter.create_eval_data(
            ground_truth=correct_letter,
            rationale={
                "correct_diagnosis": correct_answer,
                "differential_diagnosis": differential_diagnosis,
                "evidence_supporting_diagnosis": evidences_list,
                "initial_presentation": initial_evidence
            },
            metadata={
                "dataset": "ddxplus",
                "question_id": question_data.get("id", ""),
                "pathology": question_data.get("metadata", {}).get("pathology", ""),
                "patient_demographics": {"age": age, "sex": sex},
                "clinical_complexity": medical_context['complexity_level'],
                "evidence_analysis": medical_context['evidence_analysis'],
                "specialties_involved": medical_context['specialties_needed']
            }
        )
        
        return agent_task, eval_data
    
    @staticmethod
    def convert_patient_to_question(patient: Dict[str, Any], evidences: Dict, 
                                   conditions: Dict, patient_id: int) -> Optional[Dict[str, Any]]:
        """Convert DDXPlus patient to question format."""
        try:
            age = patient.get("AGE", "Unknown")
            sex = patient.get("SEX", "Unknown") 
            pathology = patient.get("PATHOLOGY", "")
            evidences_list = patient.get("EVIDENCES", [])
            initial_evidence = patient.get("INITIAL_EVIDENCE", "")
            differential_diagnosis = patient.get("DIFFERENTIAL_DIAGNOSIS", [])
            
            # Handle string representations from CSV
            if isinstance(evidences_list, str):
                try:
                    evidences_list = ast.literal_eval(evidences_list)
                except:
                    evidences_list = evidences_list.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                    evidences_list = [e.strip() for e in evidences_list.split(',') if e.strip()]
            
            if isinstance(differential_diagnosis, str):
                try:
                    differential_diagnosis = ast.literal_eval(differential_diagnosis)
                except:
                    differential_diagnosis = []
            
            symptoms_text = DDXPlusFormatter._format_evidences(evidences_list, evidences)
            
            question_text = f"Patient Information:\n"
            question_text += f"Age: {age}, Sex: {sex}\n\n"
            question_text += f"Presenting symptoms and medical history:\n{symptoms_text}\n\n"
            question_text += f"What is the most likely diagnosis?"
            
            # Create answer choices
            choices = []
            correct_answer = pathology
            
            diff_diagnoses = []
            if isinstance(differential_diagnosis, list):
                for item in differential_diagnosis:
                    if isinstance(item, list) and len(item) >= 2:
                        diff_diagnoses.append(item[0])
                    elif isinstance(item, str):
                        diff_diagnoses.append(item)
            
            if correct_answer not in diff_diagnoses:
                diff_diagnoses.insert(0, correct_answer)
            
            unique_diagnoses = []
            for diag in diff_diagnoses:
                if diag not in unique_diagnoses and diag.strip():
                    unique_diagnoses.append(diag)
                if len(unique_diagnoses) >= 4:
                    break
            
            while len(unique_diagnoses) < 4:
                filler_conditions = ["Other infectious condition", "Requires further investigation", 
                                   "Chronic inflammatory condition", "Acute viral syndrome"]
                for filler in filler_conditions:
                    if filler not in unique_diagnoses:
                        unique_diagnoses.append(filler)
                        break
                if len(unique_diagnoses) >= 4:
                    break
            
            choices = unique_diagnoses[:4]
            
            try:
                correct_idx = choices.index(correct_answer)
                correct_letter = chr(65 + correct_idx)
            except ValueError:
                correct_letter = "A"
                choices[0] = correct_answer
            
            return {
                "id": f"ddxplus_{patient_id}",
                "question": question_text,
                "choices": choices,
                "correct_answer": correct_answer,
                "correct_letter": correct_letter,
                "age": age,
                "sex": sex,
                "evidences": evidences_list,
                "initial_evidence": initial_evidence,
                "differential_diagnosis": differential_diagnosis,
                "metadata": {
                    "dataset": "ddxplus",
                    "patient_id": patient_id,
                    "pathology": pathology
                }
            }
            
        except Exception as e:
            logging.error(f"Error converting DDXPlus patient {patient_id}: {str(e)}")
            return None
    
    @staticmethod
    def _format_evidences(evidences_list: List[str], evidences_dict: Dict) -> str:
        """Format DDXPlus evidences into readable text."""
        symptoms = []
        
        for evidence in evidences_list:
            try:
                if "_@_" in evidence:
                    evidence_name, value_code = evidence.split("_@_")
                    evidence_info = evidences_dict.get(evidence_name, {})
                    
                    question_en = evidence_info.get("question_en", evidence_name)
                    value_meaning = evidence_info.get("value_meaning", {})
                    
                    if value_code in value_meaning:
                        value_text = value_meaning[value_code].get("en", value_code)
                        if value_text != "NA":
                            symptoms.append(f"{question_en}: {value_text}")
                            
                else:
                    evidence_info = evidences_dict.get(evidence, {})
                    question_en = evidence_info.get("question_en", evidence)
                    if question_en and question_en != evidence:
                        symptoms.append(question_en)
                        
            except Exception as e:
                logging.debug(f"Error formatting evidence {evidence}: {str(e)}")
                continue
        
        return "\n".join([f"- {symptom}" for symptom in symptoms]) if symptoms else "No specific symptoms recorded"
    
    @staticmethod
    def _create_medical_context(age, sex, evidences_list, initial_evidence, differential_diagnosis):
        """Create comprehensive medical context for DDXPlus cases."""
        evidence_analysis = []
        symptom_systems = set()
        evidence_types = {"binary": 0, "categorical": 0, "multi_choice": 0}
        
        for evidence in evidences_list:
            if "_@_" in evidence:
                evidence_types["categorical"] += 1
                evidence_analysis.append(f"- Categorical evidence: {evidence}")
            else:
                evidence_types["binary"] += 1
                evidence_analysis.append(f"- Binary evidence: {evidence}")
                
            # Infer medical systems
            if any(term in evidence.lower() for term in ['cardio', 'heart', 'chest']):
                symptom_systems.add("cardiovascular")
            elif any(term in evidence.lower() for term in ['neuro', 'head', 'cognitive']):
                symptom_systems.add("neurological")
            elif any(term in evidence.lower() for term in ['gastro', 'stomach', 'digest']):
                symptom_systems.add("gastrointestinal")
            elif any(term in evidence.lower() for term in ['resp', 'lung', 'breath']):
                symptom_systems.add("respiratory")
            else:
                symptom_systems.add("general_medicine")
        
        complexity_score = len(evidences_list) + (len(differential_diagnosis) if differential_diagnosis else 0)
        if complexity_score < 5:
            complexity_level = "basic"
        elif complexity_score < 15:
            complexity_level = "intermediate"
        else:
            complexity_level = "advanced"
        
        differential_probs = []
        if isinstance(differential_diagnosis, list):
            for item in differential_diagnosis:
                if isinstance(item, list) and len(item) >= 2:
                    differential_probs.append({"condition": item[0], "probability": item[1]})
        
        specialties_needed = list(symptom_systems)
        if age != "Unknown":
            try:
                age_num = int(age)
                if age_num < 18:
                    specialties_needed.append("pediatrics")
                elif age_num > 65:
                    specialties_needed.append("geriatrics")
            except:
                pass
        
        if differential_probs:
            max_prob = max(prob["probability"] for prob in differential_probs)
            if max_prob > 0.7:
                diagnostic_certainty = "high"
            elif max_prob > 0.4:
                diagnostic_certainty = "moderate"
            else:
                diagnostic_certainty = "low"
        else:
            diagnostic_certainty = "unknown"
        
        return {
            "evidence_analysis": "\n".join(evidence_analysis) if evidence_analysis else "No specific evidence analysis available",
            "symptom_categories": f"Binary: {evidence_types['binary']}, Categorical: {evidence_types['categorical']}, Multi-choice: {evidence_types['multi_choice']}",
            "differential_context": f"Differential diagnoses available: {len(differential_probs)} conditions with probabilities" if differential_probs else "No differential diagnosis probabilities available",
            "complexity_level": complexity_level,
            "diagnostic_certainty": diagnostic_certainty,
            "specialties_needed": specialties_needed,
            "evidence_types": evidence_types,
            "symptom_systems": list(symptom_systems),
            "differential_probs": differential_probs,
            "raw_symptom_categories": evidence_types
        }
    
    @staticmethod
    def _categorize_age(age):
        """Categorize age for medical context."""
        try:
            age_num = int(age)
            if age_num < 2:
                return "infant"
            elif age_num < 12:
                return "child"
            elif age_num < 18:
                return "adolescent"
            elif age_num < 65:
                return "adult"
            else:
                return "elderly"
        except:
            return "unknown"


class MedBulletsFormatter(BaseFormatter):
    """Formatter for MedBullets dataset."""
    
    @staticmethod
    def format(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format MedBullets question with comprehensive USMLE context."""
        question_text = question_data.get("question", "")
        
        choices = []
        choice_keys = ["choicesA", "choicesB", "choicesC", "choicesD", "choicesE"]
        
        for i, key in enumerate(choice_keys):
            choice_text = question_data.get(key, "")
            if choice_text and choice_text.strip():
                choices.append(f"{chr(65+i)}. {choice_text}")
        
        answer_idx = question_data.get("answer_idx", "")
        correct_answer = question_data.get("answer", "")
        explanation = question_data.get("explanation", "")
        
        if isinstance(answer_idx, (int, str)):
            try:
                if isinstance(answer_idx, str) and answer_idx.isdigit():
                    idx = int(answer_idx)
                    correct_letter = chr(64 + idx) if 1 <= idx <= 5 else "A"
                elif isinstance(answer_idx, str) and len(answer_idx) == 1 and answer_idx.upper() in "ABCDE":
                    correct_letter = answer_idx.upper()
                elif isinstance(answer_idx, int):
                    correct_letter = chr(64 + answer_idx) if 1 <= answer_idx <= 5 else "A"
                else:
                    correct_letter = "A"
            except:
                correct_letter = "A"
        else:
            correct_letter = "A"
        
        usmle_analysis = MedBulletsFormatter._create_usmle_analysis(question_text, explanation, len(choices))
        
        enhanced_description = f"""{question_text}

USMLE CLINICAL CONTEXT:

Question Characteristics:
- USMLE Step Level: {usmle_analysis['step_level']}
- Question Type: {usmle_analysis['question_type']}
- Clinical Complexity: {usmle_analysis['complexity_level']}
- Topic Category: {usmle_analysis['topic_category']}

Medical Domain Analysis:
{usmle_analysis['domain_analysis']}

Clinical Skills Required:
{usmle_analysis['skills_required']}

Question Difficulty Assessment:
- Estimated Difficulty: {usmle_analysis['difficulty_level']}
- Reasoning Type: {usmle_analysis['reasoning_type']}
- Knowledge Domain: {usmle_analysis['knowledge_domain']}
"""
        
        agent_task = MedBulletsFormatter.create_agent_task(
            name="MedBullets USMLE Clinical Question",
            description=enhanced_description,
            task_type="mcq",
            options=choices,
            expected_format=f"Single letter selection (A-{chr(64+len(choices))}) with USMLE-level clinical reasoning and step-by-step analysis",
            usmle_context={
                "exam_characteristics": {
                    "step_level": usmle_analysis['step_level'],
                    "question_type": usmle_analysis['question_type'],
                    "difficulty_level": usmle_analysis['difficulty_level'],
                    "complexity_level": usmle_analysis['complexity_level']
                },
                "clinical_domains": {
                    "primary_topic": usmle_analysis['topic_category'],
                    "knowledge_domain": usmle_analysis['knowledge_domain'],
                    "medical_specialties": usmle_analysis['medical_specialties']
                },
                "required_competencies": usmle_analysis['competencies_required'],
                "clinical_reasoning_type": usmle_analysis['reasoning_type'],
                "medical_specialties_needed": usmle_analysis['medical_specialties'],
                "clinical_skills_required": usmle_analysis['skills_list']
            },
            question_format={
                "num_options": len(choices),
                "has_explanation": bool(explanation),
                "answer_format": f"A-{chr(64+len(choices))}"
            }
        )
        
        eval_data = MedBulletsFormatter.create_eval_data(
            ground_truth=correct_letter,
            rationale={
                correct_letter: explanation if explanation else correct_answer,
                "usmle_explanation": explanation,
                "correct_answer_text": correct_answer
            },
            metadata={
                "dataset": "medbullets",
                "answer_idx_original": answer_idx,
                "usmle_analysis": {
                    "step_level": usmle_analysis['step_level'],
                    "difficulty": usmle_analysis['difficulty_level'],
                    "topic_category": usmle_analysis['topic_category'],
                    "reasoning_type": usmle_analysis['reasoning_type']
                },
                "question_characteristics": {
                    "has_explanation": bool(explanation),
                    "num_choices": len(choices),
                    "complexity": usmle_analysis['complexity_level']
                },
                "specialties_involved": usmle_analysis['medical_specialties']
            }
        )
        
        return agent_task, eval_data
    
    @staticmethod
    def _create_usmle_analysis(question_text, explanation, num_choices):
        """Create comprehensive USMLE analysis for MedBullets questions."""
        question_lower = question_text.lower()
        explanation_lower = explanation.lower() if explanation else ""
        
        # Determine USMLE Step level
        if any(term in question_lower for term in ['step 1', 'basic science', 'pathophysiology', 'anatomy']):
            step_level = "Step 1"
        elif any(term in question_lower for term in ['step 2', 'clinical', 'patient', 'diagnosis', 'treatment']):
            step_level = "Step 2 CK/CS"
        elif any(term in question_lower for term in ['step 3', 'management', 'follow-up', 'monitoring']):
            step_level = "Step 3"
        else:
            if any(term in question_lower for term in ['patient', 'year-old', 'presents', 'complains']):
                step_level = "Step 2 CK"
            else:
                step_level = "Step 2/3"
        
        # Determine question type
        if "year-old" in question_lower and "presents" in question_lower:
            question_type = "Clinical Vignette"
        elif any(term in question_lower for term in ['which of the following', 'most likely', 'best next step']):
            question_type = "Clinical Reasoning"
        elif any(term in question_lower for term in ['mechanism', 'pathway', 'process']):
            question_type = "Basic Science"
        else:
            question_type = "Clinical Knowledge"
        
        # Analyze medical specialties
        medical_specialties = []
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "coronary", "myocardial", "arrhythmia"],
            "neurology": ["brain", "neural", "seizure", "stroke", "headache", "cognitive"],
            "infectious_disease": ["infection", "fever", "antibiotic", "sepsis", "pathogen"],
            "endocrinology": ["diabetes", "thyroid", "hormone", "insulin", "glucose"],
            "gastroenterology": ["gastric", "intestinal", "liver", "hepatic", "bowel"],
            "pulmonology": ["lung", "respiratory", "pneumonia", "asthma", "breathing"],
            "nephrology": ["kidney", "renal", "urine", "creatinine", "dialysis"],
            "oncology": ["cancer", "tumor", "malignant", "metastasis", "chemotherapy"],
            "psychiatry": ["depression", "anxiety", "psychiatric", "mental", "mood"],
            "surgery": ["surgical", "operation", "procedure", "incision", "resection"]
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                medical_specialties.append(specialty)
        
        if not medical_specialties:
            medical_specialties = ["general_medicine"]
        
        # Determine complexity and difficulty
        word_count = len(question_text.split())
        if word_count < 50:
            complexity_level = "basic"
        elif word_count < 150:
            complexity_level = "intermediate"
        else:
            complexity_level = "advanced"
        
        if num_choices <= 4 and complexity_level == "basic":
            difficulty_level = "moderate"
        elif complexity_level == "advanced" or num_choices == 5:
            difficulty_level = "high"
        else:
            difficulty_level = "moderate"
        
        # Determine reasoning type and topic category
        if "best next step" in question_lower:
            reasoning_type = "clinical_management"
        elif "most likely" in question_lower:
            reasoning_type = "diagnostic_reasoning"
        elif "mechanism" in question_lower:
            reasoning_type = "pathophysiology"
        else:
            reasoning_type = "clinical_knowledge"
        
        if any(term in question_lower for term in ['diagnosis', 'differential', 'presents']):
            topic_category = "diagnosis"
        elif any(term in question_lower for term in ['treatment', 'therapy', 'management']):
            topic_category = "treatment"
        elif any(term in question_lower for term in ['prevention', 'screening', 'prophylaxis']):
            topic_category = "prevention"
        else:
            topic_category = "clinical_knowledge"
        
        # Required competencies
        competencies_required = ["medical_knowledge", "clinical_reasoning"]
        if step_level in ["Step 2 CK", "Step 2 CS", "Step 3"]:
            competencies_required.extend(["patient_care", "clinical_decision_making"])
        if reasoning_type == "diagnostic_reasoning":
            competencies_required.append("diagnostic_skills")
        if reasoning_type == "clinical_management":
            competencies_required.append("treatment_planning")
        
        return {
            "step_level": step_level,
            "question_type": question_type,
            "complexity_level": complexity_level,
            "difficulty_level": difficulty_level,
            "topic_category": topic_category,
            "reasoning_type": reasoning_type,
            "knowledge_domain": step_level,
            "medical_specialties": medical_specialties,
            "competencies_required": competencies_required,
            "domain_analysis": f"Primary specialties: {', '.join(medical_specialties)}\nClinical focus: {topic_category}",
            "skills_required": f"Reasoning type: {reasoning_type}\nCompetencies: {', '.join(competencies_required)}",
            "skills_list": competencies_required
        }


# Legacy function wrappers for backward compatibility
def format_medqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return MedQAFormatter.format(question_data)

def format_medmcqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    return MedMCQAFormatter.format(question_data)

def format_mmlupro_med_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return MMLUProMedFormatter.format(question_data)

def format_pubmedqa_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return PubMedQAFormatter.format(question_data)

def format_ddxplus_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return DDXPlusFormatter.format(question_data)

def format_medbullets_for_task(question_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return MedBulletsFormatter.format(question_data)