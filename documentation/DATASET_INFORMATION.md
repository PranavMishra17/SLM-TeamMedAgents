# Dataset Information

We evaluate our multi-agent collaboration framework and single-agent baselines across eight diverse medical question-answering datasets, encompassing a wide spectrum of clinical complexity, modality requirements, and reasoning demands. These datasets vary in question complexity, which we broadly categorize based on modality (text-only vs. vision-language), answer format (multiple-choice vs. open-ended), and the depth of medical reasoning required. Generally, questions are deemed more complex if they involve multiple modalities (e.g., medical images with text) or entail lengthy, multi-step diagnostic reasoning tasks.

We conducted **full dataset evaluations** on four datasets where all available questions were used, and employed **1,000-question subsets** (or comparable samples) for the remaining four larger-scale datasets to ensure computational feasibility while maintaining representativeness. Below, we detail each dataset, its characteristics, and provide representative samples:

---

### 1. MedBullets

**Description:** MedBullets comprises 308 USMLE Step 2/3 style clinical questions collected from open-access tweets on X (formerly Twitter) since April 2022. The dataset was introduced in research by Chen et al. (2024) to benchmark large language models on realistic clinical scenarios. The difficulty is comparable to that of USMLE Step 2/3 examinations, which emulate real-world clinical decision-making and diagnostic challenges encountered in hospital and outpatient settings.

**Complexity Level:** **High** - Questions require clinical reasoning, integration of patient history, physical examination findings, and diagnostic test interpretation.
**Format:** Multiple-choice with typically 4-5 options
**Modality:** Text-only (but may include detailed clinical vignettes)

**Sample Question:**
> A 55-year-old man with a 30-pack-year smoking history presents to the emergency department with acute onset of severe chest pain radiating to his back. His blood pressure is 180/110 mmHg in the right arm and 140/90 mmHg in the left arm. A chest X-ray shows widening of the mediastinum. What is the most likely diagnosis?
>
> **Options:** A: Acute myocardial infarction, B: Aortic dissection, C: Pulmonary embolism, D: Pneumothorax, E: Esophageal rupture

**Source:** Chen et al., "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions," arXiv:2402.18060 (2024)
**Availability:** https://github.com/HanjieChen/ChallengeClinicalQA

---

### 2. PubMedQA

**Description:** PubMedQA is a biomedical question answering dataset collected from PubMed abstracts, published at EMNLP-IJCNLP 2019. The dataset contains 1,000 expert-annotated instances, 61,200 unlabeled instances, and 211,300 artificially generated QA instances. Each instance comprises a question (typically derived from a research article title), a context (the corresponding abstract without its conclusion), and a yes/no/maybe answer summarizing the conclusion. PubMedQA is notable as the first QA dataset where reasoning over biomedical research texts, especially their quantitative contents, is required to answer questions.

**Complexity Level:** **Low-to-Moderate** - Binary or ternary (yes/no/maybe) choice format, but requires understanding of biomedical research methodology and evidence interpretation.
**Format:** Yes/No/Maybe with research abstract context
**Modality:** Text-only

**Sample Question:**
> **Question:** "Is selenium supplementation associated with reduced risk of bladder cancer?"
>
> **Context:** "We conducted a randomized, double-blind, placebo-controlled trial to evaluate the effect of selenium supplementation on cancer incidence. A total of 1,312 patients with a history of basal cell or squamous cell carcinoma were randomly assigned to receive 200 μg of selenium daily or placebo. During a mean follow-up of 7.4 years, there was no significant difference in the primary endpoint of basal or squamous cell carcinoma recurrence. However, secondary analysis revealed trends in other cancer types."
>
> **Answer:** Yes/No/Maybe

**Source:** Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering," EMNLP-IJCNLP 2019
**Availability:** https://pubmedqa.github.io
**Human Performance Baseline:** 78.0% accuracy (vs. 55.2% majority baseline)

---

### 3. MedMCQA

**Description:** MedMCQA is a large-scale Multiple-Choice Question Answering dataset designed to address real-world medical entrance exam questions from India. The dataset contains over 194,000 high-quality AIIMS (All India Institute of Medical Sciences) and NEET PG (National Eligibility cum Entrance Test for Postgraduate) entrance exam multiple-choice questions covering 2,400 healthcare topics across 21 medical subjects. Questions have an average token length of 12.77 and high topical diversity. The dataset includes detailed explanations and tests 10+ reasoning abilities across medical subjects.

**Questions Used:** 1000/194,000 questions
**Complexity Level:** **Low-to-Moderate** - Multiple-choice format with 4 options, covering foundational to advanced medical knowledge.
**Format:** 4-option multiple-choice with explanations
**Modality:** Text-only

**Data Distribution in Full Dataset:**
- Training: 183,000 examples (mock & online test series)
- Development: 6,000 examples (NEET PG exams 2001-present)
- Test: 4,000 examples (AIIMS PG exams 1991-present)

**Sample Question:**
> A 28-year-old woman presents with progressive muscle weakness and diplopia. Symptoms worsen with repeated muscle use and improve with rest. Edrophonium test shows marked improvement in muscle strength. What is the mechanism of action of the drug used for long-term treatment of this condition?
>
> **Options:** A: Inhibits acetylcholinesterase, B: Stimulates acetylcholine release, C: Blocks acetylcholine receptors, D: Enhances GABA activity

**Source:** Pal et al., "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering," ACM CHIL 2022
**Availability:** https://medmcqa.github.io

---

### 4. MMLU-Pro Medical Subset

**Description:** MMLU-Pro is an enhanced version of the Massive Multitask Language Understanding benchmark, specifically designed to be more robust and challenging. The full dataset contains 12,000 complex questions across 14 diverse domains. The medical subset encompasses health-related categories including virology, clinical knowledge, anatomy, medical genetics, nutrition, college medicine, aging, and professional medicine. Notably, MMLU-Pro increases answer choices from four to ten options per question, significantly raising difficulty and reducing the probability of success through random guessing. The dataset underwent corrections in April 2024 based on recommendations from medical professionals at Mayo Clinic and INOVA.

**Questions Used:** ~1,200 questions (from health/medicine categories)
**Complexity Level:** **Moderate-to-High** - 10-option multiple-choice requiring deep domain knowledge and reasoning.
**Format:** 10-option multiple-choice with Chain-of-Thought reasoning emphasis
**Modality:** Text-only

**Key Characteristics:**
- Increased from 4 to 10 answer options (reducing random guess success rate from 25% to 10%)
- Chain of Thought (CoT) performance 20% higher than perplexity-based approaches
- Professional medicine subcategory: 166 complex clinical scenarios
- 2024 corrections: 15 medical answers revised by specialists

**Source:** TIGER-Lab, "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark," NeurIPS 2024
**Availability:** https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro




### 5. MedQA (USMLE)

**Description:** The MedQA dataset consists of professional medical board examination questions from the United States Medical Licensing Examination (USMLE), alongside questions from Mainland China and Taiwan medical licensing exams. The complete dataset comprises 61,097 questions total (12,723 English, 34,251 Simplified Chinese, 14,123 Traditional Chinese). Our study focuses exclusively on the English USMLE test set, which contains 1,273 questions formatted as 4- or 5-option multiple-choice queries. These questions assess professional medical knowledge and clinical decision-making abilities at the level required for physician licensure in the United States.

**Total Dataset Size:** 1,273 USMLE questions (English test set)

**Complexity Level:** **Moderate** - Professional-level medical knowledge with clinical reasoning, but text-only format.
**Format:** Multiple-choice (typically 5 options)
**Modality:** Text-only

**Sample Question:**
> A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She denies fever, flank pain, or vaginal discharge. Urinalysis shows positive leukocyte esterase and nitrites. Urine culture grows >100,000 CFU/mL of E. coli. What is the most appropriate treatment?
>
> **Options:**
> A: Ciprofloxacin
> B: Trimethoprim-sulfamethoxazole
> C: Doxycycline
> D: Nitrofurantoin
> E: Observation only

**Source:** Jin et al., "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams," arXiv:2009.13081 (2020)
**Availability:** https://github.com/jind11/MedQA

---

### 6. Path-VQA

**Description:** PathVQA is the first visual question answering dataset specifically designed for pathology images. The dataset contains 32,799 questions from 4,998 pathology images (4,289 unique images after deduplication). Questions are derived from two publicly-available pathology textbooks ("Textbook of Pathology" and "Basic Pathology") and the Pathology Education Informational Resource (PEIR) digital library. The dataset includes both open-ended questions and binary "yes/no" questions, with each question manually verified for correctness. Pathology imaging is crucial for identifying disease causes and effects, and the ability to answer questions about clinical findings in images is essential for medical decision-making.

**Total Dataset Size:** 32,799 questions on 4,998 images
**Questions Used:** 1000
**Complexity Level:** **Moderate-to-High** - Requires visual pathology interpretation combined with medical knowledge.
**Format:** Yes/No questions (for evaluation consistency) and open-ended questions
**Modality:** Vision-Language (pathology images + text)

**Sample Question:**
> **Question:** "How many multi-faceted gallstones are present in the lumen?"
> **Image:** pathology_gallbladder_014.jpg (histopathology slide showing gallbladder lumen)
> **Answer:** Open-ended (e.g., "Two multi-faceted gallstones")

**Evaluation Metrics:**
- Yes/No Accuracy
- Free-form accuracy (for open-ended subset)
- Overall accuracy

**Source:** He et al., "PathVQA: 30000+ Questions for Medical Visual Question Answering," ACL 2021
**Availability:** https://github.com/UCSD-AI4H/PathVQA
**License:** MIT License

---

### 7. PMC-VQA

**Description:** PMC-VQA is a large-scale medical visual question-answering dataset containing 227,000 VQA pairs across 149,000 images from PubMed Central's OpenAccess subset. The dataset covers various imaging modalities and diseases, with radiological images comprising approximately 80% of the total. PMC-VQA significantly surpasses existing medical VQA datasets in both quantity and diversity. The dataset was created using a scalable pipeline with a generative-based model (MedVInT) that aligns visual information from a pre-trained vision encoder with a large language model. The associated model achieved over 80% accuracy on multi-choice selection tasks, outperforming existing benchmarks on VQA-RAD, SLAKE, and ImageClef-VQA-2019.

**Total Dataset Size:** 227,000 VQA pairs on 149,000 images
**Questions Used:** 1000 questions
**Complexity Level:** **High** - Requires interpretation of diverse medical imaging modalities (X-ray, CT, MRI, ultrasound, etc.) combined with medical reasoning.
**Format:** Multiple-choice questions with image input
**Modality:** Vision-Language (80% radiological images + clinical text)

**Sample Question:**
> **Question:** "Based on the chest radiograph, what is the most likely location of the opacity?"
> **Image:** PMC7623491_chest_xray.jpg (posteroanterior chest X-ray)
> **Options:**
> A: Right upper lobe
> B: Right middle lobe
> C: Left lower lobe
> D: Lingula
> E: Right lower lobe

**Source:** Zhang et al., "PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering," arXiv:2305.10415 (2023); Nature Communications Medicine (2024)
**Availability:** https://github.com/xiaoman-zhang/PMC-VQA
**Model Performance:** MedVInT achieves 80%+ accuracy on multi-choice selection

---

### 8. DDXPlus

**Description:** DDXPlus is a large-scale synthetic dataset containing approximately 1.3 million synthetic patient cases designed for automatic symptom detection and differential diagnosis research. Published at NeurIPS 2022 Datasets and Benchmarks, DDXPlus is the first large-scale dataset to include differential diagnoses alongside ground truth pathology, symptoms, and antecedents for each patient. Unlike existing datasets that only contain binary symptoms, DDXPlus includes categorical and multi-choice symptoms organized in a hierarchy, enabling more realistic and efficient patient-system interactions. Patients are synthesized using a proprietary medical knowledge base and a commercial rule-based automatic symptom detection system, characterized by socio-demographic data, a pathology, related symptoms/antecedents, and a differential diagnosis.

**Total Dataset Size:** ~1.3 million synthetic patients
**Questions Used:** 1000 cases
**Complexity Level:** **Low-to-Moderate** - Text-only multiple-choice differential diagnosis, but requires integration of multiple patient attributes.
**Format:** Multiple-choice with patient attributes (age, sex, symptoms, antecedents)
**Modality:** Text-only (structured patient data)

**Sample Patient Case:**
> **Age:** 42
> **Sex:** M
> **Chief Complaint:** Shortness of breath and chest tightness
> **Symptoms:** ['dyspnea', 'chest_tightness', 'wheezing', 'cough', 'sputum_production']
> **Antecedents:** ['smoking_history', 'allergic_rhinitis', 'family_history_asthma']
> **Initial evidence:** 'dyspnea'
> **Differential Diagnosis Options:**
> (A) Asthma
> (B) Chronic Obstructive Pulmonary Disease (COPD)
> (C) Pneumonia
> (D) Congestive Heart Failure

**Important Note:** This dataset consists of synthetic patients and is intended for research purposes only, not clinical deployment.

**Source:** Fansi Tchango et al., "DDXPlus: A New Dataset For Automatic Medical Diagnosis," NeurIPS 2022
**Availability:** https://figshare.com/articles/dataset/DDXPlus_Dataset/20043374
**GitHub:** https://github.com/mila-iqia/ddxplus

---

## Key Characteristics

### Modality Distribution
- **Text-only:** 5 datasets (MedBullets, PubMedQA, MedMCQA, MMLU-Pro Med, MedQA, DDXPlus)
- **Vision-Language:** 2 datasets (Path-VQA, PMC-VQA)

### Complexity Levels
- **Low-to-Moderate:** PubMedQA, MedMCQA, DDXPlus (text-based, binary/multiple-choice)
- **Moderate:** MedQA, MMLU-Pro Med (professional-level knowledge)
- **Moderate-to-High:** Path-VQA (pathology imaging interpretation)
- **High:** MedBullets, PMC-VQA (clinical reasoning with complex scenarios/imaging)

### Geographic & Exam Origins
- **United States:** MedQA (USMLE), MedBullets (USMLE Step 2/3)
- **India:** MedMCQA (AIIMS & NEET PG)
- **International/Research:** PubMedQA (biomedical literature), PMC-VQA (PubMed Central)
- **Synthetic:** DDXPlus (computer-generated patients)

### Evaluation Strategy Rationale

We employed **full dataset evaluation** for smaller, more challenging datasets (MedBullets, PubMedQA, MedMCQA, MMLU-Pro Med) where all questions provide high-quality assessment signal. For larger datasets (MedQA, Path-VQA, PMC-VQA, DDXPlus), we utilized **~1000-question representative subsets** to balance computational cost with statistical reliability while maintaining diversity across medical specialties, question types, and complexity levels.

---

## References

1. Jin, D., et al. (2020). "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams." *arXiv:2009.13081*.

2. Jin, Q., et al. (2019). "PubMedQA: A Dataset for Biomedical Research Question Answering." *EMNLP-IJCNLP 2019*.

3. Fansi Tchango, A., et al. (2022). "DDXPlus: A New Dataset For Automatic Medical Diagnosis." *NeurIPS 2022 Datasets and Benchmarks*.

4. Zhang, X., et al. (2023). "PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering." *arXiv:2305.10415*; *Nature Communications Medicine* (2024).

5. He, X., et al. (2020). "PathVQA: 30000+ Questions for Medical Visual Question Answering." *ACL 2021*.

6. Pal, A., et al. (2022). "MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering." *ACM CHIL 2022*.

7. Chen, H., et al. (2024). "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions." *arXiv:2402.18060*.

8. TIGER-Lab (2024). "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." *NeurIPS 2024*.

---

*This document provides comprehensive information on the eight medical QA datasets used to evaluate our multi-agent collaborative framework and single-agent baseline approaches.*
