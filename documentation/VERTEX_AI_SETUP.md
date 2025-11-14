# Vertex AI Endpoint Setup Guide

## Overview

This guide explains how to deploy MedGemma on Vertex AI and use it with the SLM-TeamMedAgents multi-agent system. This setup provides production-grade scalability while maintaining full compatibility with all existing features.

**Status:** ✅ Fully Implemented and Ready to Use

---

## Why Vertex AI Endpoints?

**Problem:** MedGemma is NOT available on Google AI Studio

**Solution:** Deploy MedGemma to Vertex AI Model Garden as a scalable endpoint

**Benefits:**
- ✅ Access to MedGemma 4B and 27B multimodal models
- ✅ Production-grade scalability and reliability
- ✅ Full multimodal support (text + images)
- ✅ Same ADK framework and algorithm structure
- ✅ All teamwork components work unchanged
- ✅ All datasets work unchanged (including vision datasets)

---

## Prerequisites

1. **Google Cloud Platform Account**
   - Active GCP project with billing enabled
   - Vertex AI API enabled

2. **Required Tools**
   ```bash
   # Install gcloud CLI
   # https://cloud.google.com/sdk/docs/install

   # Install Python dependencies
   pip install google-adk google-genai google-cloud-aiplatform
   ```

3. **Access Permissions**
   - `roles/aiplatform.user` - To deploy and use endpoints
   - `roles/aiplatform.admin` - To manage Model Garden deployments

---

## Step 1: Deploy MedGemma to Vertex AI

### Option A: Deploy via Google Cloud Console (Recommended for First Time)

1. **Navigate to Vertex AI Model Garden:**
   - Go to: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma
   - Or: Cloud Console → Vertex AI → Model Garden → Search "MedGemma"

2. **Select MedGemma Model:**
   - Choose **MedGemma 4B Multimodal** (`medgemma-4b-it`) for testing
   - OR **MedGemma 27B Multimodal** for higher performance

3. **Deploy to Endpoint:**
   - Click **"Deploy"** button
   - Select **"Vertex AI"** as deployment option
   - Configure deployment settings:
     - **Name:** `medgemma-multimodal-endpoint` (or your choice)
     - **Region:** `us-central1` (recommended)
     - **Machine type:** `g2-standard-24` (2 NVIDIA_L4 GPUs)
     - **Accelerator:** NVIDIA_L4
     - **Min replicas:** 1
     - **Max replicas:** 3 (for autoscaling)

4. **Wait for Deployment:**
   - Deployment takes 10-15 minutes
   - Status will change from "Creating" → "Deployed"

5. **Copy Endpoint Details:**
   - **Endpoint ID:** Found in endpoint details (e.g., `1234567890123456789`)
   - **Region:** `us-central1`
   - **Project ID:** Your GCP project ID

### Option B: Deploy via gcloud CLI

```bash
# Set variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export ENDPOINT_NAME="medgemma-multimodal-endpoint"
export MODEL_VERSION="medgemma-4b-it"

# Deploy model to endpoint
gcloud ai endpoints deploy-model $ENDPOINT_NAME \
  --project=$PROJECT_ID \
  --region=$REGION \
  --model=$MODEL_VERSION \
  --machine-type=g2-standard-24 \
  --accelerator=nvidia-l4,count=2 \
  --min-replica-count=1 \
  --max-replica-count=3

# Get endpoint ID
gcloud ai endpoints list \
  --project=$PROJECT_ID \
  --region=$REGION \
  --filter="displayName:$ENDPOINT_NAME"
```

---

## Step 2: Configure Authentication

Choose ONE of the following methods:

### Method A: Application Default Credentials (Recommended for Local Development)

```bash
# Login with your Google account
gcloud auth application-default login

# Set default project
gcloud config set project YOUR_PROJECT_ID
```

### Method B: Service Account (Recommended for Production)

```bash
# Create service account
gcloud iam service-accounts create medgemma-service \
  --display-name="MedGemma Service Account"

# Grant Vertex AI permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:medgemma-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create medgemma-key.json \
  --iam-account=medgemma-service@YOUR_PROJECT_ID.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/medgemma-key.json"
```

---

## Step 3: Configure Environment Variables

Create or update your `.env` file:

```bash
# Vertex AI Configuration (REQUIRED)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
VERTEX_AI_ENDPOINT_ID=1234567890123456789
GOOGLE_GENAI_USE_VERTEXAI=TRUE

# Optional: Authentication (if using service account)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/medgemma-key.json
```

**Finding Your Values:**
- **GOOGLE_CLOUD_PROJECT:** Your GCP project ID (shown in Cloud Console header)
- **GOOGLE_CLOUD_LOCATION:** Region where endpoint is deployed (default: us-central1)
- **VERTEX_AI_ENDPOINT_ID:** Numeric ID from endpoint details page
  - Example: `1234567890123456789`
  - NOT the full resource name
- **GOOGLE_GENAI_USE_VERTEXAI:** Must be set to `TRUE` (case-insensitive)

**Verify Configuration:**
```bash
# Check environment variables
echo $GOOGLE_CLOUD_PROJECT
echo $VERTEX_AI_ENDPOINT_ID
echo $GOOGLE_GENAI_USE_VERTEXAI

# Or using Python
python -c "import os; print('Project:', os.getenv('GOOGLE_CLOUD_PROJECT')); print('Endpoint:', os.getenv('VERTEX_AI_ENDPOINT_ID'))"
```

---

## Step 4: Verify Setup

Test that everything is configured correctly:

```bash
# Quick test (10 questions)
python run_simulation_vertex_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --n-agents 3

# Expected output:
# ✓ Vertex AI configuration verified:
#   Project: your-project-id
#   Location: us-central1
#   Endpoint: 1234567890123456789
# Processing questions: 100%|████████| 10/10 [02:30<00:00, 15.0s/question]
# Overall Accuracy: 35.0% (3.5/10)
```

If you see errors:
- **"GOOGLE_CLOUD_PROJECT not set"** → Check environment variables
- **"Endpoint not found"** → Verify endpoint ID is correct
- **"Permission denied"** → Check authentication and IAM roles
- **"GOOGLE_GENAI_USE_VERTEXAI not set to TRUE"** → Check spelling and case

---

## Usage Examples

### Basic Usage

```bash
# Text dataset (MedQA)
python run_simulation_vertex_adk.py \
  --dataset medqa \
  --n-questions 50 \
  --n-agents 3

# Vision dataset (PMC-VQA)
python run_simulation_vertex_adk.py \
  --dataset pmc_vqa \
  --n-questions 20 \
  --n-agents 3
```

### With Teamwork Components

```bash
# All teamwork components enabled
python run_simulation_vertex_adk.py \
  --dataset medqa \
  --n-questions 50 \
  --all-teamwork

# Individual components
python run_simulation_vertex_adk.py \
  --dataset medqa \
  --n-questions 50 \
  --smm \
  --leadership \
  --trust
```

### Command-Line Options

```bash
# Required
--dataset DATASET          # Dataset: medqa, medmcqa, pubmedqa, mmlupro, ddxplus, medbullets, pmc_vqa, path_vqa

# Optional
--n-questions N            # Number of questions (default: 10)
--n-agents N               # Fixed agent count (default: dynamic 2-4)
--output-dir DIR           # Output directory (default: multi-agent-gemma/results_vertex)
--seed N                   # Random seed (default: 42)

# Vertex AI (optional if using env vars)
--endpoint-id ID           # Vertex AI endpoint ID
--project-id ID            # GCP project ID
--location REGION          # Vertex AI region (default: us-central1)

# Teamwork Components
--smm                      # Enable Shared Mental Model
--leadership               # Enable Leadership
--team-orientation         # Enable Team Orientation
--trust                    # Enable Trust Network
--mutual-monitoring        # Enable Mutual Monitoring
--all-teamwork             # Enable ALL components
--n-turns N                # Discussion turns (2 or 3)
```

---

## Supported Datasets

All 8 datasets from the original system are fully supported:

**Text Datasets:**
- `medqa` - USMLE-style medical questions
- `medmcqa` - Indian medical entrance exams
- `pubmedqa` - Biomedical research questions
- `mmlupro` - MMLU-Pro Health subset
- `medbullets` - Clinical case questions
- `ddxplus` - Differential diagnosis

**Vision Datasets (Multimodal):**
- `pmc_vqa` - Medical visual QA (227k images)
- `path_vqa` - Pathology image analysis (33k images)

---

## Output Structure

Results are saved with the same structure as Google AI Studio version:

```
multi-agent-gemma/results_vertex/
├── medqa_50q_run1/
│   ├── questions/
│   │   ├── q001_results.json
│   │   └── ...
│   ├── summary_report.json
│   ├── accuracy_summary.json
│   ├── convergence_analysis.json
│   ├── agent_performance.json
│   ├── simulation.log
│   └── config.json
```

**Key Differences:**
- Output directory defaults to `results_vertex/` instead of `results/`
- `config.json` includes Vertex AI endpoint information
- Performance metrics include endpoint-specific timing

---

## Performance Expectations

### Timing (per question, N=3 agents, 2 turns)

| Phase | Google AI Studio | Vertex AI Endpoint |
|-------|-----------------|-------------------|
| Recruitment | 2-5s | 3-6s |
| Round 2 | 5-8s | 6-10s |
| Round 3 | 6-10s | 8-12s |
| **Total** | **15-25s** | **20-30s** |

*Vertex AI adds ~5s latency due to network overhead*

### Cost Estimate (Approximate)

**MedGemma 4B on g2-standard-24:**
- **Compute:** ~$1.50/hour
- **Per question (30s):** ~$0.0125
- **100 questions:** ~$1.25

**MedGemma 27B (larger machine):**
- **Compute:** ~$3.00/hour
- **Per question (45s):** ~$0.0375
- **100 questions:** ~$3.75

*Costs vary by region and machine type. Check current pricing at https://cloud.google.com/vertex-ai/pricing*

---

## Troubleshooting

### Common Issues

**1. "Vertex AI configuration incomplete"**
```bash
# Check all required env vars are set
python -c "
import os
print('Project:', os.getenv('GOOGLE_CLOUD_PROJECT'))
print('Location:', os.getenv('GOOGLE_CLOUD_LOCATION'))
print('Endpoint:', os.getenv('VERTEX_AI_ENDPOINT_ID'))
print('Use Vertex:', os.getenv('GOOGLE_GENAI_USE_VERTEXAI'))
"

# Set missing variables
export GOOGLE_CLOUD_PROJECT="your-project-id"
export VERTEX_AI_ENDPOINT_ID="your-endpoint-id"
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

**2. "Authentication failed"**
```bash
# Check current authentication
gcloud auth list

# Re-authenticate if needed
gcloud auth application-default login

# Or verify service account key path
echo $GOOGLE_APPLICATION_CREDENTIALS
```

**3. "Endpoint not found or not ready"**
```bash
# List all endpoints in your project
gcloud ai endpoints list \
  --project=$GOOGLE_CLOUD_PROJECT \
  --region=$GOOGLE_CLOUD_LOCATION

# Check endpoint status
gcloud ai endpoints describe $VERTEX_AI_ENDPOINT_ID \
  --project=$GOOGLE_CLOUD_PROJECT \
  --region=$GOOGLE_CLOUD_LOCATION
```

**4. "Rate limit exceeded"**
- Vertex AI endpoints have quotas
- Reduce concurrent requests or request quota increase
- Use `--seed` flag to avoid reprocessing same questions

**5. "Permission denied"**
```bash
# Check IAM permissions
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT \
  --flatten="bindings[].members" \
  --filter="bindings.members:user:YOUR_EMAIL"

# Grant necessary role if missing
gcloud projects add-iam-policy-binding $GOOGLE_CLOUD_PROJECT \
  --member="user:YOUR_EMAIL" \
  --role="roles/aiplatform.user"
```

---

## Comparison: Google AI Studio vs Vertex AI

| Feature | Google AI Studio | Vertex AI Endpoint |
|---------|-----------------|-------------------|
| **MedGemma Access** | ❌ Not Available | ✅ Available |
| **Setup Complexity** | Simple (API key) | Moderate (deployment + auth) |
| **Cost** | Free tier available | Pay per compute hour |
| **Scalability** | Limited by quotas | Auto-scaling available |
| **Production Ready** | No SLA | Enterprise SLA |
| **Multimodal Support** | ✅ Yes | ✅ Yes |
| **ADK Compatible** | ✅ Yes | ✅ Yes |
| **Code Changes** | None (use run_simulation_adk.py) | None (use run_simulation_vertex_adk.py) |

---

## Switching Between Google AI Studio and Vertex AI

Both implementations use the same codebase:

**Google AI Studio (Gemma models):**
```bash
python run_simulation_adk.py \
  --dataset medqa \
  --n-questions 50 \
  --model gemma3_4b
```

**Vertex AI (MedGemma endpoint):**
```bash
python run_simulation_vertex_adk.py \
  --dataset medqa \
  --n-questions 50
```

**Key Differences:**
- Different runner scripts
- Vertex AI requires endpoint deployment and environment configuration
- Results are compatible and comparable
- All other features (datasets, teamwork, metrics) are identical

---

## Next Steps

1. **Deploy MedGemma endpoint** (Step 1)
2. **Configure authentication** (Step 2)
3. **Set environment variables** (Step 3)
4. **Run verification test** (Step 4)
5. **Start experiments** with your datasets

**For Questions or Issues:**
- Check this documentation first
- Review error messages carefully
- Check Vertex AI logs in Cloud Console
- Ensure endpoint is deployed and running

---

## Additional Resources

- **Vertex AI Documentation:** https://cloud.google.com/vertex-ai/docs
- **MedGemma Model Card:** https://developers.google.com/health-ai-developer-foundations/medgemma
- **ADK Documentation:** https://google.github.io/adk-docs/
- **Pricing Calculator:** https://cloud.google.com/products/calculator

---

**Last Updated:** 2025-11-14
**Version:** 1.0.0
**Status:** Production Ready
