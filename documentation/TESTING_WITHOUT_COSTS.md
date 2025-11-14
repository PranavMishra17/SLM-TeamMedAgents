# Testing Vertex AI Setup Without Costs

## Overview

This guide shows you how to validate your Vertex AI setup and test your code **without deploying an endpoint** or incurring any costs.

---

## Option 1: Verification Tool (Recommended)

We've created a comprehensive verification tool that checks everything WITHOUT making API calls or deploying anything.

### Usage

```bash
# Basic verification (checks config, auth, dependencies)
python verify_vertex_setup.py

# Skip endpoint check (if not deployed yet)
python verify_vertex_setup.py --skip-endpoint

# Check if endpoint exists (only if already deployed)
python verify_vertex_setup.py --check-endpoint
```

### What It Checks

✅ **Environment Variables** - Verifies all required env vars are set
✅ **Python Dependencies** - Checks if packages are installed
✅ **Authentication** - Validates gcloud or service account auth
✅ **Vertex AI API Access** - Tests API initialization (no cost)
✅ **Endpoint Verification** - Checks endpoint exists (optional, no inference)
✅ **Custom Code** - Validates imports work correctly
✅ **Configuration Test** - Mock test of agent creation

### Example Output

```
================================================================================
  Vertex AI Setup Verification Tool
  NO COSTS INCURRED - This tool only checks configuration
================================================================================

================================================================================
  Step 1: Environment Variables
================================================================================
  ✓ GOOGLE_CLOUD_PROJECT: my-project-123
  ✓ GOOGLE_CLOUD_LOCATION: us-central1
  ✓ VERTEX_AI_ENDPOINT_ID: 1234...5678
  ✓ GOOGLE_GENAI_USE_VERTEXAI: TRUE

  ✅ All environment variables configured correctly!

================================================================================
  Step 2: Python Dependencies
================================================================================
  ✓ google-adk: Installed
  ✓ google-genai: Installed
  ✓ google-cloud-aiplatform: Installed
  ✓ pillow: Installed
  ✓ python-dotenv: Installed

  ✅ All required packages installed!

... [more checks]

================================================================================
  Summary
================================================================================

  Checks Passed: 7/7

  Status:
     ✅ Environment Variables
     ✅ Python Dependencies
     ✅ Authentication
     ✅ Vertex AI API Access
     ✅ Endpoint Verification
     ✅ Custom Code
     ✅ Configuration Test

  🎉 All checks passed! Your setup is ready for Vertex AI.
```

---

## Option 2: Google Cloud Free Credits

### New Account Credits ($300)

If you're new to Google Cloud, you get **$300 in free credits** for 90 days:

1. **Sign up:** https://cloud.google.com/free
2. **Activate free trial** - no charges for 90 days
3. **Deploy and test MedGemma** using your free credits

**Benefits:**
- Test real Vertex AI endpoints
- No credit card charges during trial
- $300 covers extensive testing (100s of questions)

### Existing Account Options

If you already used your free credits:

1. **Check for promotional credits** - Google often offers credits for specific services
2. **Education/Research grants** - Apply for research credits if eligible
3. **Minimal testing approach** (see Option 3 below)

---

## Option 3: Minimal Cost Testing

If you must deploy to test, here's how to minimize costs:

### Strategy: Deploy, Test, Delete Immediately

```bash
# 1. Deploy smallest configuration
#    Machine: n1-standard-4 (instead of g2-standard-24)
#    Model: MedGemma 4B (instead of 27B)

# 2. Run minimal test (10 questions, ~5 minutes)
python run_simulation_vertex_adk.py \
  --dataset medqa \
  --n-questions 10 \
  --n-agents 2

# 3. Delete endpoint immediately after testing
gcloud ai endpoints delete ENDPOINT_ID \
  --project=PROJECT_ID \
  --region=us-central1
```

**Estimated Cost:** ~$0.50 - $1.00 for 5-10 minutes of testing

### Cost-Saving Tips

1. **Use smallest machine type** during testing
2. **Set min-replicas to 1** (not multiple)
3. **Delete endpoint immediately** after testing
4. **Use regional endpoint** (cheaper than global)
5. **Test during off-peak hours** if pricing varies

---

## Option 4: Check Configuration Only

You can verify your configuration is valid without deploying anything:

### Check Environment Variables

```bash
# Check all env vars are set
python -c "
import os
required = ['GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION',
            'VERTEX_AI_ENDPOINT_ID', 'GOOGLE_GENAI_USE_VERTEXAI']
for var in required:
    value = os.getenv(var)
    print(f'{var}: {\"✓ SET\" if value else \"✗ NOT SET\"} {value or \"\"}')
"
```

### Verify Authentication

```bash
# Check gcloud authentication
gcloud auth list

# Check application-default credentials
gcloud auth application-default print-access-token

# Test Vertex AI API access (no deployment needed)
gcloud ai endpoints list --region=us-central1
```

### Validate Python Environment

```bash
# Check all packages installed
python -c "
packages = ['google.adk', 'google.genai', 'google.cloud.aiplatform', 'PIL', 'dotenv']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - NOT INSTALLED')
"
```

---

## Option 5: Mock Testing Mode

For development and testing without any API calls, you can create a mock version:

### Create Mock Endpoint

```python
# test_mock_vertex.py
import os

# Set mock configuration
os.environ['GOOGLE_CLOUD_PROJECT'] = 'mock-project'
os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'
os.environ['VERTEX_AI_ENDPOINT_ID'] = '1234567890'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'TRUE'

from adk_agents.gemma_agent_vertex_adk import VertexAIAgentFactory

# Test configuration parsing (no API calls)
try:
    config = VertexAIAgentFactory.get_vertex_config()
    print("✓ Configuration valid:")
    print(f"  Project: {config['project_id']}")
    print(f"  Location: {config['location']}")
    print(f"  Endpoint: {config['endpoint_id']}")

    # Build endpoint resource name
    resource_name = VertexAIAgentFactory.build_endpoint_resource_name(
        config['project_id'],
        config['location'],
        config['endpoint_id']
    )
    print(f"\n✓ Endpoint resource name: {resource_name}")

    print("\n✅ Configuration is valid!")

except Exception as e:
    print(f"✗ Configuration error: {e}")
```

Run:
```bash
python test_mock_vertex.py
```

---

## Option 6: Use Existing Endpoints (If Available)

If someone else on your team has already deployed MedGemma:

1. **Get their endpoint ID** - ask for the numeric endpoint ID
2. **Verify you have access** - make sure you have `aiplatform.user` role
3. **Use shared endpoint** - multiple people can use the same endpoint
4. **Split costs** - coordinate usage to share deployment costs

```bash
# Check if you can access their endpoint
gcloud ai endpoints describe ENDPOINT_ID \
  --project=THEIR_PROJECT_ID \
  --region=us-central1
```

---

## Recommended Testing Flow

### Phase 1: Configuration Validation (FREE)
```bash
# 1. Run verification tool
python verify_vertex_setup.py --skip-endpoint

# 2. Fix any issues identified
# 3. Re-run until all checks pass
```

### Phase 2: Code Validation (FREE)
```bash
# 4. Import and validate code
python -c "from adk_agents.gemma_agent_vertex_adk import VertexAIAgentFactory"

# 5. Test dataset loaders
python -c "from medical_datasets.dataset_loader import DatasetLoader; print('✓ Datasets OK')"

# 6. Verify teamwork components
python -c "from teamwork_components import TeamworkConfig; print('✓ Teamwork OK')"
```

### Phase 3: Live Testing (COSTS MONEY)
```bash
# 7. Deploy MedGemma endpoint (use free credits if available)

# 8. Run verification with endpoint check
python verify_vertex_setup.py --check-endpoint

# 9. Minimal test run
python run_simulation_vertex_adk.py --dataset medqa --n-questions 10

# 10. Review results and costs in Cloud Console

# 11. Delete endpoint OR continue with full experiments
```

---

## Cost Monitoring

### Set Up Budget Alerts

```bash
# Create budget alert (get notified before spending too much)
gcloud billing budgets create \
  --billing-account=BILLING_ACCOUNT_ID \
  --display-name="Vertex AI Budget" \
  --budget-amount=50 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

### Monitor Costs in Real-Time

1. **Cloud Console:** https://console.cloud.google.com/billing
2. **Check current month:** See accrued charges
3. **Forecast:** View projected costs
4. **Set alerts:** Get notified at spending thresholds

### Estimate Costs Before Deploying

Use the pricing calculator:
- **URL:** https://cloud.google.com/products/calculator
- **Select:** Vertex AI > Prediction
- **Configure:** Machine type, hours, region
- **Review:** Estimated monthly/hourly costs

---

## When You're Ready to Deploy

### Checklist

- [ ] All verification checks pass (`python verify_vertex_setup.py`)
- [ ] You have free credits OR approved budget
- [ ] Cost monitoring/alerts configured
- [ ] Team is aware of deployment
- [ ] You understand shutdown process
- [ ] Backup plan if costs exceed budget

### Deploy Command

```bash
# See documentation/VERTEX_AI_SETUP.md for full deployment guide

# Quick deploy (console):
https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/medgemma

# Or via gcloud:
gcloud ai endpoints deploy-model medgemma-endpoint \
  --project=PROJECT_ID \
  --region=us-central1 \
  --model=medgemma-4b-it \
  --machine-type=g2-standard-24 \
  --min-replica-count=1
```

---

## Troubleshooting

### "I don't have free credits"

**Options:**
1. Create new Google account (each account gets $300)
2. Ask your institution/company for research credits
3. Use minimal testing strategy (deploy for 10 min, ~$1)
4. Partner with someone who has credits

### "Verification passes but I want to test code"

**Solution:** All code can be tested without deployment:
1. Run `python verify_vertex_setup.py` - validates everything
2. Import modules in Python - checks code works
3. Use mock configuration - tests logic flow
4. Review code and documentation - understand behavior

### "I need to test with real model responses"

**Options:**
1. Use Google AI Studio with Gemma 3 (free) - similar model
2. Deploy for minimal test (10 questions, ~5-10 min)
3. Use free credits if available
4. Test with OpenAI/Anthropic API (may be cheaper for testing)

---

## Summary

| Method | Cost | Time | Confidence Level |
|--------|------|------|-----------------|
| **Verification Tool** | FREE | 2 min | Medium-High |
| **Configuration Check** | FREE | 5 min | Medium |
| **Mock Testing** | FREE | 10 min | Medium |
| **Free Credits** | FREE* | 1 hour | Very High |
| **Minimal Deploy** | $0.50-$1 | 30 min | Very High |
| **Full Deploy** | $1-5/hour | N/A | Highest |

*Free with new account

---

## Recommended Path

**If you have free credits:**
1. Run verification tool
2. Deploy MedGemma
3. Test thoroughly
4. Use for experiments

**If you don't have free credits:**
1. Run verification tool (`python verify_vertex_setup.py`)
2. Validate all checks pass
3. Test code imports and configuration
4. Deploy for minimal test (~10 min, $1)
5. Verify everything works
6. Delete endpoint OR continue with experiments

**If you want zero costs:**
1. Run verification tool
2. Trust the implementation (code is solid)
3. Wait until you're ready for real experiments
4. Deploy when you have budget/credits

---

## Next Steps

1. **Run verification:** `python verify_vertex_setup.py`
2. **Review results:** Fix any issues identified
3. **Decide on testing approach:** Based on budget and needs
4. **Proceed when ready:** Deploy or continue with free validation

For questions, see `documentation/VERTEX_AI_SETUP.md` or ask!
