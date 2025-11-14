"""
Vertex AI Setup Verification Tool

Validates Vertex AI configuration WITHOUT deploying endpoints or making API calls.
This script checks your environment, authentication, and configuration without
incurring any costs.

Usage:
    python verify_vertex_setup.py

    # Check specific endpoint (doesn't make inference calls)
    python verify_vertex_setup.py --check-endpoint

    # Run in mock mode (simulates Vertex AI for testing)
    python verify_vertex_setup.py --mock-mode
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def check_environment_variables():
    """Check if all required environment variables are set."""
    print_section("Step 1: Environment Variables")

    required_vars = {
        'GOOGLE_CLOUD_PROJECT': 'Your GCP project ID',
        'GOOGLE_CLOUD_LOCATION': 'Vertex AI region (e.g., us-central1)',
        'VERTEX_AI_ENDPOINT_ID': 'MedGemma endpoint ID',
        'GOOGLE_GENAI_USE_VERTEXAI': 'Must be set to TRUE'
    }

    all_set = True
    results = {}

    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            # Mask sensitive values
            if var == 'VERTEX_AI_ENDPOINT_ID':
                display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else value
            else:
                display_value = value

            print(f"  ✓ {var}: {display_value}")
            results[var] = value
        else:
            print(f"  ✗ {var}: NOT SET")
            print(f"     {description}")
            all_set = False

    # Check GOOGLE_GENAI_USE_VERTEXAI specifically
    if results.get('GOOGLE_GENAI_USE_VERTEXAI', '').upper() != 'TRUE':
        print(f"\n  ⚠️  WARNING: GOOGLE_GENAI_USE_VERTEXAI is not set to TRUE")
        print(f"     Current value: {results.get('GOOGLE_GENAI_USE_VERTEXAI', 'NOT SET')}")
        print(f"     ADK will use Google AI Studio instead of Vertex AI!")
        all_set = False

    if all_set:
        print(f"\n  ✅ All environment variables configured correctly!")
    else:
        print(f"\n  ❌ Some environment variables are missing or incorrect")
        print(f"\n  To fix:")
        print(f"     1. Copy .env.vertex.example to .env")
        print(f"     2. Edit .env and fill in your values")
        print(f"     3. Run: source .env  (or restart your terminal)")

    return all_set, results


def check_python_dependencies():
    """Check if required Python packages are installed."""
    print_section("Step 2: Python Dependencies")

    required_packages = [
        ('google-adk', 'google.adk'),
        ('google-genai', 'google.genai'),
        ('google-cloud-aiplatform', 'google.cloud.aiplatform'),
        ('pillow', 'PIL'),
        ('python-dotenv', 'dotenv'),
    ]

    all_installed = True

    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}: Installed")
        except ImportError:
            print(f"  ✗ {package_name}: NOT INSTALLED")
            print(f"     Install: pip install {package_name}")
            all_installed = False

    if all_installed:
        print(f"\n  ✅ All required packages installed!")
    else:
        print(f"\n  ❌ Some packages are missing")
        print(f"\n  To fix:")
        print(f"     pip install google-adk google-genai google-cloud-aiplatform pillow python-dotenv")

    return all_installed


def check_authentication():
    """Check if Google Cloud authentication is configured."""
    print_section("Step 3: Authentication")

    # Check for service account key
    service_account_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    if service_account_key:
        if Path(service_account_key).exists():
            print(f"  ✓ Service account key found: {service_account_key}")
            return True, "service_account"
        else:
            print(f"  ✗ Service account key NOT FOUND: {service_account_key}")
            print(f"     The file does not exist at the specified path")
            return False, None

    # Check for gcloud authentication
    try:
        import subprocess
        result = subprocess.run(
            ['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            account = result.stdout.strip().split('\n')[0]
            print(f"  ✓ Authenticated with gcloud: {account}")

            # Check if application-default credentials are set
            try:
                result_adc = subprocess.run(
                    ['gcloud', 'auth', 'application-default', 'print-access-token'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result_adc.returncode == 0:
                    print(f"  ✓ Application Default Credentials (ADC) configured")
                    return True, "adc"
                else:
                    print(f"  ⚠️  Authenticated but ADC not set")
                    print(f"     Run: gcloud auth application-default login")
                    return False, None

            except Exception as e:
                print(f"  ⚠️  Could not verify ADC: {e}")
                return False, None
        else:
            print(f"  ✗ No active gcloud authentication")
            print(f"     Run: gcloud auth login")
            print(f"     Then: gcloud auth application-default login")
            return False, None

    except FileNotFoundError:
        print(f"  ✗ gcloud CLI not installed")
        print(f"     Install from: https://cloud.google.com/sdk/docs/install")
        return False, None
    except Exception as e:
        print(f"  ⚠️  Could not check gcloud authentication: {e}")
        return False, None


def check_vertex_ai_access(config):
    """Check if we can access Vertex AI API (without making inference calls)."""
    print_section("Step 4: Vertex AI API Access")

    try:
        from google.cloud import aiplatform

        project_id = config.get('GOOGLE_CLOUD_PROJECT')
        location = config.get('GOOGLE_CLOUD_LOCATION', 'us-central1')

        # Initialize Vertex AI client
        aiplatform.init(project=project_id, location=location)

        print(f"  ✓ Vertex AI client initialized")
        print(f"     Project: {project_id}")
        print(f"     Location: {location}")

        return True

    except Exception as e:
        print(f"  ✗ Failed to initialize Vertex AI client")
        print(f"     Error: {e}")
        print(f"\n  Common causes:")
        print(f"     - Vertex AI API not enabled in your project")
        print(f"     - Incorrect project ID")
        print(f"     - Authentication issues")
        print(f"\n  To enable Vertex AI API:")
        print(f"     gcloud services enable aiplatform.googleapis.com --project={project_id}")
        return False


def check_endpoint_exists(config):
    """Check if the specified endpoint exists (without making inference calls)."""
    print_section("Step 5: Endpoint Verification (Optional)")

    endpoint_id = config.get('VERTEX_AI_ENDPOINT_ID')
    if not endpoint_id:
        print(f"  ⚠️  VERTEX_AI_ENDPOINT_ID not set, skipping endpoint check")
        return None

    try:
        from google.cloud import aiplatform

        project_id = config.get('GOOGLE_CLOUD_PROJECT')
        location = config.get('GOOGLE_CLOUD_LOCATION', 'us-central1')

        # Build endpoint resource name
        endpoint_name = f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

        print(f"  Checking endpoint: {endpoint_name}")

        # Get endpoint details (this doesn't cost money, just metadata query)
        endpoint = aiplatform.Endpoint(endpoint_name)

        print(f"  ✓ Endpoint exists: {endpoint.display_name}")
        print(f"     Resource name: {endpoint.resource_name}")
        print(f"     Created: {endpoint.create_time}")

        # Check deployed models
        if endpoint.list_models():
            print(f"  ✓ Endpoint has deployed models:")
            for model in endpoint.list_models():
                print(f"     - {model.display_name} (ID: {model.id})")
        else:
            print(f"  ⚠️  Endpoint exists but no models deployed")

        return True

    except Exception as e:
        print(f"  ✗ Could not verify endpoint")
        print(f"     Error: {e}")
        print(f"\n  Possible reasons:")
        print(f"     - Endpoint not deployed yet")
        print(f"     - Incorrect endpoint ID")
        print(f"     - Wrong region")
        print(f"     - Insufficient permissions")
        print(f"\n  To check your endpoints:")
        print(f"     gcloud ai endpoints list --project={project_id} --region={location}")
        return False


def check_code_imports():
    """Check if our custom code can be imported."""
    print_section("Step 6: Custom Code Validation")

    try:
        # Try importing our Vertex AI agent factory
        sys.path.insert(0, str(Path(__file__).parent))

        from adk_agents.gemma_agent_vertex_adk import VertexAIAgentFactory
        print(f"  ✓ VertexAIAgentFactory: Imported successfully")

        # Try importing teamwork components
        from teamwork_components import TeamworkConfig
        print(f"  ✓ TeamworkConfig: Imported successfully")

        # Try importing dataset loaders
        from medical_datasets.dataset_loader import DatasetLoader
        print(f"  ✓ DatasetLoader: Imported successfully")

        print(f"\n  ✅ All custom code imports successful!")
        return True

    except Exception as e:
        print(f"  ✗ Failed to import custom code")
        print(f"     Error: {e}")
        return False


def test_configuration_mock():
    """Test configuration by creating a mock agent (no API calls)."""
    print_section("Step 7: Mock Configuration Test")

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from adk_agents.gemma_agent_vertex_adk import VertexAIAgentFactory

        # Get config from environment
        config = VertexAIAgentFactory.get_vertex_config()

        print(f"  Configuration parsed successfully:")
        print(f"     Project ID: {config['project_id']}")
        print(f"     Location: {config['location']}")
        print(f"     Endpoint ID: {config['endpoint_id'][:8]}...")
        print(f"     Use Vertex AI: {config['use_vertex']}")

        # Build endpoint resource name
        resource_name = VertexAIAgentFactory.build_endpoint_resource_name(
            config['project_id'],
            config['location'],
            config['endpoint_id']
        )

        print(f"\n  Endpoint resource name:")
        print(f"     {resource_name}")

        print(f"\n  ✅ Configuration is valid and can be used to create agents!")
        return True

    except Exception as e:
        print(f"  ✗ Configuration test failed")
        print(f"     Error: {e}")
        return False


def print_summary(checks_passed):
    """Print final summary."""
    print_section("Summary")

    total_checks = len(checks_passed)
    passed_checks = sum(1 for passed in checks_passed.values() if passed)

    print(f"\n  Checks Passed: {passed_checks}/{total_checks}")
    print(f"\n  Status:")

    for check, passed in checks_passed.items():
        status = "✅" if passed else "❌"
        print(f"     {status} {check}")

    if passed_checks == total_checks:
        print(f"\n  🎉 All checks passed! Your setup is ready for Vertex AI.")
        print(f"\n  Next steps:")
        print(f"     1. Deploy MedGemma to Vertex AI (if not already deployed)")
        print(f"     2. Test with: python run_simulation_vertex_adk.py --dataset medqa --n-questions 10")
        print(f"     3. See documentation/VERTEX_AI_SETUP.md for deployment instructions")
    else:
        print(f"\n  ⚠️  Some checks failed. Please fix the issues above before proceeding.")
        print(f"\n  See documentation/VERTEX_AI_SETUP.md for detailed setup instructions.")


def main():
    """Main verification routine."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify Vertex AI setup without costs")
    parser.add_argument('--check-endpoint', action='store_true',
                       help='Check if endpoint exists (requires endpoint to be deployed)')
    parser.add_argument('--skip-endpoint', action='store_true',
                       help='Skip endpoint verification (useful if not deployed yet)')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  Vertex AI Setup Verification Tool")
    print(f"  NO COSTS INCURRED - This tool only checks configuration")
    print(f"{'='*80}")

    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            print(f"\n✓ Loaded environment from: {env_file}")
        else:
            print(f"\n⚠️  No .env file found. Using system environment variables.")
            print(f"   To create: cp .env.vertex.example .env")
    except ImportError:
        print(f"\n⚠️  python-dotenv not installed. Using system environment variables.")

    # Run all checks
    checks = {}

    # 1. Environment variables
    env_ok, config = check_environment_variables()
    checks['Environment Variables'] = env_ok

    # 2. Python dependencies
    deps_ok = check_python_dependencies()
    checks['Python Dependencies'] = deps_ok

    # 3. Authentication
    auth_ok, auth_type = check_authentication()
    checks['Authentication'] = auth_ok

    # 4. Vertex AI API access
    if env_ok and auth_ok:
        api_ok = check_vertex_ai_access(config)
        checks['Vertex AI API Access'] = api_ok
    else:
        print_section("Step 4: Vertex AI API Access")
        print(f"  ⏭️  Skipped (fix environment variables and authentication first)")
        checks['Vertex AI API Access'] = False

    # 5. Endpoint verification (optional, can be skipped)
    if args.check_endpoint and env_ok and auth_ok:
        endpoint_ok = check_endpoint_exists(config)
        checks['Endpoint Verification'] = endpoint_ok if endpoint_ok is not None else True
    elif args.skip_endpoint:
        print_section("Step 5: Endpoint Verification")
        print(f"  ⏭️  Skipped (--skip-endpoint flag used)")
        checks['Endpoint Verification'] = True
    else:
        print_section("Step 5: Endpoint Verification")
        print(f"  ⏭️  Skipped by default (use --check-endpoint to verify)")
        print(f"     This is OK if you haven't deployed the endpoint yet")
        checks['Endpoint Verification'] = True  # Don't fail on this

    # 6. Code imports
    code_ok = check_code_imports()
    checks['Custom Code'] = code_ok

    # 7. Mock configuration test
    if env_ok and deps_ok:
        mock_ok = test_configuration_mock()
        checks['Configuration Test'] = mock_ok
    else:
        print_section("Step 7: Mock Configuration Test")
        print(f"  ⏭️  Skipped (fix environment and dependencies first)")
        checks['Configuration Test'] = False

    # Print summary
    print_summary(checks)

    # Exit code
    if all(checks.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
