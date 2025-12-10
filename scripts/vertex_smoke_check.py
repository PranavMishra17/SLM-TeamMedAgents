import os
import sys
import subprocess
from pathlib import Path

REQUIRED_TRUE = [
    "GOOGLE_GENAI_USE_VERTEXAI",
]

REQUIRED_PRESENT = [
    "VERTEX_AI_ENDPOINT_ID",
    "GOOGLE_CLOUD_PROJECT",
]

OPTIONAL_PATH = [
    "GOOGLE_APPLICATION_CREDENTIALS",
]

MODULES = [
    ("google.genai", "google.genai"),
    ("google.adk", "google.adk"),
    ("google.cloud.aiplatform", "google.cloud.aiplatform"),
]

missing = []
incorrect = []

for k in REQUIRED_PRESENT:
    v = os.environ.get(k)
    if not v:
        missing.append(k)

for k in REQUIRED_TRUE:
    v = os.environ.get(k)
    if not v:
        missing.append(k)
    else:
        if v.strip().upper() != "TRUE":
            incorrect.append((k, v))

for k in OPTIONAL_PATH:
    v = os.environ.get(k)
    if v:
        if not os.path.exists(v):
            incorrect.append((k, f"path does not exist: {v}"))

import_failures = []
for mod_name, import_name in MODULES:
    try:
        __import__(import_name)
    except Exception as e:
        import_failures.append((import_name, str(e)))

# Print only missing/incorrect details exactly as requested
if missing:
    print("MISSING_ENV_VARS:")
    for k in missing:
        print(k)

if incorrect:
    print("INCORRECT_ENV_VARS:")
    for k, v in incorrect:
        print(f"{k}={v}")

if import_failures:
    print("IMPORT_ERRORS:")
    for mod, err in import_failures:
        print(f"{mod}: {err}")

if not (missing or incorrect or import_failures):
    print("OK: environment looks good for a Vertex dry smoke-check")
else:
    sys.exit(2)

# ---------------------------------------------------------------------------
# Quick functional smoke test: one text dataset + one vision dataset
# Keeps outputs small but exercises the full pipeline, including token/inference
# accounting already handled inside run_simulation_vertex_adk.py.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
runner = REPO_ROOT / "run_simulation_vertex_adk.py"

if not runner.exists():
    print(f"RUNNER_NOT_FOUND: {runner}")
    sys.exit(3)

runs = [
    {
        "label": "text-medqa",
        "args": [
            sys.executable,
            str(runner),
            "--dataset", "medqa",
            "--n-questions", "5",
            "--output-dir", "multi-agent-gemma/results_vertex_smoke"
        ]
    },
    {
        "label": "vision-pmc_vqa",
        "args": [
            sys.executable,
            str(runner),
            "--dataset", "pmc_vqa",
            "--n-questions", "5",
            "--n-agents", "3",
            "--output-dir", "multi-agent-gemma/results_vertex_smoke"
        ]
    },
]

failures = []

for run in runs:
    print(f"\n=== RUN {run['label']} ===")
    try:
        result = subprocess.run(
            run["args"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.returncode != 0:
            failures.append((run["label"], result.returncode))
    except Exception as e:
        failures.append((run["label"], str(e)))

if failures:
    print("\nSMOKE_TEST_FAILED:")
    for label, err in failures:
        print(f"{label}: {err}")
    sys.exit(4)

print("\nSMOKE_TEST_OK: text + vision (medqa, pmc_vqa) completed")
sys.exit(0)
