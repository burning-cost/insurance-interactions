"""
Run the 50-feature benchmark on Databricks serverless.

Uploads the benchmark script as a notebook, submits it, polls for completion,
and prints the output.

No timeout is set on the task — the benchmark may take 20-40 minutes on
serverless CPU compute (dominated by CANN training with n_ensemble=5,
n_epochs=300, 30k policies, 50 features).
"""
from __future__ import annotations

import base64
import os
import sys
import time

env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/burning-cost/insurance-interactions-benchmark"
NOTEBOOK_PATH = f"{WORKSPACE_PATH}/benchmark_50features"

# Read the benchmark script
bench_path = os.path.join(os.path.dirname(__file__), "benchmarks", "benchmark_50features.py")
with open(bench_path, "r") as f:
    script_content = f.read()

# Add dbutils.notebook.exit() to capture final results as JSON
script_content += """

# Capture results for Databricks notebook output
import json as _json
_exit_data = {
    "n_features": N_FEATURES,
    "n_pairs": N_PAIRS,
    "n_policies": N_POLICIES,
    "cann_time_s": round(cann_time, 1),
    "sample_pairs_time_s": round(sample_time, 1),
    "estimated_exhaustive_time_s": round(estimated_total_time, 1),
    "cann_tp": cann_tp,
    "cann_fp": cann_fp,
    "base_deviance": round(base_deviance),
    "total_elapsed_s": round(time.time() - BENCHMARK_START, 1),
}
print("\\nRESULTS_JSON:", _json.dumps(_exit_data))
try:
    dbutils.notebook.exit(_json.dumps(_exit_data))
except Exception:
    pass
"""

# Wrap in Databricks notebook format
notebook_content = (
    "# Databricks notebook source\n\n"
    "# COMMAND ----------\n\n"
    "# MAGIC %pip install \"insurance-interactions[torch]\" glum polars --quiet\n\n"
    "# COMMAND ----------\n\n"
    "dbutils.library.restartPython()\n\n"
    "# COMMAND ----------\n\n"
) + script_content

notebook_b64 = base64.b64encode(notebook_content.encode("utf-8")).decode("utf-8")

# Upload to workspace
w.workspace.mkdirs(path=WORKSPACE_PATH)
w.workspace.import_(
    path=NOTEBOOK_PATH,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=notebook_b64,
    overwrite=True,
)
print(f"Notebook uploaded to {NOTEBOOK_PATH}")

# Submit with no timeout (benchmark takes ~20-40 min)
run = w.jobs.submit(
    run_name="insurance-interactions-benchmark-50features",
    tasks=[
        jobs.SubmitTask(
            task_key="benchmark",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
                source=jobs.Source.WORKSPACE,
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=compute.Environment(client="2"),
        )
    ],
)

run_id = run.run_id
print(f"Run submitted: run_id={run_id}")
host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
print(f"URL: {host}/#job/run/{run_id}")
print()

# Poll for completion
start = time.time()
while True:
    status = w.jobs.get_run(run_id=run_id)
    lc = (status.state.life_cycle_state.value
          if status.state and status.state.life_cycle_state else "UNKNOWN")
    rs = (status.state.result_state.value
          if status.state and status.state.result_state else "")
    print(f"  [{time.strftime('%H:%M:%S')}] {lc} {rs}", flush=True)
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        # Collect task output
        if status.tasks:
            for task in status.tasks:
                if task.run_id:
                    try:
                        out = w.jobs.get_run_output(run_id=task.run_id)
                        if out.notebook_output and out.notebook_output.result:
                            print("\n--- Notebook Output ---")
                            print(out.notebook_output.result)
                        if out.error:
                            print("\n--- Error ---")
                            print(out.error)
                        if out.error_trace:
                            print("\n--- Trace ---")
                            print(out.error_trace[-5000:])
                    except Exception as exc:
                        print(f"Could not get task output: {exc}")
        elapsed = time.time() - start
        print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
        sys.exit(0 if rs == "SUCCESS" else 1)
    time.sleep(30)
