"""
Run the 30-feature benchmark on Databricks serverless.

This uploads the benchmark as a notebook, runs it, and captures the output
so we can extract the real numbers for the README Performance section.
"""
from __future__ import annotations

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
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/burning-cost/insurance-interactions-benchmark"
NOTEBOOK_PATH = f"{WORKSPACE_PATH}/benchmark_30features"

# Read the benchmark notebook source
bench_path = os.path.join(os.path.dirname(__file__), "notebooks", "benchmark_30features.py")
with open(bench_path, "rb") as f:
    notebook_source = f.read()

w.workspace.mkdirs(path=WORKSPACE_PATH)
w.workspace.upload(
    path=NOTEBOOK_PATH,
    content=notebook_source,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Notebook uploaded to {NOTEBOOK_PATH}")

run = w.jobs.submit(
    run_name="insurance-interactions-benchmark-30features",
    tasks=[
        jobs.SubmitTask(
            task_key="benchmark",
            notebook_task=jobs.NotebookTask(notebook_path=NOTEBOOK_PATH),
        )
    ],
)
run_id = run.run_id
print(f"Run submitted: run_id={run_id}")
host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
print(f"URL: {host}/#job/run/{run_id}")
print()

while True:
    status = w.jobs.get_run(run_id=run_id)
    lc = status.state.life_cycle_state.value if status.state and status.state.life_cycle_state else "UNKNOWN"
    rs = status.state.result_state.value if status.state and status.state.result_state else ""
    print(f"  [{time.strftime('%H:%M:%S')}] {lc} {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        if status.tasks:
            for task in status.tasks:
                if task.run_id:
                    try:
                        out = w.jobs.get_run_output(run_id=task.run_id)
                        if out.notebook_output and out.notebook_output.result:
                            print("\n--- Output ---")
                            print(out.notebook_output.result)
                        if out.error:
                            print("\n--- Error ---", out.error)
                        if out.error_trace:
                            print("\n--- Trace ---", out.error_trace[-3000:])
                    except Exception as exc:
                        print(f"Could not get output: {exc}")
        sys.exit(0 if rs == "SUCCESS" else 1)
    time.sleep(30)
