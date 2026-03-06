"""Submit and monitor insurance-interactions tests on Databricks (serverless)."""

from __future__ import annotations

import os
import sys
import time

# Load credentials
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

WORKSPACE_PATH = "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions"

# Fix: use --import-mode=importlib to avoid __pycache__ creation on FUSE fs
notebook_source_str = """\
# Databricks notebook source
# MAGIC %pip install torch numpy polars scipy glum pytest

# COMMAND ----------
import sys, os, io
from contextlib import redirect_stdout, redirect_stderr

src_path = "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions/src"
tests_path = "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions/tests"

# Prevent .pyc creation — Workspace filesystem doesn't support __pycache__
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

sys.path.insert(0, src_path)

buf = io.StringIO()
try:
    import pytest
    with redirect_stdout(buf), redirect_stderr(buf):
        ret = pytest.main([
            tests_path + "/test_cann.py",
            tests_path + "/test_nid.py",
            tests_path + "/test_glm_builder.py",
            tests_path + "/test_selector.py",
            "--import-mode=importlib",
            "-v", "--tb=short",
            "--no-header",
            "-p", "no:cacheprovider",
        ])
except Exception as e:
    buf.write("\\nException during pytest: " + str(e))
    ret = -1

output = buf.getvalue()
exit_code = int(ret)
result = f"EXIT:{exit_code}\\n{output}"
dbutils.notebook.exit(result[-8000:])
"""

notebook_source = notebook_source_str.encode("utf-8")

NOTEBOOK_PATH = f"{WORKSPACE_PATH}/notebooks/run_tests_notebook"

w.workspace.upload(
    path=NOTEBOOK_PATH,
    content=notebook_source,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Uploaded test notebook.")

run = w.jobs.submit(
    run_name="insurance-interactions-pytest",
    tasks=[
        jobs.SubmitTask(
            task_key="pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
        )
    ],
)
run_id = run.run_id
print(f"Job submitted, run_id={run_id}. Waiting...")

host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
print(f"URL: {host}/#job/run/{run_id}")

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
                            print("\n--- Notebook output ---")
                            print(out.notebook_output.result)
                        if out.error:
                            print("\n--- Error ---")
                            print(out.error)
                        if out.error_trace:
                            print("\n--- Trace ---")
                            print(out.error_trace[-2000:])
                    except Exception as exc:
                        print(f"Could not get output: {exc}")
        sys.exit(0 if rs == "SUCCESS" else 1)
    time.sleep(20)
