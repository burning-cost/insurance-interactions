"""
Run insurance-interactions tests on Databricks serverless for QA audit fix verification.
"""
from __future__ import annotations

import os
import sys
import time
import subprocess

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

BASE = "/home/ralph/repos/insurance-interactions"
WORKSPACE_PATH = "/Workspace/burning-cost/insurance-interactions-qa"
NOTEBOOK_PATH = f"{WORKSPACE_PATH}/run_pytest"

print(f"Uploading project to {WORKSPACE_PATH}...")
r = subprocess.run(
    ["databricks", "workspace", "import-dir", BASE, WORKSPACE_PATH, "--overwrite"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print("import-dir stderr:", r.stderr[:500])
else:
    print("Upload OK")

notebook_source = b"""\
# Databricks notebook source
# MAGIC %pip install numpy polars scipy glum packaging pyarrow torch pytest pytest-cov --quiet

# COMMAND ----------
import subprocess, sys, os, io
from contextlib import redirect_stdout, redirect_stderr

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/burning-cost/insurance-interactions-qa", "--quiet"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print("Install error:", r.stderr[:500])

import pytest

tests_path = "/Workspace/burning-cost/insurance-interactions-qa/tests"

buf = io.StringIO()
try:
    with redirect_stdout(buf), redirect_stderr(buf):
        ret = pytest.main([
            tests_path + "/test_nid.py",
            tests_path + "/test_cann.py",
            tests_path + "/test_glm_builder.py",
            tests_path + "/test_selector.py",
            "--import-mode=importlib",
            "-x", "-q", "--tb=short",
            "--no-header",
            "-p", "no:cacheprovider",
        ])
except Exception as e:
    buf.write(f"\\nException: {e}")
    ret = -1

output = buf.getvalue()
exit_code = int(ret)
result = f"EXIT:{exit_code}\\n{output}"
dbutils.notebook.exit(result[-8000:])
"""

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
    run_name="insurance-interactions-qa-pytest",
    tasks=[
        jobs.SubmitTask(
            task_key="pytest",
            notebook_task=jobs.NotebookTask(notebook_path=NOTEBOOK_PATH),
        )
    ],
)
run_id = run.run_id
print(f"Run submitted: run_id={run_id}")
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
                            print("\n--- Output ---")
                            print(out.notebook_output.result)
                        if out.error:
                            print("\n--- Error ---", out.error)
                        if out.error_trace:
                            print("\n--- Trace ---", out.error_trace[-2000:])
                    except Exception as exc:
                        print(f"Could not get output: {exc}")
        sys.exit(0 if rs == "SUCCESS" else 1)
    time.sleep(20)
