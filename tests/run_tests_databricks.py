"""Run insurance-interactions tests on Databricks.

Usage on Databricks cluster (or as a notebook cell):
    %pip install -e /path/to/insurance-interactions[dev]
    %run /Workspace/insurance-interactions/tests/run_tests_databricks.py

Or from local machine using the Databricks Jobs API via the SDK:
    python tests/run_tests_databricks.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time


def run_via_sdk() -> None:
    """Upload the project to Databricks workspace and run pytest via a job."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.expanduser("~/.config/burning-cost/databricks.env"))
    except ImportError:
        # Parse env file manually
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

    w = WorkspaceClient()
    workspace_path = "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions"

    print(f"Uploading project to {workspace_path} ...")
    import subprocess
    result = subprocess.run(
        [
            "databricks", "workspace", "import-dir",
            "--overwrite",
            str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            workspace_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Upload stderr:", result.stderr)
        sys.exit(1)
    print("Upload complete.")

    # Create and run a one-time job
    notebook_path = f"{workspace_path}/notebooks/run_tests_notebook"
    run = w.jobs.submit(
        run_name="insurance-interactions-tests",
        tasks=[
            jobs.SubmitTask(
                task_key="run-pytest",
                notebook_task=jobs.NotebookTask(
                    notebook_path=notebook_path,
                ),
                new_cluster=jobs.BaseClusterInfo(
                    spark_version="14.3.x-cpu-ml-scala2.12",
                    node_type_id="Standard_DS3_v2",
                    num_workers=0,
                    spark_conf={
                        "spark.databricks.cluster.profile": "singleNode",
                        "spark.master": "local[*]",
                    },
                    custom_tags={"ResourceClass": "SingleNode"},
                ),
            )
        ],
    )
    run_id = run.run_id
    print(f"Job submitted, run_id={run_id}. Waiting for completion...")

    while True:
        status = w.jobs.get_run(run_id=run_id)
        life_cycle = status.state.life_cycle_state.value if status.state.life_cycle_state else "UNKNOWN"
        result_state = status.state.result_state.value if status.state.result_state else ""
        print(f"  State: {life_cycle} {result_state}")
        if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            if result_state == "SUCCESS":
                print("Tests passed.")
            else:
                print(f"Tests FAILED. Result: {result_state}")
                print("Check Databricks UI for logs.")
                sys.exit(1)
            break
        time.sleep(15)


if __name__ == "__main__":
    run_via_sdk()
