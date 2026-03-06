"""Run insurance-interactions tests on Databricks.

This script is submitted as a spark_python_task (not a notebook).
It installs dependencies, then runs pytest.
"""

from __future__ import annotations

import subprocess
import sys
import os


def main() -> None:
    src_path = "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions/src"
    tests_path = "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions/tests"

    print("=== insurance-interactions test runner ===")
    print(f"Python: {sys.executable}")
    print(f"src_path exists: {os.path.isdir(src_path)}")
    print(f"tests_path exists: {os.path.isdir(tests_path)}")

    # Install dependencies
    print("\n=== Installing dependencies ===")
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "torch", "numpy", "polars", "scipy", "glum", "pytest",
         "-e", "/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions",
         ],
        capture_output=True,
        text=True,
    )
    if install.returncode != 0:
        print("pip install FAILED:")
        print(install.stdout)
        print(install.stderr)
        sys.exit(1)
    print("Dependencies installed.")

    # Run tests
    print("\n=== Running pytest ===")
    env = dict(os.environ)
    env["PYTHONPATH"] = src_path

    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         tests_path,
         "-v", "--tb=short",
         "--ignore=" + tests_path + "/run_tests_databricks.py",
         "--ignore=" + tests_path + "/submit_databricks_tests.py",
         ],
        capture_output=True,
        text=True,
        cwd="/Workspace/Users/pricing.frontier@gmail.com/insurance-interactions",
        env=env,
    )
    print(result.stdout)
    if result.stderr.strip():
        print("STDERR:", result.stderr[-3000:])
    print(f"\nReturn code: {result.returncode}")

    if result.returncode == 0:
        print("\n=== All tests PASSED ===")
        sys.exit(0)
    else:
        print(f"\n=== Tests FAILED (exit code {result.returncode}) ===")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
