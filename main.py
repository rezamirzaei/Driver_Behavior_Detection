#!/usr/bin/env python
"""
ABAX Data Science Project - Main Runner

This script executes the project notebooks to run the complete pipeline:
1. EDA & Data Processing (Classification)
2. Modeling (Classification)
3. EDA & Data Processing (Regression)
4. Modeling (Regression)

Results are saved in the notebooks themselves and in the 'results' directory.
"""

from pathlib import Path
import subprocess
import sys
import time


def run_notebook(notebook_path):
    """Run a notebook using nbconvert and save it in place."""
    print(f"Running {notebook_path.name}...")
    start_time = time.time()

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(notebook_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        print(f"✅ {notebook_path.name} completed in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {notebook_path.name} failed!")
        print(e.stderr)
        return False

def main():
    project_root = Path(__file__).parent
    notebooks_dir = project_root / "notebooks"

    notebooks_to_run = [
        "01_eda_classification.ipynb",
        "02_classification.ipynb",
        "03_eda_regression.ipynb",
        "04_regression.ipynb"
    ]

    print("🚀 Starting ABAX Data Science Pipeline...")

    success_count = 0
    for nb_name in notebooks_to_run:
        nb_path = notebooks_dir / nb_name
        if not nb_path.exists():
            print(f"⚠️ Notebook {nb_name} not found!")
            continue

        if run_notebook(nb_path):
            success_count += 1
        else:
            print("Stopping pipeline due to error.")
            break

    print(f"\nPipeline finished. {success_count}/{len(notebooks_to_run)} notebooks ran successfully.")
    print(f"Check {notebooks_dir} for executed notebooks with outputs.")

if __name__ == "__main__":
    main()
