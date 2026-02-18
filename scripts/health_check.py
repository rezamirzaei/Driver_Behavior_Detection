#!/usr/bin/env python3
"""
Project Health Check - Verify ABAX project is clean and ready.
"""

import sys
from pathlib import Path

def check_project():
    """Run comprehensive project health check."""
    import os
    os.chdir(Path(__file__).parent.parent)
    project_root = Path.cwd()

    print("🔍 ABAX Project Health Check")
    print("="*60)

    # Check essential files
    essential_files = [
        'README.md',
        'main.py',
        '.gitignore',
        'pyproject.toml',
    ]

    print("\n✅ Essential Files:")
    all_exist = True
    for file in essential_files:
        file_path = project_root / file
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if not exists:
            all_exist = False

    # Check directories
    essential_dirs = [
        'src',
        'tests',
        'notebooks',
        'data',
        'docs',
        'results',
        'scripts',
    ]

    print("\n✅ Directory Structure:")
    for directory in essential_dirs:
        dir_path = project_root / directory
        exists = dir_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {directory}/")
        if not exists:
            all_exist = False

    # Check no .txt files in root
    txt_files = list(project_root.glob('*.txt'))
    print("\n✅ No .txt Files in Root:")
    if txt_files:
        print(f"   ❌ Found {len(txt_files)} .txt files:")
        for f in txt_files:
            print(f"      - {f.name}")
        all_exist = False
    else:
        print("   ✅ Clean (no .txt files)")

    # Check notebooks
    notebook_dir = project_root / 'notebooks'
    notebooks = [
        '01_project_overview.ipynb',
        '02_classification.ipynb',
        '03_eda_regression.ipynb',
        '04_regression.ipynb',
    ]

    print("\n✅ Notebooks:")
    for nb in notebooks:
        nb_path = notebook_dir / nb
        exists = nb_path.exists()
        status = "✅" if exists else "❌"
        print(f"   {status} {nb}")
        if not exists:
            all_exist = False

    # Check dependencies can be imported
    print("\n✅ Python Environment:")
    try:
        import numpy
        print(f"   ✅ NumPy {numpy.__version__}")
    except ImportError:
        print("   ❌ NumPy not installed")
        all_exist = False

    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch not installed")
        all_exist = False

    try:
        import pandas
        print(f"   ✅ Pandas {pandas.__version__}")
    except ImportError:
        print("   ❌ Pandas not installed")
        all_exist = False

    try:
        import sklearn
        print(f"   ✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("   ❌ Scikit-learn not installed")
        all_exist = False

    # Final summary
    print("\n" + "="*60)
    if all_exist:
        print("🎉 PROJECT HEALTH: EXCELLENT")
        print("   All essential components present.")
        print("   Ready for development and deployment!")
        return 0
    else:
        print("⚠️  PROJECT HEALTH: NEEDS ATTENTION")
        print("   Some components are missing.")
        print("   Run: uv sync")
        return 1

if __name__ == '__main__':
    sys.exit(check_project())
