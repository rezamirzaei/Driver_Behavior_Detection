#!/usr/bin/env python3
"""Convert Jupyter notebooks to PyCharm-compatible Python files."""

import json
import sys

def convert_ipynb_to_pycharm(ipynb_path, py_path):
    """Convert Jupyter notebook to PyCharm-compatible Python script with cell markers."""
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        lines = []
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])

            # Handle source as list or string
            if isinstance(source, list):
                content = ''.join(source)
            else:
                content = str(source)

            if cell_type == 'markdown':
                lines.append("# %% [markdown]")
                for line in content.split('\n'):
                    lines.append(f"# {line}")
                lines.append("")
            elif cell_type == 'code':
                lines.append("# %%")
                lines.append(content)
                lines.append("")

        with open(py_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ Converted: {ipynb_path} -> {py_path}")
        return True
    except Exception as e:
        print(f"❌ Error converting {ipynb_path}: {e}")
        return False

if __name__ == '__main__':
    notebooks = [
        ('notebooks/03_eda_regression.ipynb', 'notebooks/03_eda_regression.py'),
        ('notebooks/04_regression.ipynb', 'notebooks/04_regression.py'),
    ]

    for ipynb_path, py_path in notebooks:
        convert_ipynb_to_pycharm(ipynb_path, py_path)

