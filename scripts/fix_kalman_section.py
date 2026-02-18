#!/usr/bin/env python3
"""Properly reorganize and clean up Kalman filter section in the notebook."""

import json
import sys
from pathlib import Path

def main():
    notebook_path = Path('notebooks/02_classification.ipynb')

    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}", file=sys.stderr)
        return 1

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    print(f"Loaded notebook with {len(nb['cells'])} cells")

    # Remove duplicate Kalman sections (keep only section 3)
    cells_to_remove = []
    in_old_kalman = False

    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))

        # Check for old Kalman subsections (9.x)
        if '### 9.1' in src or '### 9.2' in src or '### 9.3' in src or \
           '### 9.4' in src or '### 9.5' in src or '### 9.6' in src:
            in_old_kalman = True
            cells_to_remove.append(i)
        elif in_old_kalman:
            if src.startswith('## ') and 'Kalman' not in src:
                in_old_kalman = False
            else:
                cells_to_remove.append(i)

    # Remove cells in reverse order
    for i in sorted(cells_to_remove, reverse=True):
        del nb['cells'][i]

    print(f"Removed {len(cells_to_remove)} duplicate Kalman cells")

    # Fix duplicate "## 3. Exploratory Data Analysis" -> should be ## 4
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = cell.get('source', [])
            if isinstance(src, list):
                for j, line in enumerate(src):
                    if '## 3. Exploratory Data Analysis' in line:
                        src[j] = line.replace('## 3. Exploratory Data Analysis', '## 4. Exploratory Data Analysis')
                    elif '## 4. Data Preparation' in line:
                        src[j] = line.replace('## 4. Data Preparation', '## 5. Data Preparation')
                    elif '## 5. Train All' in line:
                        src[j] = line.replace('## 5. Train All', '## 6. Train All')
                    elif '## 6. Neural Network' in line:
                        src[j] = line.replace('## 6. Neural Network', '## 7. Neural Network')
                    elif '## 7. Model Comparison' in line:
                        src[j] = line.replace('## 7. Model Comparison', '## 8. Model Comparison')
                    elif '## 8. Confusion Matrix' in line:
                        src[j] = line.replace('## 8. Confusion Matrix', '## 9. Confusion Matrix')
                    elif '## 9. Feature Importance' in line:
                        src[j] = line.replace('## 9. Feature Importance', '## 10. Feature Importance')
                cell['source'] = src

    # Save
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"\nSaved: {notebook_path}")
    print(f"Final cell count: {len(nb['cells'])}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
