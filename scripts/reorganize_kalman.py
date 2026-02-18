"""Reorganize Kalman filter section in notebook."""

import json
from pathlib import Path

def get_notebook_structure(nb):
    """Extract markdown headers from notebook."""
    sections = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            src = cell.get('source', [])
            if isinstance(src, list):
                src = ''.join(src)
            for line in src.split('\n'):
                if line.startswith('#'):
                    sections.append((i, line.strip()))
                    break
    return sections

def reorganize_kalman_section():
    """Move Kalman filter section to proper location after data loading."""

    notebook_path = Path('notebooks/02_classification.ipynb')

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")

    # Get current structure
    sections = get_notebook_structure(nb)
    print("\nCurrent sections:")
    for idx, header in sections:
        print(f"  Cell {idx}: {header[:60]}")

    # Find Kalman filter section cells
    kalman_start = None
    kalman_end = None

    for i, cell in enumerate(nb['cells']):
        src = cell.get('source', [])
        if isinstance(src, list):
            src = ''.join(src)
        if '## 9. Kalman Filter' in src or 'Kalman Filter Signal Processing' in src:
            kalman_start = i
            # Find end of Kalman section (next ## section or end)
            for j in range(i + 1, len(nb['cells'])):
                cell_j = nb['cells'][j]
                src_j = cell_j.get('source', [])
                if isinstance(src_j, list):
                    src_j = ''.join(src_j)
                if cell_j['cell_type'] == 'markdown' and src_j.startswith('## ') and 'Kalman' not in src_j:
                    kalman_end = j
                    break
            if kalman_end is None:
                kalman_end = len(nb['cells'])
            break

    if kalman_start is not None:
        print(f"\nKalman section found: cells {kalman_start} to {kalman_end}")

        # Extract Kalman cells
        kalman_cells = nb['cells'][kalman_start:kalman_end]
        print(f"Kalman cells count: {len(kalman_cells)}")

        # Remove from current position
        del nb['cells'][kalman_start:kalman_end]

        # Find where to insert (after data loading, before model comparison)
        # Look for "## 3." or "## 4." section
        insert_idx = None
        for i, cell in enumerate(nb['cells']):
            src = cell.get('source', [])
            if isinstance(src, list):
                src = ''.join(src)
            # Insert after EDA section and before model training
            if '## 3.' in src or '## 4.' in src or 'Model' in src and '##' in src:
                insert_idx = i
                break

        if insert_idx is None:
            # Insert after data loading (around cell 10-15)
            insert_idx = min(15, len(nb['cells']) - len(kalman_cells))

        print(f"Inserting at cell {insert_idx}")

        # Insert Kalman cells at new position
        for i, cell in enumerate(kalman_cells):
            nb['cells'].insert(insert_idx + i, cell)

        # Save
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)

        print(f"\nReorganized notebook saved!")
    else:
        print("\nKalman section not found, will create new one")

if __name__ == '__main__':
    reorganize_kalman_section()
