"""Verify a notebook was actually executed. nbconvert can exit 0 having executed nothing.

    python check_executed.py symmetrization_variance.ipynb
"""
import sys

import nbformat

path = sys.argv[1] if len(sys.argv) > 1 else "symmetrization_variance.ipynb"
nb = nbformat.read(path, as_version=4)

code = [c for c in nb.cells if c.cell_type == "code"]
unrun = [i for i, c in enumerate(code) if c.get("execution_count") is None]
errors = [(i, o) for i, c in enumerate(code) for o in c.get("outputs", [])
          if o.get("output_type") == "error"]
figures = sum(1 for c in code for o in c.get("outputs", [])
              if "image/png" in o.get("data", {}))

print(f"{path}: {len(nb.cells)} cells, {len(code)} code cells")
print(f"  executed : {len(code) - len(unrun)}/{len(code)}")
print(f"  figures  : {figures}")
if unrun:
    print(f"  NOT RUN  : cells {unrun}")
for i, o in errors:
    print(f"  ERROR in code cell {i}: {o.get('ename')}: {o.get('evalue')}")

sys.exit(1 if (unrun or errors) else 0)
