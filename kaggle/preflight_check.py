"""Preflight check validating Python AST syntax of Kaggle scripts and notebooks."""

import ast
import json
import pathlib
import sys


def check_ast(path: pathlib.Path) -> bool:
    print(f"Checking AST for {path}...")
    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix == ".ipynb":
            nb = json.loads(content)
            code_cells = [
                "".join(cell["source"])
                for cell in nb.get("cells", [])
                if cell.get("cell_type") == "code"
            ]
            content = "\n\n".join(code_cells)
        ast.parse(content)
        print(f"AST valid: {path}")
        return True
    except Exception as exc:
        print(f"AST syntax error in {path}: {exc}", file=sys.stderr)
        return False


def main():
    root = pathlib.Path(__file__).parent
    targets = [root / "train_finevent.py", root / "train_finevent.ipynb"]
    all_ok = all(check_ast(p) for p in targets if p.exists())
    if not all_ok:
        sys.exit(1)
    print("All Kaggle files passed AST preflight checks!")


if __name__ == "__main__":
    main()
