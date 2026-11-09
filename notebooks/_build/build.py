"""Build lesson notebooks: one source -> a skeleton (exercises) and a solutions notebook.

In a lesson source, code cells may contain solution blocks:

        #>> TODO: what the learner has to write
        solution code ...
        #<<

The skeleton replaces each block by the TODO comment and a `...` placeholder (same indentation);
the solutions notebook keeps the code and drops the markers.

Usage:  python notebooks/_build/build.py notebooks/_build/lesson01.py
"""

import importlib.util
import re
import sys
from pathlib import Path

import nbformat as nbf

BLOCK = re.compile(r"^([ \t]*)#>> TODO:(.*?)\n(.*?)^[ \t]*#<<[ \t]*\n", re.S | re.M)


def skeleton(src: str) -> str:
    return BLOCK.sub(lambda m: "".join(f"{m.group(1)}# TODO:{line}\n" for line in m.group(2).split(" // ")) + f"{m.group(1)}...\n", src)


def solution(src: str) -> str:
    return BLOCK.sub(lambda m: m.group(3), src)


def build(lesson_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("lesson", lesson_path)
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    root = lesson_path.parent.parent
    for kind, transform, out in [
        ("skeleton", skeleton, root / f"{lesson.NAME}.ipynb"),
        ("solutions", solution, root / "solutions" / f"{lesson.NAME}_solutions.ipynb"),
    ]:
        cells = []
        for cell_type, text in lesson.CELLS:
            text = text.strip("\n") + "\n"
            if cell_type == "md":
                if kind == "solutions":
                    text = text.replace("{{SOLUTIONS_NOTE}}", "**This is the solutions notebook.**")
                else:
                    text = text.replace("{{SOLUTIONS_NOTE}}", f"Solutions: `solutions/{lesson.NAME}_solutions.ipynb` (try first!).")
                cells.append(nbf.v4.new_markdown_cell(text.rstrip("\n")))
            else:
                cells.append(nbf.v4.new_code_cell(transform(text).rstrip("\n")))
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (Isaac Lab 3.0)", "language": "python"}
        out.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(nb, out)
        print(f"{kind:>9}: {out}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        build(Path(path).resolve())
