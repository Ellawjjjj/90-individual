#!/usr/bin/env python
"""Build docs/report.docx (editable Word) from docs/report.md with pandoc.

    python tools/build_report.py            # needs pandoc on PATH (https://pandoc.org/installing.html)

Figures written as "![Figure N](path)" followed by an italic "*Figure N. caption*" line are
merged into real Word figures with captions; tall comparison grids are placed at 4.6 in width.
docs/reference.docx carries the styles (bordered tables, 11 pt body, heading fonts).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def main():
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found on PATH - install it from https://pandoc.org/installing.html")
    src = (DOCS / "report.md").read_text(encoding="utf-8")
    src = re.sub(r"^<!--.*?-->\s*", "", src, flags=re.S)  # drop the draft notes

    def repl(m):
        path, cap = m.group(1), m.group(2).replace("*", "")
        width = "4.6in" if any(k in path for k in ("top4", "top3")) else "6.3in"
        return f"![{cap}]({path}){{width={width}}}"

    src = re.sub(r"!\[Figure \d+\]\(([^)]+)\)\n\*(Figure \d+\..*?)\*", repl, src)
    build = DOCS / "_report_build.md"
    build.write_text(src, encoding="utf-8")
    out = DOCS / "report.docx"
    cmd = ["pandoc", str(build), "-o", str(out), f"--reference-doc={DOCS / 'reference.docx'}",
           f"--resource-path={DOCS}", "--shift-heading-level-by=-1",
           "--from", "markdown+pipe_tables+implicit_figures"]
    subprocess.run(cmd, check=True)
    build.unlink()
    print("wrote", out)


if __name__ == "__main__":
    main()
