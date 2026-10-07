"""List the review notes left in the tutorial notebooks.

Usage (from tutorials/v2/):
    python review_notes.py                 # all notes in all notebooks
    python review_notes.py 03_*.ipynb      # only some notebooks
    python review_notes.py --tags FIX CHECK --strict
        # only FIX/CHECK notes; exit code 1 if any are found (pre-release check)

See the "Review notes" section of README.md for the tag legend.
"""

import argparse
import json
import re
import sys
from pathlib import Path

TAGS = {
    "FIX": "🔴",
    "CHECK": "🟠",
    "TODO": "🔵",
    "QUESTION": "🟣",
    "IDEA": "🟢",
}

# Matches e.g. "🔴 <b>FIX:</b> text", "🔴 **FIX:** text" or "# 🔴 FIX: text".
NOTE_RE = re.compile(
    r"(?P<emoji>[" + "".join(TAGS.values()) + r"])\s*"
    r"(?:<b>|\*\*)?(?P<tag>" + "|".join(TAGS) + r"):?(?:</b>|\*\*)?:?\s*"
    r"(?P<text>.*)"
)
HTML_TAG_RE = re.compile(r"<[^>]+>")


def find_notes(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    for i, cell in enumerate(cells):
        source = "".join(cell["source"])
        for line_no, line in enumerate(source.splitlines(), start=1):
            match = NOTE_RE.search(line)
            if match:
                text = HTML_TAG_RE.sub("", match["text"]).strip()
                yield i, cell["cell_type"], line_no, match["tag"], text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--tags", nargs="+", choices=list(TAGS), default=list(TAGS))
    parser.add_argument(
        "--strict", action="store_true", help="exit with status 1 if any note is found"
    )
    args = parser.parse_args()

    notebooks = args.notebooks or sorted(Path(__file__).parent.glob("*.ipynb"))
    counts = dict.fromkeys(args.tags, 0)
    for nb in notebooks:
        notes = [n for n in find_notes(nb) if n[3] in args.tags]
        if not notes:
            continue
        print(f"\n{nb.name}")
        for cell, kind, line, tag, text in notes:
            counts[tag] += 1
            print(f"  cell {cell:>3} ({kind:<8} l.{line:<3}) {TAGS[tag]} {tag:<8} {text}")

    total = sum(counts.values())
    summary = ", ".join(f"{TAGS[t]} {t} {n}" for t, n in counts.items() if n)
    print(f"\n{total} note(s)" + (f": {summary}" if summary else ""))
    if args.strict and total:
        sys.exit(1)


if __name__ == "__main__":
    main()
