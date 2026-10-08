"""Generate the inputs a chapter needs from earlier chapters.

Chapters hand data to each other through files in ``outputs/``: chapter 2 writes
the truth trajectory, chapter 3 the simulated tracking data, and later chapters
read them. To start at a later chapter without running the earlier notebooks by
hand, run this script first.

Usage (from tutorials/v2/):
    python prepare.py 06            # run what chapter 6 needs (chapters 2 and 3)
    python prepare.py 06 --force    # re-run them even if their outputs already exist

The prerequisite notebooks are executed in memory; the .ipynb files are not modified.
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Notebook -> files it writes that later chapters read.
PRODUCTS = {
    "02_truth_dynamics.ipynb": ["outputs/02_truth_dynamics/truth.bsp"],
    "03_tracking_data.ipynb": [
        f"outputs/03_tracking_data/{dss}_{kind}.json"
        for dss in ("DSS-14", "DSS-43", "DSS-63")
        for kind in ("range", "range-rate")
    ],
}

# Chapter -> notebooks whose products it reads, in the order they must run.
REQUIRES = {
    "03": ["02_truth_dynamics.ipynb"],
    "04": ["02_truth_dynamics.ipynb"],
    "05": ["02_truth_dynamics.ipynb", "03_tracking_data.ipynb"],
    "06": ["02_truth_dynamics.ipynb", "03_tracking_data.ipynb"],
}


def missing_products(notebook):
    return [p for p in PRODUCTS[notebook] if not (HERE / p).exists()]


def run_notebook(notebook):
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        sys.exit("prepare.py needs Jupyter's nbformat and nbclient: pip install nbclient")

    nb = nbformat.read(HERE / notebook, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    start = time.time()
    client.execute()                     # in memory only; the notebook file is not rewritten
    print(f"  ran {notebook} in {time.time() - start:.0f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("chapter", help="chapter number, e.g. 06")
    parser.add_argument("--force", action="store_true", help="re-run prerequisites even if their outputs exist")
    args = parser.parse_args()

    chapter = args.chapter.zfill(2)
    if chapter not in REQUIRES:
        print(f"Chapter {chapter} does not read outputs from earlier chapters; nothing to prepare.")
        return

    print(f"Preparing inputs for chapter {chapter}:")
    for notebook in REQUIRES[chapter]:
        if args.force or missing_products(notebook):
            run_notebook(notebook)
        else:
            print(f"  {notebook}: outputs already present (use --force to regenerate)")

    still_missing = [p for nb in REQUIRES[chapter] for p in missing_products(nb)]
    if still_missing:
        sys.exit(f"Still missing after running prerequisites: {still_missing}")
    print("Done.")


if __name__ == "__main__":
    main()
