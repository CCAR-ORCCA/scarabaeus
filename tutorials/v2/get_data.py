"""Download the SPICE kernels the "Navigating OSIRIS-REx" series needs.

The series loads only the kernels listed in its metakernels: ``orex_approach.tm``
(chapters 0-8, about 41 MB) and ``real_data.tm`` (chapter 9 adds about 1.7 GB).
This script downloads the ones that are missing, from the links registered in
``tutorials/supplementary`` (the loader of the earlier notebooks, whose
``load_data()`` fetches every tutorial's data, 2.8 GB).

Usage (from tutorials/v2/):
    python get_data.py               # chapters 0-8
    python get_data.py --real-data   # also chapter 9
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from supplementary import file_gen  # noqa: E402
from supplementary.supp_data import Data  # noqa: E402


def kernels(metakernel):
    """Kernel paths (relative to the supplementary data folder) listed in a metakernel."""
    return ["kernels/" + k for k in re.findall(r"'\$KER/([^']+)'", (HERE / metakernel).read_text())]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real-data", action="store_true", help="also download the chapter 9 kernels (about 1.7 GB)")
    args = parser.parse_args()

    wanted = kernels("orex_approach.tm") + (kernels("real_data.tm") if args.real_data else [])
    links = {d.value.path: d.value.link for d in Data}
    for path in wanted:
        target = file_gen.DATA_PATH / path
        if target.exists():
            print(f"  present     {path}")
            continue
        link = links.get(path)
        if link is None:
            sys.exit(f"No download link registered for {path} in tutorials/supplementary/supp_data.py")
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {path} ...", end="", flush=True)
        partial = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(link, partial)  # an interrupted download leaves only the .part file
        partial.rename(target)
        print(f" {target.stat().st_size / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
