# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import re
import shutil
from pathlib import Path
from collections import defaultdict

CHANGELOG = Path.cwd() / 'CHANGELOG.md'
RELEASE_NOTES_DIR = Path.cwd() / 'docs/online_documentation/sphinx_files/release_notes'

VERSIONS_DIR  = RELEASE_NOTES_DIR / 'versions'
OVERRIDES_DIR = RELEASE_NOTES_DIR / 'overridden_versions'

# wipe and regenerate
if VERSIONS_DIR.exists():
    shutil.rmtree(VERSIONS_DIR)
VERSIONS_DIR.mkdir(parents=True)

def slugify(v):
    return v.replace(".", "-")

def md_line_to_rst(line):
    """Convert simple inline Markdown (bold, inline code, links) to RST."""
    line = re.sub(r"\*\*(.+?)\*\*", r"**\1**", line)  # bold is the same in RST
    line = re.sub(r"`([^`]+)`", r"``\1``", line)       # `code` -> ``code``
    line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"`\1 <\2>`__", line)  # [text](url)
    return line

GOLD_STYLE_BLOCK = """.. raw:: html

    <style> .gold {color:rgb(207 184 124)} </style>

.. role:: gold
"""


def md_section_to_rst(md_text, version, date):
    """
    Converts one CHANGELOG.md section's body into a full styled RST page,
    matching the hand-written release notes format.
    """
    lines = md_text.splitlines()

    title_text = f"Scarabaeus {version} Release Notes"
    title_line = f":gold:`{title_text}`"
    adornment = "=" * len(title_line)

    anchor = f"scb-v{slugify(version)}-notes"
    subtitle = f"Scarabaeus version {version} release notes"

    header = [
        ".. SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial "
        "Applications (ORCCA) Lab, University of Colorado at Boulder",
        ".. SPDX-License-Identifier: ISC",
        "",
        ".. meta::",
        "    :description lang=en:",
        f"        {subtitle}",
        "    :keywords:",
        f"        Scarabaeus, release, version {version}",
        "",
        f".. _{anchor}:",
        "",
        GOLD_STYLE_BLOCK,
        adornment,
        title_line,
        adornment,
        "",
        ".. contents::",
        "   :local:",
        "   :depth: 2",
        "",
        "----",
        "",
        "Release Notes",
        "-------------",
        "",
    ]

    if date:
        header.append(f"*Released {date}*")
        header.append("")

    body = []
    level = None  # nesting level of the previous bullet, None outside a list
    for line in lines[1:]:  # skip the original "## [x.y.z] - date" header line
        stripped = line.strip()
        if not stripped:
            body.append("")
            level = None
            continue
        h3 = re.match(r"^###\s+(.*)", stripped)
        if h3:
            heading = h3.group(1)
            if body and body[-1]:
                body.append("")
            body.append(heading)
            body.append("^" * len(heading))
            body.append("")
            level = None
            continue
        bullet = re.match(r"^(\s*)[-*]\s+(.*)", line)
        if bullet:
            # CHANGELOG nests by 4 spaces; RST nests under the parent's text
            # (2 spaces) and needs a blank line wherever the level changes
            depth = len(bullet.group(1).expandtabs(4)) // 4
            if level is not None and depth != level:
                body.append("")
            body.append(f"{'  ' * depth}- {md_line_to_rst(bullet.group(2))}")
            level = depth
            continue
        body.append(md_line_to_rst(stripped))
        level = None

    return "\n".join(header + body).rstrip() + "\n"

# wipe and regenerate so removed/renamed versions don't linger
if VERSIONS_DIR.exists():
    shutil.rmtree(VERSIONS_DIR)
VERSIONS_DIR.mkdir(parents=True)

text = CHANGELOG.read_text()
pattern = re.compile(r"^## \[(.+?)\](?:\s*-\s*(\d{4}-\d{2}-\d{2}))?\s*$", re.MULTILINE)
matches = list(pattern.finditer(text))

groups = defaultdict(list)

for i, m in enumerate(matches):
    version, date = m.group(1), m.group(2)
    if version.lower() == "unreleased":
        continue

    start = m.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    section = text[start:end].strip()

    parts = version.split(".")
    minor_key = ".".join(parts[:2]) if len(parts) >= 2 else version
    minor_slug = slugify(minor_key)
    patch_slug = slugify(version)

    patch_dir = VERSIONS_DIR / minor_slug
    patch_dir.mkdir(exist_ok=True)

    override_path = OVERRIDES_DIR / f"{patch_slug}.rst"
    dest_path = patch_dir / f"{patch_slug}.rst"

    if override_path.exists():
        shutil.copyfile(override_path, dest_path)
    else:
        rst = md_section_to_rst(section, version, date)
        dest_path.write_text(rst)

    groups[minor_key].append((version, date or "", patch_slug))


def sort_key(k):
    return tuple(int(x) if x.isdigit() else x for x in k.split("."))


sorted_minor_keys = sorted(groups.keys(), key=sort_key, reverse=True)

# --- generate a group page per minor version ---
for minor_key in sorted_minor_keys:
    minor_slug = slugify(minor_key)
    patches = groups[minor_key]

    title = f"{minor_key}.x Releases"
    lines = [title, "=" * len(title), "", ".. toctree::", "   :maxdepth: 1", "   :hidden:", ""]
    lines += [f"   {minor_slug}/{slug}" for _, _, slug in patches]
    lines.append("")
    for version, date, slug in patches:
        label = f":doc:`{version} <{minor_slug}/{slug}>`"
        lines.append(f"- {label}" + (f" - {date}" if date else ""))

    (VERSIONS_DIR / f"{minor_slug}.rst").write_text("\n".join(lines) + "\n")

# --- generate the top-level index, now showing every patch version inline ---
title = "Release Notes"
index_lines = [title, "=" * len(title), "", ".. toctree::", "   :maxdepth: 2", "   :hidden:", ""]
index_lines += [f"   versions/{slugify(k)}" for k in sorted_minor_keys]
for minor_key in sorted_minor_keys:
    minor_slug = slugify(minor_key)
    index_lines += [f"   versions/{minor_slug}/{slug}" for _, _, slug in groups[minor_key]]
index_lines.append("")

# visible content: minor version as a heading, patches as a nested list under it
for minor_key in sorted_minor_keys:
    minor_slug = slugify(minor_key)
    patches = groups[minor_key]

    heading = f"{minor_key}.x"
    index_lines.append(heading)
    index_lines.append("-" * len(heading))
    index_lines.append("")
    for version, date, slug in patches:
        label = f":doc:`{version} <versions/{minor_slug}/{slug}>`"
        index_lines.append(f"- {label}" + (f" - {date}" if date else ""))
    index_lines.append("")

(RELEASE_NOTES_DIR / "release_frontpage.rst").write_text("\n".join(index_lines) + "\n")

print('Updated release notes.')