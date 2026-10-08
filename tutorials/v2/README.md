# Scarabaeus tutorials (v2 draft)

> **Draft.** New tutorial series that will replace the notebooks in `tutorials/`. See [PLAN.md](PLAN.md) for the full structure, design rules and open questions.

## Navigating OSIRIS-REx

One scenario, start to finish: OSIRIS-REx on approach to Bennu, June 1–8, 2018.

| # | Notebook | Status |
|---|---|---|
| 0 | [Quickstart](00_quickstart.ipynb) | draft |
| 1 | [The scenario](01_scenario.ipynb) | draft |
| 2 | [Truth: dynamics & propagation](02_truth_dynamics.ipynb) | draft |
| 3 | [Tracking data I: simulating measurements](03_tracking_data.ipynb) | draft |
| 4 | [The filter's dynamics model](04_filter_dynamics.ipynb) | draft |
| 5 | [The filter's measurement model](05_filter_measurements.ipynb) | draft |
| 6 | [Your first OD solution](06_first_od_solution.ipynb) | draft |
| 7 | Sequential filtering & process noise | planned |
| 8 | Estimating more: parameters, consider, editing | planned |
| 9 | Tracking data II: real DSN data | planned |

## Running

1. Download the tutorial data once (the existing loader in `tutorials/supplementary` fetches the kernels):
   ```bash
   cd tutorials && python -c "import supplementary as s; s.load_data()"
   ```
2. Run the notebooks **from this folder** (`tutorials/v2/`): the metakernel `orex_approach.tm` uses paths relative to it.

Each notebook writes its products to `outputs/<notebook>/` (git-ignored).

## Review notes

While drafting, flag anything that needs revision with a colored note. The emoji and keyword matter: they keep the note readable where colors are stripped (GitHub) and let `review_notes.py` find it.

| Tag | Color | Meaning |
|---|---|---|
| 🔴 **FIX** | red `#dc2626` | Wrong or broken; must change before release |
| 🟠 **CHECK** | orange `#ea580c` | Check a number, claim or output |
| 🔵 **TODO** | blue `#2563eb` | Missing content to write or add |
| 🟣 **QUESTION** | purple `#9333ea` | Open question or a decision we haven't made |
| 🟢 **IDEA** | green `#16a34a` | Optional improvement, nice to have |

**Markdown cells**: a box (inline, or as its own cell), changing the color for each tag:

```html
<div style="border-left:4px solid #dc2626; background:rgba(220,38,38,0.12); padding:6px 10px; border-radius:4px">
🔴 <b>FIX:</b> the RA/Dec plot uses the wrong frame here.
</div>
```

To flag a phrase inside a sentence: `<mark style="background:rgba(234,88,12,0.3)">this value</mark> 🟠 **CHECK:** source?`

**Code cells**: a comment:

```python
# 🟠 CHECK: is 1e-12 tolerance too tight for the 30-day arc?
```

**Listing notes**:

```bash
python review_notes.py                          # every note in every notebook
python review_notes.py 03_tracking_data.ipynb   # one notebook
python review_notes.py --tags FIX CHECK --strict  # fails if any FIX/CHECK is left (pre-release)
```
