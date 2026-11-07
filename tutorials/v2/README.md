# Scarabaeus tutorials (v2)

> New tutorial series that will replace the notebooks in `tutorials/`. Chapters 0–8 are ready; chapter 9 is a preview. See [PLAN.md](PLAN.md) for the full structure, design rules and open questions.

## Navigating OSIRIS-REx

One scenario, start to finish: OSIRIS-REx on approach to Bennu, June 1–8, 2018.

| # | Notebook | Status |
|---|---|---|
| 0 | [Quickstart](00_quickstart.ipynb) | ready |
| 1 | [The scenario](01_scenario.ipynb) | ready |
| 2 | [Truth: dynamics & propagation](02_truth_dynamics.ipynb) | ready |
| 3 | [Tracking data I: simulating measurements](03_tracking_data.ipynb) | ready |
| 4 | [The filter's dynamics model](04_filter_dynamics.ipynb) | ready |
| 5 | [The filter's measurement model](05_filter_measurements.ipynb) | ready |
| 6 | [First OD solution](06_first_od_solution.ipynb) | ready |
| 7 | [Sequential filtering & process noise](07_sequential_filtering.ipynb) | ready |
| 8 | [Estimating parameters: SRP scale factor & consider covariance](08_estimating_more.ipynb) | ready |
| 9 | [Tracking data II: real DSN data](09_real_dsn_data.ipynb) | preview (ranging offset estimated, not explained; see chapter) |

## Running

0. The notebooks target **Scarabaeus 2.0** (on `main` since #10); they no longer run on 1.x. On recent macOS, build the Rust extension with `CARGO_PROFILE_RELEASE_STRIP=none` (see PLAN.md).
1. Download the SPICE kernels once, from this folder (`tutorials/v2/`):
   ```bash
   python get_data.py               # chapters 0–8: five kernels, about 41 MB
   python get_data.py --real-data   # chapter 9 as well: about 1.7 GB more
   ```
   The tracking and media-correction files for chapter 9 are in the repository. (`load_data()` in `tutorials/supplementary` fetches the data of every tutorial, 2.8 GB; the series does not need it.)
2. Run the notebooks **from this folder** (`tutorials/v2/`): the metakernels `orex_approach.tm` (chapters 0–8) and `real_data.tm` (chapter 9) and `scenario.py` use paths relative to it.

The objects chapters 1–5 build on the page (epochs, spacecraft, stations, measurement models, a-priori state and covariance) are collected in `scenario.py`. From chapter 2 on, setup cells import from it what earlier chapters built (chapters 2–5 still build their own new objects on the page; chapters 6–9 import everything they reuse). Keep the module in sync with the chapter code.

Each notebook writes its products to `outputs/<notebook>/` (git-ignored).

### Starting at a later chapter

Chapters 3–8 read files produced by earlier chapters (chapter 2's truth trajectory, chapter 3's tracking data, chapter 6's batch solution). To start at a later chapter, generate those inputs first:

```bash
python prepare.py 06            # runs chapters 2 and 3 headlessly (about 15 s)
python prepare.py 06 --force    # regenerate even if the outputs already exist
```

The prerequisite notebooks are executed in memory and are not modified. If the inputs are missing, the chapter's setup cell stops with a message pointing here (chapter 6: a comment in the setup cell, and a `SPICE` / file-not-found error).

## Known limitations

Library features the notebooks work around until they are added to Scarabaeus:

- **`get_state` takes one epoch per call.** Accepting an `EpochArray` is the main missing feature; until then, state histories are built in a loop (chapters 1, 2, 4, 7, 8 and 9).
- **Progress output.** The filters and measurement models print progress that cannot be switched off yet, so the cells that run them start with `%%capture`.
- **Plotting functions.** `plotting_prototype.py` holds the plots of this series; they are planned to move into `scb.Plotting`.
- **Results of earlier fits.** Reading `state_est` of a solution after a later fit of the same spacecraft returns the later fit's values, so the chapters read each solution right after its fit.

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
