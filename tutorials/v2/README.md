# Scarabaeus tutorials (v2)

> New tutorial series that will replace the notebooks in `tutorials/`. Chapters 0–8 are ready; chapters 9–11, on real data, are a first draft. See [PLAN.md](PLAN.md) for the full structure, design rules and open questions.

## Navigating OSIRIS-REx

One scenario, start to finish: OSIRIS-REx on approach to Bennu, June 1–8, 2018, simulated (chapters 0–8), then the real DSN passes of June 7–8 (chapters 9–11).

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
| 9 | [Real data I: from tracking files to residuals](09_real_dsn_data.ipynb) | first draft |
| 10 | [Real data II: a multi-pass fit](10_real_data_fit.ipynb) | first draft |
| 11 | [Real data III: is the solution any good?](11_real_data_prediction.ipynb) | first draft |

## Running

0. The notebooks target **Scarabaeus 2.0** (on `main` since #10); they no longer run on 1.x. On recent macOS, build the Rust extension with `CARGO_PROFILE_RELEASE_STRIP=none` (see PLAN.md).
1. Download the SPICE kernels once, from this folder (`tutorials/v2/`):
   ```bash
   python get_data.py               # chapters 0–8: five kernels, about 41 MB
   python get_data.py --real-data   # chapters 9–11 as well: about 1.7 GB more
   ```
   Of the 1.7 GB, almost all is the Jupiter and Saturn satellite ephemerides, which the real measurement models currently need for the ET − TAI time transformation; from the next Scarabaeus release, which uses the barycenters instead, they will not be needed. The tracking and media-correction files for chapters 9–11 are in the repository. (`load_data()` in `tutorials/supplementary` fetches the data of every tutorial, 2.8 GB; the series does not need it.)
2. Run the notebooks **from this folder** (`tutorials/v2/`): the metakernels `orex_approach.tm` (all chapters) and `real_data.tm` (chapters 9–11) and `scenario.py` use paths relative to it.

The objects chapters 1–5 build on the page (epochs, spacecraft, stations, measurement models, a-priori state and covariance) are collected in `scenario.py`, and so are the real-data pieces of chapters 9 and 10 (`real_tracking`, `save_editing` / `load_editing`, `real_data_fit`). From chapter 2 on, setup cells import from it what earlier chapters built (chapters 2–5 still build their own new objects on the page; chapters 6–9 import everything they reuse). Keep the module in sync with the chapter code.

Each notebook writes its products to `outputs/<notebook>/` (git-ignored).

### Starting at a later chapter

Chapters 3–8, 10 and 11 read files produced by earlier chapters (chapter 2's truth trajectory, chapter 3's tracking data, chapter 6's batch solution, chapter 9's editing of the real passes). To start at a later chapter, generate those inputs first:

```bash
python prepare.py 06            # runs chapters 2 and 3 headlessly (about 15 s)
python prepare.py 06 --force    # regenerate even if the outputs already exist
```

The prerequisite notebooks are executed in memory and are not modified. If the inputs are missing, the chapter's setup cell stops with a message pointing here (chapter 6: a comment in the setup cell, and a `SPICE` / file-not-found error).

## Known limitations

Library features the notebooks work around until they are added to Scarabaeus:

- **`get_state` takes one epoch per call.** Accepting an `EpochArray` is the main missing feature; until then, state histories are built in a loop (chapters 1, 2, 4, 7 and 8).
- **Progress output.** The filters and measurement models print progress that cannot be switched off yet, so the cells that run them start with `%%capture`.
- **Plotting functions.** `plotting_prototype.py` holds the plots of this series; they are planned to move into `scb.Plotting`.
- **Satellite ephemerides for real data.** The real measurement models evaluate ET − TAI with the Jupiter and Saturn centers, so chapters 9–11 load `jup310.bsp` and `sat360.bsp` (1.7 GB). The next release uses the barycenters, which `de424.bsp` covers; `real_data.tm` then shrinks to the stations' topocentric frames.
- **Results of earlier fits.** Reading `state_est` of a solution after a later fit of the same spacecraft returns the later fit's values, so the chapters read each solution right after its fit.
- **Real-data workflow.** No file format for an edited, weighted measurement set (chapter 9 saves flags and weights to JSON for chapter 10); no one-call prediction of a new pass (chapter 11 propagates the estimate to its own SPK and computes h P hᵀ from the model partials).

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
