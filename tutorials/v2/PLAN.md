# Tutorial series v2 — plan

Replaces the current `tutorials/` set (3 basics / 5 intermediate / 4 advanced notebooks written independently), which repeats setup in every notebook, re-implements library functionality in notebook helpers, and mixes reference material with workflow.

Audience: people already familiar with orbit determination. The tutorials teach Scarabaeus, not OD.

> **Revision 2 (restructure).** The first drafts of chapters 0–9 followed the library's subsystems (time and bodies → propagation → measurements → filter dynamics → filter measurements → filter) and reached the first `fit()` in chapter 6. This revision reorders the series around one navigation question, merges the two halves of the SRP lesson, recasts the "inside the filter" material as model validation, and moves shared setup into `scenario.py`. Most prose, plots and numbers from the drafts carry over; see [Migration from the first drafts](#migration-from-the-first-drafts).

## Target release: Scarabaeus 2.0

The series is written against **Scarabaeus 2.0** (see `CHANGELOG.md` [2.0.0]; code on `origin/sync/internal-2026-10-19`, not yet merged into this branch). The first drafts were written and executed against 1.x, so:

1. **Port before restructuring.** Done (2026-10-21, against commit `0ea519e`): the drafts in this folder carry the API port below and the text fixes from the [2.0 rerun results](#20-rerun-results), and were re-executed on 2.0 with no errors. They no longer run on 1.x.
2. **Status columns** in the two tables below come from that rerun plus targeted probes: **fixed** / **open** / **partial** are observed behavior, not changelog claims.
3. **Use the new 2.0 features where they replace notebook code** (rule 3): `MeasurementScheduler` (ch. 2), `Plotting` (ch. 3 on), `SolutionCloseness` (ch. 5–6), `fit(callback=...)` (ch. 7), the multi-pass real-data setup (ch. 8).

Building 2.0 on macOS 27: the Rust extension built with the default release profile fails to load (`dlopen: mis-aligned LINKEDIT string pool`). Building with `CARGO_PROFILE_RELEASE_STRIP=none` works. Worth a `[profile.release] strip = false` in `Cargo.toml` or a note in `DEVELOPMENT.md`.

### Porting the drafts to 2.0

| 1.x code | 2.0 | Chapters |
|---|---|---|
| `t, sec, values, flags = model.observed_measurements(...)` | returns `ObsData` with a fifth field, `aux_dict`; use the field names | 3, 5, 8 |
| measurement file block `"meas_ideal"` | `"meas_sim"` | 3 |
| residual entries `[residual, σ]` (`for r, s in ...`) | `[residual, σ, epoch_tdb_sec]` in the saved solution | 6, 7 |
| `solution.json` iterations: `covariance_diagonals` | `final` stores `covariance_full`; iterations keep diagonals (`metadata.covariance_storage`) | 7 |
| corrupting an observation tuple in place | `observed[name]._replace(meas_obs=...)` | 8 |

### 2.0 rerun results

What changed in the drafts' outputs, and what each chapter's text needs:

- **Chapters 1–5:** same numbers (noise statistics, linearization error, partials check). One text change: epochs now display their calendar time in TDB (`2018-06-01T00:01:09.1849 (TDB)` for UTC midnight), so the ch1 note "calendar representations are displayed in UTC" is wrong in 2.0, and the half-open-interval example prints 23:31:09 TDB.
- **Fits converge one or two iterations earlier.** The first batch iteration now leaves a correction of 2.3 cm (13 cm in 1.x), which already passes the deviation test: its 1e-5 threshold is the RMS of all six components, so it accepts up to ~2.5 cm of position correction. The new cost test passes too (`fit()` defaults to `convergence_criterion="any"`), so the batch fit stops after **2** iterations instead of 3. Same χ² and covariance, but the estimators now agree to ~5 cm instead of mm: LSB − SRIFB 51 mm (was 3 mm), SRIF − SRIFB 44 mm (was 6.5 mm), LKF − SRIF 53 mm (was 1.9 mm). The drafts now explain both tests and say "a few centimeters".
- **Chapter 3 (old 06) saved file:** two iterations; the metadata now carries frame, units, state labels, the a-priori, the convergence test and provenance, which makes the "what's in the file" section more useful.
- **Chapter 6 (old 07) SNC:** the old levels now do almost nothing (1e-13: 63 km error, 37σ). The same behavior moved up by about the Δt = 1800 s factor the fix removed. With the SRP-free reference, end-of-arc results:

  | σ_SNC [km/s²] | error [km] | 1σ [km] | max error/σ | post-fit RMS/σ | iterations ("any" / "deviation") |
  |---|---|---|---|---|---|
  | 1e-13 | 63.5 | 1.7 | 37.2 | 1.46 | 2 / 4 |
  | 1e-12 | 53.9 | 2.1 | 25.4 | 1.36 | 2 / 5 |
  | 1e-11 | 30.3 | 3.0 | 11.5 | 1.10 | 2 / 5 |
  | 1e-10 | 3.1 | 3.8 | 1.5 | 0.73 | 2 / 10 (cap) |
  | 1e-9 | 2.7 | 5.2 | 1.3 | 0.71 | 2 / 10 (cap) |

  The SRP acceleration being absorbed is ~2e-11 km/s², so the consistent levels (1e-10 and up) are now physically sensible. Rescan for the 15 m² story; note the deviation criterion no longer converges at the useful levels, which the chapter should mention or avoid.
- **Chapter 6 (old 08) η_SRP:** 0.7325 ± 0.0253 (was 0.7318), truth 0.7333; converges in 2 iterations (was 4). The sidecar warning is still printed.
- **Chapter 7 (old 08 §4–5):** editing and covariance-analysis numbers unchanged (with the two-pass workaround still in place).
- **Chapter 8 (old 09), real data:**
  - The **16.75 Hz** outliers are gone: 15 Doppler points are flagged instead of 45, and the remaining residuals are 3.37 mHz RMS (1.17 mHz without the end of the pass; was 3.53 / 1.25).
  - The **first 15** Doppler points are still off (−382 kHz, about −5,960 Hz, and one +3,681 Hz), so they are *not* explained by the ramp-coverage fix; the 🟠 CHECK stays.
  - The **5 placeholder ranging rows** are now flagged on read, with a message.
  - The **ranging offset changed sign and size**: median O − C is now **+1,685 RU** (was −12,386 RU), with the same ±0.6 RU spread. Still a constant offset to explain.
  - The single-iteration fit now *improves* the line-of-sight velocity: a-priori −16.9 mm/s → **−1.1 mm/s** (1σ 0.63 mm/s; was +22.5 mm/s with 1σ 0.006 mm/s), on 202 observations.
  - Iterated to 8 iterations (probe), the solution is stable: same line-of-sight velocity, flat cost, position settled ~51 km from the a-priori (≈1σ of P₀) and 58 km from the flight reconstruction, because one Doppler pass barely constrains position. With `convergence_criterion="deviation"` it never formally converges (the position keeps creeping by ~10 m per iteration); the default `"any"` (cost) criterion is the right one here, and chapter 8 can use it to motivate the cost test.
  - The same fit on a **60 s grid** gives the same answer, so the 2-second-grid workaround and its 🔴 FIX note can go.
  - Building the analytical-vs-finite-difference check needs the auxiliary data for the whole pass: slicing the epochs without slicing the aux dict raises `AuxDataError`.


## Design rules

1. **One mission, one question, start to finish.** Every core chapter continues the same scenario and moves the same number: the prediction for the June 8 DSN pass ([below](#the-thread-predicting-the-june-8-pass)). A chapter that doesn't change or explain that number needs a reason to be in the core series.
2. **Explain once, link elsewhere.** Each concept has one home. Core chapters link to *Fundamentals* and *Model catalogs* pages instead of re-explaining.
3. **No helper functions in notebooks.** If a tutorial needs a helper, that is a missing library feature: report it, and mark any temporary workaround with a `TODO` review note. *Scenario definitions are not helpers*: the objects that define the mission (epochs, spacecraft, models, a-priori) live in `scenario.py` (see [below](#scenariopy)), as they would in a real project.
4. **Data chain through files, objects come from `scenario.py`.** Each chapter writes its products (SPKs, measurement files, solutions) and later chapters load them. Setup cells rebuild nothing a reader has already seen: they import it.
5. **Uniform page layout.** Goal and prerequisites at the top, numbered steps, a *Summary* table of the API covered, the updated scoreboard (rule 1), and a link to the next page.
6. **Every notebook runs.** Notebooks are executed before review and in CI (`pytest --nbmake`), and the docs build fails on errors (`nbsphinx_allow_errors = False`).
7. **Code reads like user code.** One formatter setting for all notebooks (line length ~100), NumPy for numerics (`np.linalg.norm`, not `sum(c**2 ...)**0.5`), no stdout redirection in the final versions.
8. **The reader does something.** Each core chapter ends with one or two short *Try it* prompts that change a single input (elevation mask, a data type, a σ) and say what to look for.

## Storyline: OSIRIS-REx on approach to Bennu, June 1–8, 2018

- The tutorial data include the flight team's reconstructed OSIRIS-REx trajectory for this period and real DSN tracking. 1.x shipped one pass (June 8, 2018 DOY 159, DSS-35, ~05:45–10:40 UTC); 2.0 adds June 7 (DOY 158, DSS-65, ~18:45–23:05), a second June 8 pass (DOY 159, DSS-65, ~18:30–23:05), and three July passes (DOY 183, 185, 187, DSS-35). Note that the June 7 pass falls inside the simulated data arc (see [Open questions](#open-questions)).
- **Truth for the simulated chapters** is a trajectory propagated by Scarabaeus from the flight state on June 1. Using the flight reconstruction directly as truth does not work early in the series: the inevitable small dynamics mismatch, combined with very precise measurements, biases the estimate well outside its covariance. That effect is kept for chapter 6 as a deliberate lesson.
- Force model: Sun, planets (Mercury–Saturn), cannonball SRP (A = 11 m², C_r = 1.2, m = 1500 kg). It reproduces the flight trajectory to ~0.2 km over the week.
- SPICE IDs: flight reconstruction `-64`, simulated truth `-64000`, filter reference `-64001`.
- **Data arc:** June 1 00:00 – June 8 00:00 UTC. Range and range-rate from DSS-14, DSS-43 and DSS-63, 30-min sampling, 10° elevation mask.
- **Truth arc:** June 1 00:00 – June 9 00:00 UTC, one day longer than the data arc, so that predictions for the June 8 pass can be scored against the truth.

### The thread: predicting the June 8 pass

The simulated week ends at 23:30 on June 7; the real DSN pass begins about six hours later. That gives the series its question:

> *Given a week of tracking, how well do we know where OSIRIS-REx will be when DSS-35 acquires it on June 8?*

Every core chapter reports the same **scoreboard**, evaluated at the first Doppler epoch of the real pass (`t_pass`):

| Quantity | Simulated chapters | Chapter 8 (real data) |
|---|---|---|
| Position error, RSS | estimate mapped to `t_pass` − truth (`-64000`) | estimate − flight reconstruction (`-64`) |
| Position 1σ, RSS, and error/σ | `propagate_covariance` to `t_pass` | same |
| Line-of-sight velocity 1σ | projects the covariance on the DSS-35 line of sight | compared with the pre-fit Doppler residuals |

How the number moves across the series:

| Ch. | What changes | Expected effect on the scoreboard |
|---|---|---|
| 3 | first batch solution | baseline: error within 1σ, ~km-level plane-of-sky σ |
| 5 | sequential filter, smoothing | unchanged (same estimate); shows *why* |
| 6 | filter's SRP area wrong (15 m²) | error ≫ σ; each remedy restores consistency at a different cost |
| 7 | outliers; alternative tracking plans | outliers move the prediction; dropping Canberra inflates σ |
| 8 | real pass arrives | the prediction is tested against real data, then updated |

`SolutionOD.propagate_covariance` and the final reference SPK already provide everything the scoreboard needs; chapter 3 computes it in the open, and later chapters reuse the same few lines (or a library method, if one is added: see [API changes](#api-changes-that-shorten-the-series)).

## scenario.py

A plain module in `tutorials/v2/` holding the mission definition: the code chapters 1–3 build on the page, collected in one place.

```python
# scenario.py — the "Navigating OSIRIS-REx" mission definition (chapters 1–3)
KERNELS          = "orex_approach.tm"
data_epochs      # EpochArray, June 1–8, 30 min              (ch. 1)
truth_epochs     # EpochArray, June 1–9                       (ch. 1)
t_pass           # first Doppler epoch of the June 8 pass     (ch. 1)
orex_flight      # Spacecraft(-64)                            (ch. 1)
orex_truth       # Spacecraft(-64000, mass, area, cr)         (ch. 1)
orex_reference   # Spacecraft(-64001, same properties)        (ch. 3)
PLANETS          # third bodies                               (ch. 1)
stations         # DSS-14/43/63                               (ch. 2)
models           # RangeIdeal / RangeRateIdeal per station    (ch. 2)
apriori_state    # StateDefinition: truth + 10 km, 1 cm/s     (ch. 3)
P0               # CovarianceMatrix: 100 km, 0.1 m/s          (ch. 3)
```

Rules:

- **Definitions only.** Constants and objects; no plotting, no analysis, no functions. Anything that would need a function is a missing library feature (rule 3). Temporary exceptions (e.g. a fresh-propagator factory while filters mutate their propagator) carry a `TODO` review note and an entry in `scb_suggestions.md`.
- **Shown before it's imported.** Chapters 1–3 build every object on the page and end with *"this is what `scenario.py` contains"*. From chapter 4 on, the setup cell is `from scenario import ...` plus loading the previous chapters' files.
- **Kept in sync by review.** The chapter code and `scenario.py` must match; check it in review until a CI test does (e.g. compare the objects built in chapters 1–3 with the module's).
- **Kernel loading stays explicit** in every setup cell (`SpiceManager.load_kernel_from_mkfile(KERNELS)` plus the chapter's SPKs), so the reader always sees what is in the pool.

## Structure

```
0. Getting Started
   ├── Install + tutorial data
   └── Quickstart: the whole pipeline on one page, plus the object-flow diagram

I. Core series — "Navigating OSIRIS-REx" (in order)
   1  Scenario & truth             kernels, epochs, spacecraft; propagate the truth; force-model fidelity; SPK
   2  Tracking data                stations and visibility; simulate range/range-rate; measurement files; noise check
   3  Your first OD solution       a-priori and P0; SRIFB fit; residuals; errors vs. 3σ; covariance shape; June 8 prediction
   4  Validating a model           what fit() linearizes: STM, partials, HΦδx₀; a checklist for any new model
   5  Sequential filtering         SRIF/LKF, smoothing, equivalence with batch
   6  When the dynamics are wrong  one SRP mismodel, four responses: ignore, SNC, estimate η, consider η
   7  Data quality & planning      measurement editing; covariance analysis of tracking plans
   8  Real DSN data                predict the June 8 pass, validate the real models, edit, fit, compare with the flight

II.  Fundamentals     units, frames, time, SPICE and bodies, state definition, trajectories and SPKs, measurement files
III. Model catalogs   dynamics models, integrators, measurement types, filters, process noise, measurement editing, filter outputs
IV.  How-tos          maneuvers, B-plane, media corrections & ramp tables, n-plate and attitude, optical navigation, multi-arc OD, consistency checks
```

### 0. Quickstart

The current page, shortened once the API fixes land (target: ~30 lines of code). Add one diagram of how the objects connect, since this is the central non-obvious idea of Scarabaeus and is currently scattered across chapters 1, 2 and 4:

```
SpiceManager (kernel pool) ◄──────────── write_to_spk ◄── Trajectory ◄── Propagator ◄── StateArray ◄── StateDefinition
     │  states, frames, time                                                 ▲              ForceModelTranslation
     ▼                                                                       │
Body / Spacecraft / GroundStation ──► measurement model ──► file ──► MeasurementSpec ──► SRIFB/SRIF/LSB/LKF.fit() ──► SolutionOD
                                                                     FilterSettings ───────────┘
```

Key message: *everything is looked up through the kernel pool*, which is why every trajectory that gets measured is written to an SPK with its own SPICE ID.

### 1. Scenario & truth

From drafts 01 and 02. Keep: kernel table, the epoch half-open-interval note, the approach-geometry plot, the force-model fidelity plot and its paragraph (one of the best in the series). Move to *Fundamentals*: the units `ValueError` demo, `ArrayWFrame` slicing details, the TDB/UTC display discussion. Propagate the truth over `truth_epochs` (through June 9). Define `t_pass`.

2.0 updates: SPKs are now written as type 13 (Hermite), so rewrite the "Lagrange-interpolates" sentence and drop its 🟠 CHECK note; mention that IAS15 can't integrate the STM (`Propagator.IAS15STMError`), so the filter's reference uses DOP853; `ForceModelTranslation.list_forces()` can replace printing `dynamics_mode_list`.

**Produces:** `truth.bsp`. **Ends with:** the `scenario.py` contents introduced so far.

### 2. Tracking data

Draft 03, plus the station-visibility plot from draft 01 (it belongs next to the tracking schedule it explains). Keep the noise check.

2.0 updates: build the tracking with **`MeasurementScheduler`** (stations, types, window, cadence, elevation mask), which writes the files, reads them back and returns the `MeasurementSpec`: this replaces the model loop and the `name.replace(" ", "_")` path building. Still open one file and show its layout (the simulated block is now `meas_sim`, not `meas_ideal`), since chapter 8's real files share it. `observed_measurements` now returns an `ObsData` named tuple (`meas_time_et`, `meas_sec`, `meas_obs`, `meas_outliers`, `aux_dict`), which replaces the index table.

**Produces:** six measurement files.

### 3. Your first OD solution

Draft 06, with the a-priori state and P₀ from draft 04 set up inline (about 10 lines; the flags table moves to *Fundamentals: Defining a state*). Keep: iteration table, pre/post-fit residuals, error vs. 3σ, line-of-sight vs. plane-of-sky covariance, the LSB swap. Replace the June 10 prediction with the **first scoreboard** at `t_pass`.

2.0 updates: `verbose=False` doesn't yet silence the filter (the measurement models still print), so the redirection stays for now; `fit()` writes each iteration as `<name>_it{n}.bsp` in the filter's folder, so `traj_name` / `traj_dir` no longer need repeating; use `Plotting.plot_state_errors`, `plot_postfits_scatter` and `plot_covariance_est` instead of hand-written matplotlib where they show the same thing (check that they accept a truth trajectory and keep the figures readable); `SolutionOD.save()` now records the a-priori, state labels and provenance, so the saved-file walk-through can show them.

**Produces:** `solution.json`, `reference.bsp` (final iteration).

### 4. Validating a model

Drafts 04 §3–4 and 05 §2–4, reframed. The question is no longer "what is an STM" (the audience knows) but *"how do I check that Scarabaeus's dynamics and measurement models linearize correctly before I trust a fit?"*:

1. STM check: Φδx₀ against the actual deviation of two propagations.
2. Partials check: `compute_partials` against an independent geometric computation.
3. End-to-end check: HΦδx₀ against the actual pre-fit residuals.
4. A short checklist to run when adding a force model, a measurement type, or a state parameter.

Chapter 8 applies step 3 to `DopplerReal` (the check that exposes the current partials bug). Marked as skippable on first reading; chapter 5 doesn't depend on it.

### 5. Sequential filtering

Draft 07 §1–3, shortened: SRIF with smoothing, filtered vs. smoothed vs. batch, LKF as a one-line swap. Its job is to establish that without process noise the sequential and batch estimates are the same, so chapter 6 can use process noise. 2.0: `SolutionCloseness` (the σ at which two solutions' uncertainty ellipsoids touch) is the library's way to say "the same", replacing the hand-computed millimeter differences.

### 6. When the dynamics are wrong: the SRP story

Merges draft 07 §4–5 and draft 08 §1–3. The drafts used two different mismodels (SRP omitted entirely in 07; SRP area 15 m² instead of 11 m² in 08), so the remedies couldn't be compared. Use **one** mismodel throughout: the filter's spacecraft has A = 15 m² (a 36 % SRP error); the truth is unchanged.

| Response | Filter | What the reader sees |
|---|---|---|
| Ignore it | SRIFB, 15 m² | error ≫ σ, residuals look nearly normal: the signature of unmodeled dynamics |
| Process noise | SRIF + SNC | consistent but looser; σ scan shows the trade-off |
| Estimate it | SRIFB + `eta_srp` estimated | η recovered (~11/15), error back to baseline, σ grows slightly |
| Consider it | SRIFB + `eta_srp` considered | estimate still wrong, but the covariance now covers it |

One comparison figure (error and 3σ along the arc, four panels) and one table: the four rows of the **scoreboard at `t_pass`**. Closing paragraph: when to use each (unknown form → process noise; known form, observable → estimate; known form, not the filter's to adjust or poorly observable → consider).

The SNC discretization is fixed in 2.0 (confirmed by the rerun): the useful levels moved from ~1e-13 to ~1e-10 km/s², comparable to the SRP acceleration being absorbed, so the σ scan is now physical; rescan for the 15 m² case and delete the 🔴 FIX note. The non-PSD `covariance_est` with consider parameters is still there in 2.0 (smallest eigenvalue −6.5), so keep using `covariance_consider` and keep the note. Link to *Model catalogs: Process noise* for DMC and the 2.0 empirical accelerations (PFOGM, `FourierAcceleration`, RTN frames); the real-data chapter uses PFOGM, so name it here as the operational variant of "process noise".

### 7. Data quality & planning

Draft 08 §4–5. Measurement editing on the corrupted DSS-43 range, then covariance analysis of three tracking plans. `editing_method` still edits the a-priori pre-fits in 2.0: enabling it from the a-priori makes the fit fail (`ValueError: No state deviation history available`). Try `fit(callback=...)`, which runs between iterations, as a single-run alternative to the two-pass workaround. `MeasurementScheduler` builds the covariance-analysis plans directly. Both report the scoreboard: outliers move the June 8 prediction; dropping Canberra or range-rate inflates its σ. Covariance analysis can also mention it is the tool for choosing the plan *before* the week of chapter 2.

### 8. Real DSN data

Draft 09, reframed as the payoff of the thread:

1. **The prediction.** Load the chapter 3 solution, map the estimate and covariance to the pass. This is the a-priori, not "flight state + 10 km".
2. **The pass arrives.** Read the TRK-2-34 file, build `DopplerReal` / `SequentialRangingReal`, media corrections.
3. **Validate the models** against the flight reconstruction (chapter 4's checks, including HΦδx₀ for Doppler).
4. **Pre-fit residuals against the prediction**: does the predicted σ cover what the real data show? The simulated truth and the flight differ by ~0.2 km after a week, so this is a genuine test.
5. **Edit and fit**, then compare the update with the flight reconstruction.

The same filter code as chapter 3; what changes is the measurement models, the auxiliary data, and the care the data need.

**2.0 changes this chapter the most.** The 2.0 `advanced_RealMSR_OSIRIS_REx_OD` tutorial is already a multi-pass batch OD on the June passes (SRIFB, PFOGM, estimated η, one range bias per pass, Huber + MAD editing, a three-day forward-prediction test against OD044). Build chapter 8 from that notebook rather than from draft 09, keeping draft 09's model-validation sections and the prediction framing. Its notebook-side helpers (`run_batch_od`, `compare_od044`, `predicted_sigma`, `quiet`, …) fall under rule 3: each is either user code that belongs on the page, or a library gap to log.

What the 2.0 rerun of draft 09 showed (details in [2.0 rerun results](#20-rerun-results)):

- **fixed:** the 16.75 Hz outliers (count-interval conversion); the `DopplerReal` partials (analytical = finite difference); the oscillating iterated fit; the 2 s grid; the 5 placeholder ranging rows (flagged on read);
- **still open:** the first 15 Doppler points (not ramp coverage after all); a constant ranging offset, now **+1,685 RU**; `trk234.py` still needed, since `observed_measurements` doesn't build the auxiliary data from the tracking file;
- **not yet retested:** estimating `range_bias_RU` (the 2.0 real-data tutorial does, so it likely works), which would absorb the ranging offset in the fit but not explain it.

## Migration from the first drafts

| Draft | Goes to |
|---|---|
| `00_quickstart` | 0 (shortened, plus diagram) |
| `01_scenario` | 1; visibility plot → 2; units/time/frame detail → Fundamentals |
| `02_truth_dynamics` | 1 (truth arc extended to June 9) |
| `03_tracking_data` | 2 |
| `04_filter_dynamics` | a-priori and P₀ → 3; STM and linearization → 4; flags and parameter table → Fundamentals: Defining a state |
| `05_filter_measurements` | partials and HΦδx₀ → 4; `MeasurementSpec` → 3 |
| `06_first_od_solution` | 3 (June 10 prediction → scoreboard at `t_pass`) |
| `07_sequential_filtering` | §1–3 → 5; §4–5 → 6 (rerun with the 15 m² mismodel) |
| `08_estimating_more` | §1–3 → 6; §4–5 → 7 |
| `09_real_dsn_data` | 8 (a-priori from chapter 3's prediction) |

Also update: `README.md` (table, file names), `prepare.py` (`PRODUCTS` / `REQUIRES`), and the cross-chapter references in every notebook.

New file names: `01_scenario_and_truth`, `02_tracking_data`, `03_first_od_solution`, `04_validating_models`, `05_sequential_filtering`, `06_wrong_dynamics`, `07_data_quality_and_planning`, `08_real_dsn_data`.

## Blocking library issues

Found in 1.x while drafting; each forced a 🔴 FIX note or a workaround. Status from rerunning the drafts and targeted probes on 2.0 (2026-10-21):

| Issue | Chapters | 2.0 status |
|---|---|---|
| `StateNoiseCompensation` Γ uses [Δt³/2·I; Δt²·I] instead of [Δt²/2·I; Δt·I] | 6 | **fixed**: useful levels moved by ~Δt (1e-13 → ~1e-10 km/s²); retune |
| Consider runs: `covariance_est` not positive semi-definite | 6 | **open**: smallest eigenvalue −6.5; `covariance_consider` is fine |
| `_parameters.json` sidecar lookup matches every iteration by pattern | 6 | **open**: the "Multiple matching JSON files found" warning still appears (`SpiceManager.py:1440`) |
| Editing applied on the first iteration flags every observation | 7 | **open**: the fit fails (`No state deviation history available`); try `fit(callback=...)` |
| `DopplerReal` analytic partials ~2.35× too small | 4, 8 | **fixed**: analytical = finite difference (ratio 1.0000 on all 217 points) |
| Iterated real-data fit oscillates and diverges | 8 | **fixed**: stable over 8 iterations; LOS velocity error −1.1 mm/s (1.8σ) |
| Iterated batch filters reset the a-priori (`zx_hat_k = 0`) | 8 | **fixed, as far as observable**: the estimate settles ~1σ from the a-priori and stops moving |
| STM interpolation check rejects any grid coarser than ~4 s | 8 | **fixed**: a 60 s grid gives the same solution (now a relative tolerance, `Trajectory.max_interp_error`) |
| Doppler outliers at 16.75 Hz every ~7 min | 8 | **fixed**: gone |
| First 15 Doppler points off by −382 kHz / −5,960 Hz | 8 | **open**: unchanged |
| Sequential ranging constant offset | 8 | **open**, changed: +1,685 RU (was −12,386 RU) |
| `range_bias_RU` partials not implemented | 8 | not retested (the 2.0 real-data tutorial estimates them) |

## API changes that shorten the series

Not blocking, but each removes boilerplate or a workaround from several chapters. Ordered roughly by how much notebook code they remove (details in `scb_suggestions.md`), with their status in 2.0.

| # | Change | 2.0 status |
|---|---|---|
| 1 | A verbosity switch for propagators, measurement models and filters (removes every `%%capture`, `redirect_stdout` and `quiet()`) | **partial**: filters take `verbose`, but `SRIFB(..., verbose=False).fit(verbose=False)` still printed ~150 kB, because `Measurement.compute_partials` prints and draws progress bars unconditionally |
| 2 | `get_state` / `Trajectory` over an `EpochArray` (removes every per-epoch loop) | **open**: `Body.get_state` on a 7-epoch `EpochArray` raises `RuntimeError` |
| 3 | Filters stop mutating the `Propagator` they are given (removes the propagator factories) | not retested; nothing in the changelog |
| 4 | `fit()` defaults `traj_name` / `traj_dir` to the constructor's values | **done**: `fit()` without names writes `<name>.bsp`, `<name>_it2.bsp`, … beside the constructor's |
| 5 | `observed_measurements` returns a named object, with units stored in the file | **partial**: `ObsData` named tuple (the drafts need a small port); files still carry no units. `MeasurementScheduler` hides it for simulated data |
| 6 | A `seed=` argument for simulated measurements | **open**: no seed or generator argument |
| 7 | `StateDefinition.param(name, owner, 1.0)` accepting plain numbers | **partial**: a plain float raises `AttributeError`, but `ArrayWUnits([1.0], None)` is accepted without the `ArrayWFrame(..., J2000)` wrapper the drafts use |
| 8 | `Spacecraft` accepting area in m² | **already works, also in 1.x**: `CannonballSRP` converts the area to km² itself, so the drafts' `.convert_to(km**2)` can go |
| 9 | `SolutionOD` helpers: errors against a truth, normalized residuals, plots | **largely done** (not yet exercised): `Plotting` (`plot_state_errors`, `plot_postfits_*`, `plot_covariance_est`, `plot_corner`), `SolutionCloseness`, `state_covariance_to_elements` |
| 10 | A TRK-2-34 loader replacing `trk234.py` | **open**: `trk234.py` still runs unchanged on 2.0 and is still needed; the aux dicts must cover the whole pass (`AuxDataError` otherwise) |
| 11 | A frame `repr` that shows the vector's actual origin | **open**: Sun-relative states still print `(0 - SOLAR SYSTEM BARYCENTER)` |
| 12 | `Propagator` returns a `Trajectory` (claimed in the changelog) | **not found**: `propagate()` returns `None` and the propagator has no trajectory attribute; the drafts' `Trajectory(state_array=propagator.propagated_state_array)` still works |

## Where the current notebooks go

| Current notebook | New home |
|---|---|
| `basics_AWU_and_AWF` | Fundamentals: units; frames |
| `basics_EpochArray` | Fundamentals: time |
| `basics_SpiceManager_CelestialBody_Constants` | Fundamentals: SPICE and bodies; chapter 1 |
| `intermediate_Propagator_and_Trajectory` | chapter 1; Fundamentals: state, trajectories; catalogs: dynamics, integrators |
| `intermediate_Measurements` | chapters 2 and 4; Fundamentals: measurement files; catalog: measurement types |
| `intermediate_nPlateModel_and_Attitude` | how-to: n-plate and attitude |
| `intermediate_MissionSequence_and_FiniteBurn` | how-to: maneuvers |
| `intermediate_BPlane` | how-to: B-plane |
| `advanced_IdealMSR_BatchOD` | chapters 3, 6, 7; catalogs: dynamics, measurement editing |
| `advanced_IdealMSR_SequentialOD` | chapters 5, 6; catalogs: filters, process noise; how-to: multi-arc OD |
| `advanced_RealMSR_MediaCorrections` | how-to: media corrections & ramp tables |
| `advanced_RealMSR_OSIRIS_REx_OD` (2.0 multi-pass version) | chapter 8; how-to: multi-pass real-data OD (the July passes and PFOGM tuning, if they don't fit chapter 8) |

## Status

All drafts are ported to 2.0 and execute without errors (2026-10-21). Removed as fixed in 2.0: the SNC 🔴 FIX (old 07; levels retuned to 1e-12–1e-9 km/s²), and the STM-interpolation and `DopplerReal` 🔴 FIX notes (old 09; 60 s grid, fit iterated to convergence).

| Chapter | Status |
|---|---|
| 0–3 | content drafted (in old drafts 00–06); restructure pending; ch. 2 to move to `MeasurementScheduler` |
| 4 | content drafted (old 04–05); reframing pending |
| 5 | content drafted (old 07 §1–3); restructure pending |
| 6 | partly drafted (old 07–08); rerun with one mismodel; SNC fix in 2.0, retune |
| 7 | content drafted (old 08 §4–5); editing via `fit(callback=...)` to try |
| 8 | model validation drafted (old 09); rebuild from the 2.0 multi-pass tutorial |
| `scenario.py` | to write |
| Fundamentals, catalogs, how-tos | planned; 2.0 adds how-to candidates (thermal recoil for OSIRIS-REx, MCMC and corner plots, custom models) |

## Decisions in this revision

- **Pre-generated outputs (was an open question): yes.** Ship the chapter 1–3 products as a small download alongside the tutorial data, so any chapter can start without running earlier ones; keep `prepare.py` for regenerating them.

## Open questions

- Should the Quickstart live in the docs site rather than the notebook series? (Leaning: keep it as notebook 0 and render it in the docs with nbsphinx, so it runs in CI with the rest.)
- Scoreboard: compute it with a few visible lines in each chapter, or wait for a `SolutionOD` method (API change 9)?
- Chapter 8 a-priori: chapter 3's prediction (simulated truth) versus a prediction from a chapter 6 solution. The second is more realistic if the "estimate η" solution is the operational one; check which gives the clearer lesson.
- **The June 7 pass (2.0 data) overlaps the simulated arc.** The simulated tracking runs to June 8 00:00, but the first real pass starts June 7 ~18:45, so the chapter 3 solution doesn't *predict* it. Proposal: end the simulated data arc at June 7 18:00 (or at the last sample before the first real pass), so chapter 8 can fit all three June 7–8 passes as genuine predictions; the scoreboard epoch `t_pass` then moves to the first real pass. Every number reruns on 2.0 anyway, so the cost is small.
- 🟠 CHECK: is the predicted line-of-sight velocity σ at `t_pass` large enough to cover the simulated-truth vs. flight difference, or will the real pre-fit Doppler residuals start many σ off? Either outcome is a lesson, but chapter 8's text depends on it.
