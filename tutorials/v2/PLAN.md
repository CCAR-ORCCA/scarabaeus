# Tutorial series v2 — plan

Replaces the current `tutorials/` set (3 basics / 5 intermediate / 4 advanced notebooks written independently), which repeats setup in every notebook, re-implements library functionality in notebook helpers, and mixes reference material with workflow.

Audience: people already familiar with orbit determination. The tutorials teach Scarabaeus, not OD.

> **Revision (2026-11-05): proposal.** Two pieces of review feedback: the series spends much of its length on setting up the filter and too little on the hard part, real data; and its plots need work (prototypes in `plotting_prototype.py`, demonstrated in `plotting_prototype_demo.ipynb`). This revision proposes merging the setup chapters (9 core chapters become 8, three of them on real data) and splitting the real-data chapter into three. Sections marked **proposed** are not built yet. The restructure prototype in [`../v3/`](../v3/) (with its own plan) remains for reference; its SRP chapter is reused below.
>
> **Additions (2026-11-07).** A review against the library's user guide found Scarabaeus features the series does not use yet: finite-difference partial checks, `observability_test()`, the causal prediction products (`lookback_s`, `use_all_data`), `MeasurementSpec.build` with `MeasurementConfig` for real data, reweighting in the `fit` callback, and several how-to topics. They are folded into the chapters below and marked *(added 2026-11-07)*.

## Design rules

1. **One mission, start to finish.** Every core chapter continues the same scenario; the reader never re-learns setup.
2. **Explain once, link elsewhere.** Each concept has one home. Core chapters link to *Fundamentals* and *Model catalogs* pages instead of re-explaining.
3. **No helper functions in notebooks.** If a tutorial needs a helper, that is a missing library feature: report it, and mark any temporary workaround with a `TODO` review note. Scenario definitions live in `scenario.py`, which is not a helper.
4. **Chapters chain through files.** Each chapter writes its products (SPKs, measurement files, solutions) and later chapters load them; objects built in earlier chapters come from `scenario.py`.
5. **Uniform page layout.** Goal and prerequisites at the top, numbered steps, a *Summary* table of the API covered, and a link to the next page.
6. **Every notebook runs.** Notebooks are executed before review; the goal is to run them in CI.
7. **Length budget (new).** A core chapter is about 20–30 minutes: at most ~150 lines of code, ~1500 words, a setup cell of ~30 lines, and at least one figure per major section. Material beyond that goes to *Fundamentals*, a catalog page or a how-to.

## Storyline: OSIRIS-REx on approach to Bennu, June 2018

- **Simulated chapters:** one week of tracking, June 1–8. The truth is propagated by Scarabaeus from the flight state on June 1 and runs to June 11, so predictions can be scored. Force model: Sun, planets (Mercury–Saturn), cannonball SRP (A = 11 m², C_r = 1.2, m = 1500 kg); it reproduces the flight trajectory to ~0.2 km over the week. Range and range-rate from DSS-14, DSS-43 and DSS-63, 30-min sampling, 10° mask.
- **Real-data chapters (proposed):** the three DSN passes of June 7–8 in the tutorial data, with the flight team's reconstruction (OD044, SPICE ID `-64`) as the reference:

  | Pass | Station | UTC | Range / Doppler points | Role |
  |---|---|---|---|---|
  | June 7 (DOY 158) | DSS-65, Madrid | 18:45–23:05 | 82 / 189 | fit |
  | June 8 (DOY 159) | DSS-35, Canberra | 05:45–10:40 | 99 / 217 | fit |
  | June 8 (DOY 159) | DSS-65, Madrid | 18:30–23:05 | 88 / 206 | held out: prediction test |

  They continue the June story, and the held-out pass on the evening of June 8 closes the series' "predict the next pass" thread. The 2.0 real-data notebook uses three July passes (DOY 183/185/187, DSS-35 only); its methods carry over, and the July case becomes a how-to.
- SPICE IDs: flight reconstruction `-64`, simulated truth `-64000`, filter reference `-64001`.

## Structure (proposed)

```
0. Getting Started
   ├── Install + tutorial data
   └── Quickstart: the whole pipeline on one page

I. Core series — "Navigating OSIRIS-REx" (in order)
   1  Scenario & truth             kernels, time, spacecraft; propagate the truth; force-model fidelity; SPK
   2  Tracking data                stations and visibility; simulate range/range-rate; measurement files; noise check
   3  The linearized problem       reference trajectory and STM; a priori; pre-fit residuals, partials, H Φ δx₀
   4  First OD solution            batch fit; residuals; errors vs. covariance; prediction; covariance analysis
   5  When the dynamics are wrong  sequential filters and smoothing; one SRP error, four responses
   6  Real data I: from tracking files to residuals    models, delays, media, error budget, editing, weights
   7  Real data II: a multi-pass fit                    biases, η_SRP, stochastic accelerations, results vs. OD044
   8  Real data III: is the solution any good?          prediction, the held-out pass, tuning by prediction

II.  Fundamentals     units, frames, time, SPICE and bodies, state definition and flags, trajectories and SPKs, measurement files
III. Model catalogs   dynamics models, integrators, measurement types, filters, process noise, measurement editing, filter outputs
IV.  How-tos          covariance realism (Monte Carlo), multi-pass cruise OD (the July passes), media corrections & ramp
                      tables, maneuvers, B-plane, n-plate and attitude, optical navigation, multi-arc OD;
                      added 2026-11-07: extending Scarabaeus (CustomDynamicModel, CustomMeasurement,
                      CustomFilter), MCMC posterior sampling, station and Earth-orientation estimation,
                      small-body dynamics at Bennu (polyhedral gravity, centroiding OpNav)
```

The core series goes from 9 chapters to 8, and its real-data share from one chapter to three.

### Where the current chapters go

| Current | Proposed | Change |
|---|---|---|
| 0 Quickstart | 0 | swap in the new plots once they are in the library |
| 1 Scenario (103 lines of code) + 2 Truth dynamics (94) | **1 Scenario & truth** | merge. Move the units `ValueError` demo, the TDB/UTC display notes, `ArrayWFrame` slicing and the state flags to *Fundamentals*. Keep the kernel table, the approach-geometry plot and the force-model fidelity plot. Visibility moves to chapter 2. *(Added 2026-11-07)* A short callout on the file-based data flow: the truth is an SPK with JSON sidecars (STMs, parameters, settings), and every filter iteration later writes its own reference SPK under `traj_dir`, so readers know where files come from and where to clean up |
| 3 Tracking data I (131) | **2 Tracking data** | add the visibility from chapter 1 as `plot_tracking_timeline`; keep the file walk-through and the noise check |
| 4 Filter dynamics (137) + 5 Filter measurements (155) | **3 The linearized problem** | merge: reference trajectory and STM, the a priori and P₀, pre-fit residuals, partials, H Φ δx₀ against the actual residuals, `MeasurementSpec`. Cut the parameter table (Fundamentals) and the covariance-mapping figure (chapter 4 shows the same with data). Frame it as "what `fit()` solves, and how to check a model", so chapter 7 can apply the checks to the real models. *(Added 2026-11-07)* Check the analytical partials against `partials_method="fd"` (the same model with finite differences), and say in one sentence why the simulated chapters use the Ideal models: the Real models cannot simulate (`RealModelNotSupportedError`) |
| 6 First OD solution (105) | **4 First OD solution** | add covariance analysis from 7 §6: the same filter without data, comparing tracking plans. Use `plot_residuals`, `plot_consistency` and `plot_convergence`. *(Added 2026-11-07)* Show the prediction with the causal products of `propagate_state_covariance`: filtered (`lookback_s=0`), predicted (`lookback_s>0`) and reconstructed (`use_all_data=True`), and note that the default (`lookback_s=None`) is not causal |
| 7 Sequential filtering (216) + 8 Estimating parameters (117) | **5 When the dynamics are wrong** | merge around one SRP error (filter area 15 m², truth 11 m²), as in `../v3/06_wrong_dynamics.ipynb`: SRIF/LKF and smoothing first (short), then ignore / SNC / estimate η / consider η compared on one prediction scoreboard, with `plot_residual_summary` (the pass-to-pass drift) and `plot_prior_posterior` (the η–velocity ridge) |
| 9 Tracking data II (263) | **6, 7, 8** | split; see below |

Expected lengths after the merges: chapters 1–5 about 120–160 lines of code each (today 94–216), the three real-data chapters about 130–160 each.

## The real-data chapters (proposed)

Sources: the current chapter 9, the 2.0 notebook `advanced_RealMSR_OSIRIS_REx_OD` (sections 2–10), and the plot prototypes. Each chapter writes what the next one reads.

### 6. Real data I: from tracking files to residuals

*Goal: turn three DSN passes into clean, weighted measurements that agree with the flight reconstruction, and understand every term in between.*

1. **The tracking files.** The converted TRK-2-34 layout: observable blocks (`2W_doppler`, `2W_sranging`) and auxiliary SFDU records (ramps, count times, turnaround ratio, calibrations). `observed_measurements` now reads the auxiliary data itself (`ObsData.aux_dict`): **`trk234.py` can go** (checked on the three June passes, 2026-11-05). *(Added 2026-11-07)* Build the streams with `MeasurementSpec.build([(file, type, station), ...], config=MeasurementConfig(...))`, the entry point the measurements guide documents for real data, instead of constructing each model by hand as chapter 9 does today; if it cannot carry something the chapter needs, report it (rule 3). `plot_tracking_timeline` shows the three passes against the stations' visibility: tracking follows a schedule, not the geometry.
2. **The real models.** `DopplerReal` and `SequentialRangingReal`: ramped uplink, round-trip light time with relativistic terms, turnaround ratio, count interval, range units. Station delays from the files (`electronic_delays=True`) and the spacecraft transponder delay (1649.9 ns, estimated in house; not public). *(Added 2026-11-07)* Repeat chapter 3's finite-difference partials check on the real models: their analytical partials are new in 2.0 and their V&V is still on the roadmap.
3. **Media corrections.** Seasonal and non-seasonal troposphere (`2018_152_2018_182_tro.json` covers June); no ionosphere file for these dates.
4. **An error budget (new).** Switch each correction off in turn (`solar_corona`, `electronic_delays`, `rlt_bodies`, media) and tabulate how much it moves the computed observable. It tells the reader which terms matter at which level. At a Sun–Earth–probe angle of ~138° the corona is negligible; the delays are hundreds of meters.
5. **Residuals against the flight reconstruction.** `plot_residuals`, one row per data type, colored by station. Doppler agrees to 1.2–2.7 mHz RMS. Ranging shows a constant offset per station: **−9 m one-way at DSS-35, +52 m at DSS-65 on both of its passes** (see Open checks).
6. **Editing.** `plot_editing` previews the Huber/MAD filter, then `MeasurementSpec.edit("huber_mad_filter", ...)` runs it: gross outliers at the start of the passes, placeholder ranging rows. *(Added 2026-11-07)* One sentence on the second editing path, `FilterSettings(editing_method=...)` (`chi2`, `lasso`, `kurtosis`, date ranges), which edits inside `fit`; the *Measurement editing* catalog says which to use when.
7. **Data weights.** The refit σ measure the noise (Doppler ~1 mHz, ranging ~0.5 m); the filter is given operational weights (1 m one-way, 5.6 mHz ≈ 0.1 mm/s floor) that also cover what the models miss. Why weighting at the noise level makes a filter overconfident. *(Added 2026-11-07)* Where the weights can change: `measurement_sigmas` in the settings, or `set_measurement_sigmas` in a `fit(callback=...)` between iterations.

*Produces:* the edited, weighted measurements (format: Open checks).

### 7. Real data II: a multi-pass fit

*Goal: fit the June 7 and June 8 morning passes (two stations, about 16 hours apart) with the parameters real data need, and read the solution against the reconstruction.*

1. **The a priori.** Either the simulated week's prediction mapped to June 7 (chapter 4: it closes the thread), or OD044 plus a known error (the 2.0 notebook). The choice is open (see Open questions).
2. **The estimated state.** Position and velocity; η_SRP; one range bias per station (or per pass, `set_pass_epochs`); piecewise Gauss–Markov accelerations (PFOGM), one batch per pass plus prediction batches. A-priori sigmas and why each value: 2 m range-bias σ, PFOGM σ against the SRP acceleration.
3. **Why the range bias needs a realistic a priori.** Over hours the line of sight barely turns, so a bias and the line-of-sight position are almost the same signal: `plot_prior_posterior` and `plot_correlation` show the ρ ≈ −1 line, `plot_parameter_learning` the bias σ staying at its a priori. Two stations at different longitudes help separate them; the chapter shows by how much. *(Added 2026-11-07)* Show it before the fit too: `filter.observability_test()` returns the information matrix, whose weak direction is the bias–line-of-sight pair.
4. **The fit.** `SRIFB`; `plot_convergence` (the cost test stops it; the deviation test would not on a weakly observable arc).
5. **Results.** `plot_residuals` with `trend=True` (systematic signatures χ²/n misses), `plot_parameter_learning`, and `plot_consistency` against OD044 in RTN: meters along the line of sight, kilometers across it, each inside its own covariance, and why OD044 is a reference, not the truth.

*Produces:* `solution.json`.

### 8. Real data III: is the solution any good?

*Goal: test the solution the way navigation does, by what it predicts, and tune it with that test.*

1. **Forward prediction against OD044.** `plot_consistency` over three days past the data: the error inside its growing 3σ envelope, the Mahalanobis χ². *(Added 2026-11-07)* Use the causal products from chapter 4 (`lookback_s`), so the prediction uses only data available at the time.
2. **The held-out pass.** Predict the June 8 evening DSS-65 pass, compute its pre-fit residuals against the prediction, and compare them with the predicted measurement uncertainty (h P hᵀ + bias + weight). This is the series' scoreboard, now against real data.
3. **Tuning by prediction.** Sweep the PFOGM σ: the post-fit residuals barely change, while the prediction χ² and the held-out residuals do. Too little noise gives an optimistic covariance; too much wastes margin and absorbs the SRP signal.
4. **What a navigation team would do next:** add the held-out pass and refit, and the hand-off to the how-tos (multi-arc OD, the July passes, Monte Carlo covariance realism).

## Library prerequisites for the real-data chapters

Each is a notebook workaround today (rule 3):

| Needed | Today | Chapters |
|---|---|---|
| Attach media corrections for a pass's date in one call | a 15-line `add_media_corrections` helper (2.0 notebook). *Probably met (2026-11-07):* `MediaCorrections.for_pass` selects the files for a pass, and `MeasurementSpec.build` calls it when `MeasurementConfig(media_folder=...)` is set; check it on the June passes | 6 |
| Save and reload an edited, weighted measurement set | no file round trip for edits and weights | 6 → 7 |
| Predicted measurement and its σ for a new pass from a solution | hand-written partials and h P hᵀ (2.0 notebook `predicted_sigma`) | 8 |
| Prediction trajectory beyond the arc for a new pass's computed observables | propagate the estimate to an SPK under a third SPICE ID | 8 |
| Errors in RTN against a reference ephemeris | `state_errors` prototype | 7, 8 |
| Quiet measurement models | `%%capture` around every computation | all |
| The new plots in `scb.Plotting` | `plotting_prototype.py` | 1–8 |

## Open checks

- 🟠 **CHECK: per-station ranging offsets.** With the station and transponder delays modelled, the June ranging residuals against OD044 average −64.9 RU (−9.2 m one-way) at DSS-35 and +366 RU (+51.7 m) at DSS-65, the same on both DSS-65 passes; the July DSS-35 passes (2.0 notebook) give −2.4 to −2.6 m. The transponder delay was calibrated on DSS-35, so a DSS-65 station-delay difference is the first suspect. Explain it before chapter 6 builds on it, or present it as the reason for per-station biases.
- 🟠 **CHECK: edited Doppler counts.** The Huber/MAD editor flags 28 and 33 of 189 and 206 Doppler points on the DSS-65 passes (19 of 217 at DSS-35). Look at them with `plot_editing` before deciding the weights.
- 🟠 **CHECK: docs audience (added 2026-11-07).** `docs/.../user_guide/what_is_scb.rst` says "No prior OD experience is assumed for the tutorial series", which contradicts this plan's audience. Change the sentence when the series replaces the old notebooks.
- **Fit feasibility.** The two fitted passes are 11 hours apart; the 2.0 notebook's two were 2 days apart. Probe that η_SRP and two PFOGM batches are not over-parametrized on this arc (if they are: fix η, or one PFOGM batch).

## Open questions

- **A priori for the real data:** the simulated week's prediction (continuity; the simulated truth and OD044 differ by ~0.2 km after a week, inside the ~1.7 km σ) or OD044 plus a known error (simpler, as in the 2.0 notebook)? The first needs the simulated tracking to end before June 7 18:45, or a sentence on the overlap. *(Review 2026-11-07: the first; it closes the story, and the 0.2 km difference is well inside 1σ.)*
- **June or July** for the core real-data chapters? June keeps the story and adds a second station; July is already done in the 2.0 notebook. Proposed: June in the core series, July as a how-to.
- Ship pre-generated chapter outputs so a reader can start at a later chapter without running the earlier ones? (Proposed: yes. Review 2026-11-07: yes; `prepare.py` already regenerates them.)
- Should the Quickstart live in the docs site rather than the notebook series?

## Status

| Chapter | Status |
|---|---|
| Quickstart, 1–9 (current numbering) | drafts, ported to 2.0, execute without errors |
| Merges (proposed 1, 3, 5) | proposed; the v3 prototype of chapter 5 exists |
| Real-data chapters 6–8 | proposed; June loading checked |
| Plot prototypes | `plotting_prototype.py`, `plotting_prototype_demo.ipynb` |
| Fundamentals, catalogs, how-tos | planned |

## Where the old notebooks go

| Old notebook | New home |
|---|---|
| `basics_AWU_and_AWF` | Fundamentals: units; frames |
| `basics_EpochArray` | Fundamentals: time |
| `basics_SpiceManager_CelestialBody_Constants` | Fundamentals: SPICE and bodies; chapter 1 |
| `intermediate_Propagator_and_Trajectory` | chapter 1; Fundamentals: state, trajectories; catalogs: dynamics, integrators |
| `intermediate_Measurements` | chapters 2 and 3; Fundamentals: measurement files; catalog: measurement types |
| `intermediate_nPlateModel_and_Attitude` | how-to: n-plate and attitude |
| `intermediate_MissionSequence_and_FiniteBurn` | how-to: maneuvers |
| `intermediate_BPlane` | how-to: B-plane |
| `advanced_IdealMSR_BatchOD` | chapters 3, 4, 5; catalogs: dynamics, measurement editing |
| `advanced_IdealMSR_SequentialOD` | chapter 5; catalogs: filters, process noise; how-to: multi-arc OD |
| `advanced_RealMSR_MediaCorrections` | chapter 6; how-to: media corrections & ramp tables |
| `advanced_RealMSR_OSIRIS_REx_OD` | chapters 6–8 (methods); how-to: multi-pass cruise OD (the July passes) |
