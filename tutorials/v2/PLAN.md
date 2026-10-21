# Tutorial series v2 — plan

Replaces the current `tutorials/` set (3 basics / 5 intermediate / 4 advanced notebooks written independently), which repeats setup in every notebook, re-implements library functionality in notebook helpers, and mixes reference material with workflow.

Audience: people already familiar with orbit determination. The tutorials teach Scarabaeus, not OD.

## Design rules

1. **One mission, start to finish.** Every core chapter continues the same scenario; the reader never re-learns setup.
2. **Explain once, link elsewhere.** Each concept has one home. Core chapters link to *Fundamentals* and *Model catalogs* pages instead of re-explaining.
3. **No helper functions in notebooks.** If a tutorial needs a helper, that is a missing library feature: report it, and mark any temporary workaround with a `TODO` review note.
4. **Chapters chain through files.** Each chapter writes its products (SPKs, measurement files, solutions) and the next chapter loads them.
5. **Uniform page layout.** Goal and prerequisites at the top, numbered steps, a *Summary* table of the API covered, and a link to the next page.
6. **Every notebook runs.** Notebooks are executed before review; the goal is to run them in CI.

## Storyline: OSIRIS-REx on approach to Bennu, June 1–8, 2018

- The tutorial data already include the flight team's reconstructed OSIRIS-REx trajectory for this period and a real DSN tracking pass (2018 DOY 159), so the final chapter can process real data with the same filter setup.
- **Truth for the simulated chapters** is a trajectory propagated by Scarabaeus from the flight state on June 1. Using the flight reconstruction directly as truth does not work early in the series: the inevitable small dynamics mismatch, combined with very precise measurements, biases the estimate well outside its covariance. That effect is kept for chapter 8 as a deliberate lesson.
- Force model: Sun, planets (Mercury–Saturn), cannonball SRP (A = 11 m², C_r = 1.2, m = 1500 kg). It reproduces the flight trajectory to ~0.2 km over the week.
- SPICE IDs: flight reconstruction `-64`, simulated truth `-64000`, filter reference `-64001`.
- Tracking: range and range-rate from DSS-14, DSS-43 and DSS-63, 30-min sampling, 10° elevation mask.

## Structure

```
0. Getting Started
   ├── Install + tutorial data
   └── Quickstart: the whole pipeline on one page

I. Core series — "Navigating OSIRIS-REx" (in order)
   1  Scenario                  SPICE kernels, units, time, frames and bodies, spacecraft, DSN stations
   2  Truth dynamics            propagate the arc, compare force models with the flight reconstruction, save SPK
   3  Tracking data I           simulate range/range-rate, measurement files, noise check
   4  Filter dynamics model     reference spacecraft, a-priori state and covariance, reference trajectory and STM
   5  Filter measurement model  pre-fit residuals, measurement partials, linearized measurement equation, MeasurementSpec
   6  First OD solution         batch filter, SolutionOD, errors vs. 3σ
   7  Sequential filtering      SRIF/LKF, process noise (SNC/DMC), smoothing
   8  Estimating more           SRP scale, consider parameters, measurement editing, covariance analysis
   9  Tracking data II          real DSN data, light-time models, media corrections

II.  Fundamentals     units, frames, time, SPICE and bodies, state definition, trajectories and SPKs, measurement files
III. Model catalogs   dynamics models, integrators, measurement types, filters, process noise, measurement editing, filter outputs
IV.  How-tos          maneuvers, B-plane, media corrections, n-plate and attitude, optical navigation, multi-arc OD, consistency checks
```

| Chapter | Status |
|---|---|
| Quickstart, 1–6 | draft, in review |
| 7 | draft |
| 8–9 | planned |
| Fundamentals, catalogs, how-tos | planned |

## Where the current notebooks go

| Current notebook | New home |
|---|---|
| `basics_AWU_and_AWF` | Fundamentals: units; frames |
| `basics_EpochArray` | Fundamentals: time |
| `basics_SpiceManager_CelestialBody_Constants` | Fundamentals: SPICE and bodies; chapter 1 |
| `intermediate_Propagator_and_Trajectory` | chapter 2; Fundamentals: state, trajectories; catalogs: dynamics, integrators |
| `intermediate_Measurements` | chapters 3 and 5; Fundamentals: measurement files; catalog: measurement types |
| `intermediate_nPlateModel_and_Attitude` | how-to: n-plate and attitude |
| `intermediate_MissionSequence_and_FiniteBurn` | how-to: maneuvers |
| `intermediate_BPlane` | how-to: B-plane |
| `advanced_IdealMSR_BatchOD` | chapters 4, 6, 8; catalogs: dynamics, measurement editing |
| `advanced_IdealMSR_SequentialOD` | chapter 7; catalogs: filters, process noise; how-to: multi-arc OD |
| `advanced_RealMSR_MediaCorrections` | how-to: media corrections |
| `advanced_RealMSR_OSIRIS_REx_OD` | chapter 9 |

## Open questions

- Should we ship pre-generated chapter outputs so a reader can start at, e.g., chapter 6 without running 1–5?
- Should the Quickstart live in the docs site rather than the notebook series?
