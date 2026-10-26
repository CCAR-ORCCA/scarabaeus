# Scarabaeus tutorials (v3 prototype)

> **Prototype.** A restructured version of the [v2 drafts](../v2/README.md), built to compare the two before choosing one. See [PLAN.md](PLAN.md) for the structure and what the prototype found.

What changes from v2:

- **One question runs through the series:** how well do we know where OSIRIS-REx will be when DSS-35 acquires it on June 8? Every chapter ends with the same scoreboard for that moment.
- **`scenario.py`** holds the mission definition (epochs, spacecraft, measurement models, a-priori), so setup cells import it instead of rebuilding 70–100 lines of earlier chapters.
- **OD comes sooner:** the first fit is chapter 3, not chapter 6; the "inside the filter" material becomes chapter 4, *Validating a model*.
- **One SRP story:** chapter 6 compares four responses to the same SRP error.

## Compare side by side

| v3 prototype | v2 draft(s) it replaces |
|---|---|
| [3. First OD solution](03_first_od_solution.ipynb) | [v2 4](../v2/04_filter_dynamics.ipynb) (a-priori), [v2 6](../v2/06_first_od_solution.ipynb) |
| [6. When the dynamics are wrong](06_wrong_dynamics.ipynb) | [v2 7](../v2/07_sequential_filtering.ipynb) §4–5, [v2 8](../v2/08_estimating_more.ipynb) §1–3 |

The other chapters exist only in v2 for now.

## Running

1. Download the tutorial data once, as described in the [v2 README](../v2/README.md#running).
2. From this folder (`tutorials/v3/`), create the truth trajectory and the tracking data (a stand-in for chapters 1 and 2, about 2 s):
   ```bash
   python prepare.py
   ```
3. Run the notebooks from this folder: the metakernel `orex_approach.tm` and `scenario.py` use paths relative to it.

Each notebook writes its products to `outputs/<notebook>/` (git-ignored), separate from v2's.
