"""Generate the truth trajectory and the simulated tracking data (stand-in for chapters 1-2).

The v3 prototype has chapters 3 and 6 only. Until chapters 1 and 2 are written, this script
produces the files they will produce, with the same code the v2 drafts use (chapter 2 and 3 there):

    outputs/01_scenario_and_truth/truth.bsp     truth trajectory, June 1-9 (SPICE ID -64000)
    outputs/02_tracking_data/*.json             range and range-rate, DSS-14/43/63, June 1-8

Usage (from tutorials/v3/):
    python prepare.py            # skip if the outputs already exist
    python prepare.py --force    # regenerate
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
TRUTH = HERE / "outputs/01_scenario_and_truth/truth.bsp"
TRACKING = HERE / "outputs/02_tracking_data"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenerate even if the outputs exist")
    args = parser.parse_args()

    if TRUTH.exists() and len(list(TRACKING.glob("*.json"))) == 6 and not args.force:
        print("Outputs already exist; use --force to regenerate.")
        return

    import os

    os.chdir(HERE)  # the metakernel paths are relative to this folder
    sys.path.insert(0, str(HERE))
    import numpy as np
    import scarabaeus as scb
    import scenario as sc

    # Chapter 1: propagate the truth from the flight state and save it as an SPK.
    shutil.rmtree(TRUTH.parent, ignore_errors=True)
    TRUTH.parent.mkdir(parents=True)
    state = (
        scb.StateDefinition()
        .position(sc.orex_truth, scb.ArrayWFrame(sc.x0_flight[:3], sc.J2000))
        .velocity(sc.orex_truth, scb.ArrayWFrame(sc.x0_flight[3:], sc.J2000))
    )
    propagator = scb.Propagator(
        primary_body=sc.orex_truth,
        state_vector=scb.StateArray(epoch=sc.truth_epochs[0], origin=sc.SUN, state=state),
        tspan=sc.truth_epochs,
        force_models=scb.ForceModelTranslation(sc.orex_truth, third_bodies=sc.PLANETS, cannonball_SRP=True),
        propagate_STM=False,
        display_progress=False,
    )
    propagator.propagate()
    scb.Trajectory(state_array=propagator.propagated_state_array).write_to_spk(str(TRUTH))
    scb.SpiceManager.load_kernel(str(TRUTH))
    print(f"wrote {TRUTH.relative_to(HERE)}")

    # Chapter 2: simulate the tracking over the data arc (same seed and order as the v2 drafts).
    shutil.rmtree(TRACKING, ignore_errors=True)
    TRACKING.mkdir(parents=True)
    np.random.seed(2018)
    for model in sc.models:
        model.write_observed_measurements(
            target=sc.orex_truth,
            epoch_array=sc.data_epochs,
            noisy=True,
            check_visibility=True,
            elevation_mask=sc.ELEVATION_MASK,
            file_name=model.name.replace(" ", "_"),
            folder_path_override=str(TRACKING),
        )
    print(f"wrote {len(list(TRACKING.glob('*.json')))} files to {TRACKING.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
