"""Choose settings with the ball-masking validation (possession/validate.py). No ground truth.

    python run_validation.py                      # real data
    python run_validation.py --data-dir synthetic_data

Prints one row per configuration: balanced recovery accuracy on frames whose ball we hid, the
two halves of it, and how many possessions the decoder reports with the ball masked.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession import pipeline, validate  # noqa: E402
from possession.decode import D, runs  # noqa: E402

GRID = [
    ("v1",                              dict(use_shape=False, team_only=False)),
    ("no shape",                        dict(use_shape=False)),
    ("team 0.2 dead 0.0 missing 0.0", dict(team_w=0.2, dead_w=0.0, missing_llr=0.0)),
    ("team 0.2 dead 0.0 missing 0.15", dict(team_w=0.2, dead_w=0.0, missing_llr=0.15)),
    ("team 0.2 dead 0.0 missing 0.3", dict(team_w=0.2, dead_w=0.0, missing_llr=0.3)),
    ("team 0.2 dead 0.05 missing 0.0", dict(team_w=0.2, dead_w=0.05, missing_llr=0.0)),
    ("team 0.2 dead 0.05 missing 0.15", dict(team_w=0.2, dead_w=0.05, missing_llr=0.15)),
    ("team 0.2 dead 0.05 missing 0.3", dict(team_w=0.2, dead_w=0.05, missing_llr=0.3)),
    ("team 0.2 dead 0.1 missing 0.0", dict(team_w=0.2, dead_w=0.1, missing_llr=0.0)),
    ("team 0.2 dead 0.1 missing 0.15", dict(team_w=0.2, dead_w=0.1, missing_llr=0.15)),
    ("team 0.2 dead 0.1 missing 0.3", dict(team_w=0.2, dead_w=0.1, missing_llr=0.3)),
    ("team 0.2 dead 0.2 missing 0.0", dict(team_w=0.2, dead_w=0.2, missing_llr=0.0)),
    ("team 0.2 dead 0.2 missing 0.15", dict(team_w=0.2, dead_w=0.2, missing_llr=0.15)),
    ("team 0.2 dead 0.2 missing 0.3", dict(team_w=0.2, dead_w=0.2, missing_llr=0.3)),
    ("team 0.4 dead 0.0 missing 0.0", dict(team_w=0.4, dead_w=0.0, missing_llr=0.0)),
    ("team 0.4 dead 0.0 missing 0.15", dict(team_w=0.4, dead_w=0.0, missing_llr=0.15)),
    ("team 0.4 dead 0.0 missing 0.3", dict(team_w=0.4, dead_w=0.0, missing_llr=0.3)),
    ("team 0.4 dead 0.05 missing 0.0", dict(team_w=0.4, dead_w=0.05, missing_llr=0.0)),
    ("team 0.4 dead 0.05 missing 0.15", dict(team_w=0.4, dead_w=0.05, missing_llr=0.15)),
    ("team 0.4 dead 0.05 missing 0.3", dict(team_w=0.4, dead_w=0.05, missing_llr=0.3)),
    ("team 0.4 dead 0.1 missing 0.0", dict(team_w=0.4, dead_w=0.1, missing_llr=0.0)),
    ("team 0.4 dead 0.1 missing 0.15", dict(team_w=0.4, dead_w=0.1, missing_llr=0.15)),
    ("team 0.4 dead 0.1 missing 0.3", dict(team_w=0.4, dead_w=0.1, missing_llr=0.3)),
    ("team 0.4 dead 0.2 missing 0.0", dict(team_w=0.4, dead_w=0.2, missing_llr=0.0)),
    ("team 0.4 dead 0.2 missing 0.15", dict(team_w=0.4, dead_w=0.2, missing_llr=0.15)),
    ("team 0.4 dead 0.2 missing 0.3", dict(team_w=0.4, dead_w=0.2, missing_llr=0.3)),
    ("team 0.4 dead 0.1 ungated-dead",  dict(team_w=0.4, dead_w=0.1, dead_gate=False)),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=validate.SEED)
    args = ap.parse_args()

    data = pipeline.load(args.data_dir)
    full = pipeline.ball_stage(data)
    y_team, y_dead = validate.pseudo_targets(data, full)
    shp = pipeline.shape_stage(data, full)            # trained on unmasked pseudo-labels
    mask = validate.make_mask(data, y_team, y_dead, seed=args.seed)
    masked = pipeline.ball_stage(data, ball_mask=mask, say=pipeline.quiet)
    print("\nmask: {} graded frames hidden ({:.1f}%)\n".format(
        int(mask.sum()), 100 * mask.sum() / data["graded"].sum()))

    print("{:<34} {:>8} {:>8} {:>8} {:>6}".format("config", "balanced", "team", "dead", "poss"))
    for name, over in GRID:
        opts = dict(pipeline.DEFAULTS, snap=False)
        opts.update(over)
        _, codes = pipeline.decode_all(data, masked, shp if opts["use_shape"] else None, opts,
                                       say=pipeline.quiet)
        ta, da, bal, nt, nd = validate.score(codes, mask, y_team, y_dead)
        poss = sum(1 for _, s, e in data["blocks"] for r in runs(codes[s:e + 1]) if r[2] != D)
        print("{:<34} {:>8.4f} {:>8.4f} {:>8.4f} {:>6}".format(name, bal, ta, da, poss))
    print("\n(scored on {} masked team frames and {} masked dead frames)".format(nt, nd))
