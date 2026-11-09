"""Ball-masking validation: how the pipeline is tuned without touching any ground truth.

The hard frames are the ones where the tracker loses the ball. So: take frames where the answer is
known from the data (pseudo-labels -- a player clearly on a visible ball, or the ball visibly out
of play), hide the ball around them ourselves, re-run the whole pipeline, and count how often
it still gives the answer the hidden ball gave.

  * masks are blocks of 2-6 s, centred half on team pseudo-labels and half on dead ones, ~15% of
    graded frames, seeded
  * the ball stage is recomputed with the ball hidden; the shape models are trained cross-half
    on the unmasked pseudo-labels, so the model that predicts a masked frame never saw its half
  * score = balanced accuracy over masked pseudo-labelled frames (team frames: right team;
    dead frames: D), plus the possession count as a sanity check against over-switching

What it measures is recovery under occlusion, not agreement with the official ground truth. It
is what settings are chosen with; the public 5 minutes are only used to check the file is right.
"""

import numpy as np

from . import pseudo, tagging
from .decode import A, D, H
from .pipeline import KICKOFF_2ND_VIDEO_FRAME

SEED = 1


def make_mask(data, y_team, y_dead, share=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    n, g = data["n"], data["graded"]
    mask = np.zeros(n, dtype=bool)
    team_idx = np.flatnonzero(y_team >= 0)
    dead_idx = np.flatnonzero((y_dead == 1) & g)
    # Do not count the pre-kickoff block: it is forced to D regardless of the ball.
    dead_idx = dead_idx[(dead_idx < data["blocks"][1][1]) | (dead_idx >= KICKOFF_2ND_VIDEO_FRAME)]
    target = int(share * g.sum())
    k = 0
    while mask.sum() < target and k < 100000:
        pool = team_idx if k % 2 == 0 else dead_idx
        k += 1
        if not len(pool):
            continue
        c = int(rng.choice(pool))
        half = int(rng.integers(25, 76))   # 1-3 s either side -> 2-6 s blocks
        mask[max(0, c - half):min(n, c + half)] = True
    return mask & g


def score(codes, mask, y_team, y_dead):
    t = mask & (y_team >= 0)
    d = mask & (y_dead == 1)
    want = np.where(y_team == 0, H, A)
    team_acc = float((codes[t] == want[t]).mean()) if t.any() else float("nan")
    dead_acc = float((codes[d] == D).mean()) if d.any() else float("nan")
    return team_acc, dead_acc, 0.5 * (team_acc + dead_acc), int(t.sum()), int(d.sum())


def pseudo_targets(data, feat):
    restarts = tagging.set_piece_arrays(data["events"], data["n"], feat["crowd_speed"])[3]
    y_team = np.where(data["graded"], pseudo.team_labels(feat), -1)
    y_dead = pseudo.dead_labels(feat, restarts, data["graded"], KICKOFF_2ND_VIDEO_FRAME,
                                data["blocks"])
    return y_team, y_dead
