"""Ground-truth-free spellF1 on the competition game: does the pipeline give the same possessions
when the ball is hidden?

    python run_consistency.py

Decode the match with the ball as tracked, decode it again with the ball hidden on random 1-8 s
blocks covering 40% of frames, and score the second output against the first with the official
scorer. Unlike the frame-level ball-masking test (validate.py) this scores whole possessions, so
it catches a possession split in two by a false stoppage.

What it cannot do on its own is choose the decoder settings: a decoder that never changes its
mind agrees with itself perfectly. So the settings are chosen on real labels from other matches
(run_external.py), and this test checks that the chosen settings are stable on this match.
Only scores are printed.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import learned, pipeline, transfer  # noqa: E402
from possession import features as F  # noqa: E402

SEEDS = (101, 102, 103)


def main(configs):
    t0 = time.time()
    data = pipeline.load(say=pipeline.quiet)
    feat = pipeline.ball_stage(data, say=pipeline.quiet)
    shp = pipeline.shape_stage(data, feat, say=pipeline.quiet, with_dead=False)
    cache = os.path.join(pipeline.ROOT, "data", "external", ".cache")
    models = transfer.train(cache, pipeline.quiet) if transfer.available() else None
    tac_cache = os.path.join(pipeline.ROOT, "data", "tactical", ".cache", "learned_X.npz")
    masked = []
    for seed in SEEDS:
        mask = learned.random_blocks(data["n"], 0.4, seed) & data["graded"]
        fm = F.build(data["pm"], data["psp"], data["ball"], data["slot_team"], data["pitch"],
                     mask)
        masked.append(fm)
    for name, opts in configs:
        o = dict(pipeline.DEFAULTS)
        o.update(opts)
        s = dict(shp)
        if models is not None:
            s.update(transfer.predict(models, data, feat, tac_cache))
        ref, _ = pipeline.decode_all(data, feat, s, o, say=pipeline.quiet)
        gt = pm.label_codes(ref)
        res = []
        for fm in masked:
            sm = dict(shp)
            if models is not None:
                sm.update(transfer.predict(models, data, fm, tac_cache))
            lab, _ = pipeline.decode_all(data, fm, sm, o, say=pipeline.quiet)
            r = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)
            res.append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
        a = np.array(res)
        print("{:<40} self-consistency spellF1 {:.3f} (seeds {})  macroF1 {:.3f}  "
              "possessions {:.0f} hidden / {:.0f} visible".format(
                  name, a[:, 0].mean(), " ".join("{:.3f}".format(x) for x in a[:, 0]),
                  a[:, 1].mean(), a[:, 2].mean(), a[0, 3]), flush=True)
    print("({:.0f} s)".format(time.time() - t0))


if __name__ == "__main__":
    v2 = {"use_external": False, "snap": False, "ext_team_w": 0.0, "ext_dead_w": 0.0}
    main([("v2 (submitted)", v2),
          ("v3: + external models", {"snap": False}),
          ("v3: + external models + snapping", {})])
