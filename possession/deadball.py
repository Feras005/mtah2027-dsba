"""Dead-ball evidence: one log-likelihood ratio per frame, positive = ball out of play.

Four independent signals, added as log-odds:

  ball out of bounds        the ball is seen outside the lines                       strong
  collective player speed   stoppages are walked; live play is run                    medium
  ball moving on the pitch  a rolling ball inside the lines is almost always live     weak
  ball not seen at all      off camera, in a hand, under a wall                        weak
  set-piece tags            before / after the middle of a tagged restart              medium

The crowd-speed threshold is not a constant. A 2-component Gaussian mixture is fitted to the log
crowd speed of the graded frames, without labels, and the LLR is the slow component's likelihood
over the fast one's -- so it adapts to whatever this recording's speeds actually are.
"""

import numpy as np

from . import config as C


def fit_two_gaussians(x, iters=100):
    """EM for a 1-D 2-component GMM. Returns (means, sds, weights), slow component first."""
    x = x[np.isfinite(x)]
    if len(x) < 100:
        raise ValueError("too few frames to fit the crowd-speed mixture")
    mu = np.percentile(x, [25, 75]).astype(float)
    sd = np.array([x.std(), x.std()]) / 2 + 1e-3
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = (-0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w))
        ll -= ll.max(axis=1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(axis=1, keepdims=True)
        nk = r.sum(axis=0) + 1e-9
        mu = (r * x[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / nk) + 1e-3
        w = nk / nk.sum()
    order = np.argsort(mu)
    return mu[order], sd[order], w[order]


def crowd_llr(crowd, graded):
    lx = np.log(np.maximum(crowd, 0.0) + 0.05)
    mu, sd, w = fit_two_gaussians(lx[graded])
    z = [(lx - mu[k]) / sd[k] for k in (0, 1)]
    llr = (-0.5 * z[0] ** 2 - np.log(sd[0])) - (-0.5 * z[1] ** 2 - np.log(sd[1]))
    llr = np.clip(np.nan_to_num(llr, nan=0.0), -C.CROWD_CAP, C.CROWD_CAP) * C.CROWD_WEIGHT
    return llr, {"slow_mps": float(np.exp(mu[0]) - 0.05), "fast_mps": float(np.exp(mu[1]) - 0.05),
                 "slow_share": float(w[0])}


def dead_llr(feat, graded, set_piece_dead):
    crowd, info = crowd_llr(feat["crowd_speed"], graded)
    d = crowd.copy()
    d += np.where(feat["oob"], C.OOB_LLR, 0.0)
    moving = (feat["ball_observed"] & ~feat["oob"]
              & (np.nan_to_num(feat["ball_speed"], nan=0.0) > C.MOVING_SPEED))
    d += np.where(moving, C.MOVING_LLR, 0.0)
    d += np.where(~feat["ball_known"], C.MISSING_LLR, 0.0)
    d += set_piece_dead
    return d, info
