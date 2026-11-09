"""Models trained on real labels from other matches (external.py), applied to the competition game.

Two gradient-boosted models, each answering one of the decoder's two questions:

  dead model   P(ball out of play | players)        ball-free on purpose
  team model   P(home has it | players, ball, alive)  ball features hidden at random in training

Why the dead model never sees the ball: in the sample games the ball is not tracked while play
is stopped (seen on ~7-21% of dead frames), whereas in the competition match it is (seen on
~65%). A model allowed to look at the ball would learn "no ball = stoppage", which is an artefact
of the training data and would not transfer. So it learns stoppages the way item 3 asks, from
positive evidence in the players alone: the restart formation (a player on the touchline for a
throw-in, a crowded box and a player at the corner flag for a corner, a kick-off shape) and the
whole team slowing down and staying slow.

The team model does see the ball, but its ball features are hidden on random 1-8 s blocks
covering ~40% of training frames, so it learns to fall back on shape when the ball is gone.
"""

import os
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from . import shape
from .features import rolling_mean

SEED = 0
MASK_SHARE = 0.4
MAX_TRAIN = 150000


def _model(seed=SEED):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=seed)


def _shift(x, k):
    """x[t + k] (NaN outside)."""
    out = np.full(len(x), np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _rolling_min(x, w):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate((np.full(w // 2, np.nan), x, np.full(w - 1 - w // 2, np.nan)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(sliding_window_view(pad, w), axis=1)


def formation(pm, psp, pitch, blocks, n):
    """Restart-formation and crowd-dynamics features, ball-free. dict name -> float[N]."""
    L, W = pitch
    x, y = pm[..., 0], pm[..., 1]
    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # throw-in: someone standing on / beyond a touchline
        touch = np.minimum(y, W - y)
        cols["min_touchline_m"] = np.nanmin(touch, axis=1)
        cols["n_beyond_lines"] = np.nansum((touch < -0.5) | (x < -0.5) | (x > L + 0.5), axis=1)
        # corner: someone at a corner flag, and a crowded box
        corners = np.stack([np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)], axis=0)
        cols["min_corner_m"] = np.nanmin(np.nanmin(corners, axis=0), axis=1)
        box_y = (y > W / 2 - 20.16) & (y < W / 2 + 20.16)
        left = np.nansum((x < 16.5) & box_y, axis=1)
        right = np.nansum((x > L - 16.5) & box_y, axis=1)
        cols["box_max"] = np.maximum(left, right)
        cols["box_min"] = np.minimum(left, right)
        six_y = (y > W / 2 - 9.16) & (y < W / 2 + 9.16)
        cols["six_max"] = np.maximum(np.nansum((x < 5.5) & six_y, axis=1),
                                     np.nansum((x > L - 5.5) & six_y, axis=1))
        # goal kick: nobody near the keeper, players spread out
        cols["min_goal_line_m"] = np.nanmin(np.minimum(x, L - x), axis=1)
        # kick-off: everyone near the halfway line / in the centre circle
        cols["centre_circle_n"] = np.nansum(np.hypot(x - L / 2, y - W / 2) < 9.15, axis=1)
        cols["mean_abs_from_halfway"] = np.nanmean(np.abs(x - L / 2), axis=1)
        # crowd dynamics: slow and staying slow, then a sudden pick-up at the restart
        med = np.nanmedian(psp, axis=1)
        cols["crowd_now"] = rolling_mean(med, C.FPS)
        cols["crowd_p90"] = rolling_mean(np.nanquantile(psp, 0.9, axis=1), C.FPS)
        for s in (3, 6, 12):
            cols["crowd_avg{}s".format(s)] = rolling_mean(med, s * C.FPS)
        past = rolling_mean(med, 3 * C.FPS)
        cols["crowd_past3s"] = _shift(past, -int(1.5 * C.FPS))
        cols["crowd_next3s"] = _shift(past, int(1.5 * C.FPS))
        cols["crowd_min6s"] = _rolling_min(cols["crowd_now"], 6 * C.FPS)
        cols["n_still"] = np.nansum(psp < 1.0, axis=1)
    out = {}
    inside = np.zeros(n, bool)
    for _, s, e in blocks:
        inside[s:e + 1] = True
    for k, v in cols.items():
        out[k] = np.where(inside, v, np.nan).astype(np.float32)
    return out


def ball_features(feat):
    d = np.minimum(feat["dist"], 30.0)
    return {"ball_known": feat["ball_known"].astype(np.float32),
            "team_llr": feat["team_llr"].astype(np.float32),
            "dist_h": np.where(np.isfinite(d[:, 0]), d[:, 0], np.nan).astype(np.float32),
            "dist_a": np.where(np.isfinite(d[:, 1]), d[:, 1], np.nan).astype(np.float32),
            "ball_speed": np.minimum(feat["ball_speed"], 40.0).astype(np.float32)}


def random_blocks(n, share, seed, lo_s=1, hi_s=8):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, bool)
    while mask.mean() < share:
        c = int(rng.integers(0, n))
        h = int(rng.integers(lo_s * C.FPS, hi_s * C.FPS + 1)) // 2
        mask[max(0, c - h):c + h] = True
    return mask


def features_for(data, feat, cache=None):
    """(X_dead, names_dead, X_shape, names_shape) for one game; cached by path if given."""
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        return z["Xd"], list(z["nd"]), z["Xs"], list(z["ns"])
    Xs, ns, _ = shape.build(data["pm"], data["psp"], data["slot_team"], data["blocks"],
                            data["pitch"], data["n"])
    fm = formation(data["pm"], data["psp"], data["pitch"], data["blocks"], data["n"])
    nd = sorted(fm)
    Xd = np.concatenate([np.stack([fm[k] for k in nd], axis=1), Xs], axis=1)
    nd = nd + list(ns)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, Xd=Xd, nd=np.array(nd), Xs=Xs, ns=np.array(ns))
    return Xd, nd, Xs, list(ns)


def team_matrix(Xs, ns, feat_masked):
    bf = ball_features(feat_masked)
    nb = sorted(bf)
    return np.concatenate([Xs, np.stack([bf[k] for k in nb], axis=1)], axis=1), list(ns) + nb


def _subsample(idx, k, rng):
    return idx if len(idx) <= k else np.sort(rng.choice(idx, size=k, replace=False))


def _weights(y):
    w = np.ones(len(y))
    for c in np.unique(y):
        w[y == c] = len(y) / (2.0 * (y == c).sum())
    return w


def fit(rows_dead, y_dead, rows_team, y_team, seed=SEED):
    rng = np.random.default_rng(seed)
    i = _subsample(np.arange(len(y_dead)), MAX_TRAIN, rng)
    md = _model(seed).fit(rows_dead[i], y_dead[i], sample_weight=_weights(y_dead[i]))
    j = _subsample(np.arange(len(y_team)), MAX_TRAIN, rng)
    mt = _model(seed).fit(rows_team[j], y_team[j], sample_weight=_weights(y_team[j]))
    return md, mt


def logit(p, cap=4.0):
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
    return np.clip(np.log(p / (1 - p)), -cap, cap)
