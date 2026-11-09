"""Touch detection and boundary snapping.

spellF1 matches possessions by temporal IoU up to 0.7, where a few frames of offset already cost.
A possession changes hands at a touch, and a touch is visible in the ball track as a sudden
change of the ball's velocity next to a player. So after decoding, each boundary is moved to the
touch that explains it, if one is close enough:

  team -> other team   the first touch by the new team after the old team's last touch
  dead -> team         the first touch by the restarting team (the restart itself)
  team -> dead         the first frame the ball is seen out of bounds

A boundary with no such evidence within the window stays where Viterbi put it.
"""

import numpy as np

from . import config as C
from .decode import D, runs


def detect(feat):
    """(frames int[], teams int8[]) of detected touches."""
    xy = feat["ball_xy"]
    obs = feat["ball_observed"]
    n = len(xy)
    v = np.full((n, 2), np.nan)
    ok = obs[2:] & obs[:-2]
    v[1:-1][ok] = (xy[2:][ok] - xy[:-2][ok]) * C.FPS / 2.0
    k = C.TOUCH_LAG
    dv = np.full(n, np.nan)
    dv[k:-k] = np.hypot(*(v[2 * k:] - v[:-2 * k]).T)
    dv = np.nan_to_num(dv, nan=0.0)
    near = np.minimum(feat["dist"][:, 0], feat["dist"][:, 1]) <= C.TOUCH_RADIUS
    cand = (dv >= C.TOUCH_DV) & near
    # Local maxima only: one touch, one frame.
    w = C.TOUCH_NMS
    pad = np.concatenate((np.zeros(w), dv, np.zeros(w)))
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1).max(axis=1)
    cand &= dv >= win
    frames = np.flatnonzero(cand)
    teams = np.where(feat["dist"][frames, 0] <= feat["dist"][frames, 1], 0, 1).astype(np.int8)
    return frames, teams


def snap(codes, feat, touches, lo_limit, hi_limit):
    """Move run boundaries of `codes` (one half) to touch / out-of-bounds evidence."""
    frames, teams = touches
    out = codes.copy()
    W = C.SNAP_WINDOW
    rs = runs(out)
    oob = np.flatnonzero(feat["oob"][lo_limit:hi_limit])
    moved = 0
    for k in range(1, len(rs)):
        (ps, pe, pc), (ns, ne, nc) = rs[k - 1], rs[k]
        b = ns
        lo = max(ps + 1, b - W)
        hi = min(ne - 1, b + W)
        if hi <= lo:
            continue
        new = None
        g_lo, g_hi = lo + lo_limit, hi + lo_limit
        sel = (frames >= g_lo) & (frames <= g_hi)
        tf, tt = frames[sel] - lo_limit, teams[sel]
        if nc != D and pc != D:
            last_old = tf[tt == pc]
            first_new = tf[(tt == nc) & (tf > (last_old.max() if len(last_old) else -1))]
            if len(first_new):
                new = int(first_new[0])
        elif pc == D and nc != D:
            first_new = tf[tt == nc]
            if len(first_new):
                new = int(first_new[0])
        elif nc == D:
            o = oob[(oob >= lo) & (oob <= hi)]
            if len(o):
                new = int(o[0])
        if new is not None and new != b:
            if new < b:
                out[new:b] = nc
            else:
                out[b:new] = pc
            moved += 1
    return out, moved
