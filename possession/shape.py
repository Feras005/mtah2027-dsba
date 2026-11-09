"""Ball-free team-shape features: what the 20-odd players' geometry says about who has the ball.

A team in possession stretches the pitch -- wide, long, pushed up, moving towards goal, with
space around its players. The team without the ball compacts and presses. None of that needs
the ball, which is missing on ~41% of graded frames and disproportionately at the moments that
decide possession. These features are the input of the shape model (shape_model.py).

Attacking direction is found per half from the data, not hard-coded: each team's deepest player
(its goalkeeper, most of the time) sits near the goal it defends.
"""

import warnings

import numpy as np

from . import config as C
from .features import rolling_mean

CHUNK = 20000


def attack_directions(pm, slot_team, blocks, pitch):
    """{(half, team): +1 if the team attacks towards x = L, else -1}."""
    L = pitch[0]
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for half, s, e in blocks:
            x = pm[s:e + 1, :, 0]
            for team in (0, 1):
                xt = x[:, slot_team == team]
                left = np.nanmedian(np.nanmin(xt, axis=1))
                right = np.nanmedian(L - np.nanmax(xt, axis=1))
                out[(half, team)] = 1 if left < right else -1
    return out


def _team_stats(xo, y, W):
    """Per-frame statistics of one team, xo already oriented towards its attacking goal."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.isfinite(xo).sum(axis=1).astype(float)
        cx = np.nanmean(xo, axis=1)
        q10, q90 = np.nanquantile(xo, [0.1, 0.9], axis=1)
        y10, y90 = np.nanquantile(y, [0.1, 0.9], axis=1)
        srt = np.sort(np.where(np.isfinite(xo), xo, np.inf), axis=1)
        deepest_outfield = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)
        return {
            "n": n,
            "cx": cx,
            "cy_abs": np.nanmean(np.abs(y - W / 2.0), axis=1),
            "sx": np.nanstd(xo, axis=1),
            "sy": np.nanstd(y, axis=1),
            "length": q90 - q10,
            "width": y90 - y10,
            "line": deepest_outfield,
            "top": np.nanmax(xo, axis=1),
        }


def _free_space(pa, pb):
    """Mean over team-a players of the distance to the nearest team-b player, chunked."""
    out = np.full(len(pa), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(0, len(pa), CHUNK):
            a, b = pa[s:s + CHUNK], pb[s:s + CHUNK]
            d = np.hypot(a[:, :, None, 0] - b[:, None, :, 0], a[:, :, None, 1] - b[:, None, :, 1])
            out[s:s + CHUNK] = np.nanmean(np.nanmin(np.where(np.isfinite(d), d, np.nan), axis=2),
                                          axis=1)
    return out


def build(pm, pspeed, slot_team, blocks, pitch, n_frames):
    """(X float32[N, F], names). Rows outside the halves are NaN. Home-minus-away + both teams."""
    L, W = pitch
    dirs = attack_directions(pm, slot_team, blocks, pitch)
    cols = {}
    for half, s, e in blocks:
        sl = slice(s, e + 1)
        per = {}
        for team in (0, 1):
            m = slot_team == team
            x = pm[sl][:, m, 0]
            xo = x if dirs[(half, team)] == 1 else L - x
            st = _team_stats(xo, pm[sl][:, m, 1], W)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                st["speed"] = np.nanmean(pspeed[sl][:, m], axis=1)
                # In the opponent's half, oriented.
                st["in_opp_half"] = np.nansum(xo > L / 2.0, axis=1).astype(float)
            # Collective push towards goal: centroid velocity, oriented, over 1 s.
            cx = st["cx"]
            v = np.full(len(cx), np.nan)
            v[C.FPS:] = (cx[C.FPS:] - cx[:-C.FPS])
            st["push"] = v
            per[team] = st
        hm, am = slot_team == 0, slot_team == 1
        per[0]["free"] = _free_space(pm[sl][:, hm], pm[sl][:, am])
        per[1]["free"] = _free_space(pm[sl][:, am], pm[sl][:, hm])
        for key in per[0]:
            for name, val in (("H_" + key, per[0][key]), ("A_" + key, per[1][key]),
                              ("diff_" + key, per[0][key] - per[1][key])):
                cols.setdefault(name, np.full(n_frames, np.nan))[sl] = val

    # Temporal context: 1 s average and 2 s trend of the home-minus-away signals.
    for key in ("cx", "length", "width", "free", "push", "line", "speed"):
        base = cols["diff_" + key]
        cols["diff_{}_avg1s".format(key)] = rolling_mean(base, C.FPS)
        trend = np.full(n_frames, np.nan)
        trend[2 * C.FPS:] = base[2 * C.FPS:] - base[:-2 * C.FPS]
        cols["diff_{}_trend2s".format(key)] = trend
    # The pitch is not symmetric in time: crowd speed and overall spread help the dead model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cols["all_speed"] = np.nanmedian(pspeed, axis=1)
    cols["all_speed_avg2s"] = rolling_mean(cols["all_speed"], 2 * C.FPS)
    names = sorted(cols)
    X = np.stack([cols[k] for k in names], axis=1).astype(np.float32)
    return X, names, dirs
