"""Per-frame features, all in metres and seconds.

Collective by construction: the tracking has no identities (46 slots for 22 players, with swaps),
so nothing here follows a player. It asks, per frame, how close each *team* is to the ball, how
fast the ball and the crowd are moving, and whether the ball is on the pitch at all.
"""

import numpy as np

from . import config as C


def _runs(mask):
    """(start, end_exclusive) of each run of True."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def rolling_mean(x, window, valid=None):
    """Centred mean over `window` frames, ignoring NaN (or frames where valid is False)."""
    if valid is None:
        valid = np.isfinite(x)
    v = np.where(valid, x, 0.0)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    cn = np.concatenate(([0], np.cumsum(valid)))
    n = len(x)
    h = window // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    cnt = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)


def clean_ball(ball_m, pitch):
    """Drop implausible detections, bridge short occlusions. Returns (xy, weight, observed)."""
    L, W = pitch
    xy = ball_m.astype(np.float64).copy()
    ok = np.isfinite(xy).all(axis=1)
    m = C.BALL_PLAUSIBLE_MARGIN
    ok &= (xy[:, 0] > -m) & (xy[:, 0] < L + m) & (xy[:, 1] > -m) & (xy[:, 1] < W + m)

    # Teleports: a detection that implies a ball faster than physics allows, relative to the
    # previous kept detection, is a different object. One forward pass.
    idx = np.flatnonzero(ok)
    keep = np.ones(len(idx), dtype=bool)
    last = None
    for k, i in enumerate(idx):
        if last is not None:
            dt = (i - last) / float(C.FPS)
            if np.hypot(*(xy[i] - xy[last])) > C.MAX_BALL_SPEED * dt + 2.0:
                keep[k] = False
                continue
        last = i
    ok[idx[~keep]] = False
    xy[~ok] = np.nan

    observed = ok.copy()
    weight = observed.astype(np.float64)
    starts, ends = _runs(~ok)
    n = len(xy)
    for s, e in zip(starts, ends):
        if s == 0 or e == n or e - s > C.MAX_INTERP_FRAMES:
            continue
        t = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
        xy[s:e] = xy[s - 1] + t[:, None] * (xy[e] - xy[s - 1])
        weight[s:e] = C.INTERP_WEIGHT
    return xy, weight, observed


def ball_speed(xy):
    """m/s, central difference over +/-2 frames, NaN where the ball is unknown."""
    sp = np.full(len(xy), np.nan)
    d = xy[4:] - xy[:-4]
    sp[2:-2] = np.hypot(d[:, 0], d[:, 1]) * C.FPS / 4.0
    return sp


def player_speed(players_m, speed_field):
    """Per slot speed: the ATD's own value where present, else from positions."""
    sp = speed_field.astype(np.float64)
    pos = players_m
    diff = np.full(sp.shape, np.nan)
    d = pos[2:] - pos[:-2]
    diff[1:-1] = np.hypot(d[..., 0], d[..., 1]) * C.FPS / 2.0
    sp = np.where(np.isfinite(sp), sp, diff)
    return np.where(sp > 12.0, np.nan, sp)  # no footballer runs 12 m/s; it is a track jump


def team_distances(players_m, slot_team, ball_xy):
    """Nearest player of each team to the ball, metres (inf when unknown)."""
    d = np.hypot(players_m[..., 0] - ball_xy[:, None, 0], players_m[..., 1] - ball_xy[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    out = np.empty((len(d), 2))
    for team in (0, 1):
        cols = slot_team == team
        out[:, team] = d[:, cols].min(axis=1) if cols.any() else np.inf
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def team_llr(dist, ball_weight, bspeed):
    """Log-odds home vs away from proximity, already weighted by how much it can be believed.

    Positive = home. Zero when nobody is near the ball (a loose ball keeps its owner -- the
    decoder carries the previous state through), when the ball is unknown, or when it is in
    flight.
    """
    dh, da = dist[:, 0], dist[:, 1]
    nearest = np.minimum(dh, da)
    with np.errstate(invalid="ignore"):
        both = np.where(np.isfinite(dh) & np.isfinite(da), da - dh, 0.0)
    diff = np.where(np.isfinite(dh) & np.isfinite(da), both,
                    np.where(np.isfinite(dh), C.PROXIMITY_CAP, -C.PROXIMITY_CAP))
    raw = np.clip(C.PROXIMITY_SLOPE * diff, -C.PROXIMITY_CAP, C.PROXIMITY_CAP)
    close = sigmoid((C.CONTROL_RADIUS - nearest) / C.CONTROL_SOFTNESS)
    sp = np.nan_to_num(bspeed, nan=0.0)
    flight = C.FLIGHT_FLOOR + (1 - C.FLIGHT_FLOOR) * sigmoid((C.FLIGHT_SPEED - sp)
                                                             / C.FLIGHT_SOFTNESS)
    out = raw * close * flight * ball_weight
    return np.where(np.isfinite(nearest), out, 0.0)


def out_of_bounds(ball_xy, observed, pitch):
    L, W = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    m = C.OOB_MARGIN
    return observed & ((x < -m) | (x > L + m) | (y < -m) | (y > W + m))


def crowd_speed(pspeed):
    """Median tracked-player speed per frame, smoothed over CROWD_WINDOW_S."""
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(pspeed, axis=1)
    return rolling_mean(med, int(C.CROWD_WINDOW_S * C.FPS))


def players_in_metres(players, pitch):
    """(positions float64[N, S, 2] in metres, speeds float64[N, S] in m/s)."""
    pm = players[..., :2].astype(np.float64) * np.array(pitch)
    return pm, player_speed(pm, players[..., 2])


def build(pm, psp, ball, slot_team, pitch, ball_mask=None):
    """All ball-dependent per-frame features. ball_mask hides the ball on those frames
    (used by the occlusion validation to simulate the tracker losing it)."""
    bm = ball.astype(np.float64) * np.array(pitch)
    if ball_mask is not None:
        bm[ball_mask] = np.nan
    bxy, bw, observed = clean_ball(bm, pitch)
    bsp = ball_speed(bxy)
    dist = team_distances(pm, slot_team, bxy)
    known = np.isfinite(bxy).all(axis=1)
    return {
        "ball_xy": bxy,
        "ball_known": known,
        "ball_observed": observed,
        "ball_weight": np.where(known, bw, 0.0),
        "ball_speed": bsp,
        "dist": dist,
        "team_llr": team_llr(dist, np.where(known, bw, 0.0), bsp),
        "oob": out_of_bounds(bxy, observed, pitch),
        "crowd_speed": crowd_speed(psp),
        "n_tracked": np.isfinite(pm[..., 0]).sum(axis=1),
    }
