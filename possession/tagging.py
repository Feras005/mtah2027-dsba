"""Smart Tagging as evidence: set pieces are restart anchors, phases are a weak team prior.

The tagging is automatic and noisy, and its codes are phases, not possessions. So nothing here
copies an interval into a label. Set pieces only make a restart cheap for the decoder near where
they say one happened. Phases add a small per-frame nudge towards a team -- and how big, and in
which direction ("DEFENDING ..." tagged Barcelona plausibly means Athletic has the ball), is not
assumed: it is measured, per code, against the proximity evidence on frames where one team is
clearly on the ball. That calibration uses no labels, only the tracking.
"""

import numpy as np

from . import config as C
from .io import tag_side

SET_PIECE_CODE = "SET PIECES"
IGNORED_PHASE_CODES = {SET_PIECE_CODE, "GOALS"}
KICKOFF_START = "kick off (start)"


def set_pieces(events, n_frames):
    """[(start, end, side or None, type)] for every tagged set piece, clipped to the video."""
    out = []
    for e in events:
        if e["code"] != SET_PIECE_CODE:
            continue
        s, t = max(0, e["start"]), min(n_frames, e["end"])
        if t > s:
            out.append((s, t, tag_side(e["tags"].get("Team")), (e["tags"].get("Type") or "")))
    return out


def kickoff_side(events, frame, tolerance=10 * C.FPS):
    """Side tagged as kicking off near `frame`, or None."""
    for s, t, side, kind in set_pieces(events, 10 ** 9):
        if kind.strip().lower() == KICKOFF_START and s - tolerance <= frame <= t + tolerance:
            return side
    return None


def restart_frame(s, t, crowd):
    """Estimated restart inside a tagged set-piece window [s, t).

    Measured on this file's tags (aggregates only): kickoffs are tagged as ~5 s windows around
    the restart, every other set piece as a fixed 35 s window in which the players speed up
    only ~20-30% of the way in. So the restart is not the middle of the window. It is estimated
    per instance as the frame of the largest rise in collective speed (mean over the next 3 s
    minus mean over the previous 3 s) within the first half of the window.
    """
    if t - s <= 8 * C.FPS:
        return (s + t) // 2
    w = 3 * C.FPS
    c = np.nan_to_num(crowd[max(0, s - w):t + w], nan=0.0)
    off = s - max(0, s - w)
    cs = np.concatenate(([0.0], np.cumsum(c)))
    best, best_k = -np.inf, (s + t) // 2
    for k in range(off, off + (t - s) // 2):
        if k - w < 0 or k + w > len(c):
            continue
        rise = (cs[k + w] - cs[k]) - (cs[k] - cs[k - w])
        if rise > best:
            best, best_k = rise, s + (k - off)
    return best_k


def set_piece_arrays(events, n_frames, crowd):
    """(dead_llr, restart_anchor, side_llr, restarts) arrays from the set pieces.

    dead_llr      + from the window start to the estimated restart, - for 3 s after it
    restart_mask  True within 3 s of the estimated restart: leaving D is cheap there
    side_llr      a nudge towards the tagged restarting side for 2 s after the restart
    restarts      [(window_start, restart_frame, side)]
    """
    dead = np.zeros(n_frames)
    anchor = np.zeros(n_frames, dtype=bool)
    side_llr = np.zeros(n_frames)
    restarts = []
    a = 3 * C.FPS
    for s, t, side, _ in set_pieces(events, n_frames):
        r = restart_frame(s, t, crowd)
        restarts.append((s, r, side))
        dead[s:r] += C.SET_PIECE_LLR
        dead[r:min(t, r + a)] -= C.SET_PIECE_LLR
        anchor[max(s, r - a):min(t, r + a)] = True
        if side is not None:
            side_llr[r:min(t, r + 2 * C.FPS)] += 1.0 if side == "H" else -1.0
    return dead, anchor, side_llr, restarts


def phase_prior(events, team_llr, graded, n_frames, report=None):
    """Per-frame home-vs-away log-odds from tagged phases, self-calibrated per code."""
    confident = graded & (np.abs(team_llr) > 1.0)
    sign = np.sign(team_llr)
    by_code = {}
    for e in events:
        side = tag_side(e["tags"].get("Team"))
        if e["code"] in IGNORED_PHASE_CODES or side is None:
            continue
        s, t = max(0, e["start"]), min(n_frames, e["end"])
        if t > s:
            by_code.setdefault(e["code"], []).append((s, t, 1.0 if side == "H" else -1.0))

    prior = np.zeros(n_frames)
    for code, spans in sorted(by_code.items()):
        agree = total = 0
        for s, t, sd in spans:
            m = confident[s:t]
            total += int(m.sum())
            agree += int((sign[s:t][m] == sd).sum())
        if total < C.PHASE_MIN_FRAMES:
            weight = 0.0
        else:
            p = (agree + 1.0) / (total + 2.0)
            weight = float(np.clip(0.5 * np.log(p / (1 - p)), -C.PHASE_MAX_LLR, C.PHASE_MAX_LLR))
        if report is not None:
            report.append((code, len(spans), total, agree / total if total else float("nan"),
                           weight))
        for s, t, sd in spans:
            prior[s:t] += weight * sd
    return prior
