"""3-state HMM (H, A, D) decoded with Viterbi, then the run-level clean-up and abstentions.

Why a state model and not a vote: a possession is an interval. Viterbi places each hand-over
where the accumulated evidence says it happened, and the switch cost turns "a touch", "a
deflection", "a lost aerial duel" into exactly what CHALLENGE.md says they are -- not changes of
possession -- because a brief contrary signal never pays for the switch. A loose ball, with
nobody near it, has zero team evidence, so the state simply carries on: possession stays with
the last team that had it.

Transition costs are per frame. Going dead is cheap next to a ball seen out of bounds, coming
back to life is cheap inside a tagged set piece; everywhere else both need real evidence.
"""

import numpy as np

from . import config as C

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
NEG = -1e18


def emissions(team, dead, team_only=True):
    """[N, 3] log-likelihoods.

    team_only (default): team evidence only says *which* team -- the better-supported side
    scores 0 and the other pays for it -- and never makes "alive" more likely than "dead".
    Whether the ball is in play is the dead-ball detector's job alone. The v1 form normalised
    a neutral frame to 0 instead, which let strong team evidence argue against D.
    """
    t = np.clip(team, -30, 30)
    if team_only:
        top = np.logaddexp(0.0, -np.abs(t))
        lh = top - np.logaddexp(0.0, -t)
        la = top - np.logaddexp(0.0, t)
    else:
        lh = np.log(2.0) - np.logaddexp(0.0, -t)
        la = np.log(2.0) - np.logaddexp(0.0, t)
    return np.stack([lh, la, dead], axis=1)


def viterbi(em, stop_cost, restart_cost, init, switch_cost=None):
    """Best state path. em [N,3]; stop/restart costs [N] (cost of entering D / leaving D at t)."""
    n = len(em)
    back = np.zeros((n, 3), dtype=np.int8)
    sw = C.SWITCH_COST if switch_cost is None else switch_cost
    vh, va, vd = (init[0] + em[0, 0], init[1] + em[0, 1], init[2] + em[0, 2])
    e = em.tolist()
    sc = stop_cost.tolist()
    rc = restart_cost.tolist()
    for i in range(1, n):
        s, r = sc[i], rc[i]
        # into H
        c0, c1, c2 = vh, va - sw, vd - r
        if c0 >= c1 and c0 >= c2:
            nh, bh = c0, 0
        elif c1 >= c2:
            nh, bh = c1, 1
        else:
            nh, bh = c2, 2
        # into A
        c0, c1, c2 = vh - sw, va, vd - r
        if c1 >= c0 and c1 >= c2:
            na, ba = c1, 1
        elif c0 >= c2:
            na, ba = c0, 0
        else:
            na, ba = c2, 2
        # into D
        c0, c1, c2 = vh - s, va - s, vd
        if c2 >= c0 and c2 >= c1:
            nd, bd = c2, 2
        elif c0 >= c1:
            nd, bd = c0, 0
        else:
            nd, bd = c1, 1
        ei = e[i]
        vh, va, vd = nh + ei[0], na + ei[1], nd + ei[2]
        back[i, 0], back[i, 1], back[i, 2] = bh, ba, bd
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax([vh, va, vd]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def runs(codes):
    """[(start, end_exclusive, code)]."""
    if not len(codes):
        return []
    cut = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [len(codes)]))
    return [(int(s), int(e), int(codes[s])) for s, e in zip(starts, ends)]


def absorb_short_runs(codes, min_poss_s=None, min_dead_s=None):
    """Repeatedly hand the shortest too-short run to its longer neighbour."""
    codes = codes.copy()
    mp = C.MIN_POSSESSION_S if min_poss_s is None else min_poss_s
    md = C.MIN_DEAD_S if min_dead_s is None else min_dead_s
    min_len = {H: int(mp * C.FPS), A: int(mp * C.FPS), D: int(md * C.FPS)}
    while True:
        rs = runs(codes)
        if len(rs) < 2:
            return codes
        short = [(e - s, k) for k, (s, e, c) in enumerate(rs) if e - s < min_len[c]]
        if not short:
            return codes
        _, k = min(short)
        s, e, _ = rs[k]
        left = rs[k - 1] if k > 0 else None
        right = rs[k + 1] if k + 1 < len(rs) else None
        if left and right:
            pick = left if (left[1] - left[0]) >= (right[1] - right[0]) else right
        else:
            pick = left or right
        codes[s:e] = pick[2]


def abstain(labels, codes, ball_known):
    """X on long ball occlusions strictly inside a possession, away from its boundaries."""
    out = labels.copy()
    guard = int(C.ABSTAIN_GUARD_S * C.FPS)
    for s, e, c in runs(codes):
        if c == D:
            continue
        lo, hi = s + guard, e - guard
        if hi <= lo:
            continue
        gap = ~ball_known[lo:hi]
        if not gap.any():
            continue
        for gs, ge, gc in runs(gap.astype(np.int8)):
            if gc == 1 and ge - gs >= C.ABSTAIN_MIN_GAP_FRAMES:
                out[lo + gs:lo + ge] = "X"
    return out
