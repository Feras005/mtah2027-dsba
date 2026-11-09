"""Align the broadcast feed to the tactical timeline by player positions, and borrow its ball.

The broadcast ATD gives positions in pitch coordinates, like the tactical ATD, but on its own
timeline, with cuts, replays and close-ups, and no mapping table (DATA.md). Both are real-time
recordings of the same match, so live broadcast footage sits at a fixed frame offset from the
tactical video -- one per stretch of continuous recording -- while replays do not match anything
at that offset.

  1. Offsets: for a few hundred broadcast frames with enough players in view, compare their
     players with every tactical frame (1 s steps). The cost is how far, in median, each
     broadcast player is from the nearest tactical player: a broadcast camera sees a subset of
     the players, so the comparison goes one way. The offsets that win most often are the
     candidates; each is refined to the frame.
  2. Per frame: a broadcast frame is accepted at the candidate offset where its players match
     (median distance under MATCH_M metres, at least MIN_PLAYERS in view). Replays, close-ups and
     crowd shots fail and are dropped.
  3. Ball: on tactical frames where the tracker has no ball, the ball of an accepted broadcast
     frame fills it in.

Only offsets, match rates and coverage are printed.
"""

import os

import numpy as np

from . import config as C

STEP = C.FPS
N_SAMPLES = 400
MIN_PLAYERS = 6
MATCH_M = 1.0
MIN_VOTES = 0.05


def _directional(bro, tac):
    """Median over broadcast players of the distance to the nearest tactical player.
    bro [P, 2]; tac [M, Q, 2] -> [M]."""
    d = np.hypot(bro[None, :, None, 0] - tac[:, None, :, 0],
                 bro[None, :, None, 1] - tac[:, None, :, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    return np.median(d.min(axis=2), axis=1)


def _cost(bpm, tpm, b, t):
    bp = bpm[b]
    bp = bp[np.isfinite(bp).all(axis=1)]
    if len(bp) < MIN_PLAYERS:
        return np.inf
    return float(_directional(bp, tpm[t:t + 1])[0])


def find_offsets(bpm, tpm, seed=0, say=print):
    """Candidate offsets o (tactical = broadcast + o), refined to the frame."""
    rng = np.random.default_rng(seed)
    nb, nt = len(bpm), len(tpm)
    visible = np.isfinite(bpm[..., 0]).sum(axis=1)
    cand = np.flatnonzero(visible >= 10)
    if not len(cand):
        return []
    samples = np.sort(rng.choice(cand, size=min(N_SAMPLES, len(cand)), replace=False))
    grid = np.arange(0, nt, STEP)
    tg = tpm[grid]
    votes = []
    for b in samples:
        bp = bpm[b][np.isfinite(bpm[b]).all(axis=1)]
        c = np.concatenate([_directional(bp, tg[i:i + 2000]) for i in range(0, len(tg), 2000)])
        k = int(np.argmin(c))
        if c[k] < 2.0 * MATCH_M:
            votes.append(int(grid[k]) - int(b))
    if not votes:
        return []
    votes = np.array(votes)
    # cluster votes within 2 s
    out = []
    remaining = votes.copy()
    while len(remaining):
        vals, counts = np.unique(np.round(remaining / (2 * STEP)), return_counts=True)
        top = vals[np.argmax(counts)] * 2 * STEP
        members = remaining[np.abs(remaining - top) <= 2 * STEP]
        if len(members) < MIN_VOTES * len(samples):
            break
        out.append(int(np.median(members)))
        remaining = remaining[np.abs(remaining - top) > 2 * STEP]
    # refine each to the frame on the samples that voted for it
    refined = []
    for o in out:
        best, best_o = np.inf, o
        voters = [b for b in samples if 0 <= b + o < nt]
        for d in range(-STEP, STEP + 1):
            cs = [_cost(bpm, tpm, b, b + o + d) for b in voters[:120] if 0 <= b + o + d < nt]
            cs = [c for c in cs if np.isfinite(c)]
            if not cs:
                continue
            m = float(np.median(cs))
            if m < best:
                best, best_o = m, o + d
        refined.append(best_o)
    say("broadcast: {} of {} sampled frames matched a tactical frame; offsets {}".format(
        len(votes), len(samples), refined))
    return refined


def align(bpm, tpm, offsets):
    """tactical index -> broadcast index (or -1), for frames whose players match."""
    nb, nt = len(bpm), len(tpm)
    best_cost = np.full(nt, np.inf)
    t2b = np.full(nt, -1, dtype=np.int64)
    visible = np.isfinite(bpm[..., 0]).sum(axis=1) >= MIN_PLAYERS
    for o in offsets:
        b = np.flatnonzero(visible)
        t = b + o
        ok = (t >= 0) & (t < nt)
        b, t = b[ok], t[ok]
        cost = np.full(len(b), np.inf)
        for s in range(0, len(b), 4000):
            bb, tt = b[s:s + 4000], t[s:s + 4000]
            bp, tp = bpm[bb], tpm[tt]
            d = np.hypot(bp[:, :, None, 0] - tp[:, None, :, 0], bp[:, :, None, 1] - tp[:, None, :, 1])
            d = np.where(np.isfinite(d), d, np.inf).min(axis=2)
            d = np.where(np.isfinite(bp[..., 0]), d, np.nan)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cost[s:s + 4000] = np.nanmedian(d, axis=1)
        good = cost < MATCH_M
        better = good & (cost < best_cost[t])
        best_cost[t[better]] = cost[better]
        t2b[t[better]] = b[better]
    return t2b, best_cost


def fill_ball(tac_ball_m, bro_ball_m, t2b, graded, say=print):
    """Tactical ball (metres) with gaps filled from aligned broadcast frames."""
    out = tac_ball_m.copy()
    missing = ~np.isfinite(out).all(axis=1)
    has = t2b >= 0
    bb = np.full_like(out, np.nan)
    bb[has] = bro_ball_m[t2b[has]]
    fill = missing & np.isfinite(bb).all(axis=1)
    out[fill] = bb[fill]
    agree = has & ~missing & np.isfinite(bb).all(axis=1)
    err = np.hypot(*(bb[agree] - tac_ball_m[agree]).T) if agree.any() else np.array([np.nan])
    say("broadcast: aligned on {:.1f}% of graded frames; ball filled on {:.1f}% of graded frames "
        "(ball known {:.1f}% -> {:.1f}%); where both see it, median distance {:.2f} m".format(
            100 * has[graded].mean(), 100 * fill[graded].mean(),
            100 * (~missing)[graded].mean(), 100 * np.isfinite(out).all(axis=1)[graded].mean(),
            float(np.median(err))))
    return out, fill


def load(game, pitch, say=print):
    """Broadcast players and ball in metres, via the same ATD parser as the tactical feed."""
    import games
    from .io import read_atd_meta, read_atd_raw
    meta_p, raw_p = games.require(["atd_meta", "atd_raw"], game, None, source="broadcast")
    meta = read_atd_meta(meta_p)
    n = games.source_frames("broadcast", game)
    players, ball = read_atd_raw(raw_p, len(meta["slot_team"]), n,
                                 cache_dir=os.path.join(os.path.dirname(raw_p), ".cache"))
    pm = players[..., :2].astype(np.float64) * np.array(pitch)
    bm = ball.astype(np.float64) * np.array(pitch)
    say("broadcast: {} frames, {} slots, players in view {:.1f} per frame (median), ball on "
        "{:.1f}% of frames".format(n, len(meta["slot_team"]),
                                   float(np.median(np.isfinite(pm[..., 0]).sum(axis=1))),
                                   100 * np.isfinite(bm).all(axis=1).mean()))
    return pm, bm


def filled_ball(data, game, cache_dir, say=print):
    """data["ball"] (normalised) with gaps filled from the aligned broadcast. Cached."""
    cache = os.path.join(cache_dir, "alignment.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        t2b, bball = z["t2b"], z["bball"]
        say("broadcast: alignment from cache, offsets {}".format(z["offsets"].tolist()))
    else:
        bpm, bball = load(game, data["pitch"], say)
        offsets = find_offsets(bpm, data["pm"], say=say)
        t2b, _ = align(bpm, data["pm"], offsets)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, t2b=t2b, bball=bball, offsets=np.array(offsets))
    pitch = np.array(data["pitch"])
    tball = data["ball"].astype(np.float64) * pitch
    out, fill = fill_ball(tball, bball, t2b, data["graded"], say)
    return (out / pitch).astype(data["ball"].dtype), fill
