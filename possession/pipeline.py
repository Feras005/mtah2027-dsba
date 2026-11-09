"""data/ -> labels, in stages that the occlusion validation can re-run with the ball hidden.

  load         ATD + Smart Tagging, once
  ball stage   ball-dependent features (proximity, ball speed, out of bounds)
  shape stage  ball-free team-shape features + models trained on pseudo-labels (cross-half)
  evidence     per-frame emissions and transition costs for the HMM
  decode       Viterbi per half -> touch snapping -> short-run absorption -> labels
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

from . import config as C  # noqa: E402
from . import deadball, decode, features, pseudo, shape, shape_model, tagging, touches  # noqa
from . import broadcast, transfer  # noqa: E402
from .io import FORMAT_STATS, read_atd_meta, read_atd_raw, read_smart_tagging  # noqa: E402

# Published in FAQ.md: the second half is graded from 110,808 but kicks off at 113,742, and every
# graded frame before that is dead. A documented match boundary, not a label.
KICKOFF_2ND_VIDEO_FRAME = 113742

DEFAULTS = {
    "use_tagging": True,
    "use_shape": True,
    # On: on the 3 labelled external games, leave-one-game-out, snapping hand-overs and restarts
    # to detected touches raised spellF1 on every game (mean 0.571 -> 0.581; run_external.py).
    "snap": True,
    "team_w": C.SHAPE_TEAM_WEIGHT,
    "dead_w": C.SHAPE_DEAD_WEIGHT,
    "gate": True,
    "team_only": True,
    "missing_llr": C.MISSING_LLR,
    "dead_gate": True,
    "abstain": False,
    "switch_cost": C.SWITCH_COST,
    "stop_cost": C.STOP_COST,
    "restart_cost": C.RESTART_COST,
    "min_poss_s": C.MIN_POSSESSION_S,
    "min_dead_s": C.MIN_DEAD_S,
    # Models trained on other matches' real labels (learned.py, transfer.py). Weights chosen by
    # leave-one-game-out spellF1 on those matches (run_external.py tune): 0.540 -> 0.571.
    "ext_team_w": 0.5,
    "ext_dead_w": 0.2,
    "ext_team_gate": False,
    "use_external": True,
    # Fill ball gaps from the broadcast feed, aligned by player positions (broadcast.py). Hiding
    # the tactical ball and refilling from the broadcast recovered the possessions better on
    # every seed (run_consistency-style test, SOLUTION.md); on the real match it changes 0.6%.
    "use_broadcast": True,
}


def log(msg):
    print(msg, flush=True)


def quiet(_msg):
    pass


def load(data_dir=None, game=GAME, use_tagging=True, say=log):
    t0 = time.time()
    meta_p, raw_p = require(["atd_meta", "atd_raw"], game, data_dir)
    n = GAMES[game]["n_video_frames"]
    pitch = GAMES[game]["pitch"]
    blocks = half_video_blocks(game)
    meta = read_atd_meta(meta_p)
    slot_team = meta["slot_team"]
    say("metadata: {} track slots ({} home, {} away)".format(
        len(slot_team), int((slot_team == 0).sum()), int((slot_team == 1).sum())))
    cache = os.path.join(os.path.dirname(raw_p), ".cache")
    players, ball = read_atd_raw(raw_p, len(slot_team), n, cache_dir=cache)
    say("tracking: {} frames parsed  ({:.0f} s)".format(n, time.time() - t0))
    if FORMAT_STATS["lines"] and (FORMAT_STATS["slot_count_mismatch"]
                                  or FORMAT_STATS["short_lines"]):
        say("  WARNING format: {lines} lines, {slot_count_mismatch} with a slot count different "
            "from the metadata, {short_lines} without 3 ':' fields".format(**FORMAT_STATS))
    pm, psp = features.players_in_metres(players, pitch)
    del players
    graded = np.zeros(n, dtype=bool)
    for _, s, e in blocks:
        graded[s:e + 1] = True
    events = []
    if use_tagging:
        try:
            events = read_smart_tagging(require(["smart_tagging"], game, data_dir)[0])
        except IOError:
            say("smart tagging: not found, continuing without it")
    return {"pm": pm, "psp": psp, "ball": ball, "slot_team": slot_team, "pitch": pitch,
            "blocks": blocks, "n": n, "graded": graded, "events": events,
            "kickoff_2nd": KICKOFF_2ND_VIDEO_FRAME}


def ball_stage(data, ball_mask=None, say=log):
    feat = features.build(data["pm"], data["psp"], data["ball"], data["slot_team"],
                          data["pitch"], ball_mask)
    g = data["graded"]
    say("features: ball observed on {:.1f}% of graded frames, known (after bridging) on {:.1f}%,"
        " out of bounds on {:.1f}%".format(100 * feat["ball_observed"][g].mean(),
                                         100 * feat["ball_known"][g].mean(),
                                         100 * feat["oob"][g].mean()))
    return feat


def shape_stage(data, feat, say=log, with_dead=True):
    """Ball-free shape models, trained on pseudo-labels from the (unmasked) ball stage."""
    t0 = time.time()
    X, names, dirs = shape.build(data["pm"], data["psp"], data["slot_team"], data["blocks"],
                                 data["pitch"], data["n"])
    say("shape: {} features; attacking directions (half, team -> +1 = towards x=L): {}".format(
        len(names), {"{}{}".format(h, "HA"[t]): d for (h, t), d in sorted(dirs.items())}))
    restarts = tagging.set_piece_arrays(data["events"], data["n"], feat["crowd_speed"])[3]
    y_team = pseudo.team_labels(feat)
    y_team = np.where(data["graded"], y_team, -1)
    y_dead = pseudo.dead_labels(feat, restarts, data["graded"], data["kickoff_2nd"],
                                data["blocks"])
    say("pseudo-labels: team {} frames ({} home / {} away), dead {} frames, alive {} frames".format(
        int((y_team >= 0).sum()), int((y_team == 0).sum()), int((y_team == 1).sum()),
        int((y_dead == 1).sum()), int((y_dead == 0).sum())))
    # Team model predicts P(away); flip to P(home) so positive log-odds = home, as elsewhere.
    p_away = shape_model.cross_half(X, y_team, data["blocks"], names, "team", say,
                                    importances=True)
    p_dead = (shape_model.cross_half(X, y_dead, data["blocks"], names, "dead", say,
                                     importances=True) if with_dead
              else np.full(data["n"], 0.5))
    say("shape stage: {:.0f} s".format(time.time() - t0))
    return {"team_logit": -shape_model.logit(p_away, C.SHAPE_LOGIT_CAP),
            "dead_logit": shape_model.logit(p_dead, C.SHAPE_LOGIT_CAP)}


def decode_all(data, feat, shp, opts, say=log):
    n, g, events = data["n"], data["graded"], data["events"]
    sp_dead, sp_anchor, sp_side, _ = tagging.set_piece_arrays(events, n, feat["crowd_speed"])
    report = []
    phase = tagging.phase_prior(events, feat["team_llr"], g, n, report)
    if events:
        say("smart tagging: {} instances, {} set pieces".format(
            len(events), len(tagging.set_pieces(events, n))))
        say("  self-calibrated phase weights (code, n, evidence frames, agreement, weight):")
        for code, k, tot, agr, w in report:
            say("    {:<30} {:>4} {:>7} {:>6.2f} {:>+6.2f}".format(code, k, tot, agr, w))

    dead, crowd_info = deadball.dead_llr(feat, g, sp_dead)
    dead = dead + (opts["missing_llr"] - C.MISSING_LLR) * (~feat["ball_known"])
    say("crowd speed mixture: slow {slow_mps:.2f} m/s, fast {fast_mps:.2f} m/s, "
        "slow share {slow_share:.2f}".format(**crowd_info))
    team = feat["team_llr"] + phase + sp_side
    if shp is not None and "ext_team" in shp and opts["ext_team_w"]:
        w = opts["ext_team_w"]
        if opts["ext_team_gate"]:
            w = w * (1.0 - np.clip(np.abs(feat["team_llr"]) / C.PROXIMITY_CAP, 0.0, 1.0))
        team = team + w * shp["ext_team"]
    if shp is not None and "ext_dead" in shp and opts["ext_dead_w"]:
        dead = dead + opts["ext_dead_w"] * shp["ext_dead"]
    if shp is not None and "team_logit" in shp:
        w = opts["team_w"]
        if opts["gate"]:
            # The shape model speaks where the ball cannot: full weight with no proximity
            # evidence, none when a player is clearly on the ball.
            w = w * (1.0 - np.clip(np.abs(feat["team_llr"]) / C.PROXIMITY_CAP, 0.0, 1.0))
        team = team + w * shp["team_logit"]
        wd = opts["dead_w"]
        if opts["dead_gate"]:
            # Where the ball is seen, ball-based dead evidence (out of bounds, ball rolling)
            # is better than shape; the shape model fills in only where it is not.
            wd = wd * (~feat["ball_known"])
        if "dead_logit" in shp:
            dead = dead + wd * shp["dead_logit"]
    em = decode.emissions(team, dead, opts["team_only"])

    oob_near = features.rolling_mean(feat["oob"].astype(float), 2 * C.OOB_ANCHOR_FRAMES + 1,
                                     np.ones(n, bool)) > 0
    stop = np.where(oob_near, C.STOP_COST_AT_OOB, opts["stop_cost"])
    restart = np.where(sp_anchor, C.RESTART_COST_AT_SET_PIECE, opts["restart_cost"])
    kickoff_2nd = data["kickoff_2nd"]

    def absorb(path):
        return decode.absorb_short_runs(path, opts["min_poss_s"], opts["min_dead_s"])
    tch = touches.detect(feat) if opts["snap"] else None

    codes = np.full(n, decode.D, dtype=np.int8)
    moved = 0
    for half, s, e in data["blocks"]:
        e = e + 1
        em_h = em[s:e].copy()
        if half == 1:
            side = tagging.kickoff_side(events, s)
            init = [0.0 if side != "A" else -3.0, 0.0 if side != "H" else -3.0, decode.NEG]
        else:
            k = max(0, min(e, kickoff_2nd) - s)
            em_h[:k, :2] = decode.NEG / 1e6   # before the kickoff: only D is possible
            em_h[:k, 2] = 0.0
            init = [decode.NEG, decode.NEG, 0.0]
            if k < len(em_h):
                side = tagging.kickoff_side(events, s + k)
                if side:
                    em_h[k:k + C.FPS, 0 if side == "H" else 1] += 0.5
        path = decode.viterbi(em_h, stop[s:e], restart[s:e], np.array(init), opts["switch_cost"])
        path = absorb(path)
        if tch is not None:
            path, m = touches.snap(path, feat, tch, s, e)
            moved += m
            path = absorb(path)
        if half == 2:
            k = max(0, min(e, kickoff_2nd) - s)
            path[:k] = decode.D
        codes[s:e] = path
    if tch is not None:
        say("touches: {} detected, {} boundaries snapped".format(len(tch[0]), moved))

    labels = np.full(n, "X", dtype="<U1")
    labels[g] = decode.LABELS[codes[g]]
    if opts["abstain"]:
        labels = decode.abstain(labels, np.where(g, codes, decode.D).astype(np.int8),
                                feat["ball_known"])
        say("abstain: {} graded frames written as X".format(int((labels[g] == "X").sum())))
    rs = [r for _, s, e in data["blocks"] for r in decode.runs(codes[s:e + 1])]
    say("decoded: {} possessions, {} stoppages, D share {:.1f}%".format(
        sum(1 for r in rs if r[2] != decode.D), sum(1 for r in rs if r[2] == decode.D),
        100 * (codes[g] == decode.D).mean()))
    return labels, codes


def external_stage(data, feat, say=log):
    """Models trained on other matches' real labels (transfer.py), applied to this game."""
    t0 = time.time()
    if not transfer.available():
        raise IOError("external training data missing: run `python fetch_external.py` first "
                      "(or pass --no-external)")
    cache = os.path.join(ROOT, "data", "external", ".cache")
    models = transfer.train(cache, say)
    pred = transfer.predict(models, data, feat,
                            os.path.join(ROOT, "data", "tactical", ".cache", "learned_X.npz"))
    say("external models: P(dead) > 0.5 on {:.1f}% of graded frames, P(home | alive) > 0.5 on "
        "{:.1f}%  ({:.0f} s)".format(100 * (pred["ext_dead"][data["graded"]] > 0).mean(),
                                     100 * (pred["ext_team"][data["graded"]] > 0).mean(),
                                     time.time() - t0))
    return pred


def run(data_dir=None, game=GAME, **opts):
    o = dict(DEFAULTS)
    o.update(opts)
    t0 = time.time()
    data = load(data_dir, game, o["use_tagging"])
    if o["use_broadcast"]:
        data["ball"], _ = broadcast.filled_ball(data, game,
                                                os.path.join(ROOT, "data", "broadcast", ".cache"))
    feat = ball_stage(data)
    shp = shape_stage(data, feat, with_dead=o["dead_w"] > 0) if o["use_shape"] else {}
    if o["use_external"]:
        shp.update(external_stage(data, feat))
    labels, _ = decode_all(data, feat, shp, o)
    log("total {:.0f} s".format(time.time() - t0))
    return labels
