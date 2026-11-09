"""Benchmark and tune on labelled matches that are not the competition game.

    python run_external.py baseline     # current pipeline on Metrica's 3 public sample games
    python run_external.py study        # leave-one-game-out: learned models + decoder settings

Labels come from the games' own event data (possession/external.py), scored with the official
scorer. This is the only place a real spellF1 can be measured without touching the competition
ground truth, so it is what the learned models and the decoder settings are chosen with.
Leave-one-game-out: models that predict a game never saw it.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession import external, learned, pipeline  # noqa: E402
from possession import features as F  # noqa: E402

CACHE = os.path.join(HERE, "data", "external", ".cache")
GAMES = external.GAMES


def say(m):
    print(m, flush=True)


def prepare(game):
    d = external.load(game)
    feat = pipeline.ball_stage(d, say=pipeline.quiet)
    mask = learned.random_blocks(d["n"], learned.MASK_SHARE, seed=GAMES.index(game) + 11)
    feat_m = pipeline.features.build(d["pm"], d["psp"], d["ball"], d["slot_team"], d["pitch"],
                                     mask)
    Xd, nd, Xs, ns = learned.features_for(d, feat, os.path.join(CACHE, game + "_X.npz"))
    pc = os.path.join(CACHE, game + "_pseudo_shape.npz")
    if os.path.isfile(pc):
        team_logit = np.load(pc)["team_logit"]
    else:
        team_logit = pipeline.shape_stage(d, feat, say=pipeline.quiet,
                                          with_dead=False)["team_logit"]
        np.savez_compressed(pc, team_logit=team_logit)
    return {"data": d, "feat": feat, "feat_m": feat_m, "Xd": Xd, "nd": nd, "Xs": Xs, "ns": ns,
            "team_logit": team_logit}


def logo(G):
    """Leave-one-game-out predictions of the learned models, per game."""
    out = {}
    for test in GAMES:
        rd, yd, rt, yt = [], [], [], []
        for g in GAMES:
            if g == test:
                continue
            P = G[g]
            lab, gr = P["data"]["labels"], P["data"]["graded"]
            rd.append(P["Xd"][gr])
            yd.append((lab[gr] == external.D).astype(int))
            Xt, _ = learned.team_matrix(P["Xs"], P["ns"], P["feat_m"])
            alive = gr & (lab <= external.A)
            rt.append(Xt[alive])
            yt.append((lab[alive] == external.A).astype(int))
        md, mt = learned.fit(np.concatenate(rd), np.concatenate(yd), np.concatenate(rt),
                             np.concatenate(yt))
        P = G[test]
        Xt, _ = learned.team_matrix(P["Xs"], P["ns"], P["feat"])
        out[test] = {"ext_dead": learned.logit(md.predict_proba(P["Xd"])[:, 1]),
                     "ext_team": -learned.logit(mt.predict_proba(Xt)[:, 1])}
        lab, gr = P["data"]["labels"], P["data"]["graded"]
        from sklearn.metrics import roc_auc_score
        yd_ = (lab[gr] == external.D).astype(int)
        alive = gr & (lab <= external.A)
        say("  {}: dead model AUC {:.3f}, team model AUC {:.3f} (ball hidden: {:.3f})".format(
            test, roc_auc_score(yd_, out[test]["ext_dead"][gr]),
            roc_auc_score((lab[alive] == external.H).astype(int), out[test]["ext_team"][alive]),
            roc_auc_score((lab[alive & ~P["feat"]["ball_known"]] == external.H).astype(int),
                          out[test]["ext_team"][alive & ~P["feat"]["ball_known"]])))
    return out


def evaluate(G, preds, **opts):
    o = dict(pipeline.DEFAULTS)
    o.update(use_tagging=False)
    o.update(opts)
    res = {}
    for g in GAMES:
        P = G[g]
        shp = {"team_logit": P["team_logit"]} if o["use_shape"] else {}
        if preds:
            shp.update(preds[g])
        labels, _ = pipeline.decode_all(P["data"], P["feat"], shp, o, say=pipeline.quiet)
        r = pm.evaluate(P["data"]["labels"], pm.label_codes(labels), bootstrap_n=0)
        res[g] = (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                  r["alive_dead_f1"], r["team_acc_both_alive"])
    return res


def summary(res):
    a = np.array(list(res.values()), dtype=float)
    return {"spell": a[:, 0].mean(), "macro": a[:, 1].mean(), "poss": a[:, 2].mean(),
            "truth": a[:, 3].mean(), "ad": a[:, 4].mean(), "team": a[:, 5].mean(),
            "per_game": [round(x, 3) for x in a[:, 0]]}


def line(name, s):
    say("{:<44} spellF1 {spell:.3f} {per_game}  macroF1 {macro:.3f}  alive/dead {ad:.3f}  "
        "team {team:.3f}  poss {poss:.0f}/{truth:.0f}".format(name, **s))


def confident_labels(codes, graded, margin_s=2.0, min_run_s=6.0):
    """Self-training targets: frames deep inside long decoded runs. -1 elsewhere."""
    from possession.decode import runs
    out = np.full(len(codes), -1, dtype=np.int8)
    m, mr = int(margin_s * 25), int(min_run_s * 25)
    for s, e, c in runs(np.where(graded, codes, 3).astype(np.int8)):
        if c <= 2 and e - s >= mr:
            out[s + m:e - m] = c
    return out


def selftrain(G, opts, iters=2):
    """Leave-one-game-out self-training: decode the held-out game, add its confident stretches
    as labels, retrain, decode again. Scored against the held-out game's real labels."""
    base = {}
    rows = {}
    for g in GAMES:
        P = G[g]
        lab, gr = P["data"]["labels"], P["data"]["graded"]
        Xt, _ = learned.team_matrix(P["Xs"], P["ns"], P["feat_m"])
        alive = gr & (lab <= external.A)
        rows[g] = (P["Xd"][gr], (lab[gr] == external.D).astype(int), Xt[alive],
                   (lab[alive] == external.A).astype(int))
    history = {k: {} for k in range(iters + 1)}
    for test in GAMES:
        P = G[test]
        others = [g for g in GAMES if g != test]
        extra = None
        for it in range(iters + 1):
            rd = [rows[g][0] for g in others]
            yd = [rows[g][1] for g in others]
            rt = [rows[g][2] for g in others]
            yt = [rows[g][3] for g in others]
            if extra is not None:
                rd.append(extra[0]); yd.append(extra[1]); rt.append(extra[2]); yt.append(extra[3])
            md, mt = learned.fit(np.concatenate(rd), np.concatenate(yd), np.concatenate(rt),
                                 np.concatenate(yt))
            Xt, _ = learned.team_matrix(P["Xs"], P["ns"], P["feat"])
            pr = {"ext_dead": learned.logit(md.predict_proba(P["Xd"])[:, 1]),
                  "ext_team": -learned.logit(mt.predict_proba(Xt)[:, 1])}
            o = dict(pipeline.DEFAULTS, use_tagging=False, **opts)
            shp = {"team_logit": P["team_logit"], **pr}
            labels, codes = pipeline.decode_all(P["data"], P["feat"], shp, o, say=pipeline.quiet)
            r = pm.evaluate(P["data"]["labels"], pm.label_codes(labels), bootstrap_n=0)
            history[it][test] = (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"],
                                 r["n_gt_spells"], r["alive_dead_f1"], r["team_acc_both_alive"])
            conf = confident_labels(codes, P["data"]["graded"])
            Xtm, _ = learned.team_matrix(P["Xs"], P["ns"], P["feat_m"])
            dsel = conf >= 0
            tsel = (conf == 0) | (conf == 1)
            extra = (P["Xd"][dsel], (conf[dsel] == 2).astype(int), Xtm[tsel],
                     (conf[tsel] == 1).astype(int))
            if it == 0:
                say("  {}: self-training set {} frames ({} team)".format(
                    test, int(dsel.sum()), int(tsel.sum())))
    for it in range(iters + 1):
        line("self-training iteration {}".format(it), summary(history[it]))


GRID = {
    "ext_dead_w": [0.0, 0.1, 0.2, 0.3, 0.5],
    "ext_team_w": [0.0, 0.3, 0.5, 1.0, 1.5],
    "team_w": [0.0, 0.1, 0.2, 0.4],
    "switch_cost": [5.0, 7.0, 9.0, 11.0, 14.0, 18.0],
    "stop_cost": [3.0, 5.0, 7.0, 9.0, 12.0],
    "restart_cost": [3.0, 5.0, 7.0, 9.0, 12.0],
    "min_dead_s": [0.4, 0.8, 1.2, 2.0, 3.0],
    "min_poss_s": [0.2, 0.4, 0.8, 1.2],
}


def tune(G, preds, rounds=2):
    """Coordinate ascent on the mean leave-one-game-out spellF1, one setting at a time."""
    cur = {"ext_dead_w": 0.2, "ext_team_w": 0.5, "ext_team_gate": False}
    best = summary(evaluate(G, preds, **cur))
    line("start " + json.dumps(cur), best)
    for r in range(rounds):
        changed = False
        for key, values in GRID.items():
            scores = {}
            for v in values:
                trial = dict(cur, **{key: v})
                scores[v] = summary(evaluate(G, preds, **trial))
            v_best = max(scores, key=lambda v: scores[v]["spell"])
            say("  round {} {:<13} ".format(r + 1, key) + "  ".join(
                "{}:{:.4f}".format(v, scores[v]["spell"]) for v in values))
            if scores[v_best]["spell"] > best["spell"] + 1e-4 and cur.get(key) != v_best:
                cur[key] = v_best
                best = scores[v_best]
                changed = True
        line("after round {} {}".format(r + 1, json.dumps(cur)), best)
        if not changed:
            break
    return cur


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    t0 = time.time()
    G = {g: prepare(g) for g in GAMES}
    say("prepared {} games ({:.0f} s)".format(len(G), time.time() - t0))
    if mode == "baseline":
        line("v2 pipeline (submitted)", summary(evaluate(G, None)))
        sys.exit(0)
    pc = os.path.join(CACHE, "logo_preds.npz")
    if mode in ("tune", "extras") and os.path.isfile(pc):
        z = np.load(pc)
        preds = {g: {k: z[g + "|" + k] for k in ("ext_dead", "ext_team")} for g in GAMES}
    else:
        preds = logo(G)
        np.savez_compressed(pc, **{g + "|" + k: v for g in GAMES for k, v in preds[g].items()})
    if mode == "tune":
        tune(G, preds)
        say("({:.0f} s)".format(time.time() - t0))
        sys.exit(0)
    if mode == "extras":
        best = json.loads(sys.argv[2])
        line("tuned " + json.dumps(best), summary(evaluate(G, preds, **best)))
        line("tuned + snap to touches", summary(evaluate(G, preds, snap=True, **best)))
        selftrain(G, best)
        say("({:.0f} s)".format(time.time() - t0))
        sys.exit(0)
    line("v2 pipeline (submitted)", summary(evaluate(G, None)))
    for dw in (0.1, 0.2, 0.4):
        line("+ external dead model w={}".format(dw), summary(evaluate(G, preds, ext_dead_w=dw)))
    for tw in (0.2, 0.5):
        for gate in (True, False):
            line("+ external team model w={} gate={}".format(tw, gate),
                 summary(evaluate(G, preds, ext_team_w=tw, ext_team_gate=gate)))
    say("({:.0f} s)".format(time.time() - t0))
