"""Train the learned models on all external labelled games and apply them to the competition game.

Training data: Metrica Sports' public sample data (external.py), pinned to commit
SAMPLE_DATA_COMMIT, fetched by `python fetch_external.py` into data/external/sample-data.
Nothing about the competition game enters training.
"""

import os

import numpy as np

from . import external, learned
from . import features as F

SAMPLE_DATA_URL = "https://github.com/metrica-sports/sample-data.git"
SAMPLE_DATA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"


def available(base=external.DEFAULT_DIR):
    return all(os.path.isdir(os.path.join(base, g)) for g in external.GAMES)


def training_rows(game, cache_dir):
    """(dead rows, dead y, team rows, team y) for one external game, ball hidden at random for
    the team model. Seeded per game, so the training set is the same on every run."""
    d = external.load(game)
    feat = F.build(d["pm"], d["psp"], d["ball"], d["slot_team"], d["pitch"])
    Xd, _, Xs, ns = learned.features_for(d, feat, os.path.join(cache_dir, game + "_X.npz"))
    mask = learned.random_blocks(d["n"], learned.MASK_SHARE,
                                 seed=external.GAMES.index(game) + 11)
    feat_m = F.build(d["pm"], d["psp"], d["ball"], d["slot_team"], d["pitch"], mask)
    Xt, _ = learned.team_matrix(Xs, ns, feat_m)
    lab, gr = d["labels"], d["graded"]
    alive = gr & (lab <= external.A)
    return (Xd[gr], (lab[gr] == external.D).astype(int), Xt[alive],
            (lab[alive] == external.A).astype(int))


def train(cache_dir, say):
    rd, yd, rt, yt = [], [], [], []
    for g in external.GAMES:
        a, b, c, e = training_rows(g, cache_dir)
        rd.append(a)
        yd.append(b)
        rt.append(c)
        yt.append(e)
    yd, yt = np.concatenate(yd), np.concatenate(yt)
    say("external training data: {} games, {} frames for the dead model ({:.0%} dead), {} live "
        "frames for the team model".format(len(external.GAMES), len(yd), yd.mean(), len(yt)))
    return learned.fit(np.concatenate(rd), yd, np.concatenate(rt), yt)


def predict(models, data, feat, cache=None):
    md, mt = models
    Xd, _, Xs, ns = learned.features_for(data, feat, cache)
    Xt, _ = learned.team_matrix(Xs, ns, feat)
    return {"ext_dead": learned.logit(md.predict_proba(Xd)[:, 1]),
            "ext_team": -learned.logit(mt.predict_proba(Xt)[:, 1])}
