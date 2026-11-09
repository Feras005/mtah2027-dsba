"""Pseudo-labels: frames whose answer is obvious from the data, chosen by a rule, never by a person.

RULES.md allows "pseudo-labels produced by an algorithm, provided no person decided any of them".
These are the training targets of the shape model. They are deliberately strict -- few, clean
frames beat many noisy ones -- and they come only from the tracking and the Smart Tagging:

  team  the ball is *observed* (not bridged), slow enough to be at someone's feet, one team has a
        player within 1 m of it and the other team has nobody within 3 m
  dead  the ball is observed outside the lines; a tagged set-piece window up to 1 s before its
        estimated restart (tagging.restart_frame); graded frames before the published
        second-half kickoff
  alive the team frames above
"""

import numpy as np

from . import config as C


def team_labels(feat):
    """int8[N]: 0 home, 1 away, -1 no pseudo-label."""
    dh, da = feat["dist"][:, 0], feat["dist"][:, 1]
    sp = np.nan_to_num(feat["ball_speed"], nan=99.0)
    ok = feat["ball_observed"] & ~feat["oob"] & (sp < C.PSEUDO_MAX_BALL_SPEED)
    home = ok & (dh <= C.PSEUDO_NEAR) & (da >= C.PSEUDO_FAR)
    away = ok & (da <= C.PSEUDO_NEAR) & (dh >= C.PSEUDO_FAR)
    out = np.full(len(dh), -1, dtype=np.int8)
    out[home] = 0
    out[away] = 1
    return out


def dead_labels(feat, restarts, graded, kickoff_2nd, blocks):
    """int8[N]: 1 dead, 0 alive, -1 no pseudo-label."""
    n = len(graded)
    dead = feat["oob"].copy()
    for s, r, _ in restarts:
        dead[s:max(s, r - C.FPS)] = True
    for half, s, e in blocks:
        if half == 2:
            dead[s:min(e + 1, kickoff_2nd)] = True
    alive = team_labels(feat) >= 0
    out = np.full(n, -1, dtype=np.int8)
    out[alive & graded] = 0
    out[dead & graded] = 1
    return out
