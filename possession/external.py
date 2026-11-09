"""Labelled matches other than the competition game: Metrica Sports' public sample data.

RULES.md allows labelled data from other games in any amount. Metrica's sample repository
(https://github.com/metrica-sports/sample-data) has three anonymised matches with tracking and
hand-collected event data, synchronised to the frame:

  Sample_Game_1, Sample_Game_2   CSV tracking (home and away files) + CSV events
  Sample_Game_3                  EPTS tracking + metadata (the same format as the competition
                                 ATD, parsed with io.py) + JSON events

Each is loaded into the same dict the pipeline builds for the competition game, so every stage
(features, shape, decoder) runs on it unchanged, and its events are turned into H / A / D labels
that the official scorer can score against. Where the data lives: data/external/sample-data
(gitignored, like all data).

Labels from events, frame by frame:
  team   the team of the latest on-ball event (pass, carry, shot, recovery, ball lost, set piece):
         possession changes at the first on-ball event of the other team, e.g. at the
         interception that RECOVERY marks
  dead   from the ball going out (BALL OUT, a shot that goes out or in), a foul (FAULT RECEIVED)
         or an offside, until the restart that follows (SET PIECE, or a GOAL KICK pass)
  X      before the kick-off of each half, after its last frame
"""

import json
import os

import numpy as np
import pandas as pd

from . import features
from .io import read_atd_meta, read_atd_raw

FPS = 25
PITCH = (105.0, 68.0)
H, A, D, X = 0, 1, 2, 3
ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "SET PIECE"}
MAX_STOPPAGE_S = 120

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(ROOT, "data", "external", "sample-data", "data")
GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")


def _tracking_csv(path):
    raw = pd.read_csv(path, skiprows=2, low_memory=False)
    cols = list(raw.columns)
    frames = raw["Frame"].to_numpy(int)
    period = raw["Period"].to_numpy(int)
    players, ball = [], None
    for k, c in enumerate(cols):
        if c.startswith("Player") or c == "Ball":
            xy = raw.iloc[:, [k, k + 1]].to_numpy(np.float64)
            if c == "Ball":
                ball = xy
            else:
                players.append(xy)
    return frames, period, np.stack(players, axis=1), ball


def _events_csv(path):
    ev = pd.read_csv(path)
    out = []
    for r in ev.itertuples(index=False):
        out.append({"team": H if r.Team == "Home" else A, "type": str(r.Type).upper(),
                    "sub": str(r.Subtype).upper() if isinstance(r.Subtype, str) else "",
                    "period": int(r.Period), "start": int(r._4) - 1, "end": int(r._6) - 1})
    return out


def _events_json(path, home_id):
    d = json.load(open(path, encoding="utf-8"))["data"]
    out = []
    for e in d:
        subs = e.get("subtypes") or []
        if isinstance(subs, dict):
            subs = [subs]
        out.append({"team": H if e["team"]["id"] == home_id else A,
                    "type": e["type"]["name"].upper(),
                    "sub": "-".join(s["name"].upper() for s in subs),
                    "period": int(e["period"]), "start": int(e["start"]["frame"]) - 1,
                    "end": int(e["end"]["frame"]) - 1})
    return out


def _home_id(meta_path):
    import xml.etree.ElementTree as ET
    for el in ET.parse(meta_path).getroot().iter():
        if el.tag.rsplit("}", 1)[-1] == "Team" and el.get("id"):
            return el.get("id")
    raise ValueError("no team in " + meta_path)


def load(game, base=DEFAULT_DIR):
    """The pipeline's data dict for one sample game, plus 'labels' (int8 H/A/D/X per frame)."""
    gdir = os.path.join(base, game)
    if game in ("Sample_Game_1", "Sample_Game_2"):
        fh, per, ph, ball = _tracking_csv(os.path.join(gdir, game + "_RawTrackingData_Home_Team.csv"))
        fa, _, pa, _ = _tracking_csv(os.path.join(gdir, game + "_RawTrackingData_Away_Team.csv"))
        if not np.array_equal(fh, fa):
            raise ValueError(game + ": home and away tracking frames differ")
        n = int(fh.max())
        players = np.full((n, ph.shape[1] + pa.shape[1], 3), np.nan)
        players[fh - 1, :, :2] = np.concatenate([ph, pa], axis=1)
        b = np.full((n, 2), np.nan)
        b[fh - 1] = ball
        slot_team = np.array([0] * ph.shape[1] + [1] * pa.shape[1], dtype=np.int8)
        period = np.zeros(n, int)
        period[fh - 1] = per
        events = _events_csv(os.path.join(gdir, game + "_RawEventsData.csv"))
    else:
        meta_p = os.path.join(gdir, game + "_metadata.xml")
        raw_p = os.path.join(gdir, game + "_tracking.txt")
        meta = read_atd_meta(meta_p)
        slot_team = meta["slot_team"]
        with open(raw_p) as f:
            n = sum(1 for _ in f)
        players, b = read_atd_raw(raw_p, len(slot_team), n,
                                  cache_dir=os.path.join(gdir, ".cache"))
        players = players.astype(np.float64)
        b = b.astype(np.float64)
        events = _events_json(os.path.join(gdir, game + "_events.json"), _home_id(meta_p))
        period = np.zeros(n, int)
        for p in (1, 2):
            fr = [e["start"] for e in events if e["period"] == p]
            if fr:
                period[min(fr):max(e["end"] for e in events if e["period"] == p) + 1] = p

    events.sort(key=lambda e: (e["start"], e["end"]))
    pm, psp = features.players_in_metres(players, PITCH)
    blocks = []
    for p in (1, 2):
        ko = [e["start"] for e in events if e["period"] == p and e["type"] == "SET PIECE"
              and "KICK OFF" in e["sub"]]
        last = np.flatnonzero(period == p)
        if not len(last):
            continue
        s = min(ko) if ko else int(last.min())
        blocks.append((p, s, int(last.max())))
    graded = np.zeros(n, dtype=bool)
    for _, s, e in blocks:
        graded[s:e + 1] = True
    return {"pm": pm, "psp": psp, "ball": b, "slot_team": slot_team, "pitch": PITCH,
            "blocks": blocks, "n": n, "graded": graded, "events": [],
            "kickoff_2nd": blocks[1][1] if len(blocks) > 1 else n,
            "labels": labels_from_events(events, n, blocks), "game": game,
            "raw_events": events}


def _stops(e):
    t, s = e["type"], e["sub"]
    if t == "BALL OUT" or t == "FAULT RECEIVED":
        return True
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return True
    return t == "BALL LOST" and "OFFSIDE" in s


def _restarts(e):
    return e["type"] == "SET PIECE" or (e["type"] == "PASS" and "GOAL KICK" in e["sub"])


def labels_from_events(events, n, blocks):
    lab = np.full(n, X, dtype=np.int8)
    for half, s, e in blocks:
        evs = [v for v in events if v["period"] == half]
        team = None
        cur = s
        # team per frame: latest on-ball event
        on = [(v["start"], v["team"]) for v in evs if v["type"] in ON_BALL]
        k = 0
        seg = np.full(e + 1 - s, X, dtype=np.int8)
        for f, tm in on:
            f = min(max(f, s), e + 1)
            if team is not None:
                seg[cur - s:f - s] = team
            team, cur = tm, f
        if team is not None:
            seg[cur - s:] = team
        # dead from each stoppage to the next restart
        for i, v in enumerate(evs):
            if not _stops(v):
                continue
            begin = v["end"] if v["type"] in ("BALL OUT", "SHOT") else v["start"]
            nxt = next((w for w in evs[i + 1:] if _restarts(w) and w["start"] > begin), None)
            if nxt is None or nxt["start"] - begin > MAX_STOPPAGE_S * FPS:
                continue
            a, b = max(begin, s), min(nxt["start"], e + 1)
            if b > a:
                seg[a - s:b - s] = D
        k += 1
        lab[s:e + 1] = seg
    return lab
