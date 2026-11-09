"""Loaders for the documented input formats, and the submission writer.

Everything is parsed from the formats described in DATA.md and the reference parser in
starter/baseline.py -- nothing here was written by looking at the data itself:

  ATD metadata   FIFA/EPTS XML. <Team id=..><Name>..</Name></Team>, <Player id=.. teamId=..>.
                 The first team is home. Slot order in the raw file follows the
                 DataFormatSpecification's PlayerChannelRefs when present, else <Player> order.
  ATD raw        one line per frame: "<frame_id>:<x,y,s>;<x,y,s>;...:<ball x,y[,..]>",
                 normalised 0-1 coordinates, NaN when missing, frame_id = video_frame + 1.
  Smart Tagging  SportsCode XML <instance><start/><end/><code/><label><group/><text/></label>,
                 seconds from the first tactical video frame.

Only shapes and counts are ever printed. Raw values are not.
"""

import io
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config as C

HOME, AWAY = 0, 1
CHUNK_LINES = 20000
# Counts of lines whose shape differs from the metadata -- printed as a format check, never data.
FORMAT_STATS = {"lines": 0, "slot_count_mismatch": 0, "short_lines": 0}


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _iter(root, name):
    return (el for el in root.iter() if _local(el.tag) == name)


def read_atd_meta(path):
    """{'slot_team': int8[S], 'names': (home, away), 'pitch': (length, width) or None}."""
    root = ET.parse(path).getroot()
    teams = [t for t in _iter(root, "Team") if t.get("id")]
    if len(teams) < 2:
        raise ValueError("{}: expected 2 teams, found {}".format(path, len(teams)))
    home_id = teams[0].get("id")
    names = tuple(next((c.text for c in t if _local(c.tag).lower() == "name"), t.get("id"))
                  for t in teams[:2])
    player_team = {p.get("id"): p.get("teamId") for p in _iter(root, "Player")}

    # Prefer the order the raw file is actually written in.
    channel_player = {c.get("id"): c.get("playerId") for c in _iter(root, "PlayerChannel")}
    order = []
    for ref in _iter(root, "PlayerChannelRef"):
        pid = channel_player.get(ref.get("playerChannelId"))
        if pid is not None and (not order or order[-1] != pid):
            if pid not in order:
                order.append(pid)
    if not order:
        order = [p.get("id") for p in _iter(root, "Player")]
    slot_team = np.array([HOME if player_team.get(pid) == home_id else AWAY for pid in order],
                         dtype=np.int8)

    pitch = None
    for el in root.iter():
        if _local(el.tag) == "FieldSize":
            try:
                pitch = (float(next(c.text for c in el if _local(c.tag) == "Width")),
                         float(next(c.text for c in el if _local(c.tag) == "Height")))
            except (StopIteration, TypeError, ValueError):
                pitch = None
    return {"slot_team": slot_team, "names": names, "pitch": pitch}


def _parse_chunk(lines, n_slots):
    """Rows of floats: frame_id, then n_slots x (x, y, s), then ball x, y."""
    width = 1 + 3 * n_slots + 2
    flat = []
    for line in lines:
        parts = line.rstrip("\n").split(":")
        FORMAT_STATS["lines"] += 1
        if len(parts) < 3:
            FORMAT_STATS["short_lines"] += 1
            continue
        groups = parts[1].split(";")
        if len(groups) != n_slots:
            FORMAT_STATS["slot_count_mismatch"] += 1
        ball = parts[2].split(",")
        row = [parts[0]]
        for g in range(n_slots):
            vals = groups[g].split(",") if g < len(groups) else []
            vals = (vals + ["NaN", "NaN", "NaN"])[:3]
            row.extend(vals)
        row.extend((ball + ["NaN", "NaN"])[:2])
        flat.append(",".join(row))
    if not flat:
        return np.empty((0, width))
    return pd.read_csv(io.StringIO("\n".join(flat)), header=None, dtype=np.float64,
                       na_values=["NaN", "nan", ""]).to_numpy()


def read_atd_raw(path, n_slots, n_frames, cache_dir=None):
    """(players float32[N, S, 3] as x, y, speed; ball float32[N, 2]), normalised, NaN = missing.

    Indexed by video frame (frame_id - 1). Cached as .npz next to the data, because parsing
    199k lines of text is the slowest step of the whole pipeline.
    """
    cache = None
    if cache_dir:
        st = os.stat(path)
        cache = os.path.join(cache_dir, "atd_{}_{}_{}.npz".format(
            os.path.basename(path), st.st_size, int(st.st_mtime)))
        if os.path.isfile(cache):
            z = np.load(cache)
            return z["players"], z["ball"]

    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(path) as handle:
        while True:
            lines = [l for _, l in zip(range(CHUNK_LINES), handle)]
            if not lines:
                break
            arr = _parse_chunk(lines, n_slots)
            if not len(arr):
                continue
            idx = arr[:, 0].astype(np.int64) - 1
            ok = (idx >= 0) & (idx < n_frames)
            idx, arr = idx[ok], arr[ok]
            players[idx] = arr[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)
            ball[idx] = arr[:, 1 + 3 * n_slots:]
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, players=players, ball=ball)
    return players, ball


def read_smart_tagging(path, fps=C.FPS):
    """[{code, start, end (video frames, end exclusive), tags{group: text}}]."""
    out = []
    for inst in _iter(ET.parse(path).getroot(), "instance"):
        def text(name):
            el = next((c for c in inst if _local(c.tag) == name), None)
            return el.text if el is not None else None
        try:
            start, end = float(text("start")), float(text("end"))
        except (TypeError, ValueError):
            continue
        tags = {}
        for label in (c for c in inst if _local(c.tag) == "label"):
            group = next((c.text for c in label if _local(c.tag) == "group"), None)
            value = next((c.text for c in label if _local(c.tag) == "text"), None)
            if group:
                tags[group.strip()] = (value or "").strip()
        out.append({"code": (text("code") or "").strip().upper(),
                    "start": int(round(start * fps)), "end": int(round(end * fps)),
                    "tags": tags})
    return out


def tag_side(value):
    """'H', 'A' or None for a Smart Tagging Team value."""
    v = (value or "").strip().lower()
    if any(t in v for t in C.HOME_TAG_TOKENS):
        return "H"
    if any(t in v for t in C.AWAY_TAG_TOKENS):
        return "A"
    return None


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path
