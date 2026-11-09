"""A synthetic match in the exact on-disk formats of the real bundle, with its own ground truth.

Why: the real data is gated and may not be shown to any cloud LLM, so the pipeline was developed
end-to-end against this instead. It writes

    <out>/tactical/BAR-ATH_tactical_atd.xml             EPTS-style metadata, 46 track slots
    <out>/tactical/BAR-ATH_tactical_atd.txt             "<id>:<x,y,s;...>:<bx,by,bs>", NaN gaps
    <out>/tactical/BAR-ATH_tactical_smart_tagging.xml   SportsCode, noisy set pieces + phases
    <out>/tactical/BAR-ATH_tactical_homography.csv      constant valid matrix (pipeline ignores it)
    <out>/gt/synthetic_possession_gt.csv                frame,label truth for the whole video

The simulation is deliberately unkind in the ways DATA.md describes: ~40% ball occlusion,
burstier in flight; track slots that swap; loose balls that do not change possession; tags that
are late, missing, or name the wrong team; throw-ins tagged N/A. It is NOT a model of the real
match, so a score here proves the pipeline works, not how well it will score.

    python synthetic/make_synthetic.py --out synthetic_data --seed 7
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAMES, GAME, half_video_blocks  # noqa: E402

FPS = 25
N = GAMES[GAME]["n_video_frames"]
L, W = GAMES[GAME]["pitch"]
KICKOFF_2ND = 113742
SLOTS_PER_TEAM = 23
HOME_NAME, AWAY_NAME = "Barcelona", "Athletic Club"
TAG_NAMES = {0: "Barcelona", 1: "Atletic Club"}

# 4-3-3 anchors for a team attacking +x, in metres.
FORMATION = np.array([[5, 34], [22, 10], [20, 27], [20, 41], [22, 58], [38, 20], [36, 34],
                      [38, 48], [55, 12], [58, 34], [55, 56]], dtype=float)


def simulate(rng):
    label = np.full(N, -1, dtype=np.int8)          # 0 H, 1 A, 2 D, -1 ungraded
    ball = np.full((N, 2), np.nan)
    carrier = np.full(N, -1, dtype=np.int16)       # player index 0..21 touching / near the ball
    in_flight = np.zeros(N, dtype=bool)
    restarts = []                                  # (frame, side, type)
    phases = []                                    # (start, end, side) of possessions

    def anchor(team, half, k, bxy):
        p = FORMATION[k].copy()
        attack_pos = (team == 0) == (half == 1)
        if not attack_pos:
            p[0] = L - p[0]
        return p + 0.35 * (bxy - np.array([L / 2, W / 2]))

    for half, s, e in half_video_blocks(GAME):
        e += 1
        t = s
        if half == 2:
            label[s:KICKOFF_2ND] = 2
            ball[s:KICKOFF_2ND] = [L / 2, W / 2]
            t = KICKOFF_2ND
        team = int(rng.integers(2))
        restarts.append((t, team, "Kick Off (Start)"))
        bxy = np.array([L / 2, W / 2])
        while t < e:
            dur = int(max(2.0, rng.exponential(11.0)) * FPS)
            end = min(e, t + dur)
            phases.append((t, end, team))
            k = int(rng.integers(1, 11))
            u = t
            while u < end:
                hold = min(end - u, int(rng.uniform(0.8, 3.0) * FPS))
                v = rng.normal(0, 2.5, 2)
                for f in range(u, u + hold):
                    bxy = np.clip(bxy + v / FPS, [1, 1], [L - 1, W - 1])
                    ball[f] = bxy
                    carrier[f] = team * 11 + k
                    label[f] = team
                u += hold
                if u >= end:
                    break
                k2 = int(rng.integers(0, 11))
                dest = np.clip(anchor(team, half, k2, bxy) + rng.normal(0, 3, 2),
                               [1, 1], [L - 1, W - 1])
                fl = max(5, int(np.hypot(*(dest - bxy)) / rng.uniform(12, 20) * FPS))
                fl = min(fl, end - u)
                for j, f in enumerate(range(u, u + fl)):
                    ball[f] = bxy + (dest - bxy) * (j + 1) / fl
                    in_flight[f] = True
                    label[f] = team
                bxy = ball[u + fl - 1].copy()
                u += fl
                k = k2
            t = end
            if t >= e:
                break
            if rng.random() < 0.55:
                # Turnover: sometimes a loose, contested ball first -- possession stays put.
                if rng.random() < 0.3:
                    lo = min(e, t + int(rng.uniform(0.5, 1.5) * FPS))
                    for f in range(t, lo):
                        bxy = np.clip(bxy + rng.normal(0, 4, 2) / FPS, [1, 1], [L - 1, W - 1])
                        ball[f] = bxy
                        in_flight[f] = True
                        label[f] = team
                    t = lo
                team = 1 - team
                continue
            # Ball goes out, stoppage, restart.
            side_out = rng.integers(4)
            out = bxy.copy()
            if side_out == 0:
                out[1] = -rng.uniform(1, 3)
            elif side_out == 1:
                out[1] = W + rng.uniform(1, 3)
            elif side_out == 2:
                out[0] = -rng.uniform(1, 3)
            else:
                out[0] = L + rng.uniform(1, 3)
            fl = max(8, int(np.hypot(*(out - bxy)) / 15.0 * FPS))
            fl = min(fl, e - t)
            for j, f in enumerate(range(t, t + fl)):
                ball[f] = bxy + (out - bxy) * (j + 1) / fl
                in_flight[f] = True
                label[f] = team
            t += fl
            dead = min(e - t, int(rng.uniform(4, 25) * FPS))
            restart_xy = np.clip(out, [0, 0], [L, W])
            label[t:t + dead] = 2
            ball[t:t + dead // 2] = out
            ball[t + dead // 2:t + dead] = restart_xy
            t += dead
            bxy = restart_xy.copy()
            team = int(rng.integers(2))
            kind = ("Throw-in" if side_out < 2 else
                    str(rng.choice(["Corner Kick", "Goal Kick", "Free Kick"])))
            restarts.append((t, team, kind))
    return label, ball, carrier, in_flight, restarts, phases, anchor


def players_from(label, ball, carrier, anchor, rng):
    pos = np.full((N, 22, 2), np.nan)
    blocks = half_video_blocks(GAME)
    for half, s, e in blocks:
        e += 1
        cur = np.array([anchor(p // 11, half, p % 11, np.array([L / 2, W / 2])) for p in range(22)])
        ou = np.zeros((22, 2))
        for f in range(s, e):
            dead = label[f] == 2
            bxy = ball[f] if np.isfinite(ball[f]).all() else np.array([L / 2, W / 2])
            sd, tau = (0.6, 4.0) if dead else (2.0, 2.0)
            ou += (-ou / (tau * FPS)) + rng.normal(0, sd * np.sqrt(2 / (tau * FPS)), (22, 2))
            tgt = np.array([anchor(p // 11, half, p % 11, bxy) for p in range(22)]) + 3 * ou
            gain = np.full(22, 0.02 if dead else 0.06)
            c = carrier[f]
            if c >= 0:
                tgt[c] = bxy + np.array([0.5, 0.3])
                gain[c] = 0.5
                # nearest opponent presses
                opp = np.arange(22)[(np.arange(22) // 11) != c // 11]
                o = opp[np.argmin(np.hypot(*(cur[opp] - bxy).T))]
                tgt[o] = bxy + 3.0 * (cur[o] - bxy) / max(1e-6, np.hypot(*(cur[o] - bxy)))
                gain[o] = 0.15
            step = gain[:, None] * (tgt - cur)
            spd = np.hypot(*step.T) * FPS
            cap = np.where(spd > 8.0, 8.0 / np.maximum(spd, 1e-9), 1.0)
            cur = cur + step * cap[:, None]
            pos[f] = cur
    return pos


def observe(label, ball, in_flight, pos, rng):
    """Apply the tracker: ball occlusions, position noise, slot swaps, track dropouts."""
    graded = label >= 0
    ball_obs = ball + rng.normal(0, 0.25, ball.shape)
    vis = np.ones(N, dtype=bool)
    state = True
    for f in range(N):
        p_lose = 0.10 if in_flight[f] else (0.06 if label[f] == 2 else 0.025)
        p_find = 0.06
        state = (rng.random() > p_lose) if state else (rng.random() < p_find)
        vis[f] = state
    vis &= graded
    ball_obs[~vis] = np.nan
    junk = graded & (rng.random(N) < 0.004)
    ball_obs[junk] = rng.uniform([0, 0], [L, W], (int(junk.sum()), 2))

    slots = np.full((N, 2 * SLOTS_PER_TEAM, 3), np.nan)
    owner = np.array([p if p < 11 else SLOTS_PER_TEAM + p - 11 for p in range(22)])
    free = {0: list(range(11, SLOTS_PER_TEAM)),
            1: list(range(SLOTS_PER_TEAM + 11, 2 * SLOTS_PER_TEAM))}
    noisy = pos + rng.normal(0, 0.15, pos.shape)
    spd = np.full((N, 22), np.nan)
    d = pos[2:] - pos[:-2]
    spd[1:-1] = np.hypot(d[..., 0], d[..., 1]) * FPS / 2
    dropout = np.zeros(22, dtype=int)
    for f in range(N):
        if not graded[f]:
            continue
        for p in range(22):
            if rng.random() < 1 / (60.0 * FPS):          # identity switch / new tracklet
                team = p // 11
                free[team].append(owner[p])
                owner[p] = free[team].pop(int(rng.integers(len(free[team]))))
            if dropout[p] > 0:
                dropout[p] -= 1
                continue
            if rng.random() < 0.002:
                dropout[p] = int(rng.integers(10, 80))
            slots[f, owner[p], 0:2] = noisy[f, p] / [L, W]
            slots[f, owner[p], 2] = spd[f, p]
    return ball_obs / [L, W], slots


def write_meta(path):
    teams = "\n".join(
        '      <Team id="T{}"><Name>{}</Name></Team>'.format(i, n)
        for i, n in enumerate((HOME_NAME, AWAY_NAME)))
    players, channels, refs = [], [], []
    for slot in range(2 * SLOTS_PER_TEAM):
        team = 0 if slot < SLOTS_PER_TEAM else 1
        pid = "P{}".format(slot + 1)
        players.append('      <Player id="{}" teamId="T{}"><Name>Track {}</Name>'
                       '<ShirtNumber>{}</ShirtNumber></Player>'.format(pid, team, slot + 1,
                                                                       slot + 1))
        for ch in ("x", "y", "speed"):
            cid = "{}_{}".format(pid, ch)
            channels.append('        <PlayerChannel id="{}" playerId="{}" channelId="{}"/>'
                            .format(cid, pid, ch))
            refs.append('            <PlayerChannelRef playerChannelId="{}"/>'.format(cid))
    xml = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>25</FrameRate>
      <TrackingDataStyle><FieldSize><Width>105</Width><Height>68</Height></FieldSize></TrackingDataStyle>
    </GlobalConfig>
    <Teams>
{teams}
    </Teams>
    <Players>
{players}
    </Players>
    <Devices><Device id="device1"><Sensors><Sensor id="position"><Channels>
{channels}
    </Channels></Sensor></Sensors></Device></Devices>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification separator=":" startFrame="1" endFrame="{n}">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        <SplitRegister separator=",">
{refs}
        </SplitRegister>
      </SplitRegister>
      <SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="speed"/></SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
""".format(teams=teams, players="\n".join(players), channels="\n".join(channels),
           refs="\n".join(refs), n=N)
    with open(path, "w", encoding="utf-8") as h:
        h.write(xml)


def fmt(v):
    return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)


def write_raw(path, ball, slots):
    bsp = np.full(N, np.nan)
    with open(path, "w", newline="\n") as h:
        for f in range(N):
            groups = ";".join("{},{},{}".format(fmt(a), fmt(b), fmt(c)) for a, b, c in slots[f])
            h.write("{}:{}:{},{},{}\n".format(f + 1, groups, fmt(ball[f, 0]), fmt(ball[f, 1]),
                                             fmt(bsp[f])))


def write_tagging(path, restarts, phases, rng):
    inst = []

    def add(code, s, e, labels):
        lab = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                      for g, t in labels)
        inst.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>"
                    "{}</instance>".format(len(inst) + 1, max(0, s) / FPS, e / FPS, code, lab))

    for f, team, kind in restarts:
        if kind != "Kick Off (Start)" and rng.random() < 0.12:
            continue                                   # false negative
        c = f + rng.normal(0, 12)
        team_txt = "N/A" if kind == "Throw-in" else TAG_NAMES[team]
        add("SET PIECES", c - 3 * FPS, c + 2 * FPS, [("Team", team_txt), ("Type", kind)])
    for _ in range(25):                                # false positives
        c = rng.integers(11185, 190000)
        add("SET PIECES", c - 3 * FPS, c + 2 * FPS, [("Team", TAG_NAMES[int(rng.integers(2))]),
                                                    ("Type", "Free Kick")])
    attack = ["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES"]
    defend = ["DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD"]
    for s, e, team in phases:
        if e - s < 6 * FPS or rng.random() < 0.4:
            continue
        a = s + int(rng.uniform(0, 0.3) * (e - s))
        b = e - int(rng.uniform(0, 0.3) * (e - s))
        tagged = team if rng.random() > 0.15 else 1 - team
        if rng.random() < 0.7:
            add(str(rng.choice(attack)), a, b, [("Team", TAG_NAMES[tagged])])
        else:
            add(str(rng.choice(defend)), a, b, [("Team", TAG_NAMES[1 - tagged])])
    with open(path, "w", encoding="utf-8") as h:
        h.write('<?xml version="1.0" encoding="utf-8"?>\n<file><ALL_INSTANCES>\n')
        h.write("\n".join(inst))
        h.write("\n</ALL_INSTANCES></file>\n")


def write_homography(path):
    with open(path, "w", newline="\n") as h:
        h.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
        row = "1800,-500,40,2,170,250,-0.02,-0.5,1,0.25,1"
        h.writelines("{},{}\n".format(f, row) for f in range(N))


def write_gt(path, label):
    names = np.array(["H", "A", "D"])
    with open(path, "w", newline="\n") as h:
        h.write("frame,label\n")
        h.writelines("{},{}\n".format(f, names[c] if c >= 0 else "X")
                     for f, c in enumerate(label))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "synthetic_data"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tac = os.path.join(args.out, "tactical")
    os.makedirs(tac, exist_ok=True)
    os.makedirs(os.path.join(args.out, "gt"), exist_ok=True)

    print("simulating ...", flush=True)
    label, ball, carrier, in_flight, restarts, phases, anchor = simulate(rng)
    pos = players_from(label, ball, carrier, anchor, rng)
    ball_obs, slots = observe(label, ball, in_flight, pos, rng)
    print("writing ...", flush=True)
    stem = os.path.join(tac, "BAR-ATH_tactical")
    write_meta(stem + "_atd.xml")
    write_raw(stem + "_atd.txt", ball_obs, slots)
    write_tagging(stem + "_smart_tagging.xml", restarts, phases, rng)
    write_homography(stem + "_homography.csv")
    write_gt(os.path.join(args.out, "gt", "synthetic_possession_gt.csv"), label)
    graded = label >= 0
    print("done: {} graded frames, ball visible {:.1f}%, {} restarts".format(
        int(graded.sum()), 100 * np.isfinite(ball_obs[graded, 0]).mean(), len(restarts)))
