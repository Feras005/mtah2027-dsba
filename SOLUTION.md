# Possession without the ball — DSBA

**Team DSBA** — Feras Alsadadi, Talal Kamal, Ali Mohamed, Mohammed Hejairi

More than a Hack 2027, Phase 1: frame-level possession (`H` / `A` / `D` / `X`) for BAR-ATH.

**In one paragraph.** The ball is missing on 41% of graded frames, mostly at the moments that
decide possession. We read possession from the players as well as the ball. Two models trained
on **real labels from three other matches** (Metrica Sports' public sample data) answer the
decoder's two questions — *is the ball in play?* from restart formations and crowd dynamics, and
*which team has it?* from team shape plus the ball, trained with the ball hidden at random. A
third, ball-free shape model is trained on **pseudo-labels** from this match. Their evidence is
decoded by a **3-state HMM (H/A/D)** that pays a high price for switching teams, and hand-overs
are snapped to the touch that caused them. Where the tactical tracker loses the ball, the
**broadcast feed, aligned by player positions**, fills in. Every setting was chosen by
**leave-one-game-out spellF1 on the external matches**, and checked on this match with a
ground-truth-free **self-consistency spellF1**. No competition ground truth was used to choose
anything.

---

## Reproduce

```bash
py -m venv .venv                                   # Python 3.12 (Linux/macOS: python3, .venv/bin/python)
.venv/Scripts/python -m pip install -r requirements-solution.txt
# unpack the data bundle into ./data  (data/tactical/..., data/broadcast/...)
.venv/Scripts/python fetch_external.py             # Metrica public sample data, pinned commit (needs git)
.venv/Scripts/python run_pipeline.py               # -> evaluation/results/DSBA.csv, then validates it
```

* One command after the two downloads, about 6.5 min on a laptop the first time (parsing, the
  broadcast alignment and the feature matrices are then cached under `data/**/.cache/`,
  gitignored).
* **Deterministic.** Two runs give the identical file (MD5 `2d6c737b376dc0fd110fd681b913adb2`).
* `python run_external.py study|tune|extras` re-runs the studies on the external matches,
  `python run_consistency.py` the self-consistency test, `python run_broadcast.py` the alignment.
* `--no-external`, `--no-broadcast`, `--no-snap` switch each v3 component off; with all three
  the pipeline is the v2 that was submitted first.

## Assets used

| asset | how |
|---|---|
| tactical ATD | everything: ball, proximity, team shape, formations, speeds |
| Smart Tagging | set pieces → restart anchors; phases → weak team prior (direction learned); kickoff side |
| broadcast ATD | **ball only**: aligned to the tactical timeline by player positions, fills ball gaps |
| homography, videos | **not used**: both ATDs are already in pitch coordinates |
| public 5-min ground truth | **only** to check that frames line up and the file is well formed. No setting was chosen with it |
| Metrica Sports public sample data | **external training and validation data**, see below |

## Training data

| dataset | source | labelled by | size |
|---|---|---|---|
| Metrica Sports sample games 1–3 | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data), commit `e706dd5`, public, anonymised; used with acknowledgement as its README asks | Metrica Sports' event data, converted to H/A/D by a fixed rule (`possession/external.py`) | 3 matches, 427,894 labelled frames (34% dead), 1,044 possessions |
| pseudo-labels on BAR-ATH | the tactical ATD and Smart Tagging of this match | a fixed rule in `possession/pseudo.py`, no person | 15,283 team frames, 16,759 dead frames |

No hand labels on the competition game, no ground truth (public or private), no pre-trained
models. The unsupervised parts (crowd-speed mixture, phase-tag direction) are fitted on this
match's tracking without labels. The PFF FC World Cup 2022 data (64 matches) would be the next
addition; it needs a sign-up and was not used.

**Labels from events** (`external.py`): the team in possession is the team of the latest on-ball
event (pass, carry, shot, recovery, ball lost, set piece), so possession changes at the first
on-ball event of the other side — the interception a `RECOVERY` marks. The ball is dead from a
`BALL OUT`, a shot that goes out or in, a foul or an offside, until the restart that follows
(`SET PIECE`, or a goal-kick pass).

---

## Pipeline

```
tactical ATD --> ball features (proximity, speed, OOB) -------------------------+
 |      ^                                                                        |
 |      +-- broadcast ATD, aligned by player positions, fills ball gaps          |
 |                                                                               |
 +--> team shape --> pseudo-label shape model (this match, cross-half) ----------+
 |                                                                               |
 +--> formations + crowd dynamics --> DEAD model (trained on 3 other matches) ---+
 |                                                                               |
 +--> shape + ball (hidden at random) --> TEAM model (trained on 3 other matches)+
 |                                                                               |
 +--> collective speed --> rule-based dead-ball evidence ------------------------+
                                                                                 |
Smart Tagging --> set-piece restarts, phase prior -------------------------------+
                                                                                 v
                                   3-state HMM / Viterbi --> snap to touches --> DSBA.csv
```

### 1. Ball features (`possession/features.py`)

* **Ball cleaning:** drop implausible and teleporting detections (faster than 40 m/s). Bridge
  occlusions of ≤ 0.4 s.
* **Proximity log-odds:** `clip(d_away − d_home, ±3)`, using each team's nearest player, scaled
  down when nobody is within ~2.5 m and when the ball is in flight (> 7 m/s). A loose ball carries
  no evidence, so possession stays with the last team, as CHALLENGE.md asks.
* **Out of bounds:** the ball is seen more than 0.75 m outside the lines.

### 2. Broadcast ball (`possession/broadcast.py`)

The broadcast ATD is on its own timeline, with cuts and replays, and no mapping is provided. But
both feeds are real-time recordings of the same match, so live broadcast footage sits at a fixed
frame offset from the tactical video. We find it from the players alone: for 400 broadcast
frames, the cost of each tactical frame is the median distance from each broadcast player to the
nearest tactical player (one-way, since the TV camera sees a subset). 391 of 400 sampled frames
matched, voting for exactly two offsets, one per half (4,014 and 26,914 frames). Each broadcast
frame is then accepted only where its players match within 1 m; replays and close-ups fail.

* Aligned on 57% of graded frames; where both feeds see the ball they agree to 1.2 m (median).
* It fills the ball on 3.6% of graded frames (59.3% → 63.0%): the broadcast mostly loses the ball
  when the tactical tracker does.
* **Check:** hide the tactical ball on 40% of frames and refill from the broadcast. Against the
  decode with the ball visible, spellF1 rose on all three seeds (0.873 → 0.887, 0.838 → 0.874,
  0.851 → 0.899). On the real match it changes 0.6% of labels.

### 3. Models trained on other matches (`possession/learned.py`, `possession/transfer.py`)

Gradient-boosted trees (`HistGradientBoostingClassifier`), one per decoder question.

**Dead model — P(ball out of play), ball-free on purpose.** In the sample games the ball is
almost never tracked while play is stopped (7–21% of dead frames, against ~65% in BAR-ATH), so
a model that saw the ball would learn "no ball = stoppage", an artefact that would not transfer.
It learns stoppages from **positive evidence in the players**, as restart formations:

* throw-in: the nearest player to a touchline, players beyond the lines
* corner: the nearest player to a corner flag, players in each penalty box
* goal kick: players in the six-yard boxes, nearest player to a goal line
* kick-off: players in the centre circle, mean distance from the halfway line
* the whole team slowing down and staying slow: crowd speed now, over 3/6/12 s, the 6 s minimum,
  the 3 s before and after (the pick-up at a restart), players standing still
* plus the 55 team-shape features of section 4

**Team model — P(home | alive).** Team shape plus the ball features (proximity, distances, ball
speed, whether the ball is known). In training, the ball is **hidden on random 1–8 s blocks
covering 40% of frames**, so the model learns to fall back on shape when the ball is gone.

Leave-one-game-out (trained on two matches, tested on the third):

| held-out game | dead model AUC | team model AUC |
|---|---|---|
| Sample game 1 | 0.832 | 0.921 |
| Sample game 2 | 0.869 | 0.945 |
| Sample game 3 | 0.897 | 0.942 |

For BAR-ATH, both are trained on all three matches. Their log-odds enter the decoder at weights
0.2 (dead) and 0.5 (team), chosen as below.

### 4. Pseudo-label team-shape model (`possession/shape.py`, `possession/shape_model.py`)

Unchanged from v2: 55 ball-free features per frame (centroid, width, length, spread, defensive
line, most advanced player, players in the opponent half, speed, push towards goal, free space;
home-minus-away, 1 s mean, 2 s trend), oriented by an attacking direction inferred per half,
trained **cross-half** on this match's pseudo-labels (AUC 0.934 / 0.940 against the
pseudo-labels; logistic regression 0.906 / 0.913). Its top features say **the team with the ball
is the wider one**. Weight 0.2, gated off when a player is clearly on the ball.

Pseudo-labels (`possession/pseudo.py`): *team* where the ball is observed and slow (< 5 m/s), a
player is within 1 m and no opponent within 3 m; *dead* where the ball is seen out, in a tagged
set piece before its estimated restart, or before the published second-half kickoff.

### 5. Rule-based dead-ball evidence (`possession/deadball.py`, `possession/tagging.py`)

Log-odds of dead vs live per frame: ball seen out of bounds +4; a 2-component Gaussian mixture on
log crowd speed fitted without labels (slow 0.66 m/s, fast 1.54 m/s); ball rolling inside the
pitch −0.7, ball unseen +0.15; tagged set pieces dead before the estimated restart and live after.
The restart is estimated per set piece as the largest rise in crowd speed, because set-piece tags
are fixed 35 s windows in which play resumes only 20–30% of the way in.

### 6. Smart Tagging phases, with self-calibrated direction

At most ±0.25 log-odds per frame towards the tagged team, with the **sign learned**: on frames
where proximity is confident, every `DEFENDING …` code came out negative (the tag names the team
*without* the ball), `BUILD UP` and `PROGRESSION` positive.

### 7. Decoder (`possession/decode.py`) and snapping (`possession/touches.py`)

* **States:** H, A, D, decoded with Viterbi per half. Team evidence only decides *which* team; it
  never makes "alive" more likely than "dead", which is left to the dead-ball evidence.
* **Transition costs** (log-likelihood units): H ↔ A 14; live → dead 7 (1.5 near a ball seen
  out); dead → live 7 (2 within 3 s of an estimated restart). Possessions under 0.8 s and
  stoppages under 1.2 s are absorbed.
* **Snapping:** each hand-over is moved to the first touch by the new team after the old team's
  last touch, each restart to the first touch by the restarting team, each stoppage to the first
  frame the ball is seen out — if one lies within 1 s. A touch is a jump in the ball's velocity
  (≥ 4 m/s) with a player within 2 m.
* **Half boundaries:** half 1 starts live, biased to the tagged kickoff side; half 2 is dead until
  the published kickoff (frame 113,742, FAQ.md).

### Possession rules as implemented

* **Loose or contested ball:** no team evidence, so the last team keeps it.
* **Change of possession:** needs *sustained* control by the other side (the switch cost), and
  is then placed at the new team's first touch.
* **Restarts:** the set piece anchors the moment play resumes; the side comes from the tag where
  it names one, otherwise from proximity. Throw-ins are tagged `N/A`.
* **Dead ball:** always `D`. We never assign a team to a dead ball.

---

## How settings were chosen

### On real labels: leave-one-game-out spellF1 on the external matches (`run_external.py`)

The pipeline runs on each sample match exactly as on BAR-ATH (without Smart Tagging, which they
do not have), with the learned models trained on the other two, and is scored with the
**official scorer** against the event labels.

| configuration | mean spellF1 | per game | possessions (truth 348) |
|---|---|---|---|
| v2 pipeline, as first submitted | 0.540 | 0.540 / 0.550 / 0.528 | 260 |
| + dead model, weight 0.2 | 0.563 | 0.560 / 0.576 / 0.553 | 261 |
| + team model, weight 0.5 | 0.560 | 0.552 / 0.563 / 0.564 | 276 |
| + both | 0.571 | 0.554 / 0.577 / 0.583 | 273 |
| **+ snapping to touches (v3)** | **0.581** | 0.556 / 0.588 / 0.601 | 270 |
| + self-training, 1 round | 0.565 | 0.556 / 0.563 / 0.577 | 265 |
| + self-training, 2 rounds | 0.566 | 0.559 / 0.566 / 0.574 | 263 |

**Decoder settings.** The costs we had set by hand as football priors were tuned by coordinate
ascent over a grid (two rounds): switch cost 5–18, stop and restart costs 3–12, minimum stoppage
0.4–3 s, minimum possession 0.2–1.2 s, model weights. **The hand-set values were already the
best**: switch cost 14 beat 5 (0.558), 9 (0.560), 11 (0.565) and 18 (0.571); stop and restart
cost 7 beat 3, 5, 9 and 12. Two changes scored marginally higher (minimum stoppage 2 s, +0.0024;
shape weight 0.4, +0.003); we only adopt a change worth at least 0.005, since smaller ones cannot
be told apart from fitting three matches, so neither was taken.

### On this match, without ground truth: self-consistency spellF1 (`run_consistency.py`)

Decode the match with the ball as tracked, decode it again with the ball hidden on random 1–8 s
blocks covering 40% of frames, and score the second against the first with the official scorer.
It scores whole possessions, so a possession split by a false stoppage shows up directly. On its
own it cannot choose settings — a decoder that never changes its mind agrees with itself
perfectly — so it is a check, not the target.

| version | self-consistency spellF1 (3 seeds) | possessions, ball visible |
|---|---|---|
| v2 | 0.825 | 301 |
| + learned models | 0.851 | 325 |
| **+ snapping (v3)** | **0.854** | 323 |

v3 is more stable *and* finds more possessions, so the gain is not from switching less.

### Earlier: frame-level ball masking (`possession/validate.py`)

The v2 settings (shape weight 0.2, team-only emissions, restart estimate) were chosen by hiding
the ball around pseudo-labelled frames and scoring recovery (0.699 → 0.780 balanced accuracy).
It checks frames, not possessions, which is why the two tests above were added.

### Tried and dropped

* **Self-training** (decode, take frames ≥ 2 s inside decoded runs of ≥ 6 s as new labels,
  retrain, repeat): ~107,000 extra frames per match, but spellF1 *fell* (0.571 → 0.565 / 0.566).
  The model learns its own mistakes back.
* **A dead-ball shape model on pseudo-labels:** lowered the v2 masking score at every weight; the
  dead model trained on real labels replaces it.
* **Writing `X` mid-possession** (`--abstain`): the scorer forward-fills `X`, so spellF1 cannot
  change, while macroF1 drops.
* **Where the public ground truth places a switch** (at the touch or at the pass): not studied.
  The touch convention follows from the event labels of the external matches.

## Result

* **Output:** 327 possessions (mean 12.1 s, median 7.5 s), 121 stoppages, 38% of graded frames `D`.
* **Public 5 minutes:** spellF1 0.518 (v2: 0.422) vs 0.326 for the published floor
  `baseline_smooth50`; macroF1 0.80 (v2: 0.68). A correctness check only: 17 possessions cannot
  rank anything, and nothing was chosen with it.

## Limitations, and what we would do next

* **Labels from another provider's events** are not the competition's ground truth: where a
  switch is placed, and how a stoppage is cut, may differ. More matches (PFF FC World Cup 2022, 64
  games) would also shrink the noise on every comparison above.
* **Possessions are still under-counted** on the external matches (270 vs 348): the missing ones
  are short, and lowering the switch cost to catch them costs more than it gains.
* **The dead model sees no ball**, by design; it cannot use a ball last seen near a line, because
  in the training matches the ball is not tracked during stoppages.
* **The broadcast** adds little ball (3.6% of frames), because both feeds tend to lose it at the
  same moments.
* **Smart Tagging** cannot be validated externally, since the sample matches have none.

## Synthetic data

`synthetic/make_synthetic.py` simulates a full match in the bundle's exact file formats, with its
own ground truth. We used it to build and debug the pipeline before the gated data arrived,
because the competition terms forbid sharing the match data with cloud tools. It is not a model
of the real match and was not used to choose any setting.

## Data handling

`data/` (competition bundle, external sample data, caches) and `synthetic_data/` are gitignored.
The pipeline prints only counts, shares, fitted aggregates and model metrics, never tracking
rows.
