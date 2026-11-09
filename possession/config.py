"""Every tunable number in the pipeline, in one place, with the reason it has the value it has.

None of these were fitted to the public 5-minute ground truth (RULES.md forbids it). They are
football / physics priors, stated in metres, seconds and "evidence units" (log-likelihood ratios),
so each can be argued about on its own terms. Values that depend on the recording -- what "slow"
means for the crowd speed, how reliable each Smart Tagging code is -- are not set here at all:
they are estimated from the provided data without labels (see features.py, tagging.py).
"""

FPS = 25

# Ball track cleaning ------------------------------------------------------------------------
# A ball reported further than this outside the pitch is a false detection (crowd, sideline),
# not a ball that went out. Metres.
BALL_PLAUSIBLE_MARGIN = 6.0
# Ball occlusions up to this many frames are bridged by linear interpolation: a 0.4 s gap is a
# pass in flight or a player standing in front of the ball, not a change of state.
MAX_INTERP_FRAMES = 10
# Interpolated positions are believed less than observed ones.
INTERP_WEIGHT = 0.5
# Physical ceiling on ball speed. Faster than ~40 m/s between two detections is a jump to a
# different object, and that detection is dropped.
MAX_BALL_SPEED = 40.0

# Team-proximity evidence -------------------------------------------------------------------
# A player this close to the ball is plausibly playing it. Beyond ~2.5 m nobody controls it.
CONTROL_RADIUS = 2.5
CONTROL_SOFTNESS = 0.5
# Log-odds per metre of difference between the two teams' nearest players.
PROXIMITY_SLOPE = 1.0
PROXIMITY_CAP = 3.0
# A ball moving faster than a dribble is in flight -- a pass, a shot, a clearance -- and whoever
# happens to be near it says little about who owns it.
FLIGHT_SPEED = 7.0
FLIGHT_SOFTNESS = 1.5
FLIGHT_FLOOR = 0.15

# Dead-ball evidence (log-likelihood ratio of dead vs alive) ---------------------------------
# Ball visibly outside the pitch by more than this many metres: the ball is out.
OOB_MARGIN = 0.75
OOB_LLR = 4.0
# Collective player speed: a 2-component mixture is fitted to the log of the crowd speed over the
# graded frames, without labels. The slow component is "players walking", which is what a
# stoppage looks like. Its LLR is weighted down because slow phases also happen in live play.
CROWD_WINDOW_S = 2.0
CROWD_WEIGHT = 0.6
CROWD_CAP = 3.0
# Ball not seen and not bridged: weak evidence of a stoppage (ball off camera, in a hand).
MISSING_LLR = 0.15
# Ball visibly moving inside the pitch: weak evidence the game is on.
MOVING_SPEED = 3.0
MOVING_LLR = -0.7
# Smart Tagging set piece: the tag marks a ~5 s window around the restart. Before its middle the
# ball is still dead, after it the ball is live.
SET_PIECE_LLR = 0.8

# Smart Tagging phases as a team prior -------------------------------------------------------
# Per-frame log-odds a phase tag may add towards its team, after its reliability has been
# measured against the proximity evidence (tagging.py). Small: tags are phases, not possessions.
PHASE_MAX_LLR = 0.25
PHASE_MIN_FRAMES = 200  # a code needs this much overlapping evidence to be trusted at all

# Viterbi transition costs (negative log-probabilities, in the same evidence units) --------------
# Changing team is expensive: a touch, a deflection or a lost aerial duel must not flip
# possession, only sustained control by the other side.
SWITCH_COST = 14.0
# Going dead / coming back to life.
STOP_COST = 7.0
RESTART_COST = 7.0
# ... much cheaper where the data says it happens: ball out of bounds, tagged set piece.
STOP_COST_AT_OOB = 1.5
RESTART_COST_AT_SET_PIECE = 2.0
OOB_ANCHOR_FRAMES = 12

# Post-processing ----------------------------------------------------------------------------
# No real possession or stoppage is this short; shorter runs are absorbed by their neighbours.
MIN_POSSESSION_S = 0.8
MIN_DEAD_S = 1.2

# Abstention (only with --abstain) -----------------------------------------------------------
# Write X on ball-occluded frames inside a possession, but never within this distance of a
# decoded hand-over or stoppage boundary, where a committed label is worth more.
ABSTAIN_GUARD_S = 2.0
ABSTAIN_MIN_GAP_FRAMES = 25

# Team names in the Smart Tagging XML -> side. Matched case-insensitively by substring.
HOME_TAG_TOKENS = ("barcelona", "barça")
AWAY_TAG_TOKENS = ("atletic", "athletic", "bilbao")

# Pseudo-labels (pseudo.py) --------------------------------------------------------------------
PSEUDO_NEAR = 1.0             # m: a player this close to a slow, observed ball is on it
PSEUDO_FAR = 3.0              # m: ...and nobody from the other team is anywhere near
PSEUDO_MAX_BALL_SPEED = 5.0   # m/s: faster than this the ball is travelling, not at someone's feet

# Shape models (shape_model.py) -----------------------------------------------------------------
SHAPE_TREES = 300
SHAPE_MAX_TRAIN = 60000       # frames per fit, a seeded random subset
SHAPE_LOGIT_CAP = 4.0
# Per-frame weight of the shape models' log-odds in the HMM. Frames are strongly correlated, so a
# full-weight logit on every frame would be wildly overconfident. Chosen with the ball-masking
# validation (validate.py), never with the ground truth: 0.2 and 0.4 recover hidden-ball frames
# equally well (balanced 0.780 / 0.781) and 0.2 switches less, so 0.2. The dead-ball shape model
# lowered the validation score at every weight tried (0.05-0.2), so it is off.
SHAPE_TEAM_WEIGHT = 0.2
SHAPE_DEAD_WEIGHT = 0.0

# Touches and boundary snapping (touches.py) ---------------------------------------------------
TOUCH_LAG = 2                 # frames either side for the velocity change
TOUCH_DV = 4.0                # m/s change of ball velocity that counts as a touch
TOUCH_RADIUS = 2.0            # m: someone must be this close for it to be a touch
TOUCH_NMS = 4                 # frames: one touch per local maximum
SNAP_WINDOW = 25              # frames (1 s) either side of a decoded boundary
