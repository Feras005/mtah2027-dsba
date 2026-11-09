"""Align the broadcast feed to the tactical timeline and report how much ball it adds.

    python run_broadcast.py

Prints offsets, match rates and coverage only. Caches the alignment in data/broadcast/.cache.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from games import GAME  # noqa: E402
from possession import broadcast, pipeline  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    data = pipeline.load(say=pipeline.quiet)
    broadcast.filled_ball(data, GAME, os.path.join(HERE, "data", "broadcast", ".cache"))
    print("({:.0f} s)".format(time.time() - t0))
