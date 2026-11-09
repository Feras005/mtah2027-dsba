"""Regenerate the submission in one command.

    python run_pipeline.py                                   # data in ./data (or MTAH_DATA_DIR)
    python run_pipeline.py --data-dir synthetic_data --out evaluation/results/synthetic.csv

Writes the CSV, then runs evaluation/validate_submission.py on it. Deterministic: the same data
gives the same file, byte for byte.
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession.io import write_submission  # noqa: E402
from possession.pipeline import run  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=None, help="data bundle root (default: games.DATA_DIR)")
    p.add_argument("--out", default=os.path.join(HERE, "evaluation", "results",
                                                 "DSBA.csv"))
    p.add_argument("--no-tagging", action="store_true", help="ignore the Smart Tagging")
    p.add_argument("--abstain", action="store_true",
                   help="write X on long ball occlusions inside possessions (see SOLUTION.md: "
                        "free on spellF1, costs macroF1)")
    p.add_argument("--no-shape", action="store_true", help="v1: no team-shape models")
    p.add_argument("--snap", action="store_true", help="touch-based boundary snapping (off: "
                   "unvalidated, see SOLUTION.md)")
    args = p.parse_args()

    labels = run(args.data_dir, use_tagging=not args.no_tagging, abstain=args.abstain,
                 use_shape=not args.no_shape, snap=args.snap)
    path = write_submission(args.out, labels)
    print("wrote {}\n".format(os.path.relpath(path, HERE)))
    sys.exit(subprocess.call([sys.executable, os.path.join(HERE, "evaluation",
                                                           "validate_submission.py"), path]))
