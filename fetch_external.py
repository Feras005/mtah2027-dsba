"""Download the external training data (Metrica Sports' public sample data), pinned to a commit.

    python fetch_external.py        # -> data/external/sample-data  (gitignored)

Three anonymised matches with tracking and event data, published by Metrica Sports for public
use ("if you use it for anything public, please acknowledge the source"). Needs git.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession.transfer import SAMPLE_DATA_COMMIT, SAMPLE_DATA_URL  # noqa: E402

DEST = os.path.join(HERE, "data", "external", "sample-data")


def run(*args, cwd=None):
    subprocess.check_call(list(args), cwd=cwd)


if __name__ == "__main__":
    if not os.path.isdir(os.path.join(DEST, ".git")):
        os.makedirs(os.path.dirname(DEST), exist_ok=True)
        run("git", "clone", "--quiet", SAMPLE_DATA_URL, DEST)
    run("git", "fetch", "--quiet", "origin", cwd=DEST)
    run("git", "checkout", "--quiet", SAMPLE_DATA_COMMIT, cwd=DEST)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DEST).decode().strip()
    print("external data at {} (commit {})".format(os.path.relpath(DEST, HERE), head[:12]))
    if head != SAMPLE_DATA_COMMIT:
        sys.exit("unexpected commit")
