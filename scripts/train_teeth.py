"""Explicit Linux-only entry point; --help and imports do not load Torch."""

import argparse
from pathlib import Path
import sys

# Avoid import bytecode writes outside the authorized runtime output workspace.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teeth_local.training import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True, help="new directory; parent must exist")
    parser.add_argument("--authorize-workspace", action="store_true",
                        help="confirm prior authorization; this flag does not grant permission")
    parser.add_argument("--train-record")
    parser.add_argument("--val-record")
    parser.add_argument("--variant", choices=("B", "C"), default="B")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--synthetic-smoke", action="store_true",
                        help="two synthetic steps only; no checkpoint and no real records")
    arguments = vars(parser.parse_args())
    arguments["authorized"] = arguments.pop("authorize_workspace")
    result = run(**arguments)
    print(result["status"])


if __name__ == "__main__":
    main()
