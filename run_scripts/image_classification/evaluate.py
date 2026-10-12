"""Evaluate a previously selected candidate through the same monitored gateway as evolution.

Example: python evaluate.py --candidate runs/example/best_candidate.json --model 4b --resolution 28
Additional options are passed to run_experiment.py. A single candidate is evaluated
directly; repeated --evolution-run selects initial and descendant winners on train,
then evaluates both on test with --final-repeats and --final-max-images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int | None:
    """Resolve the saved candidate and reuse the launcher's gateway lifecycle and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    candidates = parser.add_mutually_exclusive_group(required=True)
    candidates.add_argument("--candidate", help="Exported JSON candidate containing prompt and sampling")
    candidates.add_argument("--program", help="Explicit trusted Python candidate defining entrypoint()")
    candidates.add_argument("--evolution-run", action="append", help="Evolution run for independent train selection of initial and evolved winners; repeat for a cohort")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    arguments, launcher_arguments = parser.parse_known_args(argv)
    from med_evolve.launcher import main as run_experiment

    if arguments.evolution_run:
        sources = [item for run_dir in arguments.evolution_run for item in ("--evolution-run", run_dir)]
        return run_experiment([*sources, "--split", arguments.split, *launcher_arguments])
    return run_experiment(["--evaluate", arguments.candidate or arguments.program, "--split", arguments.split, *launcher_arguments])


if __name__ == "__main__":
    sys.exit(main())
