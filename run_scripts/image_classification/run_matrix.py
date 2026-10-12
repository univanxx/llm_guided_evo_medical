"""Run two MedGemma models at four resolutions with independent search seeds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    """Execute the matrix sequentially and stop immediately on a failed search."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("4b", "27b"), default=["4b", "27b"])
    parser.add_argument("--resolutions", nargs="+", type=int, choices=(28, 64, 128, 224), default=[28, 64, 128, 224])
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--evaluator-model-4b", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--evaluator-model-27b", default="google/medgemma-27b-it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating runs or contacting endpoints")
    args, extra = parser.parse_known_args()
    reserved = {"--run-dir", "--model", "--resolution", "--seed", "--evaluator-model", "--evaluate", "--evolution-run", "--resume"}
    if any(item.split("=", 1)[0] in reserved for item in extra):
        parser.error("Matrix-owned options cannot be passed as launcher overrides")
    for name in ("models", "resolutions", "seeds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Duplicate {name} would reuse a run directory")
    for model in args.models:
        for resolution in args.resolutions:
            for seed in args.seeds:
                destination = args.output_root.resolve() / f"medgemma-{model}-r{resolution}-s{seed}"
                command = [sys.executable, "-B", str(ROOT / "run_experiment.py"), "--model", model, "--resolution", str(resolution), "--seed", str(seed), "--run-dir", str(destination), "--evaluator-model", getattr(args, f"evaluator_model_{model}"), *extra]
                if args.resume and (destination / "run_config.json").exists():
                    command.append("--resume")
                print(json.dumps(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
