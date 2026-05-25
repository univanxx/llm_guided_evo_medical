#!/usr/bin/env python3
"""Sequential multi-run launcher for image_classification.

Runs N_RUNS independent MAP-Elites evolution experiments back-to-back.
Between runs the per-run MAP-Elites archive is cleared so each run starts
fresh (evaluated programs are preserved in Redis under unique key prefixes).

Environment variables (set by run.sh or exported manually):
  EVOLUTION_LLM_URL     base URL of the evolution LLM (required)
  EVOLUTION_MODEL_NAME  served model name (required)
  REASONING_EFFORT      low | medium | high  (default: medium)
  REDIS_PORT            Redis port            (default: 44004)
  GIGAEVO_CORE_DIR      path to gigaevo-core checkout
                        (default: ../../../gigaevo-core relative to this file)
  CONDA_ENV             conda environment name (default: base)
  USE_CONDA_RUN         1 to wrap with `conda run`, 0 if already inside env
                        (default: 0)
  MAX_GENERATIONS       generations per run   (default: 30)
  N_RUNS                number of sequential runs (default: 3)
  LOGS_DIR              directory for log files
  OUTPUTS_DIR           directory for Hydra outputs
"""
import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime


# =============================================================================
# Config — overridable via environment variables
# =============================================================================

PROBLEM_NAME = "image_classification"

MAX_GENERATIONS = int(os.environ.get("MAX_GENERATIONS", "30"))
N_RUNS = int(os.environ.get("N_RUNS", "3"))

_script_dir = Path(__file__).resolve().parent
GIGAEVO_CORE_DIR = Path(
    os.environ.get("GIGAEVO_CORE_DIR", str(_script_dir.parent.parent.parent / "gigaevo-core"))
)

CONDA_ENV = os.environ.get("CONDA_ENV", "base")
USE_CONDA_RUN = os.environ.get("USE_CONDA_RUN", "0") not in ("0", "false", "False")

REDIS_PORT = int(os.environ.get("REDIS_PORT", "44004"))

LOGS_DIR = Path(os.environ.get("LOGS_DIR", str(GIGAEVO_CORE_DIR.parent / "logs")))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(GIGAEVO_CORE_DIR.parent / "outputs_sequential_runs")))

STOP_ON_ERROR = True

LEGACY_ISLAND_ID = "fitness_island"
CLEAR_LEGACY_ARCHIVE = True

# Evolution LLM — read from env (set by run.sh)
_EVOLUTION_LLM_URL = os.environ.get("EVOLUTION_LLM_URL", "")
_EVOLUTION_MODEL_NAME = os.environ.get("EVOLUTION_MODEL_NAME", "")
_REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "medium")


# =============================================================================
# Logging
# =============================================================================

def setup_main_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"run_many_{PROBLEM_NAME}_{ts}.log"

    logger = logging.getLogger("run_many")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    logger.info("Main log: %s", log_path)
    return logger


# =============================================================================
# Redis / MAP-Elites cleanup
# =============================================================================

def _archive_keys(island_id: str) -> list[str]:
    prefix = f"island_{island_id}"
    return [f"{prefix}:archive", f"{prefix}:archive:reverse"]


def clear_map_elites_archive(island_id: str, logger: logging.Logger) -> None:
    """Delete MAP-Elites archive keys for island_id (does not remove program records)."""
    keys = _archive_keys(island_id)
    logger.info("Clearing MAP-Elites archive keys: %s", keys)

    result = subprocess.run(
        ["redis-cli", "-p", str(REDIS_PORT), "DEL", *keys],
        cwd=GIGAEVO_CORE_DIR,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("Failed to clear archive for island_id=%s", island_id)
        logger.error("stderr: %s", result.stderr.strip())
        raise RuntimeError(f"Failed to clear MAP-Elites archive for island_id={island_id}")

    logger.info("Cleared archive for island_id=%s: %s", island_id, result.stdout.strip())


# =============================================================================
# Run helpers
# =============================================================================

def make_run_id(run_idx: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{PROBLEM_NAME}__run{run_idx:03d}__{ts}"


def build_command(run_id: str) -> list[str]:
    base_cmd = [
        "python", "run.py",
        f"problem.name={PROBLEM_NAME}",
        "algorithm=single_island",
        f"max_generations={MAX_GENERATIONS}",
        f"redis.port={REDIS_PORT}",
        f"llm_base_url={_EVOLUTION_LLM_URL}",
        f"model_name={_EVOLUTION_MODEL_NAME}",
        f"reasoning_effort={_REASONING_EFFORT}",
        "max_tokens=8192",
        "stage_timeout=240",
        "dag_timeout=360",
        "max_mutations_per_generation=4",
        "max_concurrent_dags=5",
        "pipeline=evolution_aware",
        f"redis_storage.config.key_prefix={run_id}",
        f"island_id={run_id}",
        f"hydra.run.dir={OUTPUTS_DIR / run_id}",
    ]
    if USE_CONDA_RUN:
        return ["conda", "run", "-n", CONDA_ENV, *base_cmd]
    return base_cmd


def run_one_experiment(run_idx: int, logger: logging.Logger) -> int:
    run_id = make_run_id(run_idx)
    experiment_log = LOGS_DIR / f"{run_id}.txt"
    run_output_dir = OUTPUTS_DIR / run_id

    cmd = build_command(run_id)

    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_output_dir.parent.mkdir(parents=True, exist_ok=True)

    clear_map_elites_archive(run_id, logger)
    if CLEAR_LEGACY_ARCHIVE:
        clear_map_elites_archive(LEGACY_ISLAND_ID, logger)

    logger.info("=" * 80)
    logger.info("Starting run %d/%d  |  RUN_ID: %s", run_idx, N_RUNS, run_id)
    logger.info("Problem: %s  |  Max generations: %d", PROBLEM_NAME, MAX_GENERATIONS)
    logger.info("LLM: %s @ %s", _EVOLUTION_MODEL_NAME, _EVOLUTION_LLM_URL)
    logger.info("Redis port: %d", REDIS_PORT)
    logger.info("Hydra run dir: %s", run_output_dir)
    logger.info("Experiment log: %s", experiment_log)
    logger.info("Command: %s", " ".join(map(str, cmd)))

    start = time.monotonic()

    with experiment_log.open("w", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            cwd=GIGAEVO_CORE_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.monotonic() - start
    logger.info("Finished RUN_ID: %s | rc=%d | %.2fh / %.1fm", run_id, result.returncode, elapsed / 3600, elapsed / 60)

    if result.returncode != 0:
        logger.error("Run failed. See: %s", experiment_log)

    return result.returncode


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    logger = setup_main_logger()

    if not _EVOLUTION_LLM_URL or not _EVOLUTION_MODEL_NAME:
        logger.error(
            "EVOLUTION_LLM_URL and EVOLUTION_MODEL_NAME must be set. "
            "Source run.sh or export them manually before calling this script."
        )
        return 1

    logger.info("Sequential runner started")
    logger.info("gigaevo-core dir: %s", GIGAEVO_CORE_DIR)
    logger.info("N_RUNS=%d  MAX_GENERATIONS=%d  REDIS_PORT=%d", N_RUNS, MAX_GENERATIONS, REDIS_PORT)

    total_start = time.monotonic()
    failed = []

    for run_idx in range(1, N_RUNS + 1):
        rc = run_one_experiment(run_idx, logger)
        if rc != 0:
            failed.append((run_idx, rc))
            if STOP_ON_ERROR:
                logger.error("STOP_ON_ERROR=True, stopping after failed run %d", run_idx)
                break

    total_elapsed = time.monotonic() - total_start
    logger.info("=" * 80)
    logger.info("Total: %.2fh / %.1fm", total_elapsed / 3600, total_elapsed / 60)

    if failed:
        logger.error("Failed runs: %s", failed)
        return 1

    logger.info("All runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
