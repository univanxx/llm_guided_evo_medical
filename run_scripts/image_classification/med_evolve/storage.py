"""Project-owned durable adapters for the pinned GigaEvo disk stores."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import uuid

from loguru import logger

from med_evolve.windows_compat import install_windows_flock

install_windows_flock()

from gigaevo.database.disk_program_storage import DiskProgramStorage
from gigaevo.exceptions import StorageError
from gigaevo.evolution.storage.archive_storage import DiskArchiveStorage
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.json import dumps as gigaevo_dumps


def _durable_replace(path: Path, content: str) -> None:
    """Replace one UTF-8 state file after file fsync and best-effort directory fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


class RecoverableDiskProgramStorage(DiskProgramStorage):
    """Fail closed on corrupt programs and tag stranded evaluations before requeue."""

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Persist program and status state through the project-owned durable writer."""
        _durable_replace(path, content)

    def _ensure_loaded(self) -> None:
        """Reject corrupt authoritative program JSON before upstream can skip it."""
        if not self._loaded and self._programs_dir.is_dir():
            for path in sorted(self._programs_dir.glob("*.json")):
                try:
                    Program.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except Exception as error:
                    raise StorageError(f"Corrupt authoritative program file {path}: {error}") from error
        super()._ensure_loaded()

    async def recover_stranded_programs(self) -> int:
        """Atomically mark recovered RUNNING programs so they resume the same pass."""
        self.require_writable("recover_stranded_programs")
        async with self._lock:
            self._ensure_loaded()
            running = ProgramState.RUNNING.value
            queued = ProgramState.QUEUED.value
            recovered = 0
            for program_id in sorted(self._status_sets.get(running, set())):
                program = self._programs.get(program_id)
                self._status_sets[running].discard(program_id)
                if program is None:
                    raise StorageError(f"RUNNING status references missing program {program_id}")
                program.state = ProgramState.QUEUED
                program.metrics = {}
                program.stage_results = {}
                program.metadata.pop("mutation_context", None)
                program.metadata["evaluation_interrupted"] = True
                self._store(program)
                self._status_sets.setdefault(queued, set()).add(program_id)
                recovered += 1
            self._persist_status_sets()
        logger.info("[RecoverableDiskProgramStorage] Recovered {} stranded RUNNING -> QUEUED", recovered)
        return recovered


class AtomicDiskArchiveStorage(DiskArchiveStorage):
    """Keep upstream archive semantics while making each map replacement durable."""

    def _persist(self) -> None:
        """Publish either the previous or the complete next archive map."""
        _durable_replace(self._path, gigaevo_dumps(self._elites))


async def repair_interrupted_archive_members(strategy: Any, storage: RecoverableDiskProgramStorage) -> int:
    """Remove interrupted evaluations from archive indexes while preserving their queued programs."""
    program_ids = sorted(await strategy.get_program_ids())
    programs = await storage.mget(program_ids)
    missing = set(program_ids) - {program.id for program in programs}
    if missing:
        raise StorageError(f"Archive references missing authoritative programs: {sorted(missing)}")
    repaired = 0
    for program in programs:
        if program.state in {ProgramState.QUEUED, ProgramState.RUNNING} and program.metadata.get("evaluation_interrupted"):
            for island in strategy.islands.values():
                if await island.remove_elite_by_id(program.id):
                    repaired += 1
                    logger.info("Removed interrupted program {} from archive {}; evaluation remains queued", program.id, island.config.island_id)
    return repaired


class AtomicDiskArchiveStorageFactory:
    """Build project-owned atomic archives over one recoverable program store."""

    def __init__(self, program_storage: RecoverableDiskProgramStorage) -> None:
        """Retain the exact storage instance Hydra already constructed."""
        self._program_storage = program_storage

    def __call__(self, key_prefix: str | None = None) -> AtomicDiskArchiveStorage:
        """Build the prefix-scoped archive expected by native strategies."""
        return AtomicDiskArchiveStorage(program_storage=self._program_storage, key_prefix=key_prefix)
