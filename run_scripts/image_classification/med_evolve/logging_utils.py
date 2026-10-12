"""Bound operational logs while preserving separate scientific audit artifacts.

Both evolution and detached-console logs retain three 10 MiB files. Large log
records are shortened and inline image payloads removed; candidate, token,
prediction and LLM I/O artifacts are intentionally outside this rotation policy.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading


LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_FILE_COUNT = 3
MAX_RECORD_CHARACTERS = 8192
IMAGE_PAYLOAD = re.compile(r"data:image/[^\s\"']+", re.IGNORECASE)


def service_text(value: str) -> str:
    """Remove image blobs and bound diagnostics without altering stored results."""
    value = IMAGE_PAYLOAD.sub("[image payload omitted from service log]", str(value))
    if len(value) > MAX_RECORD_CHARACTERS:
        value = value[:MAX_RECORD_CHARACTERS] + "\n[service record truncated; full experiment artifacts are separate]\n"
    return value


def write_console(stream, text: str) -> None:
    """Escape characters unsupported by a terminal without changing UTF-8 log files."""
    encoding = getattr(stream, "encoding", None)
    if encoding:
        text = text.encode(encoding, errors="backslashreplace").decode(encoding)
    stream.write(text)
    stream.flush()


class BoundedTextWriter:
    """Write UTF-8 operational records with a strict fixed-file disk budget."""

    def __init__(self, path: Path, max_bytes: int = LOG_MAX_BYTES, file_count: int = LOG_FILE_COUNT):
        """Open the current log and retain only the configured numbered rotations."""
        if max_bytes < 64 or file_count < 1:
            raise ValueError("Require max_bytes>=64 and file_count>=1")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.file_count = file_count
        self.lock = threading.Lock()
        prefix = self.path.name + "."
        for candidate in self.path.parent.glob(prefix + "*"):
            suffix = candidate.name[len(prefix):]
            if suffix.isascii() and suffix.isdigit() and int(suffix) >= file_count and (candidate.is_file() or candidate.is_symlink()):
                candidate.unlink(missing_ok=True)
        for index in range(file_count):
            previous = self.path if index == 0 else self.path.with_name(self.path.name + "." + str(index))
            if previous.exists() and previous.stat().st_size > max_bytes:
                with previous.open("rb") as source:
                    source.seek(-max_bytes, os.SEEK_END)
                    tail = source.read().decode("utf-8", errors="ignore").encode("utf-8")
                previous.write_bytes(tail)
        self.stream = self.path.open("ab")

    def _rotate(self):
        """Replace the oldest operational log while leaving audit paths untouched."""
        self.stream.close()
        for index in range(self.file_count - 1, 0, -1):
            source = self.path if index == 1 else self.path.with_name(self.path.name + "." + str(index - 1))
            if source.exists():
                source.replace(self.path.with_name(self.path.name + "." + str(index)))
        self.stream = self.path.open("wb")

    def write(self, message):
        """Keep complete UTF-8 records within the per-file size limit."""
        data = service_text(str(message)).encode("utf-8", errors="replace")
        if len(data) > self.max_bytes:
            marker = b"\n[oversized service record truncated]\n"
            data = data[:self.max_bytes - len(marker)].decode("utf-8", errors="ignore").encode("utf-8") + marker
        with self.lock:
            if self.stream.tell() + len(data) > self.max_bytes:
                self._rotate()
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        """Flush pending bytes for normal stream compatibility."""
        with self.lock:
            if not self.stream.closed:
                self.stream.flush()

    def stop(self):
        """Close the current file when loguru removes this sink."""
        with self.lock:
            self.stream.close()


class ServiceConsole:
    """Apply the same operational-message policy to the visible terminal."""

    def __init__(self, stream):
        """Capture the original console without changing global stdout/stderr."""
        self.stream = stream

    def write(self, message):
        """Emit bounded diagnostics with inline images removed."""
        write_console(self.stream, service_text(str(message)))


def configure_logging(run_dir: Path, *, max_bytes: int = LOG_MAX_BYTES, file_count: int = LOG_FILE_COUNT):
    """Set INFO application logs and WARNING SDK/Hydra logs with fixed retention."""
    from loguru import logger

    class ForwardHandler(logging.Handler):
        """Send standard-library warnings to the configured application sinks."""

        def emit(self, record):
            """Retain warning text without enabling verbose SDK request logging."""
            logger.opt(exception=record.exc_info).log(record.levelname, service_text(record.getMessage()))

    logger.remove()
    format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} | {message}"
    writer = BoundedTextWriter(Path(run_dir) / "evolution.log", max_bytes=max_bytes, file_count=file_count)
    logger.add(writer, level="INFO", format=format, colorize=False, backtrace=False, diagnose=False)
    logger.add(ServiceConsole(sys.stderr), level="INFO", format=format, colorize=False, backtrace=False, diagnose=False)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.addHandler(ForwardHandler(level=logging.WARNING))
    root.setLevel(logging.WARNING)
    for name in ("hydra", "httpx", "httpcore", "openai", "urllib3"):
        third_party = logging.getLogger(name)
        third_party.setLevel(logging.WARNING)
        third_party.propagate = True
        for handler in list(third_party.handlers):
            third_party.removeHandler(handler)
            handler.close()
    return writer


def capture_process(command: list[str], path: Path, *, max_bytes: int = LOG_MAX_BYTES, file_count: int = LOG_FILE_COUNT) -> int:
    """Capture a detached launcher's output with bounded memory and disk usage."""
    writer = BoundedTextWriter(path, max_bytes=max_bytes, file_count=file_count)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except BaseException:
        writer.stop()
        raise
    discarding_long_line = False
    try:
        while True:
            try:
                chunk = process.stdout.readline(MAX_RECORD_CHARACTERS + 1)
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                continue
            if not chunk:
                break
            if not discarding_long_line:
                text = chunk.decode("utf-8", errors="replace")
                if len(chunk) > MAX_RECORD_CHARACTERS and not chunk.endswith(b"\n"):
                    text += "\n[remaining oversized console line omitted]\n"
                text = service_text(text)
                writer.write(text)
                write_console(sys.stdout, text)
            discarding_long_line = not chunk.endswith(b"\n")
        result = process.wait()
        return result if result >= 0 else 128 - result
    finally:
        process.stdout.close()
        writer.stop()


def main(argv: list[str] | None = None) -> int:
    """Wrap one command for GNU screen without changing its eventual exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    return capture_process(command, args.capture)


if __name__ == "__main__":
    raise SystemExit(main())
