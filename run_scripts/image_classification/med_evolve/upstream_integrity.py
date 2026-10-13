"""Verify the external source checkout against the pinned upstream release."""
import hashlib
import json
import os
from pathlib import Path


def verify_upstream(vendor_dir: Path, manifest_path: Path) -> dict:
    """Reject missing or changed upstream sources while ignoring interpreter caches."""
    vendor_dir = Path(vendor_dir).resolve()
    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    files = []
    for directory, subdirs, names in os.walk(vendor_dir):
        subdirs[:] = [name for name in subdirs if name not in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}]
        for name in names:
            if not name.endswith((".pyc", ".pyo")):
                files.append(Path(directory) / name)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(vendor_dir).as_posix()):
        relative = path.relative_to(vendor_dir).as_posix()
        content = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
        digest.update(f"{relative}\0{hashlib.sha256(content).hexdigest()}\n".encode())
    result = {"ok": len(files) == expected["file_count"] and digest.hexdigest() == expected["source_sha256"], "commit": expected["commit"], "files": len(files), "source_sha256": digest.hexdigest()}
    if not result["ok"]:
        raise RuntimeError("External source verification failed; use the pinned download: " + json.dumps(result))
    return result
