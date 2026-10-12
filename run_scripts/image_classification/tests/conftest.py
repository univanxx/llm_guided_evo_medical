"""Resolve the task and external engine without relying on a developer checkout."""
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("GIGAEVO_DIR", str(ROOT.parents[1].parent / "gigaevo-core")))


@pytest.fixture(autouse=True)
def isolate_runtime_environment():
    """Keep launcher environment changes and pass identifiers local to each test."""
    with patch.dict(os.environ):
        for name in list(os.environ):
            if name.startswith(("PNEUMONIA_", "MEDGEMMA_", "EVOLVE_")):
                os.environ.pop(name)
        yield
