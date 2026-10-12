#!/usr/bin/env bash
# Create an isolated environment and install the verified external engine.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$task_root/../.." && pwd)"
engine_root="${GIGAEVO_DIR:-$(dirname -- "$repository_root")/gigaevo-core}"
python_bin="${PYTHON:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the validated dependency constraints"'
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" "$task_root/fetch_upstream.py" --destination "$engine_root"
if [[ ! -x "$task_root/.venv/bin/python" ]]; then
    "$python_bin" -m venv "$task_root/.venv"
fi
build_root="$(mktemp -d)"
trap 'rm -rf -- "$build_root"' EXIT
cp -a -- "$engine_root" "$build_root/engine"
"$task_root/.venv/bin/python" -m pip install --no-cache-dir "$build_root/engine" -r "$task_root/requirements.txt" -c "$task_root/constraints-ubuntu-py312.txt"
"$task_root/.venv/bin/python" -m pip freeze > "$task_root/requirements-resolved.txt"
"$python_bin" "$task_root/fetch_upstream.py" --destination "$engine_root"
echo "Ready: source $task_root/.venv/bin/activate"
