#!/usr/bin/env bash
# Run from any directory: bash scripts/setup_env.sh [auto|cu130|cu126]
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_WHEEL="${1:-auto}"

if [[ $# -gt 1 || ! "$CUDA_WHEEL" =~ ^(auto|cu130|cu126)$ ]]; then
  echo "Usage: bash scripts/setup_env.sh [auto|cu130|cu126]" >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This setup targets Linux/WSL x86_64 with NVIDIA GPUs." >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Use Python 3.10, 3.11 or 3.12"'
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is missing. Install/enable the NVIDIA driver before running setup." >&2
  exit 1
fi
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '\r ')"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
if [[ ! "$DRIVER_MAJOR" =~ ^[0-9]+$ ]]; then
  echo "Cannot determine the NVIDIA driver version: $DRIVER_VERSION" >&2
  exit 1
fi
if [[ "$CUDA_WHEEL" == auto ]]; then
  if (( DRIVER_MAJOR >= 580 )); then
    CUDA_WHEEL=cu130
  elif (( DRIVER_MAJOR >= 560 )); then
    CUDA_WHEEL=cu126
  else
    echo "This setup requires driver branch 560+ (cu126) or 580+ (cu130). Found $DRIVER_VERSION." >&2
    echo "Have the remote machine's driver upgraded, then rerun this script." >&2
    exit 1
  fi
fi
if [[ "$CUDA_WHEEL" == cu130 && "$DRIVER_MAJOR" -lt 580 ]] ||
   [[ "$CUDA_WHEEL" == cu126 && "$DRIVER_MAJOR" -lt 560 ]]; then
  echo "Driver $DRIVER_VERSION is below this setup's requirement for $CUDA_WHEEL." >&2
  exit 1
fi

echo "[setup] Driver=$DRIVER_VERSION; torch=2.14.1; wheel=$CUDA_WHEEL; venv=$VENV_DIR"
if [[ -e "$VENV_DIR" && ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  echo "Refusing to overwrite a non-venv directory: $VENV_DIR" >&2
  exit 1
fi
if [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" - "$VENV_DIR" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
assert sys.prefix != sys.base_prefix, "Python is not running inside a venv"
assert pathlib.Path(sys.prefix).resolve() == root, "Unexpected virtual environment"
assert (3, 10) <= sys.version_info[:2] < (3, 13), "Venv Python must be 3.10-3.12"
config = (root / "pyvenv.cfg").read_text().lower()
assert "include-system-site-packages = false" in config, "Use an isolated venv, without system-site-packages"
PY

"$VENV_DIR/bin/python" -m pip install --upgrade pip
# A local version suffix prevents silently retaining a CPU/different-CUDA wheel.
# Fetch only torch from its CUDA index. The requirements step then resolves ALL
# dependencies from PyPI (including NVIDIA libraries), avoiding duplicated index
# downloads. Do not import torch between these two installation steps.
"$VENV_DIR/bin/python" -m pip install --no-deps "torch==2.14.1+$CUDA_WHEEL" \
  --index-url "https://download.pytorch.org/whl/$CUDA_WHEEL"
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pip check
mkdir -p "$PROJECT_DIR/artifacts"
"$VENV_DIR/bin/python" -m pip freeze --all > "$PROJECT_DIR/artifacts/environment.freeze.txt"
"$VENV_DIR/bin/python" "$PROJECT_DIR/scripts/check_env.py" --min-gpus 1
printf '\nEnvironment ready. Activate with:\n  source "%s/bin/activate"\n' "$VENV_DIR"
