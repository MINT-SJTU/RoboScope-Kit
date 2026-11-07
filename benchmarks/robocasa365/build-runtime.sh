#!/usr/bin/env bash
set -Eeuo pipefail

# Build the isolated RoboCasa365 (PandaOmron) simulator, never the GR1 runtime.
# Usage: build-runtime.sh <install-dir> [--environment-only]
# The optional mode prepares imports while assets are being downloaded; it does
# not install a benchmark manifest or claim that a usable runtime was built.
# Normal builds require all committed asset checksums and finish with --self-test.
# KIT_PYTHON selects a Python containing this Kit checkout (default: python3).
# ROBOSCOPE_ROBOCASA365_ASSETS_DIR selects the persistent ZIP download cache.
# ROBOSCOPE_ROBOCASA365_SOURCE_CACHE selects the verified source archive cache.
# ROBOSCOPE_ROBOCASA365_PYTHON may select an existing absolute Python 3.11
# interpreter path. Its installation must remain available to the resulting venv.
# ROBOSCOPE_PYPI_INDEX may select a package mirror (default: official PyPI);
# package versions and simulator pins remain fixed.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -ge 1 && $# -le 2 ]] || {
  echo "usage: $0 <install-dir> [--environment-only]" >&2
  exit 2
}
ENVIRONMENT_ONLY=0
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--environment-only" ]] || { echo "unknown option: $2" >&2; exit 2; }
  ENVIRONMENT_ONLY=1
fi
[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
  echo "RoboCasa365 runtime builds require Linux x86_64" >&2
  exit 3
}
command -v uv >/dev/null 2>&1 || {
  echo "uv is required" >&2; exit 3;
}
UV_BIN="$(command -v uv)"
[[ "$("${UV_BIN}" --version)" == "uv 0.11.14 "* ]] || {
  echo "uv 0.11.14 is required" >&2
  exit 3
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
PYTHON="${INSTALL_DIR}/env/bin/python"
SOURCE_ROOT="${INSTALL_DIR}/src"
KIT_PYTHON="${KIT_PYTHON:-python3}"
EXISTING_PYTHON="${ROBOSCOPE_ROBOCASA365_PYTHON:-}"
PYPI_INDEX="${ROBOSCOPE_PYPI_INDEX:-https://pypi.org/simple}"
export UV_PYTHON_INSTALL_DIR="${INSTALL_DIR}/python"
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

if [[ "${ENVIRONMENT_ONLY}" -eq 0 ]]; then
  "${KIT_PYTHON}" -I - "${SCRIPT_DIR}/../../packages/roboscope-kit/pyproject.toml" <<'PY'
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from roboscope_kit.benchmark_specs import BenchmarkName

expected = tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["project"]["version"]
actual = version("roboscope-kit")
if actual != expected or BenchmarkName.ROBOCASA365.value != "robocasa365":
    raise SystemExit(f"KIT_PYTHON must contain this Kit version ({expected}), found {actual}; reinstall the package")
PY
  "${KIT_PYTHON}" - "${SCRIPT_DIR}/asset-inputs.json" <<'PY'
import json
import re
import sys
from pathlib import Path

inputs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for name, spec in inputs["archives"].items():
    if not re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256") or ""):
        raise SystemExit(f"Missing reviewed SHA256 for {name}; full runtime build refused")
PY
fi
# Refuse altered source or incompatible patch provenance before installing dependencies.
"${KIT_PYTHON}" -I "${SCRIPT_DIR}/apply-runtime-patches.py" \
  --source-root "${SOURCE_ROOT}" --check-only
"${KIT_PYTHON}" -I "${SCRIPT_DIR}/install-sources.py" \
  --source-root "${SOURCE_ROOT}" \
  --cache-dir "${ROBOSCOPE_ROBOCASA365_SOURCE_CACHE:-${INSTALL_DIR}/source-cache}" \
  --check-inputs
if [[ -n "${EXISTING_PYTHON}" ]]; then
  [[ "${EXISTING_PYTHON}" == /* && -x "${EXISTING_PYTHON}" ]] || {
    echo "ROBOSCOPE_ROBOCASA365_PYTHON must be an absolute executable path" >&2; exit 3;
  }
  "${EXISTING_PYTHON}" -I -c \
    'import sys; assert sys.version_info[:2] == (3, 11), f"Python 3.11 required: {sys.version}"'
  if [[ ! -x "${PYTHON}" ]]; then
    "${UV_BIN}" venv --python "${EXISTING_PYTHON}" --no-python-downloads \
      --relocatable "${INSTALL_DIR}/env"
  fi
else
  "${UV_BIN}" python install --no-bin --no-progress 3.11
  if [[ ! -x "${PYTHON}" ]]; then
    "${UV_BIN}" venv --python 3.11 --managed-python --no-python-downloads \
      --relocatable "${INSTALL_DIR}/env"
  fi
fi
# CPU torch is required by RoboCasa's imported playback utilities, although
# simulation itself uses MuJoCo/EGL. Policy inference has its own environment.
"${UV_BIN}" pip install --python "${PYTHON}" --no-progress --no-deps \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
"${UV_BIN}" pip install --python "${PYTHON}" --no-progress \
  --index-url "${PYPI_INDEX}" \
  --requirements "${SCRIPT_DIR}/runtime-requirements.txt"

# Source transport uses canonical hash-verified archives, independent of Git mirrors.
"${PYTHON}" -I "${SCRIPT_DIR}/install-sources.py" \
  --source-root "${SOURCE_ROOT}" \
  --cache-dir "${ROBOSCOPE_ROBOCASA365_SOURCE_CACHE:-${INSTALL_DIR}/source-cache}"
"${PYTHON}" -I "${SCRIPT_DIR}/apply-runtime-patches.py" --source-root "${SOURCE_ROOT}"
# These checkouts declare training, dataset-conversion and device extras that
# evaluation never imports. Use the explicit runtime lock instead of resolving
# lerobot / tianshou / pynput / hidapi / pygame or changing the pinned simulator's
# package requirements. In particular, the headless Gym path does not import
# robosuite.devices, so evdev and its native compiler are unnecessary.
"${UV_BIN}" pip install --python "${PYTHON}" --no-progress --no-deps \
  --no-build-isolation -e "${SOURCE_ROOT}/robosuite" -e "${SOURCE_ROOT}/robocasa"

env -u PYTHONPATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${PYTHON}" -I - <<'PY'
import sys
from importlib.metadata import version

if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, got {sys.version}")
for name, expected in (
    ("robocasa", "1.0.1"), ("robosuite", "1.5.2"),
    ("mujoco", "3.3.1"), ("numpy", "2.2.5"), ("torch", "2.5.1+cpu"),
):
    if version(name) != expected:
        raise SystemExit(f"{name}: expected {expected}, found {version(name)}")
import robocasa
import robocasa.wrappers.gym_wrapper
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robosuite.utils.binding_utils import MjRenderContext
from robosuite.renderers.context.egl_context import EGLGLContext

if MjRenderContext._roboscope_owning_context_cleanup != 2 or not callable(EGLGLContext.preserve_current):
    raise SystemExit("The reviewed owning-EGL-context cleanup patches are missing")
if [len(TASK_SET_REGISTRY[k]) for k in ("atomic_seen", "composite_seen", "composite_unseen")] != [18, 16, 16]:
    raise SystemExit("RoboCasa365 official task sets do not match 18/16/16")
print("RoboCasa365 source pins and simulator imports verified")
PY
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"
install -m 0644 "${SCRIPT_DIR}/runtime-requirements.txt" "${INSTALL_DIR}/runtime-requirements.txt"
install -m 0644 "${SCRIPT_DIR}/source-inputs.json" "${INSTALL_DIR}/source-inputs.json"
install -m 0644 "${SCRIPT_DIR}/runtime-patches.json" "${INSTALL_DIR}/runtime-patches.json"
if [[ "${ENVIRONMENT_ONLY}" -eq 1 ]]; then
  echo "Prepared RoboCasa365 environment only: ${INSTALL_DIR}; assets and self-test remain"
  exit 0
fi
"${PYTHON}" -I "${SCRIPT_DIR}/build-assets.py" \
  --source-root "${SOURCE_ROOT}/robocasa" \
  --cache-dir "${ROBOSCOPE_ROBOCASA365_ASSETS_DIR:-${INSTALL_DIR}/downloads}"
install -m 0644 "${SCRIPT_DIR}/benchmark-manifest.json" "${INSTALL_DIR}/benchmark-manifest.json"
install -m 0644 "${SCRIPT_DIR}/asset-inputs.json" "${INSTALL_DIR}/asset-inputs.json"
SELF_TEST_OUTPUT="$(mktemp -d "${INSTALL_DIR}/self-test.XXXXXX")"
"${KIT_PYTHON}" -I -m roboscope_kit.runtime.benchmark --benchmark robocasa365 --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
  --output-dir "${SELF_TEST_OUTPUT}"
echo "Built RoboCasa365 runtime and passed Kit self-test: ${INSTALL_DIR}"
echo "Self-test evidence: ${SELF_TEST_OUTPUT}"
