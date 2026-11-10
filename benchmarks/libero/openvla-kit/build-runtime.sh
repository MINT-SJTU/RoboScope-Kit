#!/usr/bin/env bash
set -Eeuo pipefail

# Supported Kit Policy + simulator environment matching OpenVLA's package versions.
# Entry point: benchmarks/libero/build-runtime.sh --openvla-kit <install-dir>
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || { echo "Linux x86_64 required" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
mkdir -p -- "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
PYTHON="${INSTALL_DIR}/env/bin/python"
INDEX_URL="${OPENVLA_INDEX_URL:-https://pypi.org/simple}"
FLASH_WHEEL=flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
FLASH_SHA256=beb6cb7d86f348636b7fb10ec40bbcd0452d45e83c1833931b517a794e98957c
CACHE="${DOWNLOAD_CACHE:-${INSTALL_DIR}/downloads}"
mkdir -p -- "${CACHE}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null

if [[ ! -x "${PYTHON}" ]]; then
  uv venv --python 3.11.15 "${INSTALL_DIR}/env"
fi
"${PYTHON}" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
uv pip sync --python "${PYTHON}" --index-url "${INDEX_URL}" --require-hashes "${SCRIPT_DIR}/requirements.lock"

SOURCE_CACHE_ARGS=()
if [[ -n "${LIBERO_SOURCE_CACHE:-}" ]]; then
  SOURCE_CACHE_ARGS=(--cache "${LIBERO_SOURCE_CACHE}")
fi
"${PYTHON}" "${SCRIPT_DIR}/materialize_source.py" "${SCRIPT_DIR}/libero-source.json" \
  "${INSTALL_DIR}/LIBERO" "${SOURCE_CACHE_ARGS[@]}"
uv pip install --python "${PYTHON}" --no-deps --no-build-isolation "${INSTALL_DIR}/LIBERO"
uv pip install --python "${PYTHON}" --no-deps "${REPO_ROOT}/packages/roboscope-kit"
if [[ ! -f "${CACHE}/${FLASH_WHEEL}" ]]; then
  if ! curl --fail --location --retry 2 --connect-timeout 20 \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    --output "${CACHE}/${FLASH_WHEEL}.partial"; then
    curl --fail --location --retry 2 --connect-timeout 20 \
      --header 'Accept: application/octet-stream' \
      'https://api.github.com/repos/Dao-AILab/flash-attention/releases/assets/152900428?download=1' \
      --output "${CACHE}/${FLASH_WHEEL}.partial"
  fi
  echo "${FLASH_SHA256}  ${CACHE}/${FLASH_WHEEL}.partial" | sha256sum --check --status
  mv -- "${CACHE}/${FLASH_WHEEL}.partial" "${CACHE}/${FLASH_WHEEL}"
fi
echo "${FLASH_SHA256}  ${CACHE}/${FLASH_WHEEL}" | sha256sum --check --status
uv pip install --python "${PYTHON}" --no-deps "${CACHE}/${FLASH_WHEEL}"

uv pip check --python "${PYTHON}"

mkdir -p -- "${INSTALL_DIR}/config" "${INSTALL_DIR}/LIBERO/libero/datasets"
"${PYTHON}" - "${INSTALL_DIR}" "${SCRIPT_DIR}" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

from roboscope_kit.runtime.benchmarks.libero.adapter import write_libero_config

root, inputs = map(Path, sys.argv[1:])
write_libero_config(root / "config/config.yaml", root / "LIBERO")
manifest = json.loads((inputs.parent / "benchmark-manifest.json").read_text())
manifest["defaults"]["ready_timeout_s"] = 1800
(root / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
build = {
    "profile": "openvla-kit", "python": sys.version,
    "libero_commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "python_executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
    "packages": {n: version(n) for n in ("roboscope-kit", "torch", "transformers", "tensorflow", "flash-attn", "libero", "robosuite", "mujoco", "numpy")},
    "build_inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()},
}
(root / "build-manifest.json").write_text(json.dumps(build, indent=2) + "\n")
PY
env PYTHONPATH="${INSTALL_DIR}/LIBERO" LIBERO_CONFIG_PATH="${INSTALL_DIR}/config" \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${PYTHON}" - <<'PY'
from importlib.metadata import version

import flash_attn
import torch
from roboscope_kit.runtime.benchmarks.libero import worker

assert version("robosuite") == "1.4.1"
assert version("transformers") == "4.40.1"
assert flash_attn.__version__ == "2.5.5"
print("Supported Kit worker and model imports passed", torch.__version__, worker.__file__)
PY
"${PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark libero --self-test \
  --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" --output-dir "${INSTALL_DIR}/self-test"
