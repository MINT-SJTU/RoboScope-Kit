#!/usr/bin/env bash
set -Eeuo pipefail

# Build only the OpenVLA Policy and benchmark CLI environment.
# Build simulator Workers separately with benchmarks/libero/build-runtime.sh.
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || { echo "Linux x86_64 required" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
for required in uv curl sha256sum; do
  command -v "${required}" >/dev/null || { echo "${required} is required" >&2; exit 2; }
done
mkdir -p -- "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
if [[ -e "${INSTALL_DIR}/benchmark-manifest.json" || -e "${INSTALL_DIR}/LIBERO" ]]; then
  echo "Use a separate Policy install directory, not an existing simulator runtime" >&2
  exit 2
fi
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
"${PYTHON}" -I -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
uv pip sync --python "${PYTHON}" --index-url "${INDEX_URL}" --require-hashes "${SCRIPT_DIR}/requirements.lock"
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

"${PYTHON}" -I - "${INSTALL_DIR}" "${SCRIPT_DIR}" <<'PYTHON'
import hashlib
import importlib.util
import json
import sys
from importlib.metadata import version
from pathlib import Path

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import flash_attn
import numpy as np
import torch
import transformers
from roboscope_kit.policy import Policy

root, inputs = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("openvla_policy", inputs / "openvla_policy.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert issubclass(module.OpenVLAPolicy, Policy)
image = module.prepare_image(np.zeros((256, 256, 3), dtype=np.uint8))
assert image.shape == (224, 224, 3) and image.dtype == np.uint8
assert not tf.config.get_visible_devices("GPU")
assert transformers.__version__ == "4.40.1"
assert flash_attn.__version__ == "2.5.5"
build = {
    "profile": "openvla-policy",
    "python": sys.version,
    "tensorflow_visible_gpus": [],
    "preprocessing_device": "CPU",
    "python_executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
    "packages": {n: version(n) for n in (
        "roboscope-kit", "torch", "transformers", "tensorflow", "flash-attn", "numpy",
    )},
    "build_inputs_sha256": {
        name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
        for name in ("build-policy-env.sh", "requirements.in", "requirements.lock", "openvla_policy.py")
    },
    "validation": "Policy imports and CPU preprocessing only; no model loading or simulator rollout",
}
(root / "build-manifest.json").write_text(json.dumps(build, indent=2) + "\n")
print("OpenVLA Policy imports and CPU preprocessing passed", torch.__version__)
PYTHON

echo "Built OpenVLA Policy environment in ${INSTALL_DIR}"
echo "Build Workers separately with benchmarks/libero/build-runtime.sh and KIT_PYTHON=${PYTHON}"
