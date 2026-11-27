#!/usr/bin/env bash
set -Eeuo pipefail

# Build a CPython 3.11 inference environment for the Evo-DexBench DP-C / DP-T checkpoints.
# It installs the upstream example's dp_policy package and pins from the same Evo-DexBench
# revision as the runtime, plus the unmodified official Diffusion Policy checkout it loads.
#   TORCH_INDEX_URL  default https://download.pytorch.org/whl/cu124
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
ENV_DIR="${INSTALL_DIR}/env"
EVO_COMMIT="818ac28e42e9199f0cbccce8617cb686519d8abf"
EVO_SHA256="b3bb479ca93d976745436261f22be33b1902d72f3fdb724916634107369d5d33"
EVO_ROOT="${INSTALL_DIR}/Evo-DexBench"
# dp_policy.upstream finds the checkout here by default, so the server needs no environment.
DP_ROOT="${EVO_ROOT}/.third_party/DiffusionPolicy-official"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 3; }

if [[ ! -d "${EVO_ROOT}" ]]; then
  ARCHIVE="${INSTALL_DIR}/evo-dexbench-${EVO_COMMIT}.tar.gz"
  [[ -f "${ARCHIVE}" ]] || curl --fail --location --retry 3 --output "${ARCHIVE}" \
    "https://codeload.github.com/MINT-SJTU/Evo-DexBench/tar.gz/${EVO_COMMIT}"
  [[ "$(sha256sum "${ARCHIVE}" | awk '{print $1}')" == "${EVO_SHA256}" ]] || {
    echo "Evo-DexBench source archive checksum mismatch" >&2; exit 4;
  }
  install -d "${EVO_ROOT}"
  tar -xzf "${ARCHIVE}" --strip-components=1 -C "${EVO_ROOT}"
fi
EXAMPLE="${EVO_ROOT}/examples/policies/diffusion_policy"
DP_COMMIT="$(tail -n 1 "${EXAMPLE}/dp_policy/UPSTREAM_COMMIT" | tr -d '[:space:]')"
if [[ ! -d "${DP_ROOT}/.git" ]]; then
  git init -q "${DP_ROOT}"
  git -C "${DP_ROOT}" remote add origin https://github.com/real-stanford/diffusion_policy.git
fi
# Only the pinned commit is needed; a shallow fetch avoids cloning the full history.
git -C "${DP_ROOT}" cat-file -e "${DP_COMMIT}^{commit}" 2>/dev/null \
  || git -C "${DP_ROOT}" -c http.version=HTTP/1.1 fetch --depth 1 --no-tags origin "${DP_COMMIT}"
git -C "${DP_ROOT}" checkout -q --detach "${DP_COMMIT}"
[[ -z "$(git -C "${DP_ROOT}" status --porcelain)" ]] || { echo "Diffusion Policy checkout is modified" >&2; exit 4; }

[[ -x "${ENV_DIR}/bin/python" ]] || uv venv --python 3.11 "${ENV_DIR}"
PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
uv pip install --python "${PYTHON}" --index-url "${TORCH_INDEX_URL}" torch==2.5.1 torchvision==0.20.1
# Upstream pins target Python 3.10; numcodecs 0.10.2 has no 3.11 wheel and no longer compiles
# there, so it is lifted to the first release that does (zarr 2.12 accepts it). It only serves
# zarr datasets, not inference.
printf 'numcodecs==0.12.1\n' > "${INSTALL_DIR}/py311-overrides.txt"
uv pip install --python "${PYTHON}" -r "${EXAMPLE}/requirements.txt" --override "${INSTALL_DIR}/py311-overrides.txt"
# As upstream: LeRobot's dataset API only, without its robot/UI dependency stack.
uv pip install --python "${PYTHON}" --no-deps lerobot==0.4.0
uv pip install --python "${PYTHON}" --no-deps -e "${EVO_ROOT}" -e "${EXAMPLE}"
# six: imported by the hydra workspace that checkpoints name, absent from the upstream pins.
uv pip install --python "${PYTHON}" pyyaml six "${KIT_ROOT}/packages/roboscope-kit"
"${PYTHON}" -I - "${DP_ROOT}" <<'PY'
import sys

from dp_policy.upstream import configure_upstream

configure_upstream(sys.argv[1])
from dp_policy.server import _load_policy  # noqa: F401
# The workspace class DP-T checkpoints are saved with, so missing training-side imports fail here.
from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (  # noqa: F401
    TrainDiffusionTransformerHybridWorkspace,
)

import roboscope_kit  # noqa: F401

print("Evo-DexBench DP imports passed")
PY
echo "Built inference environment: ${PYTHON}"
