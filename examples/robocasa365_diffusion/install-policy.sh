#!/usr/bin/env bash
set -Eeuo pipefail

# Build a separate CPython 3.11 inference environment; simulator pins stay separate.
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
ENV_DIR="${INSTALL_DIR}/env"
command -v uv >/dev/null || { echo "uv is required" >&2; exit 3; }
# An existing supported interpreter avoids an additional standalone Python download.
POLICY_PYTHON="${ROBOSCOPE_POLICY_PYTHON:-3.11}"
export UV_DEFAULT_INDEX="${ROBOSCOPE_PYPI_INDEX:-${UV_DEFAULT_INDEX:-https://pypi.org/simple}}"
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  uv venv --python "${POLICY_PYTHON}" "${ENV_DIR}"
fi
PYTHON="${ENV_DIR}/bin/python"
"${PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
uv pip install --python "${PYTHON}" --index-url https://download.pytorch.org/whl/cu124 \
  --constraint "${SCRIPT_DIR}/requirements-lock.txt" torch==2.6.0 torchvision==0.21.0
uv pip install --python "${PYTHON}" --constraint "${SCRIPT_DIR}/requirements-lock.txt" \
  -r "${SCRIPT_DIR}/requirements.txt"

"${PYTHON}" "${SCRIPT_DIR}/install_sources.py" "${INSTALL_DIR}" \
  --archives-dir "${ROBOSCOPE_POLICY_SOURCE_ARCHIVES_DIR:-${INSTALL_DIR}/source-archives}"
uv pip install --python "${PYTHON}" --no-deps -e "${INSTALL_DIR}/diffusion_policy" \
  -e "${INSTALL_DIR}/robomimic"
# The pinned source is an implicit namespace package, but its setup.py uses
# find_packages(), so a modern editable install exposes no Diffusion Policy modules.
"${PYTHON}" -I - "${INSTALL_DIR}/diffusion_policy" <<'PY'
import sys
import sysconfig
from pathlib import Path

source = Path(sys.argv[1]).resolve()
if not (source / "diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py").is_file():
    raise SystemExit(f"verified Diffusion Policy source is missing: {source}")
path_entry = Path(sysconfig.get_path("purelib")) / "roboscope_robocasa365_diffusion.pth"
path_entry.write_text(str(source) + "\n", encoding="utf-8")
PY
uv pip install --python "${PYTHON}" --constraint "${SCRIPT_DIR}/requirements-lock.txt" \
  "${KIT_ROOT}/packages/roboscope-kit"
"${PYTHON}" -I - <<'PY'
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from robomimic.utils.lang_utils import LangEncoder

import roboscope_kit

print("RoboCasa365 model imports passed independently of cwd and PYTHONPATH")
PY
echo "Built inference environment: ${PYTHON}"
