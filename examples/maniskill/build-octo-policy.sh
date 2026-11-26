#!/usr/bin/env bash
set -Eeuo pipefail

# Build a separate Octo inference environment. The simulator has its own builder.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || {
  echo "Octo policy builds require Linux x86_64" >&2; exit 3;
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
SOURCE_COMMIT="653c54acde686fde619855f2eac0dd6edad7116b"
SOURCE_SHA256="e1e9a3078cf8d4bd7353f7d9147fe87d93e84e482396b43830eef0a7cadae3dd"
SOURCE_URL="https://codeload.github.com/octo-models/octo/tar.gz/$SOURCE_COMMIT"
SOURCE_ROOT="$INSTALL_DIR/octo"
KIT_SOURCE="$(cd -- "$SCRIPT_DIR/../../packages/roboscope-kit" && pwd -P)"
ENV_PYTHON="$INSTALL_DIR/env/bin/python"
UV_BIN="${UV_BIN:-uv}"
[[ "$("$UV_BIN" --version | awk '{print $2}')" == 0.11.14 ]] || {
  echo "uv 0.11.14 is required; set UV_BIN to its executable" >&2; exit 3;
}
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/python"
export UV_NO_PROGRESS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PIP_CONFIG_FILE=/dev/null

ARCHIVE="${OCTO_SOURCE_ARCHIVE:-$INSTALL_DIR/downloads/octo-$SOURCE_COMMIT.tar.gz}"
if [[ ! -f "$ARCHIVE" ]]; then
  [[ -z "${OCTO_SOURCE_ARCHIVE:-}" ]] || {
    echo "supplied Octo source archive is missing: $ARCHIVE" >&2; exit 4;
  }
  install -d "$INSTALL_DIR/downloads"
  curl --fail --location --retry 3 --output "$ARCHIVE.part" "$SOURCE_URL"
  mv -- "$ARCHIVE.part" "$ARCHIVE"
fi
[[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || {
  echo "Octo source archive checksum mismatch" >&2; exit 4;
}
if [[ ! -d "$SOURCE_ROOT" ]]; then
  install -d "$SOURCE_ROOT"
  tar -xzf "$ARCHIVE" --strip-components=1 -C "$SOURCE_ROOT"
  printf '%s\n' "$SOURCE_SHA256" > "$SOURCE_ROOT/.roboscope-source-sha256"
else
  [[ -f "$SOURCE_ROOT/.roboscope-source-sha256" ]] &&
    [[ "$(cat "$SOURCE_ROOT/.roboscope-source-sha256")" == "$SOURCE_SHA256" ]] || {
      echo "existing Octo source has no matching provenance; use a fresh prefix" >&2; exit 4;
    }
fi
"$UV_BIN" python install 3.11.15
PYTHON_BIN="$("$UV_BIN" python find --managed-python 3.11.15)"
if [[ ! -x "$ENV_PYTHON" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$INSTALL_DIR/env"
fi
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
WHEEL_ARGUMENTS=()
if [[ -n "${OCTO_WHEEL_CACHE:-}" ]]; then
  [[ -d "$OCTO_WHEEL_CACHE" ]] || {
    echo "supplied Octo wheel cache is missing: $OCTO_WHEEL_CACHE" >&2; exit 4;
  }
  WHEEL_ARGUMENTS+=(--find-links "$OCTO_WHEEL_CACHE")
fi
"$UV_BIN" pip install --python "$ENV_PYTHON" \
  --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  "${WHEEL_ARGUMENTS[@]}" \
  --requirement "$SCRIPT_DIR/octo-requirements.txt"
# The pinned upstream setup lists only its root package and documents editable installation.
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation --editable "$SOURCE_ROOT"
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation "$KIT_SOURCE"
"$UV_BIN" pip check --python "$ENV_PYTHON"
"$UV_BIN" pip freeze --python "$ENV_PYTHON" > "$INSTALL_DIR/requirements-resolved.txt"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$ENV_PYTHON" -I - \
  "$INSTALL_DIR" "$SOURCE_COMMIT" "$SOURCE_SHA256" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import flax
import jax
import numpy as np
import tensorflow as tf
from octo.model.octo_model import OctoModel
from roboscope_kit import Policy

assert jax.__version__ == "0.4.20"
assert flax.__version__ == "0.7.5"
assert tf.__version__ == "2.15.1"
assert np.__version__ == "1.26.4"
assert jax.default_backend() == "cpu"
root = Path(sys.argv[1])
records = {
    name: importlib.metadata.version(name)
    for name in ("octo", "roboscope-kit", "jax", "jaxlib", "flax", "orbax-checkpoint",
                 "tensorflow-cpu", "transformers", "numpy", "ml-dtypes", "tensorstore")
}
data = {
    "python": sys.version, "octo_source_commit": sys.argv[2],
    "octo_source_archive_sha256": sys.argv[3], "versions": records,
    "octo_model_import": f"{OctoModel.__module__}.{OctoModel.__name__}",
    "sdk_policy_import": f"{Policy.__module__}.{Policy.__name__}",
    "cpu_import_check": True, "gpu_model_inference_validated": False,
    "requirements_sha256": hashlib.sha256((root / "requirements-resolved.txt").read_bytes()).hexdigest(),
}
(root / "build-provenance.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(json.dumps(data, indent=2))
PY
echo "Built Octo policy environment in $INSTALL_DIR"
