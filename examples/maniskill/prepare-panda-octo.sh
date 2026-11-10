#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare isolated v1.5 imports on the existing pinned Octo inference Python.
[[ $# -eq 2 ]] || { echo "usage: $0 <octo-python> <checkpoint-dir>" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
POLICY_PYTHON="$1"
CHECKPOINT_DIR="$(cd -- "$2" && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
[[ "$("$UV_BIN" --version | awk '{print $2}')" == 0.11.14 ]] || {
  echo "uv 0.11.14 is required" >&2; exit 3;
}
"$POLICY_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
DEPENDENCY_DIR="${PANDA_OCTO_DEPENDENCY_DIR:-$CHECKPOINT_DIR/python-deps}"
if [[ "$DEPENDENCY_DIR" != "$CHECKPOINT_DIR/python-deps" ]]; then
  install -d "$DEPENDENCY_DIR"
  DEPENDENCY_DIR="$(cd -- "$DEPENDENCY_DIR" && pwd -P)"
  if [[ ! -e "$CHECKPOINT_DIR/python-deps" && ! -L "$CHECKPOINT_DIR/python-deps" ]]; then
    ln -s "$DEPENDENCY_DIR" "$CHECKPOINT_DIR/python-deps"
  fi
  [[ "$(readlink -f "$CHECKPOINT_DIR/python-deps")" == "$DEPENDENCY_DIR" ]] || {
    echo "Existing checkpoint import overlay points elsewhere" >&2; exit 3;
  }
fi
WHEEL_ARGS=()
if [[ -n "${PANDA_OCTO_WHEEL_CACHE:-}" ]]; then
  [[ -d "$PANDA_OCTO_WHEEL_CACHE" ]] || { echo "Wheel cache does not exist" >&2; exit 3; }
  WHEEL_ARGS=(--offline --find-links "$PANDA_OCTO_WHEEL_CACHE")
fi
"$UV_BIN" pip install --python "$POLICY_PYTHON" --no-deps --no-build-isolation \
  --target "$DEPENDENCY_DIR" --requirement "$SCRIPT_DIR/panda-octo-requirements.txt" \
  "${WHEEL_ARGS[@]}"
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$POLICY_PYTHON" -I - "$CHECKPOINT_DIR" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [str(root / name) for name in ("python-deps", "octo-source", "dlimp-source")]
import jax
import flax
import tensorflow as tf
import octo.model.octo_model
import dlimp

assert jax.__version__ == "0.4.20"
assert flax.__version__ == "0.7.5"
assert tf.__version__ == "2.15.1"
assert jax.default_backend() == "cpu"
assert Path(octo.model.octo_model.__file__).is_relative_to(root / "octo-source")
assert Path(dlimp.__file__).is_relative_to(root / "dlimp-source")
data = {
    "python": sys.version, "cpu_import_check": True,
    "octo_source": octo.model.octo_model.__file__, "dlimp_source": dlimp.__file__,
    "versions": {name: importlib.metadata.version(name) for name in (
        "jax", "flax", "tensorflow-cpu", "tensorflow-datasets", "tensorflow-metadata",
        "array-record", "click", "googleapis-common-protos", "promise", "psutil", "toml",
    )},
}
(root / "panda-octo-import-provenance.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PY
