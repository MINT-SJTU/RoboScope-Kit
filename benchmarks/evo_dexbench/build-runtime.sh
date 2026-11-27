#!/usr/bin/env bash
set -Eeuo pipefail

# Build the pinned Evo-DexBench evaluation runtime at any chosen install prefix.
#
#   <install-dir>/
#     Evo-DexBench/             pinned upstream source, installed editable, with .roboscope-revision
#     python/                   the uv-managed CPython 3.11 the environment runs on
#     env/                      runtime environment: torch 2.5.1, runtime-requirements.txt, Kit
#     assets/                   upstream asset root (public bundle + PartNet-Mobility), verified
#     benchmark-manifest.json   runtime paths filled in for this prefix
#
# Needs Linux x86_64, an NVIDIA driver with Vulkan (rendering), uv 0.11.14, curl and tar.
# Environment:
#   EVO_DEXBENCH_SOURCE_ARCHIVE  an already downloaded source tarball (checked against SOURCE_SHA256)
#   EVO_DEXBENCH_ASSET_DIR       an existing, complete upstream asset root to use instead of downloading
#   EVO_DEXBENCH_PARTNET_DIR     the user's own PartNet-Mobility download (one directory per model id);
#                                PartNet is gated, so this script never downloads it without HF_TOKEN
#   HF_TOKEN                     a token whose account accepted the PartNet-Mobility terms
#   TORCH_INDEX_URL              default https://download.pytorch.org/whl/cu124: physics steps on the CPU,
#                                but ManiSkill hands rendered camera textures to torch on the GPU
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || { echo "Evo-DexBench builds require Linux x86_64" >&2; exit 3; }
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
SOURCE_COMMIT="818ac28e42e9199f0cbccce8617cb686519d8abf"
SOURCE_SHA256="b3bb479ca93d976745436261f22be33b1902d72f3fdb724916634107369d5d33"
SOURCE_URL="https://codeload.github.com/MINT-SJTU/Evo-DexBench/tar.gz/$SOURCE_COMMIT"
ASSET_BUNDLE_REVISION="09868871eafe6fffdbd8e37b053b6b1f7e9ea6ec"
SOURCE_ROOT="$INSTALL_DIR/Evo-DexBench"
ENV_PREFIX="$INSTALL_DIR/env"
ENV_PYTHON="$ENV_PREFIX/bin/python"
ASSET_ROOT="$INSTALL_DIR/assets"
KIT_SOURCE="$(cd -- "$SCRIPT_DIR/../../packages/roboscope-kit" && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
[[ "$("$UV_BIN" --version | awk '{print $2}')" == 0.11.14 ]] || {
  echo "uv 0.11.14 is required; set UV_BIN to its executable" >&2; exit 3;
}
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/python"
export UV_NO_PROGRESS=1
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

STAGING="$(mktemp -d "$INSTALL_DIR/.build.XXXXXX")"
trap 'rm -rf -- "$STAGING"' EXIT
ARCHIVE="${EVO_DEXBENCH_SOURCE_ARCHIVE:-}"
if [[ -z "$ARCHIVE" ]]; then
  install -d "$INSTALL_DIR/downloads"
  ARCHIVE="$INSTALL_DIR/downloads/evo-dexbench-$SOURCE_COMMIT.tar.gz"
  if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 3 --output "$ARCHIVE.part" "$SOURCE_URL"
    mv -- "$ARCHIVE.part" "$ARCHIVE"
  fi
fi
[[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || {
  echo "Evo-DexBench source archive checksum mismatch" >&2; exit 4;
}
if [[ ! -d "$SOURCE_ROOT" ]]; then
  install -d "$STAGING/source"
  tar -xzf "$ARCHIVE" --strip-components=1 -C "$STAGING/source"
  for path in pyproject.toml dex_benchmark/evaluation/protocol_parallel.py \
    dex_benchmark/resources/eval/protocol_v1.yaml assets/manifest.yaml; do
    [[ -f "$STAGING/source/$path" ]] || { echo "source is missing $path" >&2; exit 4; }
  done
  printf '%s\n' "$SOURCE_COMMIT" > "$STAGING/source/.roboscope-revision"
  mv -- "$STAGING/source" "$SOURCE_ROOT"
else
  [[ "$(cat "$SOURCE_ROOT/.roboscope-revision" 2>/dev/null)" == "$SOURCE_COMMIT" ]] || {
    echo "existing source has no matching provenance; use a fresh install directory" >&2; exit 4;
  }
fi

"$UV_BIN" python install 3.11.15
PYTHON_BIN="$("$UV_BIN" python find --managed-python 3.11.15)"
[[ -x "$ENV_PYTHON" ]] || "$UV_BIN" venv --python "$PYTHON_BIN" "$ENV_PREFIX"
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
"$UV_BIN" pip install --python "$ENV_PYTHON" --index-url "$TORCH_INDEX_URL" torch==2.5.1
"$UV_BIN" pip install --python "$ENV_PYTHON" --requirement "$SCRIPT_DIR/runtime-requirements.txt"
# Editable, so dex_benchmark resolves to the pinned source tree and its revision marker.
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation --editable "$SOURCE_ROOT"
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps "$KIT_SOURCE"
"$UV_BIN" pip check --python "$ENV_PYTHON"
"$UV_BIN" pip freeze --python "$ENV_PYTHON" > "$INSTALL_DIR/runtime-requirements-resolved.txt"

if [[ -n "${EVO_DEXBENCH_ASSET_DIR:-}" ]]; then
  EXTERNAL_ASSETS="$(cd -- "$EVO_DEXBENCH_ASSET_DIR" && pwd -P)"
  if [[ -e "$ASSET_ROOT" || -L "$ASSET_ROOT" ]]; then
    [[ "$(readlink -f -- "$ASSET_ROOT")" == "$EXTERNAL_ASSETS" ]] || {
      echo "existing runtime assets point to another directory; use a fresh prefix" >&2; exit 4;
    }
  else
    ln -s -- "$EXTERNAL_ASSETS" "$ASSET_ROOT"
  fi
else
  install -d "$ASSET_ROOT"
  PARTNET_ARGS=(--skip-partnet)
  if [[ -n "${EVO_DEXBENCH_PARTNET_DIR:-}" ]]; then
    "$ENV_PYTHON" - "$EVO_DEXBENCH_PARTNET_DIR" "$ASSET_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path

from dex_benchmark.asset_download import required_partnet_model_ids
from dex_benchmark.partnet_overrides import generate_partnet_overrides

source, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
missing = [model for model in required_partnet_model_ids() if not (source / model / "mobility.urdf").is_file()]
if missing:
    raise SystemExit(f"EVO_DEXBENCH_PARTNET_DIR lacks PartNet-Mobility models: {', '.join(missing)}")
for model in required_partnet_model_ids():
    target = root / "partnet_mobility" / model
    if not target.exists():
        shutil.copytree(source / model, target)
generate_partnet_overrides(root / "partnet_mobility", root / "partnet_mobility_modified")
PY
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    PARTNET_ARGS=()
  else
    echo "PartNet-Mobility is gated: set EVO_DEXBENCH_PARTNET_DIR to your own download, or HF_TOKEN" >&2
    echo "for an account that accepted its terms (https://huggingface.co/datasets/sapien-sim/PartNetMobility)." >&2
    exit 5
  fi
  "$ENV_PYTHON" -m dex_benchmark assets download --asset-dir "$ASSET_ROOT" \
    --bundle-revision "$ASSET_BUNDLE_REVISION" --no-hf-mirror "${PARTNET_ARGS[@]}"
fi
"$ENV_PYTHON" -m dex_benchmark assets verify --asset-dir "$ASSET_ROOT"

"$ENV_PYTHON" - "$SCRIPT_DIR/benchmark-manifest.json" "$INSTALL_DIR" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
manifest["runtime"] = {
    "python": str(root / "env/bin/python"), "benchmark_root": str(root / "Evo-DexBench"),
    "asset_root": str(root / "assets"),
}
(root / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
echo "Evo-DexBench runtime ready: $INSTALL_DIR/benchmark-manifest.json"
