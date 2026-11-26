#!/usr/bin/env bash
set -Eeuo pipefail

# Build the pinned ManiSkill multi-task runtime at any chosen install prefix.
# Needs Linux x86_64, NVIDIA GPU/driver, Vulkan loader, uv 0.11.14, curl and tar.
# Rebuild at a new prefix to change location; an existing Python env is not relocatable.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == Linux:x86_64 ]] || {
  echo "ManiSkill runtime builds require Linux x86_64" >&2; exit 3;
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
SOURCE_COMMIT="baab60ede2e89167c1b7aaed41a9aa8e690a9d1e"
SOURCE_SHA256="db7f1fb49038a1a11612876022ba31fdb8c9095e4e3f4a408728abf8b821f49e"
SOURCE_URL="https://codeload.github.com/mani-skill/ManiSkill/tar.gz/$SOURCE_COMMIT"
SOURCE_ROOT="$INSTALL_DIR/ManiSkill"
ENV_PREFIX="$INSTALL_DIR/env"
ENV_PYTHON="$ENV_PREFIX/bin/python"
RUNTIME_HOME="$INSTALL_DIR/home"
ASSET_ROOT="$INSTALL_DIR/assets"
PHYSX_VERSION="105.1-physx-5.3.1.patch0"
PHYSX_PARENT="$RUNTIME_HOME/.sapien/physx/$PHYSX_VERSION"
PHYSX_LIBRARY="$PHYSX_PARENT/libPhysXGpu_64.so"
PHYSX_URL="https://github.com/sapien-sim/physx-precompiled/releases/download/$PHYSX_VERSION/linux-so.zip"
KIT_SOURCE="$(cd -- "$SCRIPT_DIR/../../packages/roboscope-kit" && pwd -P)"
UV_BIN="${UV_BIN:-uv}"
[[ "$("$UV_BIN" --version | awk '{print $2}')" == 0.11.14 ]] || {
  echo "uv 0.11.14 is required; set UV_BIN to its executable" >&2; exit 3;
}
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/python"
export UV_NO_PROGRESS=1
export PIP_CONFIG_FILE=/dev/null
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${MANISKILL_ASSET_DIR:-}" ]]; then
  install -d "$MANISKILL_ASSET_DIR"
  EXTERNAL_ASSET_ROOT="$(cd -- "$MANISKILL_ASSET_DIR" && pwd -P)"
  if [[ "$EXTERNAL_ASSET_ROOT" != "$ASSET_ROOT" ]]; then
    if [[ -e "$ASSET_ROOT" || -L "$ASSET_ROOT" ]]; then
      [[ "$(readlink -f -- "$ASSET_ROOT")" == "$EXTERNAL_ASSET_ROOT" ]] || {
        echo "existing runtime assets point to another directory; use a fresh prefix" >&2; exit 4;
      }
    else
      ln -s -- "$EXTERNAL_ASSET_ROOT" "$ASSET_ROOT"
    fi
  fi
else
  install -d "$ASSET_ROOT"
fi
export MS_ASSET_DIR="$ASSET_ROOT"

STAGING="$(mktemp -d "$INSTALL_DIR/.build.XXXXXX")"
trap 'rm -rf -- "$STAGING"' EXIT
ARCHIVE="${MANISKILL_SOURCE_ARCHIVE:-}"
if [[ -z "$ARCHIVE" ]]; then
  install -d "$INSTALL_DIR/downloads"
  ARCHIVE="$INSTALL_DIR/downloads/maniskill-$SOURCE_COMMIT.tar.gz"
  if [[ ! -f "$ARCHIVE" ]]; then
    curl --fail --location --retry 3 --output "$ARCHIVE.part" "$SOURCE_URL"
    mv -- "$ARCHIVE.part" "$ARCHIVE"
  fi
fi
[[ -f "$ARCHIVE" ]] || { echo "source archive not found: $ARCHIVE" >&2; exit 4; }
[[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" == "$SOURCE_SHA256" ]] || {
  echo "ManiSkill source archive checksum mismatch" >&2; exit 4;
}
if [[ ! -d "$SOURCE_ROOT" ]]; then
  install -d "$STAGING/source"
  tar -xzf "$ARCHIVE" --strip-components=1 -C "$STAGING/source"
  for path in setup.py mani_skill/envs/tasks/tabletop/pick_cube.py \
    mani_skill/agents/robots/panda/panda.py \
    mani_skill/utils/scene_builder/table/assets/table.glb; do
    [[ -f "$STAGING/source/$path" ]] || { echo "source is missing $path" >&2; exit 4; }
  done
  printf '%s\n' "$SOURCE_SHA256" > "$STAGING/source/.roboscope-source-sha256"
  mv -- "$STAGING/source" "$SOURCE_ROOT"
else
  [[ -f "$SOURCE_ROOT/.roboscope-source-sha256" ]] &&
    [[ "$(cat "$SOURCE_ROOT/.roboscope-source-sha256")" == "$SOURCE_SHA256" ]] || {
      echo "existing source has no matching provenance; use a fresh install directory" >&2; exit 4;
    }
fi
"$UV_BIN" python install 3.11.15
PYTHON_BIN="$("$UV_BIN" python find --managed-python 3.11.15)"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
if [[ ! -x "$ENV_PYTHON" ]]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$ENV_PREFIX"
fi
"$ENV_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3, 11, 15), sys.version'
PHYSX_SHA256="$("$ENV_PYTHON" - "$SCRIPT_DIR/benchmark-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

digest = json.loads(Path(sys.argv[1]).read_text())["benchmark_protocols"]["maniskill"]["physx_gpu_library_sha256"]
if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    raise ValueError("ManiSkill manifest requires a valid physx_gpu_library_sha256")
print(digest)
PY
)"
install -d "$PHYSX_PARENT"
if [[ ! -f "$PHYSX_LIBRARY" ]]; then
  if [[ -n "${MANISKILL_PHYSX_GPU_LIBRARY:-}" ]]; then
    [[ "$(sha256sum "$MANISKILL_PHYSX_GPU_LIBRARY" | awk '{print $1}')" == "$PHYSX_SHA256" ]] || {
      echo "supplied PhysX GPU library checksum mismatch" >&2; exit 4;
    }
    install -m 0644 "$MANISKILL_PHYSX_GPU_LIBRARY" "$PHYSX_LIBRARY"
  else
    install -d "$INSTALL_DIR/downloads"
    PHYSX_ARCHIVE="$INSTALL_DIR/downloads/physx-$PHYSX_VERSION-linux-so.zip"
    if [[ ! -f "$PHYSX_ARCHIVE" ]]; then
      curl --fail --location --retry 3 --output "$PHYSX_ARCHIVE.part" "$PHYSX_URL"
      mv -- "$PHYSX_ARCHIVE.part" "$PHYSX_ARCHIVE"
    fi
    "$ENV_PYTHON" - "$PHYSX_ARCHIVE" "$PHYSX_LIBRARY" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

with zipfile.ZipFile(sys.argv[1]) as archive:
    members = [name for name in archive.namelist() if Path(name).name == "libPhysXGpu_64.so"]
    if len(members) != 1:
        raise RuntimeError("PhysX release ZIP does not contain exactly one GPU library")
    with archive.open(members[0]) as source, open(sys.argv[2], "wb") as destination:
        shutil.copyfileobj(source, destination)
PY
  fi
fi
[[ "$(sha256sum "$PHYSX_LIBRARY" | awk '{print $1}')" == "$PHYSX_SHA256" ]] || {
  echo "PhysX GPU library checksum mismatch" >&2; exit 4;
}
"$UV_BIN" pip install --python "$ENV_PYTHON" --requirement "$SCRIPT_DIR/runtime-requirements.txt"
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation "$SOURCE_ROOT"
"$UV_BIN" pip install --python "$ENV_PYTHON" --no-deps --no-build-isolation "$KIT_SOURCE"
"$UV_BIN" pip check --python "$ENV_PYTHON"
"$UV_BIN" pip freeze --python "$ENV_PYTHON" > "$INSTALL_DIR/runtime-requirements-resolved.txt"
install -m 0644 "$SCRIPT_DIR/benchmark-manifest.json" "$INSTALL_DIR/benchmark-manifest.json"
install -m 0755 "$SCRIPT_DIR/prepare-assets.py" "$INSTALL_DIR/prepare-assets.py"
IFS=',' read -r -a ASSET_SUITES <<< "${MANISKILL_ASSET_SUITES:-baseline_state_v1}"
IFS=',' read -r -a ASSET_TASKS <<< "${MANISKILL_ASSET_TASKS:-}"
ASSET_ARGUMENTS=()
if [[ -n "${MANISKILL_ASSET_ARCHIVE_DIR:-}" ]]; then
  ASSET_ARGUMENTS+=(--archive-dir "$MANISKILL_ASSET_ARCHIVE_DIR")
fi
for suite in "${ASSET_SUITES[@]}"; do
  [[ -n "$suite" ]] && ASSET_ARGUMENTS+=(--suite "$suite")
done
for task in "${ASSET_TASKS[@]}"; do
  [[ -n "$task" ]] && ASSET_ARGUMENTS+=(--task-id "$task")
done
HOME="$RUNTIME_HOME" "$ENV_PYTHON" -I "$INSTALL_DIR/prepare-assets.py" \
  --benchmark-manifest "$INSTALL_DIR/benchmark-manifest.json" --asset-dir "$ASSET_ROOT" \
  "${ASSET_ARGUMENTS[@]}" --download --output "$INSTALL_DIR/asset-provenance.json"
HOME="$RUNTIME_HOME" "$ENV_PYTHON" -I - "$INSTALL_DIR" "$SOURCE_COMMIT" "$SOURCE_SHA256" <<'PY'
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mani_skill
import roboscope_kit
import sapien
import sapien.physx
import torch

expected = {
    "mani_skill": "3.0.0b15", "sapien": "3.0.0b1", "gymnasium": "0.29.1",
    "torch": "2.6.0", "numpy": "1.26.4",
}
versions = {name: version(name) for name in expected}
for name, wanted in expected.items():
    if versions[name].split("+")[0] != wanted:
        raise RuntimeError(f"{name} is {versions[name]}, expected {wanted}")
if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime build requires torch.cuda.is_available()")
if sapien.physx.version() != "105.1-physx-5.3.1.patch0":
    raise RuntimeError(f"Unexpected PhysX version: {sapien.physx.version()}")
root = Path(sys.argv[1])
records = []
asset_root = Path(mani_skill.__file__).parent
for relative in (
    "assets/robots/panda/panda_v2.urdf",
    "assets/robots/panda/panda_stick.urdf",
    "utils/scene_builder/table/assets/table.glb",
):
    path = asset_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Bundled profile asset is missing: {path}")
    records.append({
        "path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
(root / "build-provenance.json").write_text(json.dumps({
    "source_commit": sys.argv[2], "source_archive_sha256": sys.argv[3],
    "python": sys.version, "versions": versions, "kit_version": version("roboscope-kit"),
    "kit_module": roboscope_kit.__file__, "maniskill_module": mani_skill.__file__,
    "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
    "profile_asset_checksums": records,
    "physx_gpu_library": {
        "version": sapien.physx.version(),
        "relative_path": "home/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so",
        "sha256": hashlib.sha256((root / "home/.sapien/physx/105.1-physx-5.3.1.patch0/libPhysXGpu_64.so").read_bytes()).hexdigest(),
    },
    "manifest_sha256": hashlib.sha256((root / "benchmark-manifest.json").read_bytes()).hexdigest(),
    "asset_provenance": "asset-provenance.json",
    "scope": "Manifest-driven ManiSkill state/RGB suites; representative runtime validation",
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(versions))
PY
SELF_TEST_OUTPUT="$INSTALL_DIR/self-test-$(date -u +%Y%m%dT%H%M%SZ)-$$"
HOME="$RUNTIME_HOME" "$ENV_PYTHON" -I -m roboscope_kit.runtime.benchmark --benchmark maniskill --self-test \
  --benchmark-manifest "$INSTALL_DIR/benchmark-manifest.json" --output-dir "$SELF_TEST_OUTPUT"
echo "Built ManiSkill $SOURCE_COMMIT in $INSTALL_DIR"
echo "Self-test evidence: $SELF_TEST_OUTPUT"
