#!/usr/bin/env bash
set -Eeuo pipefail

# Build the BEHAVIOR Challenge 2026 runtime (BEHAVIOR-1K v3.9.3-post1, OmniGibson on Isaac Sim 5.1)
# under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     python/                   the uv-managed CPython 3.11 the environment runs on
#     env/                      venv: exactly runtime-requirements.txt, then BDDL, OmniGibson and
#                               JoyLo (the evaluator imports its gello package) from the checkout
#     src/BEHAVIOR-1K/          StanfordVL/BEHAVIOR-1K at the v3.9.3-post1 commit
#     data/                     OMNIGIBSON_DATA_PATH: behavior-1k-assets 3.9.0, omnigibson-robot-assets
#                               3.8.2, 2026-challenge-task-instances, and the dataset key
#     pip-freeze.txt            what env/ actually contains
#     check.json                the data check and one rendered instance load (check.py)
#
# Usage: build-runtime.sh --accept-licenses <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark behavior --suite challenge_2026 \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Licences. The runtime runs NVIDIA Isaac Sim, distributed under the NVIDIA Isaac Sim End User
# License Agreement, and the BEHAVIOR Data Bundle, whose assets are encrypted and licensed for
# non-commercial academic use inside OmniGibson only, without redistribution. Passing
# --accept-licenses means that you have read both and accept them: the build then downloads the
# dataset key with BEHAVIOR-1K's own download_key() and the kit runs Isaac Sim with
# OMNI_KIT_ACCEPT_EULA=YES. Without it the build stops before downloading anything. Nothing
# licensed is committed to the repository.
#
# Host prerequisites: Linux x86_64 with an NVIDIA driver that supports Isaac Sim 5.1 and a working
# Vulkan ICD (vulkaninfo lists the GPU), git, curl, sha256sum, and uv 0.11.14. The datasets need
# about 70 GB while the archives are unpacked and 36 GB afterwards; env/ needs about 25 GB.
#
# Sources. Every package comes from runtime-requirements.txt, an exact lock resolved with
# BEHAVIOR-1K's setup.sh (--omnigibson --bddl --joylo --eval) on this commit. Indexes can be
# replaced by mirrors that serve the same files: BEHAVIOR_PYPI_INDEX (PyPI), BEHAVIOR_TORCH_INDEX
# (download.pytorch.org/whl/cu128 layout), BEHAVIOR_TORCH_FIND_LINKS (a flat directory of the same
# wheels), BEHAVIOR_NVIDIA_INDEX (pypi.nvidia.com), and HF_ENDPOINT (Hugging Face). Git ignores the
# user's global and system configuration; git, curl, uv and Python honour https_proxy. Behind a
# TLS-intercepting proxy, set UV_SYSTEM_CERTS=1 and SSL_CERT_FILE to a bundle with the proxy's CA.
# The dataset archives are checked against their Hugging Face SHA-256 whatever the endpoint. Set
# BEHAVIOR_ZIPS to a directory already holding them to skip their download; otherwise they are
# downloaded into <install-dir>/downloads and removed after unpacking. Re-running the script on an
# existing <install-dir> reuses what is already verified.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
usage() {
  echo "usage: $0 --accept-licenses <install-dir>" >&2
  exit 2
}
ACCEPT_LICENSES=0
INSTALL_ARG=""
for arg in "$@"; do
  case "${arg}" in
    --accept-licenses) ACCEPT_LICENSES=1 ;;
    -*) usage ;;
    *)
      [[ -z "${INSTALL_ARG}" ]] || usage
      INSTALL_ARG="${arg}"
      ;;
  esac
done
[[ -n "${INSTALL_ARG}" ]] || usage
set -- "${INSTALL_ARG}"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "BEHAVIOR runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ "${ACCEPT_LICENSES}" != 1 ]]; then
  echo "The BEHAVIOR runtime uses NVIDIA Isaac Sim and the BEHAVIOR Data Bundle, which have their own licences:" >&2
  echo "  https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html" >&2
  echo "  https://behavior.stanford.edu/ (BEHAVIOR Data Bundle License Agreement, printed by OmniGibson)" >&2
  echo "Read them, then rerun with --accept-licenses; passing it means that you accept both." >&2
  exit 3
fi

MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
LOCK="${SCRIPT_DIR}/runtime-requirements.txt"
KIT_SOURCE="${KIT_ROOT}/packages/roboscope-kit/src"
for input in "${MANIFEST}" "${LOCK}"; do
  [[ -f "${input}" ]] || {
    echo "BEHAVIOR build input is missing: ${input}" >&2
    exit 3
  }
done
[[ -d "${KIT_SOURCE}/roboscope_kit" ]] || {
  echo "roboscope-kit source is missing: ${KIT_SOURCE}" >&2
  exit 3
}
for executable in git curl sha256sum; do
  command -v "${executable}" >/dev/null 2>&1 || {
    echo "Missing required executable: ${executable}" >&2
    exit 3
  }
done
command -v uv >/dev/null 2>&1 || {
  echo "uv 0.11.14 is required to build the BEHAVIOR runtime" >&2
  exit 3
}
UV_BIN="$(command -v uv)"
if [[ "$("${UV_BIN}" --version)" != "uv 0.11.14 "* ]]; then
  echo "uv 0.11.14 is required to build the BEHAVIOR runtime" >&2
  exit 3
fi
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1

install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
PYTHON_INSTALL_DIR="${INSTALL_DIR}/python"
VENV="${INSTALL_DIR}/env"
PYTHON="${VENV}/bin/python"
PYTHON_VERSION="3.11.15"
SOURCE="${INSTALL_DIR}/src/BEHAVIOR-1K"
DATA="${INSTALL_DIR}/data"
ZIP_DIR="${BEHAVIOR_ZIPS:-${INSTALL_DIR}/downloads}"

# BEHAVIOR-1K v3.9.3-post1: the release the organizers use for the 2026 final evaluation.
BEHAVIOR_1K_URL="https://github.com/StanfordVL/BEHAVIOR-1K.git"
BEHAVIOR_1K_COMMIT="bd049de3119acdcdf2334fe9e1ebe060fa20c108"
# behavior-1k/zipped-datasets: the archives v3.9.3-post1's asset_utils downloads
# (BEHAVIOR_1K_ASSET_VERSION 3.9.0, OMNIGIBSON_ROBOT_ASSETS_VERSION 3.8.2, the 2026 task instances).
ZIPPED_DATASETS_REVISION="9f0d57d465726976ed98138d3f8b8ca3e2186775"
ZIPS=(
  "2026-challenge-task-instances 2026-challenge-task-instances.zip 108543852 d4ac3d72dd585178e85d542f28d1b48233b886c80e196e2f9d3aa5993ba21f81"
  "omnigibson-robot-assets omnigibson-robot-assets-3.8.2.zip 641004353 3d813b2181e0581cf2300a40892de70f8475fe59346ceeea4fb9bf7ff21ce126"
  "behavior-1k-assets behavior-1k-assets-3.9.0.zip 31457673073 09e9fce600f841dc611aa96c0b1b9f9074f56f0e67898b37b39bd00c38a0095e"
)
PYPI_INDEX="${BEHAVIOR_PYPI_INDEX:-https://pypi.org/simple}"
TORCH_INDEX="${BEHAVIOR_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
NVIDIA_INDEX="${BEHAVIOR_NVIDIA_INDEX:-https://pypi.nvidia.com}"
PYG_LINKS="https://data.pyg.org/whl/torch-2.7.0+cu128.html"
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"

for pin in "torch==2.7.0+cu128" "isaacsim==5.1.0.0" "numpy==1." "cffi==1.17.1" "torch-cluster==" \
  "lerobot @ git+https://github.com/wensi-ai/lerobot@436812bd8ee39b768c645c248c91f1330834e687"; do
  grep -qF "${pin}" "${LOCK}" || {
    echo "BEHAVIOR runtime lock must pin ${pin}" >&2
    exit 3
  }
done

fetch() {
  # fetch <url> <sha256> <target>: download with resume until the file has the pinned digest.
  local url="$1" sha256="$2" target="$3" attempt
  if [[ -f "${target}" ]] && printf '%s  %s\n' "${sha256}" "${target}" | sha256sum -c - >/dev/null 2>&1; then
    return 0
  fi
  mkdir -p "$(dirname -- "${target}")"
  for ((attempt = 1; attempt <= 20; attempt++)); do
    if curl -fLsS --retry 2 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
      -C - --output "${target}.part" "${url}"; then
      if printf '%s  %s\n' "${sha256}" "${target}.part" | sha256sum -c - >/dev/null 2>&1; then
        mv -- "${target}.part" "${target}"
        return 0
      fi
      # A complete transfer with the wrong digest cannot be resumed into the right one.
      rm -f -- "${target}.part"
    else
      sleep 10
    fi
  done
  echo "Could not download ${url} with SHA-256 ${sha256}" >&2
  return 1
}

# Source: a clean checkout at the pinned commit; never reuse modified source.
if [[ ! -d "${SOURCE}/.git" ]]; then
  mkdir -p "${SOURCE}"
  git init -q "${SOURCE}"
  git -C "${SOURCE}" remote add origin "${BEHAVIOR_1K_URL}"
fi
if ! git -C "${SOURCE}" cat-file -e "${BEHAVIOR_1K_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${SOURCE}" fetch -q --depth 1 origin "${BEHAVIOR_1K_COMMIT}"
fi
if [[ -n "$(git -C "${SOURCE}" status --porcelain --untracked-files=normal | grep -v '\.egg-info/$' || true)" ]]; then
  echo "${SOURCE} contains local changes; use a clean install directory" >&2
  exit 5
fi
git -C "${SOURCE}" -c advice.detachedHead=false checkout -q --detach "${BEHAVIOR_1K_COMMIT}"
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${BEHAVIOR_1K_COMMIT}" ]] || {
  echo "${SOURCE} is not at ${BEHAVIOR_1K_COMMIT}" >&2
  exit 5
}

# Python environment. Keep the interpreter inside <install-dir> so the environment does not
# depend on the user's uv cache.
export UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}"
"${UV_BIN}" python install --no-bin --no-progress "${PYTHON_VERSION}"
if [[ ! -x "${PYTHON}" ]]; then
  "${UV_BIN}" venv --python "${PYTHON_VERSION}" --managed-python --no-python-downloads "${VENV}"
fi
torch_sources=(--extra-index-url "${TORCH_INDEX}")
if [[ -n "${BEHAVIOR_TORCH_FIND_LINKS:-}" ]]; then
  torch_sources=(--find-links "${BEHAVIOR_TORCH_FIND_LINKS}")
fi
# Exactly the lock, without resolving: the lock already holds what setup.sh's sequence of installs
# leaves behind (Isaac Sim's pins overridden by cffi 1.17.1 and websockets 15, as setup.sh does).
OMNI_KIT_ACCEPT_EULA=YES "${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --no-deps \
  --index-strategy unsafe-best-match \
  --index-url "${PYPI_INDEX}" \
  --extra-index-url "${NVIDIA_INDEX}" \
  "${torch_sources[@]}" \
  --find-links "${PYG_LINKS}" \
  --requirements "${LOCK}"
# The evaluator's own packages, in development mode as setup.sh installs them.
"${UV_BIN}" pip install --python "${PYTHON}" --no-progress --no-deps --no-build-isolation \
  --editable "${SOURCE}/bddl3" --editable "${SOURCE}/OmniGibson" --editable "${SOURCE}/joylo"
# setup.sh: drop the websockets and packaging copies bundled with Isaac Sim's extensions, which
# shadow the environment's websockets 15 and packaging.
ISAAC_PATH="$(OMNI_KIT_ACCEPT_EULA=YES "${PYTHON}" -I -c "import isaacsim, os; print(os.environ.get('ISAAC_PATH', ''))")"
[[ -n "${ISAAC_PATH}" && -d "${ISAAC_PATH}/extscache" ]] || {
  echo "isaacsim did not report its ISAAC_PATH" >&2
  exit 5
}
find "${ISAAC_PATH}/extscache" -type d -name websockets -path "*/pip_prebundle/*" -prune -exec rm -rf {} +
rm -rf -- "${ISAAC_PATH}/extscache/omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/pip_prebundle/packaging"
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"

# The environment is exactly the lock plus the three source packages.
"${PYTHON}" -I - "${LOCK}" "${INSTALL_DIR}/pip-freeze.txt" "${SOURCE}" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

lock_path, freeze_path, source = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()


def requirements(path):
    lines = (line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("-e ")}


lock = requirements(lock_path)
frozen = requirements(freeze_path)
if frozen != lock:
    raise SystemExit(f"env differs from the lock: extra {sorted(frozen - lock)}, missing {sorted(lock - frozen)}")
for name, expected in (("torch", "2.7.0+cu128"), ("isaacsim", "5.1.0.0"), ("cffi", "1.17.1")):
    if version(name) != expected:
        raise SystemExit(f"BEHAVIOR runtime {name} is {version(name)}, expected {expected}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"BEHAVIOR runtime Python is {sys.version}, expected 3.11")
import bddl
import gello
import omnigibson

for module in (bddl, gello, omnigibson):
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise SystemExit(f"{module.__name__} is not imported from {source}")
print("BEHAVIOR runtime versions verified")
PY

# Datasets: download or reuse each pinned archive, verify it, and unpack it once.
mkdir -p "${DATA}"
for entry in "${ZIPS[@]}"; do
  read -r dataset filename size sha256 <<<"${entry}"
  marker="${DATA}/${dataset}/.zip-sha256"
  if [[ -f "${marker}" && "$(cat -- "${marker}")" == "${sha256}" ]]; then
    continue
  fi
  archive="${ZIP_DIR}/${filename}"
  fetch "${HF_BASE}/datasets/behavior-1k/zipped-datasets/resolve/${ZIPPED_DATASETS_REVISION}/${filename}" \
    "${sha256}" "${archive}"
  [[ "$(stat -c %s -- "${archive}")" == "${size}" ]] || {
    echo "${archive} is not ${size} bytes" >&2
    exit 5
  }
  rm -rf -- "${DATA:?}/${dataset}" "${DATA:?}/${dataset}.partial"
  # asset_utils.download_and_unpack_zipped_dataset: the archive's members go into data/<dataset>/.
  "${PYTHON}" -I - "${archive}" "${DATA}/${dataset}.partial" <<'PY'
import sys
import zipfile
from pathlib import Path

archive, root = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        if not (root / member.filename).resolve().is_relative_to(root):
            raise SystemExit(f"{archive} member {member.filename} is outside the dataset")
    bundle.extractall(root)
PY
  mv -- "${DATA}/${dataset}.partial" "${DATA}/${dataset}"
  printf '%s\n' "${sha256}" > "${marker}"
  if [[ -z "${BEHAVIOR_ZIPS:-}" ]]; then
    rm -f -- "${archive}"
  fi
done
# The dataset key, with BEHAVIOR-1K's own download (the licence was accepted above).
if [[ ! -f "${DATA}/omnigibson.key" ]]; then
  OMNIGIBSON_DATA_PATH="${DATA}" "${PYTHON}" -I -c \
    "from omnigibson.utils.asset_utils import download_key; download_key()"
fi

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"

# The pinned data, then a headless launch: the quick-test task's scene, its first public test
# instance, and the first observation mapped to the SDK features, through the worker's own code.
env -u PYTHONPATH PYTHONPATH="${KIT_SOURCE}" OMNIGIBSON_DATA_PATH="${DATA}" OMNI_KIT_ACCEPT_EULA=YES \
  PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON}" -m roboscope_kit.runtime.benchmarks.behavior.check --report "${INSTALL_DIR}/check.json"
echo "Built BEHAVIOR challenge_2026 in ${INSTALL_DIR}"

# The kit runs its own check against the installed manifest. It needs a Python with
# roboscope-kit installed; the BEHAVIOR environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  rm -rf -- "${INSTALL_DIR}/self-test"
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark behavior --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi
