#!/usr/bin/env bash
set -Eeuo pipefail

# Build the SimplerEnv evaluation environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     python/                   the uv-managed CPython 3.11 the environment runs on
#     env/                      venv with exactly runtime-requirements.txt, SimplerEnv, and ManiSkill2_real2sim
#     src/SimplerEnv/           SimplerEnv at the pinned commit with its ManiSkill2_real2sim submodule
#                               (which carries the scene, object, and real-image overlay assets)
#     pip-freeze.txt            what env/ actually contains
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark simpler_env \
#          --suite google_robot_visual_matching \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Host prerequisites: git, uv 0.11.14, and an NVIDIA GPU with a working Vulkan driver
# (SAPIEN 2.2 renders through Vulkan; the build renders one reset per check). When the
# driver also installs its Vulkan SC ICD (nvidia_icd_vksc.json), the Vulkan loader loads it
# too and every SAPIEN renderer teardown segfaults while unloading it; export
# VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json (the regular NVIDIA ICD) for the
# build and every evaluation. The build refuses to start without it on such a host.
#
# SimplerEnv and ManiSkill2_real2sim are installed editable from the pinned checkout, as
# SimplerEnv's README installs them: ManiSkill2_real2sim finds its assets in the checkout's
# data/ directory, and the official scripts' relative overlay paths resolve from src/SimplerEnv.
# Re-running the script on an existing <install-dir> reuses the venv and re-applies the lock.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ $# -eq 1 ]] || {
  echo "usage: $0 <install-dir>" >&2
  exit 2
}
install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"

MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
LOCK="${SCRIPT_DIR}/runtime-requirements.txt"
KIT_SOURCE="${KIT_ROOT}/packages/roboscope-kit/src"
PYTHON_INSTALL_DIR="${INSTALL_DIR}/python"
VENV="${INSTALL_DIR}/env"
PYTHON="${VENV}/bin/python"
SOURCE="${INSTALL_DIR}/src/SimplerEnv"
PYTHON_VERSION="3.11.15"
SIMPLER_ENV_URL="https://github.com/simpler-env/SimplerEnv.git"
# Head of SimplerEnv main on 2026-09-30 ("pin transformers<5").
SIMPLER_ENV_COMMIT="eddb5690749a1e34267807c0f7b5eede323918d7"
# The ManiSkill2_real2sim submodule commit that SimplerEnv's commit records.
REAL2SIM_COMMIT="ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "SimplerEnv runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ ! -f "${MANIFEST}" || ! -f "${LOCK}" || ! -d "${KIT_SOURCE}/roboscope_kit" ]]; then
  echo "SimplerEnv manifest, runtime lock or kit source is missing" >&2
  exit 3
fi
for pin in "sapien==2.2.2" "numpy==1.24.4" "tensorflow==2.15.1" "tf-agents==0.19.0" \
  "tensorflow-hub==0.16.0" "gymnasium==0.29.1" "setuptools=="; do
  grep -q -- "^${pin//./\\.}" "${LOCK}" || {
    echo "SimplerEnv runtime lock must pin ${pin}" >&2
    exit 3
  }
done
command -v git >/dev/null 2>&1 || {
  echo "git is required to fetch the pinned SimplerEnv and ManiSkill2_real2sim commits" >&2
  exit 3
}
command -v uv >/dev/null 2>&1 || {
  echo "uv 0.11.14 is required to build the SimplerEnv runtime" >&2
  exit 3
}
UV_BIN="$(command -v uv)"
if [[ "$("${UV_BIN}" --version)" != "uv 0.11.14 "* ]]; then
  echo "uv 0.11.14 is required to build the SimplerEnv runtime" >&2
  exit 3
fi

if [[ -z "${VK_ICD_FILENAMES:-}${VK_DRIVER_FILES:-}" ]]; then
  for icd_dir in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    if [[ -f "${icd_dir}/nvidia_icd_vksc.json" ]]; then
      echo "${icd_dir}/nvidia_icd_vksc.json (NVIDIA Vulkan SC) crashes SAPIEN's renderer teardown;" >&2
      echo "export VK_ICD_FILENAMES=${icd_dir}/nvidia_icd.json and re-run" >&2
      exit 3
    fi
  done
fi

# SimplerEnv and its submodule at the pinned commits. A clean re-run checks the existing
# checkout instead of fetching again.
if [[ ! -d "${SOURCE}/.git" ]]; then
  rm -rf -- "${SOURCE}"
  install -d -m 0755 "${SOURCE}"
  git -C "${SOURCE}" init -q
  git -C "${SOURCE}" remote add origin "${SIMPLER_ENV_URL}"
  git -C "${SOURCE}" fetch -q --depth 1 origin "${SIMPLER_ENV_COMMIT}"
  git -C "${SOURCE}" -c advice.detachedHead=false checkout -q FETCH_HEAD
  git -C "${SOURCE}" submodule update -q --init --depth 1 ManiSkill2_real2sim
fi
actual="$(git -C "${SOURCE}" rev-parse HEAD)"
[[ "${actual}" == "${SIMPLER_ENV_COMMIT}" ]] || {
  echo "SimplerEnv checkout is at ${actual}, expected ${SIMPLER_ENV_COMMIT}" >&2
  exit 4
}
recorded="$(git -C "${SOURCE}" ls-tree HEAD ManiSkill2_real2sim | awk '{print $3}')"
actual="$(git -C "${SOURCE}/ManiSkill2_real2sim" rev-parse HEAD)"
if [[ "${recorded}" != "${REAL2SIM_COMMIT}" || "${actual}" != "${REAL2SIM_COMMIT}" ]]; then
  echo "ManiSkill2_real2sim is at ${actual} (recorded ${recorded}), expected ${REAL2SIM_COMMIT}" >&2
  exit 4
fi
for checkout in "${SOURCE}" "${SOURCE}/ManiSkill2_real2sim"; do
  if [[ -n "$(git -C "${checkout}" status --porcelain --untracked-files=no)" ]]; then
    echo "${checkout} has modified tracked files; remove ${SOURCE} and rebuild" >&2
    exit 4
  fi
done

# Keep the interpreter inside <install-dir> so the environment does not depend
# on the user's uv cache.
export UV_PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}"
"${UV_BIN}" python install --no-bin --no-progress "${PYTHON_VERSION}"
if [[ ! -x "${PYTHON}" ]]; then
  "${UV_BIN}" venv \
    --python "${PYTHON_VERSION}" \
    --managed-python \
    --no-python-downloads \
    --relocatable \
    "${VENV}"
fi
# Every package, including the TensorFlow CUDA wheels, comes from PyPI. On a host where PyPI is slow,
# ROBOSCOPE_PYPI_INDEX_URL may name a full PyPI mirror; the lock pins every version either way.
"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --index-url "${ROBOSCOPE_PYPI_INDEX_URL:-https://pypi.org/simple}" \
  --requirements "${LOCK}"
"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --no-deps \
  --editable "${SOURCE}/ManiSkill2_real2sim" \
  --editable "${SOURCE}"
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3 \
  "${PYTHON}" -I - "${SOURCE}" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

source = Path(sys.argv[1])
for name, locked in (
    ("sapien", "2.2.2"),
    ("numpy", "1.24.4"),
    ("gymnasium", "0.29.1"),
    ("tensorflow", "2.15.1"),
    ("tensorflow-hub", "0.16.0"),
    ("tf-agents", "0.19.0"),
    ("mani_skill2_real2sim", "0.5.3"),
    ("simpler_env", "0.0.1"),
):
    actual = version(name)
    if actual != locked:
        raise SystemExit(f"SimplerEnv runtime {name} is {actual}, expected {locked}")

import mani_skill2_real2sim
import simpler_env
from simpler_env.evaluation.argparse import get_args  # noqa: F401
from simpler_env.policies.rt1.rt1_model import RT1Inference  # noqa: F401

if Path(simpler_env.__file__).resolve().parent != source / "simpler_env":
    raise SystemExit(f"simpler_env imports from {simpler_env.__file__}, not {source}")
if mani_skill2_real2sim.ASSET_DIR.resolve() != source / "ManiSkill2_real2sim" / "data":
    raise SystemExit(f"ManiSkill2_real2sim assets resolve to {mani_skill2_real2sim.ASSET_DIR}")
if not (source / "simpler_env" / "main_inference.py").is_file():
    raise SystemExit("SimplerEnv's main_inference.py is missing")
print("SimplerEnv runtime versions verified")
PY

# Parse, build, reset, render, and step the quick-test variant of every suite and one
# ray-traced drawer variant through the kit worker's own path.
(
  cd "${SOURCE}"
  env PYTHONPATH="${KIT_SOURCE}" PYTHONDONTWRITEBYTECODE=1 DISPLAY="" \
    "${PYTHON}" -m roboscope_kit.runtime.benchmarks.simpler_env.check
)

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built SimplerEnv in ${INSTALL_DIR}"

# The kit runs its own check against the installed manifest. It needs a Python with
# roboscope-kit installed; the SimplerEnv environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  rm -rf -- "${INSTALL_DIR}/self-test"
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark simpler_env --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi
