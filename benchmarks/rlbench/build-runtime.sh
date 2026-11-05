#!/usr/bin/env bash
set -Eeuo pipefail

# Build the RLBench runtime of the PerAct 18-task reproduction profile under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     coppeliasim/              CoppeliaSim Edu 4.1.0 (Ubuntu 20.04 build), from Coppelia Robotics
#     python/                   the uv-managed CPython 3.11 the environment runs on
#     env/                      venv: runtime-requirements.txt, PyRep, RLBench and YARR
#     src/                      PyRep, the PerAct RLBench and YARR forks, and PerAct, at pinned commits
#     peract-test/test/         PerAct's released RLBench test split (18 tasks x 25 episodes)
#     bin/Xvfb                  link to the host's Xvfb, used for headless rendering
#     pip-freeze.txt            what env/ actually contains
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark rlbench --suite peract_18 \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# Licences. CoppeliaSim Edu is free only for educational entities (hobbyists, students,
# teachers, schools and universities) and non-commercial use; RLBench is licensed by Imperial
# College London for non-commercial, internal or academic research only, and the PerAct test
# set was generated with it. Read both and set ROBOSCOPE_ACCEPT_COPPELIASIM_EDU_LICENSE=YES and
# ROBOSCOPE_ACCEPT_RLBENCH_LICENSE=YES yourself; the build does not accept them for you.
# Nothing licensed is committed to the repository: everything is downloaded from its
# official source and checked against the SHA-256 or commit pinned here.
#
# Host prerequisites: git, curl, sha256sum, tar with xz, uv 0.11.14, and an Xvfb server on
# PATH or in XVFB (for example Ubuntu's xvfb package). CoppeliaSim renders the cameras with
# Mesa's llvmpipe on a private Xvfb display per worker; no GPU is used by the simulator.
#
# The test set is about 15 GB of zip files. Set RLBENCH_PERACT_TEST_ZIPS to a directory that
# already holds verified <task>.zip files to skip their download; otherwise they are
# downloaded into <install-dir>/downloads and removed after extraction. Re-running the script
# on an existing <install-dir> reuses what is already verified.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
KIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
[[ $# -eq 1 ]] || {
  echo "usage: $0 <install-dir>" >&2
  exit 2
}

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "RLBench runtime builds require Linux x86_64" >&2
  exit 3
fi
case "${ROBOSCOPE_ACCEPT_COPPELIASIM_EDU_LICENSE:-}:${ROBOSCOPE_ACCEPT_RLBENCH_LICENSE:-}" in
  YES:YES) ;;
  *)
    echo "The RLBench runtime uses CoppeliaSim Edu and RLBench, which have non-commercial licences." >&2
    echo "Read them and set ROBOSCOPE_ACCEPT_COPPELIASIM_EDU_LICENSE=YES and ROBOSCOPE_ACCEPT_RLBENCH_LICENSE=YES:" >&2
    echo "  https://www.coppeliarobotics.com/helpFiles/en/licensing.htm" >&2
    echo "  https://github.com/stepjam/RLBench/blob/master/LICENSE" >&2
    exit 3
    ;;
esac

MANIFEST="${SCRIPT_DIR}/benchmark-manifest.json"
LOCK="${SCRIPT_DIR}/runtime-requirements.txt"
TEST_SET_PINS="${SCRIPT_DIR}/peract-test-set.txt"
PERACT_PATCH="${SCRIPT_DIR}/patches/peract-python311.patch"
KIT_SOURCE="${KIT_ROOT}/packages/roboscope-kit/src"
for input in "${MANIFEST}" "${LOCK}" "${TEST_SET_PINS}" "${PERACT_PATCH}"; do
  [[ -f "${input}" ]] || {
    echo "RLBench build input is missing: ${input}" >&2
    exit 3
  }
done
[[ -d "${KIT_SOURCE}/roboscope_kit" ]] || {
  echo "roboscope-kit source is missing: ${KIT_SOURCE}" >&2
  exit 3
}
for executable in git curl sha256sum tar xz; do
  command -v "${executable}" >/dev/null 2>&1 || {
    echo "Missing required executable: ${executable}" >&2
    exit 3
  }
done
command -v uv >/dev/null 2>&1 || {
  echo "uv 0.11.14 is required to build the RLBench runtime" >&2
  exit 3
}
UV_BIN="$(command -v uv)"
if [[ "$("${UV_BIN}" --version)" != "uv 0.11.14 "* ]]; then
  echo "uv 0.11.14 is required to build the RLBench runtime" >&2
  exit 3
fi
XVFB_BIN="${XVFB:-$(command -v Xvfb || true)}"
if [[ -z "${XVFB_BIN}" || ! -x "${XVFB_BIN}" ]]; then
  echo "Xvfb is required for headless rendering: install it (Ubuntu: xvfb) or set XVFB=/path/to/Xvfb" >&2
  exit 3
fi

install -d -m 0755 "$1"
INSTALL_DIR="$(cd -- "$1" && pwd -P)"
PYTHON_INSTALL_DIR="${INSTALL_DIR}/python"
VENV="${INSTALL_DIR}/env"
PYTHON="${VENV}/bin/python"
PYTHON_VERSION="3.11.15"
COPPELIASIM_ROOT="${INSTALL_DIR}/coppeliasim"
SOURCE_ROOT="${INSTALL_DIR}/src"
DOWNLOADS="${INSTALL_DIR}/downloads"
TEST_SET="${INSTALL_DIR}/peract-test/test"

# CoppeliaSim Edu 4.1.0, the version PyRep 4.1 requires, from Coppelia Robotics' archive.
COPPELIASIM_URL="https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
COPPELIASIM_SHA256="512de3a7347387fcc1b12fa675195a912265753f7901808382865bbf51a2a7f8"
# PyRep 4.1.0.3: the last commit before PyRep added robots PerAct never used (Feb 2022).
PYREP_URL="https://github.com/stepjam/PyRep.git"
PYREP_COMMIT="231a1ac6b0a179cff53c1d403d379260b9f05f2f"
# Head of the 'peract' branches PerAct's README installs.
RLBENCH_URL="https://github.com/MohitShridhar/RLBench.git"
RLBENCH_COMMIT="ad991951bc53e4f3b73b803a75cf4b7d55295cf7"
YARR_URL="https://github.com/MohitShridhar/YARR.git"
YARR_COMMIT="8d0f2e228b97efa207eeccedfc6dac9334342ba3"
# PerAct main; its eval.py is the native entry point the Kit is compared with.
PERACT_URL="https://github.com/peract/peract.git"
PERACT_COMMIT="02fb87681c5a47be9dbf20141bedb836ee2d3ef9"

for pin in "torch==" "numpy==" "cffi==" "clip @ git+https://github.com/openai/CLIP.git@"; do
  grep -qF "${pin}" "${LOCK}" || {
    echo "RLBench runtime lock must pin ${pin}" >&2
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
  for ((attempt = 1; attempt <= 8; attempt++)); do
    if [[ -f "${target}.part" ]] && printf '%s  %s\n' "${sha256}" "${target}.part" | sha256sum -c - >/dev/null 2>&1; then
      mv -- "${target}.part" "${target}"
      return 0
    fi
    if curl -fLsS --retry 2 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
      -C - --output "${target}.part" "${url}"; then
      if ! printf '%s  %s\n' "${sha256}" "${target}.part" | sha256sum -c - >/dev/null 2>&1; then
        # A complete transfer with the wrong digest cannot be resumed into the right one.
        rm -f -- "${target}.part"
      fi
    else
      sleep 10
    fi
  done
  if [[ -f "${target}.part" ]] && printf '%s  %s\n' "${sha256}" "${target}.part" | sha256sum -c - >/dev/null 2>&1; then
    mv -- "${target}.part" "${target}"
    return 0
  fi
  echo "Could not download ${url} with SHA-256 ${sha256}" >&2
  return 1
}

checkout() {
  # checkout <url> <commit> <dir>: a detached checkout of exactly <commit>.
  local url="$1" commit="$2" dir="$3"
  if [[ ! -d "${dir}/.git" ]]; then
    git init -q "${dir}"
    git -C "${dir}" remote add origin "${url}"
  fi
  if ! git -C "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${dir}" fetch -q --depth 1 origin "${commit}"
  fi
  git -C "${dir}" -c advice.detachedHead=false checkout -q --detach "${commit}"
  [[ "$(git -C "${dir}" rev-parse HEAD)" == "${commit}" ]] || {
    echo "${dir} is not at ${commit}" >&2
    exit 5
  }
}

# CoppeliaSim.
if [[ ! -x "${COPPELIASIM_ROOT}/coppeliaSim" ]]; then
  fetch "${COPPELIASIM_URL}" "${COPPELIASIM_SHA256}" "${DOWNLOADS}/$(basename -- "${COPPELIASIM_URL}")"
  rm -rf -- "${COPPELIASIM_ROOT}.partial"
  mkdir -p "${COPPELIASIM_ROOT}.partial"
  tar -xJf "${DOWNLOADS}/$(basename -- "${COPPELIASIM_URL}")" -C "${COPPELIASIM_ROOT}.partial" --strip-components=1
  mv -- "${COPPELIASIM_ROOT}.partial" "${COPPELIASIM_ROOT}"
  rm -f -- "${DOWNLOADS}/$(basename -- "${COPPELIASIM_URL}")"
fi

# Sources.
checkout "${PYREP_URL}" "${PYREP_COMMIT}" "${SOURCE_ROOT}/PyRep"
checkout "${RLBENCH_URL}" "${RLBENCH_COMMIT}" "${SOURCE_ROOT}/RLBench"
checkout "${YARR_URL}" "${YARR_COMMIT}" "${SOURCE_ROOT}/YARR"
checkout "${PERACT_URL}" "${PERACT_COMMIT}" "${SOURCE_ROOT}/peract"
# PerAct imports pytorch3d at module level only for training-time augmentation; pytorch3d has
# no Python 3.11 build. The patch defers that import; evaluation code is unchanged.
if git -C "${SOURCE_ROOT}/peract" apply --check "${PERACT_PATCH}" 2>/dev/null; then
  git -C "${SOURCE_ROOT}/peract" apply "${PERACT_PATCH}"
elif ! git -C "${SOURCE_ROOT}/peract" apply --reverse --check "${PERACT_PATCH}" 2>/dev/null; then
  echo "${SOURCE_ROOT}/peract has local changes; remove it and build again" >&2
  exit 5
fi

# Python environment. Keep the interpreter inside <install-dir> so the environment does not
# depend on the user's uv cache.
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
# Every package, including the torch CUDA wheels, comes from PyPI; CLIP from its pinned commit.
"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --index-url https://pypi.org/simple \
  --requirements "${LOCK}"
# PyRep compiles its CFFI bindings against this CoppeliaSim and adds allowOldEduRelease to its
# system/usrset.txt. RLBench and YARR are installed in development mode, as PerAct's README
# does (RLBench's setup.py omits rlbench.action_modes from a regular install).
COPPELIASIM_ROOT="${COPPELIASIM_ROOT}" "${UV_BIN}" pip install \
  --python "${PYTHON}" --no-progress --no-deps --no-build-isolation --reinstall-package pyrep \
  "${SOURCE_ROOT}/PyRep"
"${UV_BIN}" pip install --python "${PYTHON}" --no-progress --no-deps --no-build-isolation \
  --editable "${SOURCE_ROOT}/RLBench" --editable "${SOURCE_ROOT}/YARR"
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"

runtime_env=(
  env -u PYTHONPATH
  COPPELIASIM_ROOT="${COPPELIASIM_ROOT}"
  LD_LIBRARY_PATH="${COPPELIASIM_ROOT}"
  QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
  CUDA_VISIBLE_DEVICES=""
  PYTHONDONTWRITEBYTECODE=1
)
"${runtime_env[@]}" "${PYTHON}" -I - "${SOURCE_ROOT}" "${RLBENCH_COMMIT}" "${YARR_COMMIT}" <<'PY'
import sys
from importlib.metadata import version
from pathlib import Path

source_root = Path(sys.argv[1])
for name, locked in (("numpy", "1.26.4"), ("torch", "2.4.1"), ("pyrep", "4.1.0.3"), ("rlbench", "1.2.0")):
    actual = version(name)
    if actual != locked:
        raise SystemExit(f"RLBench runtime {name} is {actual}, expected {locked}")

import clip  # noqa: F401  (YARR's RLBench environment tokenizes with OpenAI CLIP)
import pyrep
import rlbench
import yarr

for module, checkout in ((rlbench, "RLBench"), (yarr, "YARR")):
    if Path(module.__file__).resolve().parents[1] != (source_root / checkout).resolve():
        raise SystemExit(f"{module.__name__} is not imported from {source_root / checkout}")
if not Path(pyrep.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise SystemExit("pyrep is not installed in the runtime environment")
print("RLBench runtime versions verified")
PY

# PerAct test set: download or reuse each pinned zip, verify it, and extract it once.
mkdir -p "${TEST_SET}"
zip_dir="${RLBENCH_PERACT_TEST_ZIPS:-${DOWNLOADS}/peract-test}"
while read -r task drive_id size sha256; do
  [[ -z "${task}" || "${task}" == \#* ]] && continue
  marker="${TEST_SET}/${task}/.zip-sha256"
  if [[ -f "${marker}" && "$(cat -- "${marker}")" == "${sha256}" ]]; then
    continue
  fi
  archive="${zip_dir}/${task}.zip"
  fetch "https://drive.usercontent.google.com/download?id=${drive_id}&export=download&confirm=t" \
    "${sha256}" "${archive}"
  [[ "$(stat -c %s -- "${archive}")" == "${size}" ]] || {
    echo "${archive} is not ${size} bytes" >&2
    exit 5
  }
  rm -rf -- "${TEST_SET:?}/${task}"
  "${PYTHON}" -I - "${archive}" "${TEST_SET}" "${task}" <<'PY'
import sys
import zipfile
from pathlib import Path

archive, root, task = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), sys.argv[3]
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        target = (root / member.filename).resolve()
        if not target.is_relative_to(root / task):
            raise SystemExit(f"{archive} member {member.filename} is outside {task}/")
    bundle.extractall(root)
PY
  printf '%s\n' "${sha256}" > "${marker}"
  if [[ -z "${RLBENCH_PERACT_TEST_ZIPS:-}" ]]; then
    rm -f -- "${archive}"
  fi
done < "${TEST_SET_PINS}"

install -d -m 0755 "${INSTALL_DIR}/bin"
ln -sfn "$(readlink -f -- "${XVFB_BIN}")" "${INSTALL_DIR}/bin/Xvfb"

# Every task's 25 test episodes against the frozen digests, then a headless launch and one
# rendered reset through the kit worker's own code.
"${runtime_env[@]}" PYTHONPATH="${KIT_SOURCE}" LP_NUM_THREADS=2 \
  "${PYTHON}" -m roboscope_kit.runtime.benchmarks.rlbench.check \
  --test-set "${TEST_SET}" --xvfb "${INSTALL_DIR}/bin/Xvfb"

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built RLBench peract_18 in ${INSTALL_DIR}"

# The kit runs its own check against the installed manifest. It needs a Python with
# roboscope-kit installed; the RLBench environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  rm -rf -- "${INSTALL_DIR}/self-test"
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark rlbench --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi
