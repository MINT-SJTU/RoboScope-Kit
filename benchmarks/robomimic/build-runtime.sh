#!/usr/bin/env bash
set -Eeuo pipefail

# Build the robomimic v0.1 rollout environment under <install-dir>:
#
#   <install-dir>/
#     benchmark-manifest.json   this directory's manifest; runtime paths are relative to it
#     python/                   the uv-managed CPython 3.11 the environment runs on
#     env/                      venv with exactly runtime-requirements.txt
#     pip-freeze.txt            what env/ actually contains
#
# Usage: build-runtime.sh <install-dir>
# Then:  python -m roboscope_kit.runtime.benchmark --benchmark robomimic --suite ph_low_dim \
#          --benchmark-manifest <install-dir>/benchmark-manifest.json ...
#
# The environment pairs robomimic v0.4.0 with the robosuite v1.4.1 branch on the DeepMind
# MuJoCo 2.3.7 bindings. The robomimic v0.1 model zoo was trained on the mujoco-py based
# robosuite offline_study branch, which needs MuJoCo 2.0 and mujoco-py 2.0 and does not
# build on Python 3.11; see ROBOMIMIC_EVALUATION_PROTOCOL.md for that boundary.
#
# The build regenerates the 50 seeded initial states of every task through the kit
# worker's own reset path and rejects an environment whose digests differ from the
# constants in the kit's robomimic task table. Re-running the script on an existing
# <install-dir> reuses the venv and re-applies the lock.

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
PYTHON_VERSION="3.11"
ROBOMIMIC_URL="https://github.com/ARISE-Initiative/robomimic.git"
# robomimic v0.4.0
ROBOMIMIC_COMMIT="7c66e7a41b5d9dcc905b1a68346bfee1b49b79c9"
# Head of robosuite's v1.4.1 branch, which robomimic v0.4.0 documents for its datasets.
ROBOSUITE_COMMIT="c848ca848020d0c4ccdd10c5056bd06f2a195ba2"

if [[ "$(uname -s):$(uname -m)" != "Linux:x86_64" ]]; then
  echo "robomimic runtime builds require Linux x86_64" >&2
  exit 3
fi
if [[ ! -f "${MANIFEST}" || ! -f "${LOCK}" || ! -d "${KIT_SOURCE}/roboscope_kit" ]]; then
  echo "robomimic manifest, runtime lock or kit source is missing" >&2
  exit 3
fi
for pin in "mujoco==2.3.7" "torch==" "numpy==" "msgpack==1.2.4" \
  "robomimic @ git+${ROBOMIMIC_URL}@${ROBOMIMIC_COMMIT}" \
  "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@${ROBOSUITE_COMMIT}"; do
  grep -qF "${pin}" "${LOCK}" || {
    echo "robomimic runtime lock must pin ${pin}" >&2
    exit 3
  }
done
command -v git >/dev/null 2>&1 || {
  echo "git is required to install the pinned robomimic and robosuite commits" >&2
  exit 3
}
command -v uv >/dev/null 2>&1 || {
  echo "uv 0.11.14 is required to build the robomimic runtime" >&2
  exit 3
}
UV_BIN="$(command -v uv)"
if [[ "$("${UV_BIN}" --version)" != "uv 0.11.14 "* ]]; then
  echo "uv 0.11.14 is required to build the robomimic runtime" >&2
  exit 3
fi

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
# Every package, including the torch CUDA wheels, comes from PyPI.
"${UV_BIN}" pip install \
  --python "${PYTHON}" \
  --no-progress \
  --index-url https://pypi.org/simple \
  --requirements "${LOCK}"
"${UV_BIN}" pip freeze --python "${PYTHON}" > "${INSTALL_DIR}/pip-freeze.txt"

# Apply the pinned source patch only through this builder. Replace the file atomically
# so uv's hard-linked package cache and other runtime environments remain untouched.
env -u PYTHONPATH "${PYTHON}" -I - "${SCRIPT_DIR}/runtime-patches.json" \
  "${INSTALL_DIR}/runtime-patches.json" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import distribution
from pathlib import Path

pins_path, report = map(Path, sys.argv[1:])
pins = json.loads(pins_path.read_text())
for pin in pins["patches"]:
    patch = pins_path.parent / pin["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != pin["sha256"]:
        raise SystemExit(f"source patch checksum differs from its pin: {patch}")
    package = distribution(pin["distribution"])
    targets = [(entry, Path(package.locate_file(entry["path"]))) for entry in pin["files"]]
    states = [hashlib.sha256(target.read_bytes()).hexdigest() for _, target in targets]
    if states == [entry["patched_sha256"] for entry, _ in targets]:
        continue
    if states != [entry["original_sha256"] for entry, _ in targets]:
        raise SystemExit(f"installed files differ from the pinned original or patched set: {patch}")
    with tempfile.TemporaryDirectory(prefix="roboscope-robomimic-patch-") as temporary:
        for entry, target in targets:
            source = Path(temporary) / entry["path"]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(target.read_bytes())
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=temporary, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=temporary, check=True)
        patched_files = [(Path(temporary) / entry["path"]).read_bytes() for entry, _ in targets]
    for (entry, _), patched in zip(targets, patched_files, strict=True):
        if hashlib.sha256(patched).hexdigest() != entry["patched_sha256"]:
            raise SystemExit(f"source patch produced unexpected bytes: {entry['path']}")
    for (_, target), patched in zip(targets, patched_files, strict=True):
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            replacement = Path(output.name)
            output.write(patched)
        try:
            replacement.chmod(target.stat().st_mode & 0o777)
            os.replace(replacement, target)
        finally:
            replacement.unlink(missing_ok=True)
report.write_text(json.dumps(pins, indent=2) + "\n")
print("robomimic simulator source patches verified")
PY

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -I - "${ROBOMIMIC_COMMIT}" "${ROBOSUITE_COMMIT}" <<'PY'
import json
import sys
from importlib.metadata import distribution, version

robomimic_commit, robosuite_commit = sys.argv[1:3]
for name, locked in (
    ("mujoco", "2.3.7"),
    ("numpy", "1.26.4"),
    ("msgpack", "1.2.4"),
    ("torch", "2.4.1"),
    ("robomimic", "0.4.0"),
    ("robosuite", "1.4.1"),
):
    actual = version(name)
    if actual != locked:
        raise SystemExit(f"robomimic runtime {name} is {actual}, expected {locked}")
for name, commit in (("robomimic", robomimic_commit), ("robosuite", robosuite_commit)):
    direct_url = json.loads(distribution(name).read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != commit:
        raise SystemExit(f"robomimic runtime {name} was not installed from commit {commit}")

import importlib.util

import robomimic.utils.file_utils  # noqa: F401
import robosuite

# The upstream rollout entry point used for the native comparison.
if importlib.util.find_spec("robomimic.scripts.run_trained_agent") is None:
    raise SystemExit("robomimic.scripts.run_trained_agent is not installed")

if robosuite.__version__ != "1.4.1":
    raise SystemExit(f"robosuite reports version {robosuite.__version__}, expected 1.4.1")
print("robomimic runtime versions verified")
PY

# Regenerate every task's seeded initial states with the kit worker's reset path.
env PYTHONPATH="${KIT_SOURCE}" CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON}" -m roboscope_kit.runtime.benchmarks.robomimic.initial_states

install -m 0644 "${MANIFEST}" "${INSTALL_DIR}/benchmark-manifest.json"
echo "Built robomimic ph_low_dim in ${INSTALL_DIR}"

# The kit runs its own check against the installed manifest. It needs a Python with
# roboscope-kit installed; the robomimic environment does not need it.
if [[ -n "${KIT_PYTHON:-}" ]] || python3 -c 'import roboscope_kit' >/dev/null 2>&1; then
  KIT_PYTHON="${KIT_PYTHON:-python3}"
  "${KIT_PYTHON}" -c 'import roboscope_kit' || {
    echo "KIT_PYTHON=${KIT_PYTHON} cannot import roboscope_kit; install roboscope-kit into it" >&2
    exit 6
  }
  rm -rf -- "${INSTALL_DIR}/self-test"
  "${KIT_PYTHON}" -m roboscope_kit.runtime.benchmark --benchmark robomimic --self-test \
    --benchmark-manifest "${INSTALL_DIR}/benchmark-manifest.json" \
    --output-dir "${INSTALL_DIR}/self-test"
else
  echo "Skipped the kit self-test: python3 cannot import roboscope_kit; set KIT_PYTHON to run it" >&2
fi
