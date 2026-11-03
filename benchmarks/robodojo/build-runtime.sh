#!/usr/bin/env bash
set -Eeuo pipefail

# Build a pinned RoboDojo simulation runtime and its public benchmark manifest.
# Usage: build-runtime.sh <install-dir>
BUILDER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="$(cd -- "${BUILDER_DIR}/../.." && pwd -P)"
[[ $# -eq 1 ]] || { echo "usage: $0 <install-dir>" >&2; exit 2; }
[[ "$(uname -s):$(uname -m)" == "Linux:x86_64" ]] || {
  echo "RoboDojo runtime builds require Linux x86_64" >&2; exit 3;
}
case "${OMNI_KIT_ACCEPT_EULA:-}" in
  Y|YES|y|yes) ;;
  *)
    echo "RoboDojo uses NVIDIA Isaac Sim. Read its license and set OMNI_KIT_ACCEPT_EULA=YES to accept it:" >&2
    echo "https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html" >&2
    exit 3 ;;
esac
for executable in git uv cmake ffmpeg; do
  command -v "${executable}" >/dev/null || { echo "Missing required executable: ${executable}" >&2; exit 3; }
done

mkdir -p "$1"
INSTALL_ROOT="$(cd -- "$1" && pwd -P)"
SOURCE_ROOT="${INSTALL_ROOT}/RoboDojo"
ENV_ROOT="${INSTALL_ROOT}/env"
SOURCE_URL="https://github.com/RoboDojo-Benchmark/RoboDojo.git"
SOURCE_COMMIT="f57e1633ffed59761bc8e568cfe7530c4b33381c"

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --no-checkout "${SOURCE_URL}" "${SOURCE_ROOT}"
  git -C "${SOURCE_ROOT}" checkout --detach "${SOURCE_COMMIT}"
fi
[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]] || {
  echo "Existing RoboDojo checkout is not at ${SOURCE_COMMIT}: ${SOURCE_ROOT}" >&2; exit 5;
}
git -C "${SOURCE_ROOT}" submodule update --init third_party/IsaacLab third_party/curobo XPolicyLab

apply_once() {
  local repository="$1" patch_file="$2" strip="${3:-1}"
  if git -C "${repository}" apply --check -p"${strip}" "${patch_file}"; then
    git -C "${repository}" apply -p"${strip}" "${patch_file}"
  elif ! git -C "${repository}" apply --reverse --check -p"${strip}" "${patch_file}"; then
    echo "Cannot apply ${patch_file}; the existing source differs from its expected state" >&2
    exit 5
  fi
}
apply_once "${SOURCE_ROOT}" "${BUILDER_DIR}/patches/layout-shards.patch"
apply_once "${SOURCE_ROOT}" "${BUILDER_DIR}/patches/policy-bridge.patch"
for name in async-video unpaced-render physics-threads control-state; do
  apply_once "${SOURCE_ROOT}" "${BUILDER_DIR}/patches/${name}.patch"
done
apply_once "${SOURCE_ROOT}/third_party/IsaacLab" "${BUILDER_DIR}/patches/isaaclab-deps.patch" 3
install -d "${SOURCE_ROOT}/XPolicyLab/policy/RoboScopeKit"
for file in __init__.py client.py deploy.py deploy.yml; do
  install -m 0644 "${BUILDER_DIR}/roboscope_policy/${file}" "${SOURCE_ROOT}/XPolicyLab/policy/RoboScopeKit/${file}"
done

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  uv venv --python 3.11 "${ENV_ROOT}"
fi
# shellcheck source=/dev/null
source "${ENV_ROOT}/bin/activate"
export PIP_USER=0 PYTHONNOUSERSITE=1 TERM=xterm-256color
export PIP_CONSTRAINT="${BUILDER_DIR}/runtime-requirements.txt"
export UV_CONSTRAINT="${BUILDER_DIR}/runtime-requirements.txt"
# The caller has accepted the Isaac Sim EULA above; privacy consent is left to them.
export ACCEPT_EULA=Y
# Import upstream's install functions without its conda, system-package, or moving-submodule steps.
# shellcheck source=/dev/null
source "${SOURCE_ROOT}/scripts/install.sh" --help
pip_install() { uv pip install --python "${ENV_ROOT}/bin/python" "$@"; }
pip_install 'pip==26.2.2' 'setuptools==79.0.1' 'wheel==0.45.1'
setup_base_deps
setup_isaacsim
setup_isaaclab
python -m pip install --no-deps --no-build-isolation -e "${SOURCE_ROOT}/third_party/IsaacLab/source/isaaclab"
setup_curobo
pip_install 'pydantic>=2.5' 'h5py>=3.8' 'pyyaml>=6' 'msgpack>=1.0.8' msgpack-numpy
python -m pip install --no-deps "${REPO_DIR}/packages/roboscope-kit"
python -m pip check

# Verify every asset against the immutable upstream inventory. The optional
# mirror is a byte transport only; it cannot change the accepted hashes.
asset_args=(--transport "${ROBODOJO_ASSET_TRANSPORT:-primary}" --workers "${ROBODOJO_ASSET_WORKERS:-8}")
if [[ -n "${ROBODOJO_ASSET_INVENTORY:-}" ]]; then
  asset_args+=(--inventory "${ROBODOJO_ASSET_INVENTORY}")
fi
python "${BUILDER_DIR}/download_assets.py" "${SOURCE_ROOT}" "${asset_args[@]}"
cd "${SOURCE_ROOT}"
python utils/update_embodiment_config_path.py
install -m 0644 "${BUILDER_DIR}/benchmark-manifest.json" "${INSTALL_ROOT}/benchmark-manifest.json"
echo "RoboDojo runtime built at ${INSTALL_ROOT}"
