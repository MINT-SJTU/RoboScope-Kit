#!/usr/bin/env bash
set -Eeuo pipefail

# Invoked by build-runtime.sh when a pinned simulator environment already exists.
# The new venv shadows packages locally and never installs into the base runtime.
builder_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd -- "${builder_dir}/../.." && pwd -P)"
[[ $# -eq 2 ]] || { echo 'usage: build-reused-runtime.sh <install-dir> <base-runtime>' >&2; exit 2; }
base_root="$(cd -- "$2" && pwd -P)"
base_source="${base_root}/RoboDojo"
base_python="${base_root}/env/bin/python"
asset_source="${ROBODOJO_ASSET_SOURCE:?set ROBODOJO_ASSET_SOURCE to a read-only pinned asset tree}"
wheelhouse="${ROBODOJO_WHEELHOUSE:?set ROBODOJO_WHEELHOUSE to the Kit wheel and missing dependency wheels}"
inventory="${ROBODOJO_ASSET_INVENTORY:?set ROBODOJO_ASSET_INVENTORY to the verified asset inventory}"
source_commit=f57e1633ffed59761bc8e568cfe7530c4b33381c
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL || true
export PYTHONNOUSERSITE=1
[[ -x "${base_python}" && -d "${asset_source}" && -d "${wheelhouse}" && -f "${inventory}" ]] || {
  echo 'Base Python, asset source, wheelhouse or inventory is missing' >&2; exit 3;
}
[[ "$(git -C "${base_source}" rev-parse HEAD)" == "${source_commit}" ]] || {
  echo 'Base RoboDojo revision does not match the pinned revision' >&2; exit 5;
}
for spec in \
  'third_party/IsaacLab:afca7b09d60d8beb9c1cb28b43066499940b969b' \
  'third_party/curobo:895c6517243f8cb091c73c018c8167192d39599a' \
  'XPolicyLab:c7479d6fb014b0a23dfc292ef214eb1990f27029'; do
  path="${spec%%:*}" expected="${spec#*:}"
  [[ "$(git -C "${base_source}/${path}" rev-parse HEAD)" == "${expected}" ]] || {
    echo "Base submodule revision mismatch: ${path}" >&2; exit 5;
  }
done
[[ -z "$(git -C "${base_source}/third_party/curobo" status --porcelain)" ]] || {
  echo 'Base CuRobo source is modified or has untracked files' >&2; exit 5;
}
"${base_python}" -I -c 'import sys; assert sys.version_info[:3] == (3, 11, 18)'
"${base_python}" -I -m pip check
mkdir -p "$1"
install_root="$(cd -- "$1" && pwd -P)"
source_root="${install_root}/RoboDojo"
env_root="${install_root}/env"
[[ ! -e "${source_root}" && ! -e "${env_root}" ]] || {
  echo 'Destination already contains a source tree or environment' >&2; exit 5;
}

# Check that the wheel was built from exactly the staged SDK source.
kit_wheel="$("${base_python}" -I - "${repo_dir}/packages/roboscope-kit" "${wheelhouse}" <<'PY'
from pathlib import Path
import sys, tomllib, zipfile
root, wheels = map(Path, sys.argv[1:])
version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
wheel = wheels / f'roboscope_kit-{version}-py3-none-any.whl'
source = root / 'src'
files = {str(p.relative_to(source)) for p in (source / 'roboscope_kit').rglob('*.py')}
with zipfile.ZipFile(wheel) as archive:
    packed = {name for name in archive.namelist() if name.startswith('roboscope_kit/') and name.endswith('.py')}
    if files != packed or any(archive.read(name) != (source / name).read_bytes() for name in files):
        raise SystemExit('Kit wheel does not match the staged source')
print(wheel)
PY
)"

git clone --no-hardlinks --no-checkout "${base_source}" "${source_root}"
git -C "${source_root}" checkout --detach "${source_commit}"
git -C "${source_root}" submodule init third_party/IsaacLab third_party/curobo XPolicyLab
for path in third_party/IsaacLab third_party/curobo XPolicyLab; do
  git -C "${source_root}" config "submodule.${path}.url" "${base_source}/${path}"
done
git -c protocol.file.allow=always -C "${source_root}" submodule update --init \
  third_party/IsaacLab third_party/curobo XPolicyLab
git -C "${source_root}/third_party/IsaacLab" checkout --detach afca7b09d60d8beb9c1cb28b43066499940b969b
git -C "${source_root}/third_party/curobo" checkout --detach 895c6517243f8cb091c73c018c8167192d39599a
git -C "${source_root}/XPolicyLab" checkout --detach c7479d6fb014b0a23dfc292ef214eb1990f27029

apply_once() {
  local repository="$1" patch_file="$2" strip="${3:-1}"
  if git -C "${repository}" apply --check -p"${strip}" "${patch_file}"; then
    git -C "${repository}" apply -p"${strip}" "${patch_file}"
  elif ! git -C "${repository}" apply --reverse --check -p"${strip}" "${patch_file}"; then
    echo "Cannot apply ${patch_file}" >&2; exit 5
  fi
}
apply_once "${source_root}" "${builder_dir}/patches/layout-shards.patch"
apply_once "${source_root}" "${builder_dir}/patches/policy-bridge.patch"
for name in async-video unpaced-render physics-threads control-state; do
  apply_once "${source_root}" "${builder_dir}/patches/${name}.patch"
done
apply_once "${source_root}/third_party/IsaacLab" "${builder_dir}/patches/isaaclab-deps.patch" 3
install -d "${source_root}/XPolicyLab/policy/RoboScopeKit"
for file in __init__.py client.py deploy.py deploy.yml; do
  install -m 0644 "${builder_dir}/roboscope_policy/${file}" \
    "${source_root}/XPolicyLab/policy/RoboScopeKit/${file}"
done

"${base_python}" -I "${builder_dir}/stage_asset_links.py" "${asset_source}" "${source_root}/Assets"
"${base_python}" -m venv --system-site-packages "${env_root}"
python="${env_root}/bin/python"
"${python}" -m pip install --no-index --no-deps --find-links "${wheelhouse}" \
  -r "${builder_dir}/runtime-requirements.txt"
"${python}" -m pip install --no-index --no-deps "${kit_wheel}"
"${python}" -m pip install --no-deps --no-build-isolation -e \
  "${source_root}/third_party/IsaacLab/source/isaaclab"
"${python}" -m pip check
"${python}" -I - "${builder_dir}/runtime-requirements.txt" "${env_root}" \
  "${repo_dir}/packages/roboscope-kit/pyproject.toml" <<'PY'
import importlib.metadata as md, pathlib, sys, tomllib
from packaging.version import Version
requirements, env, project = map(pathlib.Path, sys.argv[1:])
for line in requirements.read_text().splitlines():
    if not line or line.startswith('#'):
        continue
    name, wanted = line.split('==', 1)
    if Version(md.version(name)) != Version(wanted):
        raise SystemExit(f'installed version differs from pinned requirement: {name}')
import roboscope_kit
expected = tomllib.loads(project.read_text())['project']['version']
if md.version('roboscope-kit') != expected or not pathlib.Path(roboscope_kit.__file__).is_relative_to(env):
    raise SystemExit('Kit version or installation location does not match the new environment')
print('Pinned dependency versions and local Kit installation verified')
PY
"${python}" "${builder_dir}/download_assets.py" "${source_root}" \
  --inventory "${inventory}" --transport "${ROBODOJO_ASSET_TRANSPORT:-primary}" \
  --workers "${ROBODOJO_ASSET_WORKERS:-8}"
( cd "${source_root}" && "${python}" utils/update_embodiment_config_path.py )
install -m 0644 "${builder_dir}/benchmark-manifest.json" "${install_root}/benchmark-manifest.json"
"${python}" -I - "${install_root}/base-provenance.json" "${base_root}" "${inventory}" \
  "${kit_wheel}" <<'PY'
import hashlib, importlib.metadata as md, json, pathlib, sys
path, base, inventory, wheel = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
packages = sorted((d.metadata['Name'], d.version) for d in md.distributions() if d.metadata.get('Name'))
payload = {'base_runtime': base, 'base_prefix': sys.base_prefix, 'packages': packages,
           'robodojo_revision': 'f57e1633ffed59761bc8e568cfe7530c4b33381c',
           'isaaclab_revision': 'afca7b09d60d8beb9c1cb28b43066499940b969b',
           'curobo_revision': '895c6517243f8cb091c73c018c8167192d39599a',
           'xpolicylab_revision': 'c7479d6fb014b0a23dfc292ef214eb1990f27029',
           'inventory_sha256': hashlib.sha256(inventory.read_bytes()).hexdigest(),
           'kit_wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
path.write_text(json.dumps(payload, indent=2))
PY
"${python}" -m roboscope_kit.runtime.benchmark --benchmark robodojo --self-test \
  --benchmark-manifest "${install_root}/benchmark-manifest.json" --output-dir "${install_root}/self-test"
echo "RoboDojo runtime built at ${install_root}, using read-only dependencies from ${base_root}"
