#!/usr/bin/env bash
set -Eeuo pipefail

# Build the BEHAVIOR2026VLA model environment of this example under <dir>, and fetch the checkpoint:
#
#   <dir>/behavior-1k-solution-2026/   Robot-Learning-Collective/behavior-1k-solution-2026 at the pinned
#                                      commit, with its two submodules at their recorded commits
#                                      (openpi: wensi-ai/openpi, BEHAVIOR-1K: v3.9.3-post1) and its
#                                      openpi patches applied, as its setup_remote.sh does
#   <dir>/python/                      the uv-managed CPython 3.11 the solution requires (==3.11.*)
#   <dir>/behavior-1k-solution-2026/.venv/
#                                      the solution's uv.lock for Linux x86_64 (hashes checked) and the
#                                      solution package; then, as setup_remote.sh installs them,
#                                      BEHAVIOR-1K's bddl3 and OmniGibson[eval] (resolved by uv; this
#                                      downgrades pillow and cffi to OmniGibson's pins), av<16 and
#                                      warp-lang 1.12.0. Isaac Sim and JoyLo are not installed: the
#                                      model never starts the simulator, and the native evaluator runs
#                                      in the BEHAVIOR runtime.
#   <dir>/checkpoints/BEHAVIOR2026VLA/ IliaLarchenko/BEHAVIOR2026VLA at the pinned revision (params/ and
#                                      assets/, 10.7 GB; not the README's replay video)
#
# Usage: build-model-env.sh <dir>
#
# Licences: the solution code and the checkpoint are Apache-2.0. The checkpoint is a fine-tune of
# Physical Intelligence's π0.5 (openpi), whose PaliGemma backbone is under the Gemma terms of use.
#
# Indexes can be replaced by mirrors that serve the same files: B1K_PYPI_INDEX (PyPI),
# B1K_TORCH_FIND_LINKS (a flat directory of the download.pytorch.org/whl/cu128 wheels) and
# HF_ENDPOINT; BEHAVIOR_1K_GIT (for example the BEHAVIOR runtime's own checkout) replaces the
# BEHAVIOR-1K repository. Every locked package is installed at the lock's exact version and checked
# against the lock's hash wherever the index publishes one. Git ignores the user's global
# configuration; set https_proxy for GitHub if needed, and behind a TLS-intercepting proxy
# UV_SYSTEM_CERTS=1.

[[ $# -eq 1 ]] || {
  echo "usage: $0 <dir>" >&2
  exit 2
}
for executable in git curl sha256sum uv; do
  command -v "${executable}" >/dev/null 2>&1 || {
    echo "Missing required executable: ${executable}" >&2
    exit 3
  }
done
[[ "$(uv --version)" == "uv 0.11.14 "* ]] || {
  echo "uv 0.11.14 is required" >&2
  exit 3
}
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_LFS_SKIP_SMUDGE=1
install -d -m 0755 "$1"
ROOT="$(cd -- "$1" && pwd -P)"
SOURCE="${ROOT}/behavior-1k-solution-2026"
SOLUTION_URL="https://github.com/Robot-Learning-Collective/behavior-1k-solution-2026.git"
SOLUTION_COMMIT="68c7f912f521d25804a2eaf3245c993ea75411b4"
OPENPI_URL="https://github.com/wensi-ai/openpi.git"
OPENPI_COMMIT="01177e0242a1c7e8fad2547caa0e987def614cda"
BEHAVIOR_1K_URL="${BEHAVIOR_1K_GIT:-https://github.com/StanfordVL/BEHAVIOR-1K.git}"
BEHAVIOR_1K_COMMIT="bd049de3119acdcdf2334fe9e1ebe060fa20c108"
PYTHON_VERSION="3.11.15"
PYPI_INDEX="${B1K_PYPI_INDEX:-https://pypi.org/simple}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_REPO="IliaLarchenko/BEHAVIOR2026VLA"
MODEL_REVISION="dc8c443f21dd2d138e3779c26b142cfde058739f"
CHECKPOINT="${ROOT}/checkpoints/BEHAVIOR2026VLA"
# path, Hugging Face git blob SHA-1 (regular files) or LFS SHA-256.
CHECKPOINT_FILES=(
  "assets/b1k_2026/fast_tokenizer/bpe_tokenizer/tokenizer.json b9186356ed33ac559b98b37c243b00d4290fb8d6"
  "assets/b1k_2026/fast_tokenizer/bpe_tokenizer/tokenizer_config.json b7556ccf00c7a1328b84ccab8309dd029b028568"
  "assets/b1k_2026/fast_tokenizer/metadata.json fc5853d40198eea3aac76a70013c2d912b27820b"
  "assets/b1k_2026/fast_tokenizer/processing_action_tokenizer.py 0819e6c2e5cf5c900a49c8705755b6bf0fb995ea"
  "assets/b1k_2026/fast_tokenizer/processor_config.json 6421a06ad307a2575fa941d8b17a53dd7193c9a1"
  "assets/b1k_2026/norm_stats.json 42abbc8468503d04fc8250a1fc9a2bce65d0884b"
  "params/_METADATA 6c12f1a014ca4b8958a7c8a5228dccd42ef99be3"
  "params/_sharding cd5f518950d0560c14c3476045a5d63919c4dd14"
  "params/array_metadatas/process_0 ad7859085dff9354cf40ce9c33214e78b93ac860"
  "params/d/6244aa236b3d3dcf38bc2f5cfa8683cd 87000581d6572e993dc00041e61e1400a8ad4a40711784a730b8b4fdf02c268e"
  "params/manifest.ocdbt 00d7017ae5e11d689eaaad70b5887c3074aa9f91"
  "params/ocdbt.process_0/manifest.ocdbt e13f4213faccd8bda30915bd643d1dfa64b23a5e"
  "params/ocdbt.process_0/d/05baa478fecb9a5cee1a3603361ebdfd ccf6d26087e754a40d71c397a8d02a518990aa7c"
  "params/ocdbt.process_0/d/1883ed6db0ccea368008127116ef66d5 b4a4f19b626dd94b59f02e20e84bcb37282ac267"
  "params/ocdbt.process_0/d/83f6b358937f5a30d26ef9c2df8aaf26 8abd0c100fe8e46cb9d6669ff62c13eaf983ba31"
  "params/ocdbt.process_0/d/0714eaeccddcbda4b2ec28893064fd8f 5a06f51ced151b9c448d0f98657a2c535a0b1a695dfd437c81dd579816f64fd5"
  "params/ocdbt.process_0/d/0afbd57cfc650112c7e4c5a798f1d72d f920ec7ab7e12a594f45dab74659c7853111143913c933b259cd56081139c73e"
  "params/ocdbt.process_0/d/17f24171d809e2ed987a0f8466248d48 1ef328773d6bd3c14b7020af42312655188bf2a1ad0e844f2dd798563ca96be9"
  "params/ocdbt.process_0/d/2586148c44b233a0e505634734d66d60 909c6efabf7be860cf08a657fdbccd2afbc3e6ef41ad3dfcd490a5ec1e865341"
  "params/ocdbt.process_0/d/3612fb390b69f9f50e1555c7a796f338 04da84a577c3d1b9bedfad003a6875a2c487148ba1fda5d54453804def907c55"
  "params/ocdbt.process_0/d/3e08443c44c15947cc6d90a18538ddc6 d8d6325eb8ea4ee0a6bd9f8ebe9f6b14cde9d412ae2522d11918fe426001a91a"
  "params/ocdbt.process_0/d/563882be234baf5f21e4263c66b3a7e7 e2d28860bf4328cfda2a38aa647709620e6ab111c09ccca44ff3aeb4697b0f64"
  "params/ocdbt.process_0/d/66cc1e931765d743b08b8b350f6117ee 841fc6e75177fe19fc3b2f3e295ca8a991434c9af2624443bb804b1fe99c1873"
  "params/ocdbt.process_0/d/71ec460c0859dde538daaac61a3c1abc e7dabc92ec5ae42f3c2f55e8c62562d99814bc337caa991badb3e3896a932816"
  "params/ocdbt.process_0/d/7aa49462ec4ef761280640ff39d7045a 5082a7810514f02edd01df72e5955872a5122e9a3732c4431429e45db2c2d30c"
  "params/ocdbt.process_0/d/b875d8862014cf40cfaa44bb0fef92b6 76962e429bfdabb5a265a6a3c221f8d69c224622cdd03abeaf4570b5ab328678"
  "params/ocdbt.process_0/d/d0e9fb32db9ce02ca8727c8cbf8ecfc8 7813ed0ae1d9b058a1546b6523bf850b1881631b65a47c02bbc0370cf182ed2c"
  "params/ocdbt.process_0/d/e1bf57b91bab40fba3ce50cd4a518ed6 c164b4fe9af383499f2397640e7715993d6c0d3e90f36b308f8e465fa79d5fd7"
  "params/ocdbt.process_0/d/e2d69009df673c695afa1944d0d3be2a 23f0a1b085aecf2b13084abdbc38c06fce868bfc29f05ba2c78f68e5641e8e8c"
  "params/ocdbt.process_0/d/e3ce53c7d8ec3821b47feaa0493fb983 59915c7df66df833262702687f935392ec81ea39d01b9f1a415c9c871a8364ac"
  "params/ocdbt.process_0/d/f855d8718d2f75fc4d8146e8336961f6 4795a34846d4118d2f3bf6c0dd110a506b874cf2f6c2a01cc722515f51145464"
)

fetch() {
  # fetch <url> <digest> <target>: resume into <target>.part until it has the digest, then move it in place.
  local url="$1" digest="$2" target="$3" part="$3.part" attempt status
  matches "${target}" "${digest}" && return 0
  mkdir -p "$(dirname -- "${target}")"
  for ((attempt = 1; attempt <= 20; attempt++)); do
    status=0
    curl -fLsS --retry 2 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
      -C - --output "${part}" "${url}" || status=$?
    if matches "${part}" "${digest}"; then
      mv -- "${part}" "${target}"
      return 0
    fi
    # A complete transfer with the wrong digest, or an HTTP error such as 416 for a part at least as
    # long as the file, cannot be resumed into the right file: start over. Otherwise resume.
    if [[ "${status}" == 0 || "${status}" == 22 ]]; then
      rm -f -- "${part}"
    else
      sleep 10
    fi
  done
  echo "Could not download ${url} with digest ${digest}" >&2
  return 1
}

matches() {
  # matches <file> <digest>: a 40-character digest is a git blob SHA-1, a 64-character one a SHA-256.
  local file="$1" digest="$2"
  [[ -f "${file}" ]] || return 1
  if [[ ${#digest} -eq 40 ]]; then
    [[ "$(git hash-object -- "${file}")" == "${digest}" ]]
  else
    printf '%s  %s\n' "${digest}" "${file}" | sha256sum -c - >/dev/null 2>&1
  fi
}

checkout() {
  # checkout <dir> <url> <commit>: a clean detached checkout of one commit.
  local dir="$1" url="$2" commit="$3"
  if [[ ! -d "${dir}/.git" && ! -f "${dir}/.git" ]]; then
    mkdir -p "${dir}"
    git init -q "${dir}"
    git -C "${dir}" remote add origin "${url}"
  fi
  if ! git -C "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${dir}" fetch -q --depth 1 origin "${commit}"
  fi
  git -C "${dir}" -c advice.detachedHead=false checkout -q --detach "${commit}"
}

# Source: the solution and its submodules at the commits it records.
checkout "${SOURCE}" "${SOLUTION_URL}" "${SOLUTION_COMMIT}"
for entry in "openpi ${OPENPI_COMMIT}" "BEHAVIOR-1K ${BEHAVIOR_1K_COMMIT}"; do
  read -r path commit <<<"${entry}"
  [[ "$(git -C "${SOURCE}" ls-tree HEAD "${path}" | awk '{print $3}')" == "${commit}" ]] || {
    echo "the solution records another ${path} commit" >&2
    exit 5
  }
done
if [[ -n "$(git -C "${SOURCE}" status --porcelain --untracked-files=normal --ignore-submodules=all \
  | grep -v -E '^\?\? (\.venv/|.*\.egg-info/$)' || true)" ]]; then
  echo "${SOURCE} contains local changes; use a clean directory" >&2
  exit 5
fi
checkout "${SOURCE}/openpi" "${OPENPI_URL}" "${OPENPI_COMMIT}"
checkout "${SOURCE}/BEHAVIOR-1K" "${BEHAVIOR_1K_URL}" "${BEHAVIOR_1K_COMMIT}"
# setup_remote.sh's local openpi patches (the 2026 OmniGibson import path; attention sink, SigLIP
# registers, small token table), applied to the clean openpi checkout.
git -C "${SOURCE}/openpi" reset -q --hard "${OPENPI_COMMIT}"
for patch_file in "${SOURCE}"/patches/openpi-*.patch; do
  git -C "${SOURCE}/openpi" apply "${patch_file}"
done

# Environment: the solution's uv.lock, exported with hashes and installed without resolution. The installs
# pass --no-config: in the solution checkout uv would otherwise apply the project's override-dependencies,
# whose lerobot@release/b1k branch has moved past the locked commit (to one that needs Python 3.12).
export UV_PYTHON_INSTALL_DIR="${ROOT}/python"
VENV="${SOURCE}/.venv"
uv python install --no-bin --no-progress "${PYTHON_VERSION}"
[[ -x "${VENV}/bin/python" ]] || \
  uv venv --python "${PYTHON_VERSION}" --managed-python --no-python-downloads "${VENV}"
requirements="${ROOT}/b1k-requirements.txt"
(cd "${SOURCE}" && uv export --frozen --no-dev --no-emit-project --format requirements-txt \
  --python "${PYTHON_VERSION}" --output-file "${requirements}" >/dev/null)
torch_sources=(--extra-index-url "${TORCH_INDEX}")
if [[ -n "${B1K_TORCH_FIND_LINKS:-}" ]]; then
  torch_sources=(--find-links "${B1K_TORCH_FIND_LINKS}")
fi
(cd "${SOURCE}" && uv pip install --no-config --python "${VENV}/bin/python" --no-progress --no-deps \
  --verify-hashes --index-strategy unsafe-best-match --index-url "${PYPI_INDEX}" "${torch_sources[@]}" \
  --requirements "${requirements}")
uv pip install --no-config --python "${VENV}/bin/python" --no-progress --no-deps --index-url "${PYPI_INDEX}" \
  --editable "${SOURCE}"
# BEHAVIOR-1K's evaluation packages, as setup_remote.sh installs them (outside the solution's lock).
# OmniGibson requires lerobot from the moving release/b1k branch; keep the commit the lock installed.
overrides="${ROOT}/b1k-overrides.txt"
grep -E '^lerobot @ git\+' "${requirements}" > "${overrides}"
uv pip install --no-config --python "${VENV}/bin/python" --no-progress --index-url "${PYPI_INDEX}" \
  --overrides "${overrides}" --editable "${SOURCE}/BEHAVIOR-1K/bddl3" \
  --editable "${SOURCE}/BEHAVIOR-1K/OmniGibson[eval]" "av>=15,<16"
uv pip install --no-config --python "${VENV}/bin/python" --no-progress --no-deps --index-url "${PYPI_INDEX}" \
  "warp-lang==1.12.0"
uv pip freeze --python "${VENV}/bin/python" > "${ROOT}/b1k-pip-freeze.txt"
"${VENV}/bin/python" -I -c "
import shutil, sys
from pathlib import Path
import b1k, openpi, jax, omnigibson
shutil.rmtree(omnigibson.tempdir)
assert sys.version_info[:2] == (3, 11), sys.version
assert Path(b1k.__file__).resolve().is_relative_to(Path('${SOURCE}/src').resolve()), b1k.__file__
assert Path(openpi.__file__).resolve().is_relative_to(Path('${SOURCE}/openpi').resolve()), openpi.__file__
assert Path(omnigibson.__file__).resolve().is_relative_to(Path('${SOURCE}/BEHAVIOR-1K').resolve())
assert jax.__version__ == '0.5.3', jax.__version__
print('BEHAVIOR2026VLA environment verified')
"

# Checkpoint snapshot.
for entry in "${CHECKPOINT_FILES[@]}"; do
  read -r path digest <<<"${entry}"
  fetch "${HF_BASE}/${MODEL_REPO}/resolve/${MODEL_REVISION}/${path}" "${digest}" "${CHECKPOINT}/${path}" ||
    exit 5
done

echo "B1K_SOLUTION_ROOT=${SOURCE}"
echo "B1K_PYTHON=${VENV}/bin/python"
echo "checkpoint: ${CHECKPOINT}"
