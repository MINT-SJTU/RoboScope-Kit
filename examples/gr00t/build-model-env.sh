#!/usr/bin/env bash
set -Eeuo pipefail

# Build the GR00T model environment of this example under <dir>, and fetch the checkpoint:
#
#   <dir>/Isaac-GR00T/              kmy17518/Isaac-GR00T at the pinned commit: the wensi-ai BEHAVIOR
#                                   fork with serve_b1k.py's per-task prompts (resolved from task_id)
#   <dir>/python/                   the uv-managed CPython 3.10 the fork requires (==3.10.*)
#   <dir>/Isaac-GR00T/.venv/        exactly the fork's uv.lock for Linux x86_64 (hashes checked), plus
#                                   websockets 16.0 and patches/transformers-4.57.3-offline-tokenizer.patch
#   <dir>/checkpoints/gr00t-n1.7-b1k-multitask/checkpoint-238000/
#                                   the final snapshot of kmy17518/gr00t-n1.7-b1k-multitask
#   <dir>/hf-home/                  HF_HOME holding nvidia/Cosmos-Reason2-2B, the frozen backbone the
#                                   GR00T processor and model are built from (served offline)
#
# Usage: build-model-env.sh <dir>
#
# Licences: GR00T N1.7 and Cosmos-Reason2 are distributed under NVIDIA's model licences (the
# NVIDIA Open Model License); the checkpoint card defers to them. nvidia/Cosmos-Reason2-2B is gated
# on Hugging Face: with HF_TOKEN set to a token that has accepted it, the script downloads it from
# there; otherwise from NVIDIA's own ModelScope organization (nv-community/Cosmos-Reason2-2B), where
# every file except the weights is checked against the Hugging Face git blob digest and the weights
# against ModelScope's SHA-256 and the Hugging Face size.
#
# Indexes can be replaced by mirrors that serve the same files: GR00T_PYPI_INDEX (PyPI),
# GR00T_TORCH_FIND_LINKS (a flat directory of the download.pytorch.org/whl/cu128 wheels),
# GR00T_NVIDIA_INDEX (pypi.nvidia.com, which the lock takes TensorRT from), and HF_ENDPOINT.
# Every package is installed at the lock's exact version and checked against the lock's hash
# wherever the lock has one (the PyTorch index publishes none for some wheels).
# Git ignores the user's global configuration; set https_proxy for GitHub if needed, and behind a
# TLS-intercepting proxy UV_SYSTEM_CERTS=1.

[[ $# -eq 1 ]] || {
  echo "usage: $0 <dir>" >&2
  exit 2
}
for executable in git curl sha256sum patch uv; do
  command -v "${executable}" >/dev/null 2>&1 || {
    echo "Missing required executable: ${executable}" >&2
    exit 3
  }
done
[[ "$(uv --version)" == "uv 0.11.14 "* ]] || {
  echo "uv 0.11.14 is required" >&2
  exit 3
}
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
install -d -m 0755 "$1"
ROOT="$(cd -- "$1" && pwd -P)"
SOURCE="${ROOT}/Isaac-GR00T"
PATCH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/patches/transformers-4.57.3-offline-tokenizer.patch"
GR00T_URL="https://github.com/kmy17518/Isaac-GR00T.git"
GR00T_COMMIT="080d830dcee2a0cefc7873cb0e4adf42cf71b472"
PYTHON_VERSION="3.10.20"
PYPI_INDEX="${GR00T_PYPI_INDEX:-https://pypi.org/simple}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
NVIDIA_INDEX="${GR00T_NVIDIA_INDEX:-https://pypi.nvidia.com}"
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_REPO="kmy17518/gr00t-n1.7-b1k-multitask"
MODEL_REVISION="5831e9dc0ec1212e9aaa3d96c8e27d2548718a82"
SNAPSHOT="checkpoint-238000"
CHECKPOINT="${ROOT}/checkpoints/gr00t-n1.7-b1k-multitask/${SNAPSHOT}"
# path, Hugging Face git blob SHA-1 (small files) or LFS SHA-256 (weights).
CHECKPOINT_FILES=(
  "config.json c435ae135b5416de0a26d5ce67c9953c17d2ca35"
  "embodiment_id.json 9a5b40fe7cd0266bec65666448a0afb51bbe1138"
  "experiment_cfg/conf.yaml 4e508c6c405be0bf0a9581c1f8322e994f66afde"
  "experiment_cfg/config.yaml fc48423ee2b6431db5e969dc050ee760e9b790cc"
  "experiment_cfg/dataset_statistics.json 77c51beaa16296607f0057170c2a628dd8ecfba1"
  "experiment_cfg/final_model_config.json cf3bcfd0587735699df9c097b9bcdd148070e997"
  "experiment_cfg/final_processor_config.json 3f316b88953b53ff387aeb66e1cdeb14a7f77ab8"
  "model.safetensors.index.json 4761851d9515098ac203474f61221a26cd3a98b6"
  "processor_config.json d4bc10908562b05aef98eb69956b2616d74ce10d"
  "statistics.json fd5357e6b8f594b349f49ebfc9b006c8ff3c740a"
  "wandb_config.json 2b3c76bb0fa038f9d4d1b95d799a86b8dbb34747"
  "model-00002-of-00002.safetensors eb667619bd5b0e7b978d50ddb72aa0e46fd51f55252218dc59c5fe8c01847c3b"
  "model-00001-of-00002.safetensors 3eb85a770b0a6d4b66810e5005ac99066d25ca975939c73b7ce4a006438d98b6"
)
COSMOS_REVISION="9ce19a195e423419c349abfc86fd07178b230561"
COSMOS_SNAPSHOT="${ROOT}/hf-home/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/${COSMOS_REVISION}"
# file, Hugging Face git blob SHA-1 ("-" for the LFS weights), ModelScope SHA-256.
COSMOS_FILES=(
  "chat_template.json e49e75bdda104c8de941eb20e0b3765184fb27bc 6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4"
  "config.json 0cd8c646eb55794594b14fefcd1d3acea5cc000f bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
  "generation_config.json 7bb37933d2d86e4ceaabcc6540be989658f238e9 1e241830b48b397cb0900101421df5450baddc7adf01e5fc86b5615865f3bae4"
  "merges.txt 20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0 599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
  "preprocessor_config.json 2ea84a437d448ff71b08df68fdd949d5cc4ebb64 27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
  "tokenizer.json c6cc1014128b19d1fc46b1d30a23e3b1d35db421 a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
  "tokenizer_config.json d3d3763207692c78780f4bf42d4dadf49a5c8012 c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399"
  "video_preprocessor_config.json 3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe 7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
  "vocab.json 4783fe10ac3adce15ac8f358ef5462739852c569 ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
  "model.safetensors - fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede"
)
COSMOS_WEIGHTS_SIZE=4877470304

download() {
  # download <url> <target> [header]: resumable.
  local url="$1" target="$2" attempt
  shift 2
  mkdir -p "$(dirname -- "${target}")"
  for ((attempt = 1; attempt <= 20; attempt++)); do
    if curl -fLsS --retry 2 --retry-all-errors --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
      -C - "$@" --output "${target}" "${url}"; then
      return 0
    fi
    sleep 10
  done
  echo "Could not download ${url}" >&2
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

# Source.
if [[ ! -d "${SOURCE}/.git" ]]; then
  git init -q "${SOURCE}"
  git -C "${SOURCE}" remote add origin "${GR00T_URL}"
fi
if ! git -C "${SOURCE}" cat-file -e "${GR00T_COMMIT}^{commit}" 2>/dev/null; then
  GIT_LFS_SKIP_SMUDGE=1 git -C "${SOURCE}" fetch -q --depth 1 origin "${GR00T_COMMIT}"
fi
if [[ -n "$(git -C "${SOURCE}" status --porcelain --untracked-files=normal | grep -v -E '^\?\? (\.venv/|.*\.egg-info/$)' || true)" ]]; then
  echo "${SOURCE} contains local changes; use a clean directory" >&2
  exit 5
fi
GIT_LFS_SKIP_SMUDGE=1 git -C "${SOURCE}" -c advice.detachedHead=false checkout -q --detach "${GR00T_COMMIT}"

# Environment: the fork's uv.lock, exported with hashes and installed without resolution.
export UV_PYTHON_INSTALL_DIR="${ROOT}/python"
uv python install --no-bin --no-progress "${PYTHON_VERSION}"
[[ -x "${SOURCE}/.venv/bin/python" ]] || \
  uv venv --python "${PYTHON_VERSION}" --managed-python --no-python-downloads "${SOURCE}/.venv"
requirements="${ROOT}/gr00t-requirements.txt"
(cd "${SOURCE}" && uv export --frozen --no-dev --no-emit-project --format requirements-txt \
  --python "${PYTHON_VERSION}" --output-file "${requirements}" >/dev/null)
torch_sources=(--extra-index-url "${TORCH_INDEX}")
if [[ -n "${GR00T_TORCH_FIND_LINKS:-}" ]]; then
  torch_sources=(--find-links "${GR00T_TORCH_FIND_LINKS}")
fi
uv pip install --python "${SOURCE}/.venv/bin/python" --no-progress --no-deps --verify-hashes \
  --index-strategy unsafe-best-match --index-url "${PYPI_INDEX}" --extra-index-url "${NVIDIA_INDEX}" \
  "${torch_sources[@]}" \
  --requirements "${requirements}"
# The fork's pyproject pins websockets==16.0 for serve_b1k.py, but its uv.lock predates that pin.
printf '%s\n' 'websockets==16.0 --hash=sha256:7d837379b647c0c4c2355c2499723f82f1635fd2c26510e1f587d89bc2199e72' \
  > "${ROOT}/gr00t-requirements-extra.txt"
uv pip install --python "${SOURCE}/.venv/bin/python" --no-progress --no-deps --require-hashes \
  --index-url "${PYPI_INDEX}" --requirements "${ROOT}/gr00t-requirements-extra.txt"
uv pip install --python "${SOURCE}/.venv/bin/python" --no-progress --no-deps --no-build-isolation \
  --editable "${SOURCE}"
# transformers 4.57.3 asks the Hub whether a tokenizer is a Mistral one even in offline mode, so the
# offline Cosmos backbone cannot load. Backport transformers 4.57.4's fix (offline means local).
SITE_PACKAGES="$("${SOURCE}/.venv/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if patch -p1 --forward --dry-run --silent -d "${SITE_PACKAGES}" < "${PATCH}" >/dev/null; then
  patch -p1 --forward --silent -d "${SITE_PACKAGES}" < "${PATCH}"
elif ! patch -p1 --reverse --dry-run --silent -d "${SITE_PACKAGES}" < "${PATCH}" >/dev/null; then
  echo "transformers in ${SITE_PACKAGES} differs from 4.57.3; rebuild the environment" >&2
  exit 5
fi
uv pip freeze --python "${SOURCE}/.venv/bin/python" > "${ROOT}/gr00t-pip-freeze.txt"
"${SOURCE}/.venv/bin/python" -I -c "
import sys, gr00t, torch, flash_attn, transformers, websockets
from pathlib import Path
assert sys.version_info[:2] == (3, 10), sys.version
assert Path(gr00t.__file__).resolve().is_relative_to(Path('${SOURCE}').resolve()), gr00t.__file__
assert torch.__version__ == '2.7.1+cu128', torch.__version__
assert transformers.__version__ == '4.57.3', transformers.__version__
assert websockets.__version__ == '16.0', websockets.__version__
print('GR00T environment verified')
"

# Checkpoint snapshot.
for entry in "${CHECKPOINT_FILES[@]}"; do
  read -r path digest <<<"${entry}"
  target="${CHECKPOINT}/${path}"
  if ! matches "${target}" "${digest}"; then
    download "${HF_BASE}/${MODEL_REPO}/resolve/${MODEL_REVISION}/${SNAPSHOT}/${path}" "${target}"
    matches "${target}" "${digest}" || {
      rm -f -- "${target}"
      echo "${target} does not match ${digest}" >&2
      exit 5
    }
  fi
done

# Cosmos-Reason2-2B, as an offline Hugging Face cache snapshot.
mkdir -p "${COSMOS_SNAPSHOT}" "$(dirname -- "$(dirname -- "${COSMOS_SNAPSHOT}")")/refs"
printf '%s' "${COSMOS_REVISION}" > "$(dirname -- "$(dirname -- "${COSMOS_SNAPSHOT}")")/refs/main"
for entry in "${COSMOS_FILES[@]}"; do
  read -r file hf_sha1 ms_sha256 <<<"${entry}"
  target="${COSMOS_SNAPSHOT}/${file}"
  if matches "${target}" "${ms_sha256}"; then
    continue
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    download "${HF_BASE}/nvidia/Cosmos-Reason2-2B/resolve/${COSMOS_REVISION}/${file}" "${target}" \
      -H "Authorization: Bearer ${HF_TOKEN}"
  else
    download "https://modelscope.cn/api/v1/models/nv-community/Cosmos-Reason2-2B/repo?Revision=master&FilePath=${file}" \
      "${target}"
  fi
  matches "${target}" "${ms_sha256}" || {
    rm -f -- "${target}"
    echo "${target} does not match ${ms_sha256}" >&2
    exit 5
  }
  if [[ "${hf_sha1}" != - ]]; then
    matches "${target}" "${hf_sha1}" || {
      echo "${target} differs from the Hugging Face file" >&2
      exit 5
    }
  fi
done
[[ "$(stat -c %s -- "${COSMOS_SNAPSHOT}/model.safetensors")" == "${COSMOS_WEIGHTS_SIZE}" ]] || {
  echo "Cosmos-Reason2-2B weights are not ${COSMOS_WEIGHTS_SIZE} bytes" >&2
  exit 5
}

echo "GR00T_ROOT=${SOURCE}"
echo "GR00T_PYTHON=${SOURCE}/.venv/bin/python"
echo "HF_HOME=${ROOT}/hf-home"
echo "checkpoint: ${CHECKPOINT}"
