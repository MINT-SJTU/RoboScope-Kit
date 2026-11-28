#!/usr/bin/env bash
set -euo pipefail
: "${ROBOINF_ASSETS:?provide the licensed and expanded RoboTwin assets directory}"
root=$(realpath "${1:?usage: build-runtime.sh INSTALL_DIR}")
src=$(dirname "$(realpath "$0")")
assets=$(realpath -e "$ROBOINF_ASSETS")
test -d "$assets/embodiments" && test -d "$assets/background_texture"
mkdir -p "$root"
if [[ -n $(ls -A "$root") ]]; then echo "Installation directory must be empty" >&2; exit 1; fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH
python=${PYTHON310:-python3.10}
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
"$python" -m venv "$root/sim-env"
py="$root/sim-env/bin/python"
"$py" -m pip install -r "$src/requirements-sim.txt"
git clone --no-checkout https://github.com/NVlabs/curobo.git "$root/curobo"
git -C "$root/curobo" checkout --detach d64c4b005459db10c5dd867d8b30a87d5bda9bdb
"$py" -m pip install --no-build-isolation "$root/curobo"
mkdir -p "$root/data/evaluation"
git clone --no-checkout https://github.com/SJTU-DeepVisionLab/RoboInF.git "$root/RoboInF"
git -C "$root/RoboInF" checkout --detach fa162e107f0dc725fd4539e9a1d3a5d0e3fcdd93
cp -R "$root/RoboInF/third_party/robotwin" "$root/RoboTwin"
ln -s "$assets" "$root/RoboTwin/assets"
(cd "$root/RoboTwin" && sha256sum -c "$root/RoboInF/roboinf_benchmark/robotwin.sha256")
for file in test-cases test-scenes; do
  if [[ -n ${ROBOINF_EVALUATION_DIR:-} ]]; then
    cp "$ROBOINF_EVALUATION_DIR/$file.jsonl" "$root/data/evaluation/"
  else
    curl -fLsS --retry 3 \
    "${HF_ENDPOINT:-https://huggingface.co}/datasets/JGuo666/RoboInF-4K/resolve/1893df2369cd9316f4ef56928252ae309d90471d/evaluation/$file.jsonl" \
    -o "$root/data/evaluation/$file.jsonl"
  fi
done
PYTHONPATH="$root/RoboInF" "$root/sim-env/bin/python" -m roboinf_benchmark inspect --domain test --data "$root/data"
install -m644 "$src"/{sim_case.py,benchmark-manifest.json} "$root"
install -m755 "$src/replay-judge.sh" "$root"
