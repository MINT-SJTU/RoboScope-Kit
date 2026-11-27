#!/usr/bin/env bash
set -euo pipefail
# Assemble the public evaluator around a provisioned RoboTwin simulator and its licensed assets.
: "${ROBOINF_SIM_ENV:?provide the simulator virtual environment}"
: "${ROBOINF_ROBOTWIN_ROOT:?provide the RoboTwin source and assets}"
root=$(realpath "${1:?usage: build-runtime.sh INSTALL_DIR}")
src=$(dirname "$(realpath "$0")")
test "$(git -C "$ROBOINF_ROBOTWIN_ROOT" rev-parse HEAD)" = 266f3aadf505a4f7fe9af0faa41a20f5f47cd123
"$ROBOINF_SIM_ENV/bin/python" -c 'import msgpack, numpy, sapien, mplib; assert numpy.__version__ == "1.26.4"'
mkdir -p "$root/data/evaluation"
git clone --no-checkout https://github.com/SJTU-DeepVisionLab/RoboInF.git "$root/RoboInF"
git -C "$root/RoboInF" checkout --detach af9d8d583885df48b300ae97f1a45bd07d318b6a
ln -s "$(realpath "$ROBOINF_SIM_ENV")" "$root/sim-env"
ln -s "$(realpath "$ROBOINF_ROBOTWIN_ROOT")" "$root/RoboTwin"
for file in test-cases test-scenes; do
  curl -fLsS --retry 3 \
    "${HF_ENDPOINT:-https://huggingface.co}/datasets/JGuo666/RoboInF-4K/resolve/1893df2369cd9316f4ef56928252ae309d90471d/evaluation/$file.jsonl" \
    -o "$root/data/evaluation/$file.jsonl"
done
PYTHONPATH="$root/RoboInF" "$root/sim-env/bin/python" -m roboinf_benchmark inspect --domain test --data "$root/data"
install -m644 "$src"/{sim_case.py,benchmark-manifest.json} "$root"
