#!/usr/bin/env bash
set -euo pipefail
root=$(dirname "$(realpath "$0")")
export PYTHONNOUSERSITE=1 PYTHONPATH="$root/RoboInF"
exec "$root/sim-env/bin/python" -m roboinf_benchmark.checker.replay \
    --data "$root/data" --robotwin "$root/RoboTwin" "$@"
