"""Install private simulator/policy environments through the pinned upstream installers."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = {
    "simulator": r'''
set -euo pipefail
if [[ ! -x "$ENV_ROOT/simulator/bin/python" ]]; then
  "$PYTHON311" -m venv "$ENV_ROOT/simulator"
fi
source "$ENV_ROOT/simulator/bin/activate"
# --help defines the upstream functions without running its moving-submodule
# update or creating its hard-coded RoboDojo environment.
source "$ROBODOJO_ROOT/scripts/install.sh" --help
pip_install() { uv pip install --python "$VIRTUAL_ENV/bin/python" "$@"; }
pip_install 'pip==26.2.2' 'setuptools==79.0.1' 'wheel==0.45.1'
APT=(apt-get -o Dir::Etc::sourceparts=- -o Dir::Etc::sourcelist=/etc/apt/sources.list)
"${APT[@]}" update
"${APT[@]}" install -y --no-install-recommends cmake build-essential ffmpeg pkg-config \
  libgl1 libglu1-mesa libegl1 libvulkan1 \
  libxt6 libxrandr2 libxi6 libxcursor1 libxinerama1 vulkan-tools ninja-build
setup_base_deps
setup_isaacsim
setup_isaaclab
# Refresh metadata even when the upstream installer reuses an existing package.
python -m pip install --no-deps --no-build-isolation -e "$ROBODOJO_ROOT/third_party/IsaacLab/source/isaaclab"
setup_curobo
# The simulator uses XPolicyLab from PYTHONPATH, as in the upstream Docker image.
# Its client supports Isaac Sim's websockets 12 pin; the separate policy server
# installs XPolicyLab's newer websocket dependency in its own environment.
python -m pip install 'pydantic>=2.5' 'h5py>=3.8' 'pyyaml>=6' 'msgpack>=1.0.8' msgpack-numpy
python -m pip check
python -m pip freeze
''',
    "policy": r'''
set -euo pipefail
if [[ ! -x "$ENV_ROOT/policy/bin/python" ]]; then
  "$PYTHON311" -m venv "$ENV_ROOT/policy"
fi
source "$ENV_ROOT/policy/bin/activate"
# The packages are the pinned adapter install.sh recipe, in a private venv.
uv pip install --python "$VIRTUAL_ENV/bin/python" 'pip==24.0' 'setuptools==79.0.1' 'wheel==0.48.0'
uv pip install --python "$VIRTUAL_ENV/bin/python" 'torch==2.7.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VIRTUAL_ENV/bin/python" \
  'numpy>=1.26,<2' 'sentencepiece==0.2.2' 'safetensors>=0.4.3' \
  'huggingface_hub>=0.34.0' pillow 'opencv-python-headless==4.11.0.86' \
  --no-build-isolation -e "$ROBODOJO_ROOT/XPolicyLab"
python -m pip check
PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab" python - <<'PY'
from XPolicyLab.policy.Simate_beta.model import Model
from client_server.ws.model_server import PolicyServer
PY
python -m pip freeze
''',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=SCRIPTS)
    parser.add_argument("root", type=Path)
    parser.add_argument("--env-root", type=Path, default=Path("/workspace/robodojo-envs"))
    parser.add_argument("--cache-root", type=Path, default=Path("/workspace/robodojo-cache"))
    parser.add_argument("--python311", type=Path, required=True, help="existing Python 3.11 interpreter")
    args = parser.parse_args()
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() not in {"YES", "Y"}:
        parser.error(
            "Isaac Sim is NVIDIA software under its own EULA "
            "(https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html). "
            "Read it and export OMNI_KIT_ACCEPT_EULA=YES to accept it yourself; this recipe does not accept it for you."
        )
    subprocess.run(
        [str(args.python311), "-c", "import sys; assert sys.version_info[:2] == (3, 11)"], check=True
    )
    root = args.root.resolve()
    if args.component == "simulator":
        state = subprocess.check_output(
            [sys.executable, str(HERE / "apply_runtime_patch.py"), "check", str(root),
             "--patch", "isaac51-deps"], text=True,
        ).strip()
        if state != "isaac51-deps patch: after":
            parser.error("Apply the committed isaac51-deps patch before installing Isaac Sim 5.1")
    evidence = root.parent / "evidence" / "runtime-setup"
    evidence.mkdir(parents=True, exist_ok=True)
    args.env_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        ROBODOJO_ROOT=str(root), ENV_ROOT=str(args.env_root.resolve()),
        PYTHON311=str(args.python311.resolve()),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="2",
        PIP_USER="0", PYTHONNOUSERSITE="1",
        # uv streams and unpacks many small blocks; use local storage, not AutoFS.
        PIP_CACHE_DIR=str(args.cache_root.resolve() / "pip"),
        UV_CACHE_DIR=str(args.cache_root.resolve() / "uv"), UV_CONCURRENT_DOWNLOADS="16",
        UV_HTTP_TIMEOUT="120",
        # The caller accepted the NVIDIA EULA (checked above); PRIVACY_CONSENT stays theirs to set.
        ACCEPT_EULA="Y",
        TERM="xterm-256color", CUDA_HOME="/usr/local/cuda",
        PIP_CONSTRAINT=str(HERE / f"constraints-{args.component}.txt"),
        UV_CONSTRAINT=str(HERE / f"constraints-{args.component}.txt"),
    )
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    env["PATH"] = f"/usr/local/cuda/bin:{env['PATH']}"
    started = time.time()
    log = evidence / f"install-{args.component}-{int(started)}.log"
    print(f"Installing {args.component}; log: {log}", flush=True)
    with log.open("w") as output:
        result = subprocess.run(
            ["bash", "-c", SCRIPTS[args.component]], cwd=root, env=env,
            stdout=output, stderr=subprocess.STDOUT, check=False,
        )
    record = {
        "component": args.component, "started_unix_s": started,
        "elapsed_s": time.time() - started, "exit_code": result.returncode,
        "log": str(log), "env_root": str(args.env_root.resolve()),
    }
    log.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
