"""Run the StarVLA Policy with a CPU budget suited to concurrent RoboCasa workers."""

from __future__ import annotations

import os


def main() -> None:
    threads = int(os.environ.get("ROBOSCOPE_POLICY_CPU_THREADS", "1"))
    if threads < 1:
        raise ValueError("ROBOSCOPE_POLICY_CPU_THREADS must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import cv2
    import torch

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    cv2.setNumThreads(1)
    print(
        f"StarVLA CPU threads: intra={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()} opencv={cv2.getNumThreads()}",
        flush=True,
    )

    from starvla_policy import main as run_policy

    run_policy()


if __name__ == "__main__":
    main()
