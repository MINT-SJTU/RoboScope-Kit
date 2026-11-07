"""Run the StarVLA Policy with a CPU budget suited to concurrent RoboCasa workers."""

from __future__ import annotations

import os
from pathlib import Path


def release_checkpoint_cache(checkpoint: Path) -> None:
    """Hint that the loaded checkpoint's clean file pages need not stay cached."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        with checkpoint.open("rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError as error:
        print(f"StarVLA checkpoint cache hint unavailable: {error}", flush=True)
    else:
        print("StarVLA checkpoint cache drop requested", flush=True)


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

    from starvla_policy import StarVLAPolicy, resolve_starvla_run_snapshot

    from roboscope_kit.policy import PolicyCapabilities, SessionContext
    from roboscope_kit.server import run

    class RoboCasaStarVLAPolicy(StarVLAPolicy):
        def load(self, context: SessionContext) -> PolicyCapabilities:
            checkpoint, _ = resolve_starvla_run_snapshot(context.checkpoint_path)
            capabilities = super().load(context)
            release_checkpoint_cache(checkpoint)
            return capabilities

    run(RoboCasaStarVLAPolicy())


if __name__ == "__main__":
    main()
