# Copyright 2026 RoboScope Team
# SPDX-License-Identifier: Apache-2.0

"""Official RoboTwin work manifest and SDK Policy worker adapter."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscope_kit.runtime.benchmark import ProcessSpec, WorkerContext, WorkItem, WorkResult

from .render_service import RoboTwinRenderServices

ROBOTWIN_ALL_TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)

# Observation feature -> worker flag that creates its RoboTwin camera at a negotiated size.
ROBOTWIN_CAMERA_SIZE_FLAGS = (
    ("images.head", "--head-size"),
    ("images.left_wrist", "--left-wrist-size"),
    ("images.right_wrist", "--right-wrist-size"),
)


@dataclass(frozen=True)
class RoboTwinRuntimeConfig:
    sdk_python: Path
    benchmark_python: Path
    robotwin_root: Path
    checkpoint_path: Path
    deploy_policy_path: Path
    launcher_path: Path
    suite: str
    tasks: tuple[str, ...]
    episodes_per_task: int
    episode_shard_size: int
    seed_cache_dir: Path | None = None
    protocol_id: str | None = None
    seed_generation_digest: str | None = None
    instruction_seed_limit: int | None = None
    seed: int = 0
    save_videos: bool = False
    cpu_threads: int = 1
    # "native": every evaluator renders on its GPU. "central": evaluators simulate on the CPU and a few
    # render services per GPU render for them (render_service_path, render_services_per_gpu).
    render_mode: str = "native"
    render_services_per_gpu: int = 2
    render_service_path: Path | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.tasks) - set(ROBOTWIN_ALL_TASKS))
        if unknown:
            raise ValueError(f"unknown RoboTwin tasks: {', '.join(unknown)}")
        if (
            not self.tasks
            or self.episodes_per_task < 1
            or self.episode_shard_size < 1
            or self.cpu_threads < 1
        ):
            raise ValueError(
                "RoboTwin tasks, episodes, episode_shard_size, and cpu_threads must be positive"
            )
        if self.suite not in {"demo_clean", "demo_randomized"}:
            raise ValueError(f"unsupported RoboTwin suite: {self.suite}")
        if self.instruction_seed_limit is not None and self.instruction_seed_limit < 1:
            raise ValueError("RoboTwin instruction_seed_limit must be positive")
        if (self.protocol_id is None) != (self.seed_generation_digest is None):
            raise ValueError("RoboTwin protocol and seed generation identity must be paired")
        if self.render_mode not in {"native", "central"}:
            raise ValueError(f"RoboTwin render_mode must be native or central; got {self.render_mode!r}")
        if self.render_mode == "central":
            if self.render_service_path is None or self.render_services_per_gpu < 1:
                raise ValueError(
                    "RoboTwin central rendering needs runtime.render_service_path and a positive "
                    "render_services_per_gpu"
                )
            if self.suite != "demo_clean" or self.seed_cache_dir is None or self.save_videos:
                # demo_randomized changes light colors during episodes, which the services do not replay;
                # without a seed cache the expert path needs cuRobo on the GPU; videos read every camera.
                raise ValueError(
                    "RoboTwin central rendering supports demo_clean with a seed cache and no videos"
                )


class RoboTwinBenchmarkAdapter:
    def __init__(self, config: RoboTwinRuntimeConfig) -> None:
        self.config = config
        self.render_services: RoboTwinRenderServices | None = None

    def work_items(self) -> Sequence[WorkItem]:
        items = []
        ordinal = 0
        total = self.config.episodes_per_task
        for episode_start in range(0, total, self.config.episode_shard_size):
            episodes = min(self.config.episode_shard_size, total - episode_start)
            episode_stop = episode_start + episodes
            for task in self.config.tasks:
                item_id = (
                    f"{self.config.suite}-{task}-"
                    f"episodes-{episode_start:04d}-{episode_stop:04d}"
                )
                items.append(
                    WorkItem(
                        ordinal=ordinal,
                        item_id=item_id,
                        # A unique group disables WorkerLoop's same-group
                        # affinity after this window completes. Task aggregation
                        # uses result metadata, so later windows stay available
                        # to every idle worker instead of sticking to one lane.
                        group_id=item_id,
                        payload={
                            "task": task,
                            "suite": self.config.suite,
                            "seed": self.config.seed,
                            "episode_start": episode_start,
                            "episodes": episodes,
                            "total_episodes": total,
                            # Instructions are drawn from the protocol's limit, not from
                            # this run's episode count, so a run of fewer episodes per
                            # task draws the same instruction for each seed.
                            "instruction_seed_limit": self.config.instruction_seed_limit or total,
                        },
                        expected_results=episodes,
                    )
                )
                ordinal += 1
        return tuple(items)

    def worker_process(self, context: WorkerContext) -> ProcessSpec:
        action_type = {
            "robotwin.qpos-v1": "qpos",
            "robotwin.ee-compact-quat-v1": "ee",
        }[context.execution_contract.action_representation]
        threads = str(self.config.cpu_threads)
        image_sizes = context.execution_contract.image_sizes
        camera_sizes: tuple[str, ...] = ()
        for feature, flag in ROBOTWIN_CAMERA_SIZE_FLAGS:
            if feature in image_sizes:
                height, width = image_sizes[feature]
                camera_sizes += (flag, f"{height}x{width}")
        render_service = ()
        if self.config.render_mode == "central":
            if action_type != "qpos":
                raise ValueError("RoboTwin central rendering supports qpos actions only")
            if image_sizes:
                # benchmark_camera_support offers RoboTwin's camera ranges for native rendering only.
                raise ValueError("RoboTwin central rendering renders the task config's camera sizes only")
            if self.render_services is None:
                assert self.config.render_service_path is not None
                self.render_services = RoboTwinRenderServices(
                    python=self.config.benchmark_python,
                    script=self.config.render_service_path,
                    robotwin_root=self.config.robotwin_root,
                    services_per_gpu=self.config.render_services_per_gpu,
                    log_dir=context.output_dir / "render-services",
                )
            socket = self.render_services.socket_for(context.gpu_device, context.gpu_slot, context.worker_id)
            render_service = ("--render-service", str(socket))
        return ProcessSpec(
            command=(
                str(self.config.sdk_python),
                "-I",
                "-m",
                "roboscope_kit.runtime.benchmarks.robotwin.worker",
                "--worker-id", str(context.worker_id),
                "--gpu-id", context.gpu_device,
                "--policy-socket", str(context.replica_socket),
                "--robotwin-dir", str(self.config.robotwin_root),
                "--robotwin-python", str(self.config.benchmark_python),
                "--checkpoint-path", str(self.config.checkpoint_path),
                "--deploy-policy", str(self.config.deploy_policy_path),
                "--launcher", str(self.config.launcher_path),
                "--queue-dir", str(context.queue_dir),
                "--worker-state-dir", str(context.state_dir),
                "--output-dir", str(context.output_dir),
                *(("--disable-video",) if not self.config.save_videos else ()),
                *(
                    ("--seed-cache-dir", str(self.config.seed_cache_dir))
                    if self.config.seed_cache_dir is not None
                    else ()
                ),
                # With a seed cache every task skips the expert play_once path,
                # so one launcher process can serve consecutive tasks safely;
                # the shared CuroboPlanner patch then pays planner construction
                # once per worker instead of once per task.
                *(("--eval-session",) if self.config.seed_cache_dir is not None else ()),
                *render_service,
                *camera_sizes,
            ),
            cwd=self.config.robotwin_root,
            env={
                "PYTHONPATH": ":".join(
                    (
                        str(self.config.robotwin_root),
                        str(self.config.robotwin_root / "script"),
                        str(self.config.robotwin_root / "policy"),
                        str(self.config.robotwin_root / "description" / "utils"),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "TOKENIZERS_PARALLELISM": "false",
                "ROBOTWIN_ACTION_TYPE": action_type,
                "ROBOTWIN_OBS_STRIDE": str(context.execution_contract.observation_stride),
            },
        )

    def close(self) -> None:
        """Stop the render services after the pass's Workers have stopped."""

        if self.render_services is not None:
            services, self.render_services = self.render_services, None
            services.close()

    def aggregate(self, results: Sequence[WorkResult]) -> dict[str, Any]:
        tasks: dict[str, dict[str, Any]] = {}
        for result in results:
            task_name = str(result.metadata["task"])
            row = tasks.setdefault(task_name, {"task": task_name, "episodes": 0, "successes": 0})
            row["episodes"] += len(result.records)
            row["successes"] += sum(record.success for record in result.records)
        rows = []
        for task_name in self.config.tasks:
            row = tasks.get(task_name, {"task": task_name, "episodes": 0, "successes": 0})
            row["success_rate"] = row["successes"] / row["episodes"] if row["episodes"] else 0.0
            rows.append(row)
        return {"tasks": rows}

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "adapter": type(self).__name__,
            "benchmark": "robotwin",
            "suite": self.config.suite,
            "protocol": "robotwin2-v1",
            "bound_hint": "gpu_infer",
            "tasks": list(self.config.tasks),
            "episodes_per_task": self.config.episodes_per_task,
            "episode_shard_size": self.config.episode_shard_size,
            "seed": self.config.seed,
            "video_logging": self.config.save_videos,
            "render_mode": self.config.render_mode,
        }
        if self.config.render_mode == "central":
            metadata["render_services_per_gpu"] = self.config.render_services_per_gpu
        if self.config.protocol_id is not None:
            metadata["benchmark_protocol_id"] = self.config.protocol_id
            metadata["seed_generation_digest"] = self.config.seed_generation_digest
        return metadata


__all__ = [
    "ROBOTWIN_ALL_TASKS",
    "ROBOTWIN_CAMERA_SIZE_FLAGS",
    "RoboTwinBenchmarkAdapter",
    "RoboTwinRuntimeConfig",
]
