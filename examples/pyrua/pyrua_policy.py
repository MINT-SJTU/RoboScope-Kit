"""Serve PyRUA-Lean's native RoboDojo agent through RoboScope Kit.

Install the pinned PyRUA-Lean policy dependencies separately from the simulator.
The policy preserves PyRUA's own Bridge, tools, prompt and episode judge; only
the transport from the RoboDojo client to that bridge is adapted here.
"""

from __future__ import annotations

import argparse
import json
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboscope_kit import (
    ActionChunk,
    EpisodeContext,
    Observation,
    Policy,
    PolicyCapabilities,
    SessionContext,
    run,
)
from roboscope_kit.policy import EpisodeControl

CAMERAS = {"cam_head": "images.head", "cam_left_wrist": "images.left_wrist",
           "cam_right_wrist": "images.right_wrist"}
ACTION_KEYS = ("left_arm_joint_state", "left_ee_joint_state",
               "right_arm_joint_state", "right_ee_joint_state")
# Once the simulator sends its verdict, the agent may still be completing a
# tool call and shutting down its model session. Keep that cleanup bounded but
# allow time for slow image-bearing responses before rejecting the episode.
# This remains below the RoboDojo PolicyClient's 900-second socket timeout.
OUTCOME_JOIN_TIMEOUT_S = 720.0


def native_observation(observation: Observation) -> dict[str, Any]:
    extras = observation.metadata.get("robodojo_native")
    if not isinstance(extras, dict):
        raise TypeError("interactive RoboDojo observation is missing native metadata")
    raw = deepcopy(extras["other"])
    raw["state"] = deepcopy(extras["state"])
    raw["vision"] = deepcopy(extras["vision"])
    for native, feature in CAMERAS.items():
        raw["vision"][native]["color"] = observation.features[feature]
    raw["instruction"] = observation.features["language.instruction"]
    return raw


def action_or_control(actions: list[dict[str, Any]]) -> ActionChunk | EpisodeControl:
    if not actions:
        return EpisodeControl("finish")
    if len(actions) == 1 and actions[0] == {"pyrualean": "wait"}:
        return EpisodeControl("wait")
    if len(actions) != 1:
        raise ValueError("PyRUA's RoboDojo bridge must return exactly one action")
    action = actions[0]
    if set(action) != set(ACTION_KEYS):
        raise ValueError("PyRUA returned unexpected joint action fields")
    pieces = [np.asarray(action[key], dtype=np.float32).reshape(-1) for key in ACTION_KEYS]
    if tuple(len(piece) for piece in pieces) != (6, 1, 6, 1):
        raise ValueError("PyRUA returned unexpected joint action dimensions")
    row = np.concatenate(pieces)
    if not np.isfinite(row).all():
        raise ValueError("PyRUA returned a non-finite joint action")
    return ActionChunk(row)


def validate_result(result: dict[str, Any] | None, event_path: Path, verdict: bool) -> None:
    if (not isinstance(result, dict) or result.get("status") != "completed"
            or result.get("codex_returncode") != 0
            or type(result.get("environment_success")) is not bool
            or result["environment_success"] != verdict):
        raise RuntimeError("PyRUA agent result does not match the simulator verdict")
    completed = {"python": 0, "finish": 0}
    with event_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            item = event.get("item") or {}
            if (event.get("type") != "item.completed" or item.get("type") != "mcp_tool_call"
                    or item.get("server") != "pyrualean"):
                continue
            if item.get("status") == "failed":
                detail = json.dumps({"error": item.get("error"), "result": item.get("result")})
                if "Transport send error" in detail or "validation errors for CallToolResult" in detail:
                    raise RuntimeError("PyRUA tool transport or result serialization failed")
            if item.get("status") == "completed" and item.get("tool") in completed:
                completed[item["tool"]] += 1
    finish = result.get("finish")
    if not completed["python"]:
        raise RuntimeError("PyRUA did not complete a Python tool call")
    if finish is None:
        if not verdict:
            raise RuntimeError("unsuccessful PyRUA episode has no finish tool call")
    elif (not isinstance(finish, dict) or finish.get("status") not in ("success", "failure", "stuck")
          or not completed["finish"]):
        raise RuntimeError("PyRUA finish tool result is missing or invalid")


@dataclass
class Episode:
    context: EpisodeContext
    bridge: Any
    thread: threading.Thread | None = None
    result: dict[str, Any] | None = None
    error: BaseException | None = None
    outcome_received: bool = False


class PyRUAPolicy(Policy):
    def __init__(self, *, model: str, reasoning: str, codex_bin: str,
                 result_dir: Path, max_turns: int, budget_s: float) -> None:
        self.model, self.reasoning, self.codex_bin = model, reasoning, codex_bin
        self.result_dir = result_dir.resolve()
        self.max_turns, self.budget_s = max_turns, budget_s
        self.suite = ""
        self.episodes: dict[str, Episode] = {}

    def load(self, context: SessionContext) -> PolicyCapabilities:
        from pyrualean.hosts.xpolicylab_bridge import Bridge  # noqa: F401 - fail before evaluation
        from pyrualean.play import play_episode  # noqa: F401 - fail before evaluation

        if context.benchmark.name != "robodojo":
            raise ValueError("PyRUA policy only supports RoboDojo")
        self.suite = context.benchmark.suite
        self.result_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        return PolicyCapabilities(
            supported_benchmarks=("robodojo",),
            required_features=("images.head", "images.left_wrist", "images.right_wrist",
                               "state.joints", "state.ee_pose", "language.instruction"),
            max_batch_size=1,
            supported_action_chunk_sizes=(1,),
            action_representations={"robodojo": "robodojo.x5-joint-v1"},
            interactive_benchmarks=("robodojo",),
            max_workers_per_replica=1,
        )

    def reset(self, context: EpisodeContext) -> None:
        from pyrualean.hosts.xpolicylab_bridge import Bridge

        if self.episodes:
            raise ValueError("PyRUA requires one active episode per replica; use one replica per Worker")
        if Path(context.episode_id).name != context.episode_id or context.episode_id in (".", ".."):
            raise ValueError("PyRUA episode ID must be a single path component")
        episode_dir = self.result_dir / context.episode_id
        episode_dir.mkdir(mode=0o700, exist_ok=False)
        identity = {"episode_id": context.episode_id, "task": context.task_id,
                    "layout_id": context.metadata.get("layout_id"),
                    "robodojo_run_id": context.metadata.get("robodojo_run_id")}
        (episode_dir / "kit_episode.json").write_text(json.dumps(identity) + "\n", encoding="utf-8")
        self.episodes[context.episode_id] = Episode(context, Bridge(poll_s=0.01))

    def act(self, observation: Observation) -> ActionChunk | EpisodeControl:
        episode = self.episodes[observation.episode_id]
        raw = native_observation(observation)
        status = (raw.get("additional_info") or {}).get("pyrualean") or {}
        if (status.get("task_name") != episode.context.task_id
                or status.get("layout") != episode.context.metadata.get("layout_id")):
            raise ValueError("native episode metadata does not match the active Kit episode")
        episode.bridge.update_obs(raw)
        if episode.thread is None:
            episode.thread = threading.Thread(target=self._play, args=(episode, dict(status)),
                                              name="pyrualean-episode", daemon=True)
            episode.thread.start()
        return self._poll(episode)

    def poll(self, episode_id: str) -> ActionChunk | EpisodeControl:
        return self._poll(self.episodes[episode_id])

    @staticmethod
    def _poll(episode: Episode) -> ActionChunk | EpisodeControl:
        answer = action_or_control(episode.bridge.get_action())
        if isinstance(answer, EpisodeControl) and answer.kind == "finish" and episode.error:
            raise RuntimeError("PyRUA agent failed before completing the episode") from episode.error
        return answer

    def episode_outcome(self, episode_id: str, outcome: dict[str, Any]) -> None:
        episode = self.episodes[episode_id]
        if episode.outcome_received:
            raise ValueError("episode outcome was already delivered")
        episode.outcome_received = True
        episode.bridge.episode_end(outcome)
        if episode.thread is None:
            raise RuntimeError("PyRUA agent did not start")
        episode.thread.join(timeout=OUTCOME_JOIN_TIMEOUT_S)
        if episode.thread.is_alive():
            raise TimeoutError("PyRUA did not record the episode verdict")
        if episode.error is not None:
            raise RuntimeError("PyRUA agent failed") from episode.error
        validate_result(episode.result, self.result_dir / episode_id / "codex_events.jsonl",
                        outcome["success"])

    def end_episode(self, episode_id: str) -> None:
        episode = self.episodes.pop(episode_id, None)
        if episode is None:
            return
        if not episode.outcome_received:
            episode.bridge.abandon()
        if episode.thread is not None and episode.thread.is_alive():
            episode.thread.join(timeout=30)
            if episode.thread.is_alive():
                raise TimeoutError("PyRUA agent did not stop after episode end")

    def unload(self) -> None:
        for episode_id in list(self.episodes):
            self.end_episode(episode_id)

    def _play(self, episode: Episode, status: dict[str, Any]) -> None:
        from pyrualean.hosts.xpolicylab_robodojo import Cell
        from pyrualean.play import play_episode

        context = episode.context
        cell = Cell(suite=self.suite, task=context.task_id, seed=int(status["layout"]), cuda_device=0)
        out = self.result_dir / context.episode_id
        try:
            episode.result = play_episode(
                cell, out, arm="cells", model=self.model, reasoning_effort=self.reasoning,
                budget_s=self.budget_s, codex_bin=self.codex_bin, images="on-demand",
                feedback="pure", max_turns=self.max_turns, robot="robodojo_xpl_rgb",
                boot_options={"bridge": episode.bridge, "eval_seed": int(status.get("eval_seed", 0))},
                runtime="codex", primitives="no-vla",
            )
        except BaseException as exc:  # noqa: BLE001 - evaluator must stop if the agent crashes
            episode.error = exc
        finally:
            episode.bridge.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", default="max")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--budget-s", type=float, default=5400.0)
    args = parser.parse_args()
    if args.max_turns < 1 or args.budget_s <= 0:
        parser.error("max-turns and budget-s must be positive")
    run(PyRUAPolicy(model=args.model, reasoning=args.reasoning, codex_bin=args.codex_bin,
                    result_dir=args.result_dir, max_turns=args.max_turns, budget_s=args.budget_s))


if __name__ == "__main__":
    main()
