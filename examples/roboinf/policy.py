"""Serve the released RoboInF policy clients or an original experiment method."""

import argparse
import json
from pathlib import Path

from roboinf_benchmark.contracts import CAMERAS, EpisodeContext, Observation
from roboinf_benchmark.policies import POLICY_NAMES, load_policy

from roboscope_kit import ActionChunk, Policy, PolicyCapabilities, run


class RoboInFPolicy(Policy):
    def __init__(self, model, client_config, method, chunk):
        self.model, self.config, self.method, self.chunk = model, client_config, method, chunk
        self.sessions = {}

    def load(self, context):
        if self.method:
            # Import this dependency only in the original experiment environment.
            from benchmarking.methods import create_policy, load_method

            method = load_method(self.method)
            key = "model_bundle" if self.model == "lingbot-va" else "checkpoint"
            if method["kind"] != "python" or not Path(method["config"][key]).samefile(
                context.checkpoint_path
            ):
                raise ValueError("The supplied checkpoint must match the original local method")
            if method["description"]["input_mode"] != "task_instruction":
                raise ValueError("Expected a task-instruction method")
            self.native = create_policy(method)
        else:
            self.native = load_policy(self.model, self.config)
        self.spec = self.native.describe().action_spec
        features = [*("images." + key for key in CAMERAS), "state.joints", "language.instruction"]
        if self.spec == "eef16_absolute_wxyz":
            features.append("state.eef")
        return PolicyCapabilities(
            ("roboinf",),
            tuple(features),
            {"roboinf": ("test", "test_random")},
            supported_action_chunk_sizes=(self.chunk,),
            action_representations={"roboinf": "roboinf." + self.spec},
        )

    def reset(self, context):
        if context.episode_id in self.sessions:
            raise ValueError("Duplicate episode reset")
        if hasattr(self.native, "new_session"):
            session = self.native.new_session()
        elif self.sessions:
            raise ValueError("Stateful model endpoint requires one active episode per replica")
        else:
            session = self.native
        self.sessions[context.episode_id] = (session, context.metadata["seed"])

    def act(self, obs):
        session, seed = self.sessions[obs.episode_id]
        f = obs.features
        if seed is not None:
            session.reset(EpisodeContext(f["language.instruction"], seed))
            self.sessions[obs.episode_id] = (session, None)
        observation = Observation(
            {k: f["images." + k] for k in CAMERAS},
            f["state.joints"],
            f.get("state.eef"),
            obs.step,
            obs.timestamp_s,
        )
        return ActionChunk(session.predict(observation).checked(self.spec))

    def end_episode(self, episode_id):
        session, _ = self.sessions.pop(episode_id)
        if session is not self.native:
            session.close()

    def unload(self):
        for episode_id in list(self.sessions):
            self.end_episode(episode_id)
        self.native.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=POLICY_NAMES)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--method", type=Path, help="Original trusted local benchmarking method manifest")
    group.add_argument("--client-config", type=Path, help="Upstream model server client configuration JSON")
    parser.add_argument("--chunk", type=int, help="Exact prediction horizon for the selected checkpoint")
    args = parser.parse_args()
    horizons = {
        "pi_05": 50,
        "openwam": 32 if args.method else 1,
        "gr00t": 40,
        "lingbot-va": 4 if args.method else 1,
    }
    config = json.loads(args.client_config.read_text()) if args.client_config else {}
    run(
        RoboInFPolicy(
            args.model, config, args.method, horizons[args.model] if args.chunk is None else args.chunk
        )
    )
