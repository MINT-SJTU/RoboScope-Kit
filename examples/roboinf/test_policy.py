"""Run with the pinned RoboInF checkout on PYTHONPATH; no model service required."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from policy import RoboInFPolicy
from roboinf_benchmark.contracts import CAMERAS, EEF, JOINT
from roboinf_benchmark.policies.adapters import Gr00t, LingBotVA, OpenWAM, Pi05

from roboscope_kit import EpisodeContext, Observation


@pytest.mark.parametrize(
    "name,horizon,width", [("pi_05", 50, 14), ("gr00t", 40, 14), ("openwam", 1, 16), ("lingbot-va", 1, 16)]
)
def test_kit_to_native_policy(name, horizon, width):
    pose = np.zeros(16, np.float32)
    pose[[3, 11]] = 1
    features = {"images." + k: np.zeros((4, 5, 3), np.uint8) for k in CAMERAS}
    features.update(
        {"state.joints": np.zeros(14, np.float32), "state.eef": pose, "language.instruction": "open"}
    )
    client = Mock()
    if name == "pi_05":
        client.infer.return_value = {"actions": np.zeros((50, 14))}
        native = Pi05(client)
    elif name == "gr00t":
        client.get_action.return_value = ({k: np.zeros((1, 40, 7)) for k in ("left_arm", "right_arm")}, {})
        native = Gr00t(client)
    elif name == "openwam":
        client.step.return_value = np.zeros(20)
        conversion = Mock()
        conversion.eef20d_to_ee16d.return_value = pose
        native = OpenWAM(client, EEF, conversion, layout="legacy_wxyz_values_as_xyzw")
    else:
        raw = np.zeros((16, 2, 4))
        raw[[6, 14]] = 1
        client.infer.return_value = {"action": raw}
        native = LingBotVA(client)
    wrapper = RoboInFPolicy(name, {}, None, horizon)
    with patch("policy.load_policy", return_value=native):
        caps = wrapper.load(SimpleNamespace())
    expected = JOINT if width == 14 else EEF
    assert caps.action_representations == {"roboinf": "roboinf." + expected.name}
    assert caps.supported_suites["roboinf"] == ("test", "test_random")
    wrapper.reset(EpisodeContext("one", "task", {"seed": 0}))
    with pytest.raises(ValueError, match="one active episode"):
        wrapper.reset(EpisodeContext("two", "task", {"seed": 0}))
    action = wrapper.act(Observation("one", 0, features, 0.0))
    assert action.actions.shape == (horizon, width)
    wrapper.end_episode("one")
    wrapper.reset(EpisodeContext("two", "task", {"seed": 0}))
    wrapper.unload()
