"""Probe reset history without a Policy or installed-environment changes."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

ROOT = Path(__file__).resolve().parent.parent / "reset-probe"
ROOT.mkdir(exist_ok=True)
PLAN = {"spatial": [(1,3)], "object": [(2,4)], "goal": [(0,2),(4,1)], "10": [(1,2)]}

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

results=[]
for suite, cases in PLAN.items():
    tasks=benchmark.get_benchmark_dict()["libero_"+suite]()
    for task_id, state_index in cases:
        for mode in ["isolated", "prior_resets"]:
            np.random.seed(7)
            env, _ = get_libero_env(tasks.get_task(task_id), "openvla", resolution=256)
            env.seed(7)
            states=tasks.get_task_init_states(task_id)
            indices=[state_index] if mode=="isolated" else list(range(state_index+1))
            for index in indices:
                env.reset()
                obs=env.set_init_state(states[index])
                if index != state_index:
                    for _ in range(10):env.step([0.0]*6+[-1.0])
            folder=ROOT/f"{suite}-task-{task_id}-state-{state_index}-{mode}"
            folder.mkdir(exist_ok=False)
            Image.fromarray(obs["agentview_image"][::-1,::-1]).save(folder/"initial.png")
            model_pos=np.asarray(env.sim.model.body_pos).copy()
            model_quat=np.asarray(env.sim.model.body_quat).copy()
            np.savez_compressed(folder/"state.npz",body_pos=model_pos,body_quat=model_quat,qpos=np.asarray(env.sim.data.qpos),qvel=np.asarray(env.sim.data.qvel))
            record={"suite":suite,"task_id":task_id,"state_index":state_index,"mode":mode,
                "initial_state_sha256":digest(states[state_index]), "initial_image_sha256":digest(obs["agentview_image"]),
                "body_pos_sha256":digest(model_pos),"body_quat_sha256":digest(model_quat)}
            for _ in range(10):obs,_,_,_=env.step([0.0]*6+[-1.0])
            Image.fromarray(obs["agentview_image"][::-1,::-1]).save(folder/"after-settling.png")
            record["after_settling_image_sha256"]=digest(obs["agentview_image"])
            results.append(record)
            (ROOT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
            env.close()
print(json.dumps(results,indent=2))
