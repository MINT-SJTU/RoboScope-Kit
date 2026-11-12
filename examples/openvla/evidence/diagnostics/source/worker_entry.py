"""Observe the installed Kit Worker without replacing rollout decisions."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from roboscope_kit.runtime.benchmarks.libero import worker

original = worker.run_episode

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class Observed:
    def __init__(self, env, folder, settling, reset_count):
        self.wrapped, self.folder, self.settling = env, folder, settling
        self.count, self.reset_count, self.obs = 0, reset_count, None
    def __getattr__(self, name):return getattr(self.wrapped, name)
    def snapshot(self, name):
        sim=self.wrapped.sim
        arrays={"body_pos":np.asarray(sim.model.body_pos).copy(),"body_quat":np.asarray(sim.model.body_quat).copy()}
        for key in ["qpos","qvel","ctrl","qacc_warmstart","act"]:
            if hasattr(sim.data,key):arrays[key]=np.asarray(getattr(sim.data,key)).copy()
        np.savez_compressed(self.folder/(name+".npz"),**arrays)
        Image.fromarray(self.obs["agentview_image"][::-1,::-1]).save(self.folder/(name+".png"))
    def set_init_state(self, state):
        self.obs=self.wrapped.set_init_state(state)
        self.snapshot("initial")
        (self.folder/"initial.json").write_text(json.dumps({"initial_state_sha256":digest(state),"initial_image_sha256":digest(self.obs["agentview_image"]),"reset_count_in_task_runtime":self.reset_count},indent=2)+"\n")
        return self.obs
    def step(self, action):
        before=digest(self.obs["agentview_image"])
        result=self.wrapped.step(action)
        self.obs=result[0]
        row={"simulation_step":self.count,"policy_step":self.count-self.settling,"action":np.asarray(action).tolist(),"dtype":str(np.asarray(action).dtype),"before_image_sha256":before,"after_image_sha256":digest(self.obs["agentview_image"]),"reward":float(result[1]),"task_done":bool(result[2]),"env_done":bool(self.wrapped.env.done)}
        with (self.folder/"steps.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        self.count+=1
        if self.count==self.settling:self.snapshot("after-settling")
        return result

def observed(runtime, policy, episode_index, **kwargs):
    folder=Path(os.environ["DIAGNOSTIC_WORKER_TRACE"])/f"task-{runtime.task_id:04d}-episode-{episode_index:04d}"
    folder.mkdir(parents=True,exist_ok=False)
    env=runtime.env
    reset_count=getattr(env,"_diagnostic_reset_count",0)+1
    env._diagnostic_reset_count=reset_count
    proxy=Observed(env,folder,kwargs["settle_steps"],reset_count)
    runtime.env=proxy
    try:
        result=original(runtime,policy,episode_index,**kwargs)
        proxy.snapshot("final")
        (folder/"result.json").write_text(json.dumps({"success":result[0],"steps":result[1],"elapsed_s":result[2]})+"\n")
        return result
    finally:runtime.env=env

worker.run_episode=observed
worker.main()
