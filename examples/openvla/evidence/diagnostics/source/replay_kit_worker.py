"""Replay recorded float32 actions through the actual Kit Worker rollout."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from libero.libero import benchmark
from roboscope_kit.runtime.benchmarks.libero import worker

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"action-replay/kit-worker-float32"
OUT.mkdir(parents=True,exist_ok=False)
EP="task-0007-episode-0000"
trace=[json.loads(l) for l in (ROOT/"spatial-native/trace/steps.jsonl").read_text().splitlines() if json.loads(l)["episode_id"]==EP]
reference=[json.loads(l) for l in (ROOT/"action-replay/float32/steps.jsonl").read_text().splitlines()]
records=[]
class RecordedActions:
    def reset(self,context):assert context.episode_id==EP
    def end_episode(self):pass
    def step(self,obs,instruction,index):
        image=obs["agentview_image"]
        sha=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        action=np.asarray(trace[index]["environment_action"],dtype=np.float32)
        Image.fromarray(image[::-1,::-1]).save(OUT/f"step-{index:04d}.png")
        records.append({"step":index,"raw_image_sha256":sha,"action":action.tolist()})
        return action
np.random.seed(7)
runtime=worker.create_task_runtime(benchmark.get_benchmark_dict()["libero_spatial"](),7,7)
try:
    result=worker.run_episode(runtime,RecordedActions(),0,max_steps=len(trace),settle_steps=10,save_frames=False)
finally:runtime.env.close()
first=next((a["step"] for a,b in zip(records,reference) if a["raw_image_sha256"]!=b["raw_image_sha256"]),None)
summary={"case":EP,"suite":"libero_spatial","scope":"open-loop replay, no model inference",
         "success":result[0],"executed_steps":result[1],"native_reference_steps":len(reference),
         "first_image_difference_from_native_float32_replay":first,
         "all_observations_match":len(records)==len(reference) and first is None,
         "python":__import__("sys").version,"worker_source":worker.__file__}
(OUT/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
(OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
