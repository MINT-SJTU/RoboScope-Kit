"""Replay an observed Native action sequence with and without float32 conversion."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env

ROOT=Path(__file__).resolve().parent.parent
EP="task-0007-episode-0000"
trace=[json.loads(l) for l in (ROOT/"spatial-native/trace/steps.jsonl").read_text().splitlines() if json.loads(l)["episode_id"]==EP]
results=[]
for mode in ["native_precision","float32"]:
    out=ROOT/"action-replay"/mode;out.mkdir(parents=True,exist_ok=False)
    np.random.seed(7)
    tasks=benchmark.get_benchmark_dict()["libero_spatial"]()
    env,_=get_libero_env(tasks.get_task(7),"openvla",resolution=256);env.seed(7)
    env.reset();obs=env.set_init_state(tasks.get_task_init_states(7)[0])
    for _ in range(10):obs,_,_,_=env.step([0.0]*6+[-1.0])
    first=None;records=[]
    for row in trace:
        image=np.asarray(obs["agentview_image"])
        sha=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        if sha!=row["images"]["raw_primary"]["sha256"] and first is None:first=row["step"]
        Image.fromarray(image[::-1,::-1]).save(out/f"step-{row['step']:04d}.png")
        action=np.asarray(row["environment_action"],dtype=np.float32 if mode=="float32" else np.float64)
        obs,reward,done,info=env.step(action.tolist() if mode=="native_precision" else action)
        records.append({"step":row["step"],"raw_image_sha256":sha,"done":bool(done),"action":action.tolist(),"action_dtype":str(action.dtype)})
        if done:break
    env.close()
    result={"mode":mode,"first_image_difference_from_native_trace":first,"executed_steps":len(records),"success":bool(done),
            "scope":"open-loop replay of recorded actions; no model or feedback recomputation"}
    results.append(result)
    (out/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
(ROOT/"action-replay/summary.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps(results,indent=2))
