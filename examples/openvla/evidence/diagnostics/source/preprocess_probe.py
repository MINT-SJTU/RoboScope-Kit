"""Replay saved camera inputs through CPU/GPU preprocessing without loading weights."""
import json
import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from kit_policy import prepare_image
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"preprocess-probe"
OUT.mkdir(exist_ok=True)
results=[]
for batch,ep,step in [("spatial-native","task-0007-episode-0000",0),("spatial-native","task-0007-episode-0000",16),("spatial-native-extra","task-0005-episode-0001",14),("spatial-native-extra","task-0008-episode-0001",0)]:
    nd=ROOT/batch/"trace"/ep/f"step-{step:04d}"
    kd=ROOT/"spatial-kit/policy-trace"/ep/f"step-{step:04d}"
    nraw=np.array(Image.open(nd/"canonical_primary.png"))
    kraw=np.array(Image.open(kd/"canonical_primary.png"))
    row={"episode_id":ep,"step":step,"canonical_equal":bool(np.array_equal(nraw,kraw)),"devices":{}}
    for source,raw in [("native",nraw),("kit",kraw)]:
        expected={"native":np.array(Image.open(nd/"model_image.png")),"kit":np.array(Image.open(kd/"model_image.png"))}
        for device in ["/CPU:0","/GPU:0"]:
            with tf.device(device):
                resized,cropped=prepare_image(raw)
                actual_device=tf.constant(0.).device
            key=source+"-"+device.split(':')[0].strip('/')
            record={"actual_device":actual_device}
            for target,ref in expected.items():
                diff=np.abs(cropped.astype(int)-ref.astype(int))
                record[target]={"equal":bool(np.array_equal(cropped,ref)),"different_components":int(np.count_nonzero(diff)),"max_abs":int(diff.max())}
            row["devices"][key]=record
            Image.fromarray(cropped).save(OUT/f"{ep}-step-{step:04d}-{key}.png")
    results.append(row)
(OUT/"summary.json").write_text(json.dumps({"python":os.sys.version,"tensorflow":tf.__version__,"physical_devices":[str(d) for d in tf.config.list_physical_devices()],"cases":results},indent=2)+"\n")
print(json.dumps(results,indent=2))
