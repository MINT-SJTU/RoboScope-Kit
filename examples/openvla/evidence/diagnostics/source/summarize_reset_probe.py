import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parent.parent
probe=ROOT/"reset-probe"
records=json.loads((probe/"results.json").read_text())
summary=[]
for a in records:
    if a["mode"]!="isolated":continue
    b=next(r for r in records if r["suite"]==a["suite"] and r["task_id"]==a["task_id"] and r["state_index"]==a["state_index"] and r["mode"]=="prior_resets")
    stem=f"{a['suite']}-task-{a['task_id']}-state-{a['state_index']}"
    da,db=probe/(stem+"-isolated"),probe/(stem+"-prior_resets")
    with np.load(da/"state.npz") as na,np.load(db/"state.npz") as nb:
        delta=np.linalg.norm(na["body_pos"]-nb["body_pos"],axis=1)
        positions={str(i):float(d) for i,d in enumerate(delta) if d>0}
        qpos_equal=np.array_equal(na["qpos"],nb["qpos"])
    ia=np.asarray(Image.open(da/"initial.png"));ib=np.asarray(Image.open(db/"initial.png"))
    changed=np.any(ia!=ib,axis=2)
    historical=Path("/root/autodl-tmp/openvla-native-200-seed7-20261109")/("libero_"+a["suite"])/"episodes.jsonl"
    old=next(r for r in map(json.loads,historical.read_text().splitlines()) if r["task_id"]==a["task_id"] and r["initial_state_index"]==a["state_index"])
    row={"case":stem,"qpos_equal":bool(qpos_equal),"changed_model_body_positions_m":positions,
         "initial_image_changed_pixels":int(changed.sum()),"initial_image_total_pixels":int(changed.size),
         "isolated_matches_historical":a["initial_image_sha256"]==old["initial_primary_image"]["sha256"],
         "prior_resets_matches_historical":b["initial_image_sha256"]==old["initial_primary_image"]["sha256"]}
    summary.append(row)
    canvas=Image.new("RGB",(768,286),"white");draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate([(Image.fromarray(ia),"Isolated reset"),(Image.fromarray(ib),"Prior resets"),(Image.fromarray((changed*255).astype("uint8")).convert("RGB"),"Changed pixels (white)")]):
        canvas.paste(im,(256*i,30));draw.text((256*i+6,8),label,fill="black")
    canvas.save(probe/(stem+"-comparison.png"))
(probe/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
