"""Check CPU preprocessing at observed same-camera divergence milestones."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
tf.config.set_visible_devices([],"GPU")
from kit_policy import prepare_image
ROOT=Path(__file__).resolve().parent.parent
report=[]
for row in json.loads((ROOT/'comparison.json').read_text()):
    milestones={row['first_divergence'][key] for key in ['model_image','model_action_gt_1e-6'] if key in row['first_divergence']}
    for step in sorted(milestones):
        nd=ROOT/row['native_source_batch']/'trace'/row['episode_id']/f"step-{step:04d}"
        kd=ROOT/f"{row['suite']}-kit"/'policy-trace'/row['episode_id']/f"step-{step:04d}"
        raw=np.array(Image.open(nd/'canonical_primary.png'))
        if not np.array_equal(raw,np.array(Image.open(kd/'canonical_primary.png'))):continue
        native=np.array(Image.open(nd/'model_image.png'))
        kit=np.array(Image.open(kd/'model_image.png'))
        if np.array_equal(native,kit):continue
        with tf.device('/CPU:0'):resized,model=prepare_image(raw)
        delta=np.abs(native.astype(int)-kit.astype(int))
        report.append({'suite':row['suite'],'episode_id':row['episode_id'],'step':step,'same_canonical_camera':True,'cpu_model_image_matches_native':bool(np.array_equal(model,native)),'cpu_resized_image_matches_native':bool(np.array_equal(resized,np.array(Image.open(nd/'resized_primary.png')))),'different_color_components':int(np.count_nonzero(delta)),'max_abs_uint8':int(delta.max())})
output={'scope':'Only first cropped-image / first differing-model-action milestones with equal canonical camera arrays and unequal cropped images. No model inference or rollout intervention.','paired_records_available':len(json.loads((ROOT/'comparison.json').read_text())),'cases':report,'all_cpu_outputs_match_native':all(r['cpu_model_image_matches_native'] and r['cpu_resized_image_matches_native'] for r in report)}
(ROOT/'preprocess-probe/coverage.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
