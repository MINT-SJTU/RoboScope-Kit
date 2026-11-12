"""Compare saved historical Native initial records with the selected diagnostic."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
HISTORY=Path("/root/autodl-tmp/openvla-native-200-seed7-20261109")
rows=[]; selected=[]
for suite in ["spatial","object","goal","10"]:
    old={r['episode_id']:r for r in map(json.loads,(HISTORY/f"libero_{suite}/episodes.jsonl").read_text().splitlines())}
    for suffix in ['native','native-extra']:
        path=ROOT/f"{suite}-{suffix}/episodes.jsonl"
        if not path.exists():continue
        for r in map(json.loads,path.read_text().splitlines()):
            h=old[r['episode_id']]
            selected.append(h)
            rows.append({'suite':suite,'episode_id':r['episode_id'],'initial_state_equal':r['initial_state']==h['initial_state'],'initial_image_equal':r['initial_primary_image']==h['initial_primary_image'],'first_processor_equal':r['first_processor_inputs']==h['first_processor_inputs'],'first_action_equal':r['first_environment_action']==h['first_environment_action'],'historical_success':h['success'],'diagnostic_success':r['success'],'historical_steps':h['policy_steps'],'diagnostic_steps':r['policy_steps']})
assert len(rows)==36
(ROOT/'native-historical-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
(ROOT/'historical-native-selected.json').write_text(json.dumps({'source_root':str(HISTORY),'episodes':selected},indent=2)+'\n')
for suite in ['spatial','object','goal','10']:
    rr=[r for r in rows if r['suite']==suite]
    print(suite,{'episodes':len(rr),'different_initial_images':sum(not r['initial_image_equal'] for r in rr),'different_outcomes':sum(r['historical_success']!=r['diagnostic_success'] for r in rr),'different_outcomes_with_equal_first_input_and_action':[r['episode_id'] for r in rr if r['historical_success']!=r['diagnostic_success'] and r['first_processor_equal'] and r['first_action_equal']]})
