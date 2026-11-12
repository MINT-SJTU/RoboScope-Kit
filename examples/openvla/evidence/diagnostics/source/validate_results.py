"""Verify final diagnostic coverage and outcomes independently of report rendering."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
selection=json.loads((ROOT/"selection.json").read_text())
paired=json.loads((ROOT/"comparison.json").read_text())
assert len(paired)==36
checks=[]
for suite,tasks in selection.items():
    expected={f"task-{int(t):04d}-episode-{i:04d}" for t,indices in tasks.items() for i in indices}
    nr={}
    for suffix in ["native","native-extra"]:
        batch=ROOT/f"{suite}-{suffix}"
        if not (batch/"episodes.jsonl").exists():continue
        summary=json.loads((batch/"summary.json").read_text())
        assert summary["execution_valid"] and summary["complete_unique_coverage"]
        assert summary["errors"]==0 and summary["retries"]==0
        for line in (batch/"episodes.jsonl").read_text().splitlines():
            r=json.loads(line)
            assert r['episode_id'] not in nr
            nr[r['episode_id']]=r
    kr=ROOT/f"{suite}-kit"
    result=json.loads((kr/"result.json").read_text())
    assert result['result']['state']=='complete'
    assert not result['recovery']['worker_failures'] and result['recovery']['oom_events']==0
    assert result['runtime']['concurrency']==2 and result['runtime']['replicas']==2
    assert result['result']['episodes']==len(expected)
    assert set(nr)==expected==set(result['record_success'])
    rr=[r for r in paired if r['suite']==suite]
    assert {r['episode_id'] for r in rr}==expected and len(rr)==len(expected)
    for r in rr:
        ep=r['episode_id']
        assert r['native_success']==nr[ep]['success']
        assert r['native_steps']==nr[ep]['policy_steps']
        assert r['kit_success']==result['record_success'][ep]
        assert r['initial_state_arrays_equal'] and r['worker_actions_match_policy_trace']
        assert 'prompt' not in r['first_divergence'] and 'processor_input_ids' not in r['first_divergence']
    checks.append({'suite':suite,'paired_episodes':len(rr),'native_steps':sum(r['native_steps'] for r in rr),'kit_steps':sum(r['kit_steps'] for r in rr),'outcome_disagreements':sum(r['native_success']!=r['kit_success'] for r in rr),'equal_initial_state_arrays':len(rr),'worker_actions_verified':len(rr),'different_action_with_equal_saved_inputs_steps':sum(len(r['different_model_action_with_equal_saved_model_inputs']) for r in rr)})
assert json.loads((ROOT/'status.json').read_text())['state']=='complete'
(ROOT/'validation.json').write_text(json.dumps({'valid':True,'pairs':len(paired),'rollouts':2*len(paired),'checks':checks},indent=2)+'\n')
print(json.dumps(checks,indent=2))
