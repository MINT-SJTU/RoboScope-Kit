import json,csv,math,statistics
from pathlib import Path
roots={'native':'openvla-native-200-seed7-20261109','serial':'openvla-kit-200-serial-seed7-20261109','fixed4':'openvla-kit-200-fixed4-seed7-20261109'}
def stats(xs):
 xs=sorted(xs)
 if not xs:return None
 def q(p):
  i=(len(xs)-1)*p;lo=int(i);hi=math.ceil(i)
  return xs[lo]+(xs[hi]-xs[lo])*(i-lo)
 return {'count':len(xs),'mean':statistics.fmean(xs),'p50':q(.5),'p95':q(.95),'p99':q(.99),'max':xs[-1]}
out={}
for arm,name in roots.items():
 root=Path('/root/autodl-tmp')/name
 all_requests=[];formal_resources=[];all_resources=[];suites={}
 for suite in ('spatial','object','goal','10'):
  run=root/('libero_'+suite)
  measured=run if arm=='native' else root/'measurements'/('libero_'+suite)
  files=[run/'requests.jsonl'] if arm=='native' else sorted(measured.rglob('requests.jsonl'))
  requests=[]
  for f in files:
   with f.open() as stream:requests += [json.loads(x) for x in stream]
  assert all(r.get('phase','formal')=='formal' for r in requests)
  with (measured/'resources.csv').open() as stream:resources=list(csv.DictReader(stream))
  phase='evaluation' if arm=='native' else 'evaluation_s'
  formal=[r for r in resources if r['phase']==phase]
  def resource_stats(rows):
   fields=('gpu_util_pct','gpu_memory_mib','process_tree_rss_bytes','cgroup_memory_bytes')
   return {f:stats([float(r[f]) for r in rows if r.get(f) not in (None,'') and math.isfinite(float(r[f]))]) for f in fields}
  suites[suite]={'request_count':len(requests),'request_errors':sum(not r['success'] for r in requests),'service_ms':stats([r['service_ms'] for r in requests if r['success']]),'first_request_ms':requests[0]['service_ms'],'formal_resources':resource_stats(formal),'whole_resources':resource_stats(resources),'resource_samples':len(resources),'resource_error_samples':sum(bool(r.get('error')) for r in resources),'formal_resource_errors':sum(bool(r.get('error')) for r in formal)}
  all_requests+=requests;formal_resources+=formal;all_resources+=resources
 out[arm]={'suites':suites,'aggregate':{'request_count':len(all_requests),'request_errors':sum(not r['success'] for r in all_requests),'latency_ms':{key:stats([r[key] for r in all_requests if r['success'] and key in r]) for key in ('service_ms','predict_action_ms','image_preprocess_ms','action_postprocess_ms')},'formal_resources':resource_stats(formal_resources),'whole_resources':resource_stats(all_resources),'resource_samples':len(all_resources),'resource_error_samples':sum(bool(r.get('error')) for r in all_resources),'formal_resource_errors':sum(bool(r.get('error')) for r in formal_resources)}}
print(json.dumps(out))
