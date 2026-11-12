import json, pathlib, subprocess, time, os
root=pathlib.Path('/root/autodl-tmp')
p=root/'openvla-kit-200-autotune-seed7-20261109'
previous=root/'openvla-kit-200-fixed4-seed7-20261109'
def state(stage, **kw):
 f=p/'followup-status.json'; tmp=f.with_suffix('.tmp'); tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated_unix_s=time.time(),**kw),indent=2)+'\n');tmp.replace(f)
try:
 state('waiting-fixed4')
 while True:
  s=json.loads((previous/'status.json').read_text())
  if s['stage']=='failed': raise RuntimeError('Fixed4 failed: '+str(s.get('error')))
  if s['stage']=='complete': break
  time.sleep(30)
 state('waiting-idle-gpu')
 while subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(): time.sleep(10)
 if (p/'status.json').exists(): raise RuntimeError('Autotune was already started')
 state('starting-autotune')
 with (p/'launcher.log').open('x') as log:
  child=subprocess.Popen(['/root/autodl-tmp/openvla-kit-20261109/runtime/env/bin/python','-u',str(p/'launch.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=p)
  state('running-autotune',child_pid=child.pid)
  rc=child.wait()
 if rc: raise RuntimeError('Autotune launcher exit '+str(rc))
 s=json.loads((p/'status.json').read_text())
 if s['stage']!='complete': raise RuntimeError('Autotune not complete')
 state('complete')
except BaseException as exc:
 state('failed',error=repr(exc)); raise
