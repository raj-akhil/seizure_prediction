"""Bounded real-data pipeline verification through the live dashboard, without copying EDFs."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-sec',type=float,default=180)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 0<args.timeout_sec<=300:raise ValueError('Use a timeout between zero and 300 seconds')
    root=Path(__file__).resolve().parents[3]
    base=f'http://127.0.0.1:{args.port}'
    def request(path,body=None):
        headers={'Content-Type':'application/json'}
        if body is not None:headers['X-Dashboard-Token']=token
        req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
        with urllib.request.urlopen(req,timeout=15) as response:return json.load(response)
    token=request('/api/bootstrap')['token']
    config=yaml.safe_load((root/'src/config/real_data_check.yaml').read_text())
    started=time.monotonic()
    state=request('/api/jobs',{'config':config,'mode':'train','name':'Real-data pipeline check · four recordings'})
    job=state['id'];print(f'Job: {job}',flush=True)
    previous=None;timed_out=False
    while state['status'] in ('queued','running','cancelling'):
        if time.monotonic()-started>args.timeout_sec:
            request(f'/api/jobs/{job}/cancel',{});timed_out=True;break
        if state['current_stage']!=previous:
            previous=state['current_stage'];print(f'Stage: {previous}',flush=True)
        time.sleep(.5);state=request('/api/jobs/'+job)
    results=request(f'/api/jobs/{job}/results')
    summary={'status':'timeout_cancel_requested' if timed_out else state['status'],
        'job_id':job,'elapsed_sec':round(time.monotonic()-started,2),'config_file':'src/config/real_data_check.yaml',
        'purpose':'Small real-data execution check, not a clinical efficacy study or a 30-minute forecast validation',
        'preparation':results['preparation'],'report_partitions':list(results['reports']),'error':state.get('error')}
    output=root/'audit/runs/reports/real_data_check';output.mkdir(parents=True,exist_ok=True)
    (output/'latest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    if summary['status']!='complete':
        print(request(f'/api/jobs/{job}/log')['text'][-5000:])
        raise SystemExit(2)
    assert set(results['reports'])=={'validation','test'}


if __name__=='__main__':main()
