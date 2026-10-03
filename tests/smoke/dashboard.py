"""Run the existing synthetic EDF cohort through the live dashboard API."""
import json
import argparse
from pathlib import Path
import time
import urllib.request
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eeg-only',action='store_true',help='Verify a second experiment with different input dimensions')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    base='http://127.0.0.1:8765'
    with urllib.request.urlopen(base+'/api/bootstrap') as response:
        token=json.load(response)['token']
    latest=json.loads((root/'outputs/smoke_tests/latest.json').read_text())
    config=yaml.safe_load((Path(latest['report']).parent/'synthetic_config.yaml').read_text())
    if args.eeg_only:
        config['sensors']['enabled']=['eeg']
    request=urllib.request.Request(base+'/api/jobs',
        data=json.dumps({'config':config,'mode':'train','name':'Synthetic EEG-only training test' if args.eeg_only else 'Synthetic dashboard smoke test'}).encode(),
        headers={'Content-Type':'application/json','X-Dashboard-Token':token})
    with urllib.request.urlopen(request) as response:job=json.load(response)['id']
    print(f'Job: {job}',flush=True)
    deadline=time.monotonic()+240
    while time.monotonic()<deadline:
        with urllib.request.urlopen(base+'/api/jobs/'+job) as response:state=json.load(response)
        if state['status'] not in ('queued','running','cancelling'):break
        time.sleep(.5)
    if state['status']!='complete':raise RuntimeError(state)
    with urllib.request.urlopen(base+'/api/jobs/'+job+'/results') as response:result=json.load(response)
    assert set(result['reports'])=={'validation','test'}
    assert result['reports']['test']['window']['windows']==(65 if args.eeg_only else 47)
    figure=next(f for f in result['files'] if f.endswith('/test/curves.png'))
    with urllib.request.urlopen(base+'/api/jobs/'+job+'/files/'+figure) as response:
        assert response.read(8)==b'\x89PNG\r\n\x1a\n'
    summary={'status':'passed','job_id':job,'stages':state['stages'],
        'modalities':config['sensors']['enabled'],
        'report_partitions':list(result['reports']),'plot_served':True,
        'purpose':'Synthetic engineering test; not real seizure-prediction efficacy'}
    target=root/'outputs/dashboard'/job/'DASHBOARD_SMOKE.json'
    target.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
