import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
import yaml
from src.dashboard.server import Server


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'data'/'raw').mkdir(parents=True)
        (self.root / 'src' / 'config').mkdir(parents=True)
        self.config=yaml.safe_load(Path('src/config/config.yaml').read_text())
        self.config['training']['splits']={'train':['a','b','c'],'validation':['d'],'test':['e']}
        (self.root / 'src' / 'config'/'config.yaml').write_text(yaml.safe_dump(self.config))
        (self.root / 'src' / 'commands').mkdir()
        (self.root/'src/commands/__init__.py').write_text('')
        for script in ['preprocess.py','prepare_features.py','train.py']:
            (self.root / 'src' / 'commands'/script).write_text("print('stage executed', flush=True)\n")
        self.server=Server(self.root,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        if self.server.jobs.active:
            try:self.server.jobs.cancel(self.server.jobs.active)
            except ValueError:pass
            deadline=time.time()+10
            while self.server.jobs.active and time.time()<deadline:time.sleep(.05)
        self.server.shutdown();self.server.server_close();self.temp.cleanup()

    def request(self,path,body=None,headers=None):
        h={'Content-Type':'application/json','X-Dashboard-Token':self.server.token,**(headers or {})}
        req=urllib.request.Request(self.url+path,data=json.dumps(body).encode() if body is not None else None,headers=h)
        with urllib.request.urlopen(req,timeout=10) as r:return json.loads(r.read())

    def wait(self,job):
        deadline=time.time()+15
        while time.time()<deadline:
            result=self.request('/api/jobs/'+job)
            if result['status'] not in ['queued','running','cancelling']:return result
            time.sleep(.05)
        self.fail('Job did not finish')

    def test_job_stages_and_immutable_configuration(self):
        state=self.request('/api/jobs',{'config':self.config,'name':'test','mode':'train'})
        self.assertEqual(self.wait(state['id'])['status'],'complete')
        saved=self.request('/api/jobs/'+state['id']+'/config')
        self.assertIn(state['id'],saved['dataset']['interim_dir'])
        self.assertIn('audit/runs/prepared',saved['dataset']['interim_dir'].replace('\\','/'))
        preparation=yaml.safe_load((self.server.jobs.directory/state['id']/'preparation_config.yaml').read_text())
        training=yaml.safe_load((self.server.jobs.directory/state['id']/'training_config.yaml').read_text())
        self.assertIn('features',preparation)
        self.assertNotIn('training',preparation)
        self.assertEqual(set(training),{'training','validation'})
        self.assertEqual(yaml.safe_load((self.root/'src/config/config.yaml').read_text()),self.config)
        self.assertEqual(self.request('/api/jobs/'+state['id']+'/log')['text'].count('stage executed'),4)
        folder=self.server.jobs.directory/state['id']/'training'/'example'
        folder.mkdir(parents=True)
        (folder/'model_comparison.json').write_text('{"ok":true}')
        self.assertEqual(self.request('/api/jobs/'+state['id']+'/files/training/example/model_comparison.json'),{'ok':True})

    def test_failed_stage_stops_pipeline(self):
        (self.root/'src/commands/preprocess.py').write_text('raise SystemExit(7)')
        state=self.request('/api/jobs',{'config':self.config,'mode':'train'})
        done=self.wait(state['id'])
        self.assertEqual(done['status'],'failed')
        self.assertEqual(done['stages'][1]['status'],'pending')

    def test_cancel_and_concurrent_job_guard(self):
        (self.root/'src/commands/preprocess.py').write_text('import time\ntime.sleep(30)')
        state=self.request('/api/jobs',{'config':self.config,'mode':'preprocess'})
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/jobs',{'config':self.config,'mode':'preprocess'})
        self.request('/api/jobs/'+state['id']+'/cancel',{})
        self.assertEqual(self.wait(state['id'])['status'],'cancelled')

    def test_origin_token_and_paths(self):
        for headers in [{'X-Dashboard-Token':'wrong'},{'Origin':'https://example.com'},{'Host':'example.com'}]:
            with self.assertRaises(urllib.error.HTTPError) as exc:
                self.request('/api/config/validate',{'config':self.config},headers)
            self.assertEqual(exc.exception.code,403)
        cfg=copy.deepcopy(self.config);cfg['dataset']['raw_dir']=str(self.root.parent)
        with self.assertRaises(urllib.error.HTTPError):self.request('/api/config/validate',{'config':cfg})
        with self.assertRaises(urllib.error.HTTPError):self.request('/api/jobs/../../config')

    def test_bootstrap_and_validation(self):
        self.assertEqual(self.request('/api/bootstrap')['config'],self.config)
        self.assertTrue(self.request('/api/config/validate',{'config':self.config})['valid'])
        cfg=copy.deepcopy(self.config);cfg['sensors']['enabled']=['invalid']
        with self.assertRaises(urllib.error.HTTPError):self.request('/api/config/validate',{'config':cfg})

    def test_missing_partitions_block_before_processing(self):
        cfg=copy.deepcopy(self.config);cfg['training']['splits']['train']=[]
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/jobs',{'config':cfg,'mode':'train'})
        self.assertEqual(self.server.jobs.list(),[])

    def test_results_identify_the_job_status_when_no_model_report_exists(self):
        state=self.request('/api/jobs',{'config':self.config,'mode':'preprocess'})
        self.assertEqual(self.wait(state['id'])['status'],'complete')
        result=self.request('/api/jobs/'+state['id']+'/results')
        self.assertEqual(result['job']['status'],'complete')
        self.assertEqual(result['job']['mode'],'preprocess')
        self.assertEqual(result['job']['stages'][0]['name'],'preprocessing')
        self.assertEqual(result['reports'],{})

    def test_results_expose_saved_training_report_metrics(self):
        state=self.request('/api/jobs',{'config':self.config,'mode':'preprocess'})
        self.assertEqual(self.wait(state['id'])['status'],'complete')
        report=self.server.jobs.directory/state['id']/'training'/'example'/'validation_reports'/'test'
        report.mkdir(parents=True)
        (report/'metrics.json').write_text(json.dumps({'partition':'test','window':{'precision':.8,'f1':.75},
                                                       'alarm':{'event_sensitivity':.5}}))
        result=self.request('/api/jobs/'+state['id']+'/results')
        self.assertEqual(result['reports']['test']['window']['f1'],.75)
        self.assertEqual(result['reports']['test']['alarm']['event_sensitivity'],.5)

    def test_cleanup_preserves_raw_and_project_files(self):
        raw=self.root/'data'/'raw'/'sub-keep';raw.mkdir()
        (raw/'recording.edf').write_text('raw')
        (self.root / 'audit' / 'runs'/'README.md').write_text('keep')
        for relative in ('audit/runs/cli_prepared/cache.bin','audit/runs/dashboard/old/pipeline.log',
                         'audit/runs/prepared/snapshot/features.pkl','audit/runs/reports/old.json',
                         'audit/runs/training/model.bin'):
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('generated')
        response=self.request('/api/cleanup',{})
        self.assertGreaterEqual(response['removed_entries'],4)
        self.assertEqual((raw/'recording.edf').read_text(),'raw')
        self.assertEqual((self.root / 'audit' / 'runs'/'README.md').read_text(),'keep')
        self.assertEqual(self.server.jobs.list(),[])
        self.assertTrue((self.root/'audit'/'runs'/'cli_prepared').is_dir())
        self.assertEqual(list((self.root/'audit/runs/prepared').iterdir()),[])

    def test_patient_api_uses_configured_folder_and_refreshes_without_launching(self):
        cfg=copy.deepcopy(self.config)
        cfg['training']['splits']={'train':[],'validation':[],'test':[]}
        raw=self.root/'alternate-data';raw.mkdir()
        cfg['dataset']['raw_dir']='alternate-data'
        (raw/'sub-501').mkdir()
        catalog=self.request('/api/patients',{'config':cfg})
        self.assertEqual([row['id'] for row in catalog['patients']],['sub-501'])
        self.assertFalse(catalog['timing_checked'])
        (raw/'sub-502').mkdir()
        self.assertEqual(len(self.request('/api/patients',{'config':cfg})['patients']),2)
        detailed=self.request('/api/patients/details',{'config':cfg})
        self.assertTrue(detailed['timing_checked'])
        self.assertIsNone(self.request('/api/patients/suggest',{'config':cfg})['suggestion']['splits'])
        self.assertEqual(self.server.jobs.list(),[])
        cfg['dataset']['raw_dir']=str(self.root.parent)
        with self.assertRaises(urllib.error.HTTPError):self.request('/api/patients',{'config':cfg})

    def prepared_fixture(self):
        source=self.request('/api/jobs',{'config':self.config,'mode':'features'})
        self.assertEqual(self.wait(source['id'])['status'],'complete')
        saved=self.request('/api/jobs/'+source['id']+'/config')
        interim=Path(saved['dataset']['interim_dir'])
        interim.mkdir(exist_ok=True)
        (interim/'feature_manifest.json').write_text(json.dumps([{'run':'sub-a_ses-01_task-test_run-01','subject':'sub-a'}]))
        return source['id'],self.request('/api/jobs/'+source['id']+'/prepared-config')

    def test_reuses_only_prepared_stages_with_isolated_outputs_and_fixed_snapshot(self):
        source,cfg=self.prepared_fixture()
        original=(self.server.jobs.directory/source/'config.yaml').read_bytes()
        self.assertEqual(cfg['dataset']['subjects'],['sub-a'])
        self.assertEqual(cfg['dataset']['recordings'],['sub-a_ses-01_task-test_run-01'])
        child=self.request('/api/jobs',{'config':cfg,'mode':'check','prepared_job':source})
        done=self.wait(child['id'])
        self.assertEqual(done['status'],'complete')
        self.assertEqual([stage['name'] for stage in done['stages']],['readiness'])
        saved=self.request('/api/jobs/'+child['id']+'/config')
        self.assertEqual(saved['dataset']['interim_dir'],cfg['dataset']['interim_dir'])
        self.assertIn(child['id'],saved['training']['output_dir'])
        self.assertEqual((self.server.jobs.directory/source/'config.yaml').read_bytes(),original)
        self.assertEqual(self.request('/api/jobs/'+child['id']+'/log')['text'].count('stage executed'),1)

    def test_prepared_reuse_rejects_changed_labels_and_incomplete_source(self):
        source,cfg=self.prepared_fixture()
        changed=copy.deepcopy(cfg);changed['labels']['minimum_warning_min']+=1
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/jobs',{'config':changed,'mode':'check','prepared_job':source})
        changed=copy.deepcopy(cfg);changed['dataset']['recordings']=None
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/jobs',{'config':changed,'mode':'check','prepared_job':source})
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/jobs',{'config':cfg,'mode':'preprocess','prepared_job':source})
        state=self.server.jobs.get(source);state['stages'][1]['status']='failed';self.server.jobs._save(state)
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/jobs/'+source+'/prepared-config')

    def test_repair_rejects_source_with_prior_training(self):
        source,cfg=self.prepared_fixture()
        state=self.server.jobs.get(source);state['stages'].append({'name':'training','status':'failed'})
        self.server.jobs._save(state)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request('/api/jobs/'+source+'/repair-split',{})
        self.assertIn('before model fitting',error.exception.read().decode())


if __name__=='__main__':unittest.main()
