"""Serialized subprocess jobs with immutable configuration and isolated outputs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import uuid
from datetime import datetime, timezone
import yaml
from src.utils.config import load_config
from src.training.registry import candidates


MODES = {'preprocess':['preprocessing'], 'features':['preprocessing','features'],
         'check':['preprocessing','features','readiness'],
         'train':['preprocessing','features','readiness','training']}
SCRIPTS = {'preprocessing':'scripts.preprocess','features':'scripts.prepare_features',
           'readiness':'scripts.train','training':'scripts.train'}
PREPARATION_SECTIONS=('dataset','sensors','alignment','preprocessing','labels','execution','qualification','features')
TRAINING_SECTIONS=('training','validation')


def now():
    return datetime.now(timezone.utc).isoformat()


class Jobs:
    def __init__(self, root):
        self.root=Path(root).resolve()
        self.directory=self.root/'outputs'/'dashboard'
        self.directory.mkdir(parents=True,exist_ok=True)
        # Prepared arrays/features are intentionally outside individual dashboard jobs.
        # They are immutable snapshots that can be selected by many later training jobs.
        self.prepared_directory=self.root/'outputs'/'prepared'
        self.prepared_directory.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.process=None;self.active=None;self.tree=None
        for path in self.directory.glob('*/state.json'):
            state=json.loads(path.read_text())
            if state['status'] in ('queued','running','cancelling'):
                state.update(status='interrupted',error='Dashboard stopped before this job completed',finished=now())
                self._save(state)

    def _save(self,state):
        folder=self.directory/state['id'];folder.mkdir(exist_ok=True)
        temp=folder/'state.tmp'
        temp.write_text(json.dumps(state,indent=2),encoding='utf-8')
        temp.replace(folder/'state.json')

    @staticmethod
    def _phase_config(config, sections):
        """Keep a small, human-readable record beside each phase's artifacts."""
        return {section:copy.deepcopy(config[section]) for section in sections if section in config}

    def _write_phase_configs(self, config, folder, shared_snapshot=None):
        preparation=self._phase_config(config,PREPARATION_SECTIONS)
        (folder/'preparation_config.yaml').write_text(yaml.safe_dump(preparation,sort_keys=False),encoding='utf-8')
        if shared_snapshot:
            shared_snapshot.mkdir(parents=True,exist_ok=True)
            (shared_snapshot/'preparation_config.yaml').write_text(yaml.safe_dump(preparation,sort_keys=False),encoding='utf-8')
        training=self._phase_config(config,TRAINING_SECTIONS)
        (folder/'training_config.yaml').write_text(yaml.safe_dump(training,sort_keys=False),encoding='utf-8')

    def get(self,job):
        if len(job)!=32 or any(ch not in '0123456789abcdef' for ch in job):
            raise ValueError('Invalid job identifier')
        with self.lock:
            return json.loads((self.directory/job/'state.json').read_text())

    def list(self):
        with self.lock:
            return sorted([json.loads(p.read_text()) for p in self.directory.glob('*/state.json')],
                          key=lambda s:s['created'],reverse=True)

    def cleanup_generated(self):
        """Clear only named generated-artifact roots; source and raw data are never targets."""
        with self.lock:
            if self.active:
                raise ValueError('Cancel or wait for the active experiment before clearing generated files')
            targets=[self.root/'data'/'interim', self.root/'outputs'/'dashboard', self.prepared_directory,
                     self.root/'outputs'/'reports', self.root/'outputs'/'training',
                     self.root/'outputs'/'figures', self.root/'outputs'/'logs',
                     self.root/'outputs'/'smoke_tests', self.root/'outputs'/'real_data_check']
            removed=0;cleared=[]
            for target in targets:
                target=target.resolve()
                if not target.is_relative_to(self.root):
                    raise PermissionError('Refusing to clear a path outside this project')
                target.mkdir(parents=True,exist_ok=True)
                for child in target.iterdir():
                    if child.is_symlink() or child.is_file():
                        child.unlink()
                    else:
                        shutil.rmtree(child)
                    removed+=1
                cleared.append(str(target.relative_to(self.root)).replace('\\','/'))
            return {'removed_entries':removed,'cleared':cleared,
                    'message':'Generated artifacts and experiment history cleared. Raw EDF data was preserved.'}

    def raw_directory(self,config):
        raw=Path(config['dataset']['raw_dir'])
        raw=(self.root/raw).resolve() if not raw.is_absolute() else raw.resolve()
        if not raw.is_relative_to(self.root) or not raw.is_dir():
            raise ValueError('Raw data must be an existing directory inside this project')
        if raw==self.directory or self.directory.is_relative_to(raw):
            raise ValueError('Raw data cannot contain dashboard outputs')
        return raw

    def prepared_source(self,job):
        state=self.get(job)
        if state['status'] in ('queued','running','cancelling'):
            raise ValueError('Wait for the source experiment to stop before reusing its features')
        saved=yaml.safe_load((self.directory/job/'config.yaml').read_text())
        interim=Path(saved['dataset']['interim_dir']).resolve()
        # Older dashboard experiments stored their snapshot inside the job directory.
        # Keep them reusable, while new snapshots live in the shared prepared workspace.
        if not (interim.is_relative_to(self.prepared_directory) or interim.is_relative_to(self.directory)):
            raise ValueError('Prepared source must belong to this project')
        if not any(s['name']=='features' and s['status']=='complete' for s in state['stages']) and not state.get('prepared_job'):
            raise ValueError('The source experiment has no completed feature stage')
        if not (interim/'feature_manifest.json').is_file():
            raise ValueError('The source feature manifest is unavailable')
        # Freeze the source cohort: new downloads must not silently join an old snapshot.
        entries=json.loads((interim/'feature_manifest.json').read_text())
        saved['dataset']['recordings']=sorted(e['run'] for e in entries)
        saved['dataset']['subjects']=sorted({e['subject'] for e in entries})
        return saved

    def check_prepared_compatibility(self,config,job):
        saved=self.prepared_source(job)
        if self.raw_directory(saved)!=self.raw_directory(config):
            raise ValueError('Raw data folder changed. Start a fresh preprocessing experiment.')
        for key in ('subjects','runs','recordings'):
            if saved['dataset'].get(key)!=config['dataset'].get(key):
                raise ValueError(f'dataset.{key} changed. Start a fresh preprocessing experiment.')
        for section in ('sensors','alignment','preprocessing','labels','features'):
            if saved[section]!=config[section]:
                raise ValueError(f'{section} changed. Prepared features cannot be reused; start a fresh preprocessing experiment.')
        return saved

    def repair(self,job):
        saved=self.prepared_source(job)
        source=job;seen=set()
        while source:
            if source in seen:
                raise ValueError('Invalid prepared-source chain')
            seen.add(source);state=self.get(source)
            if any(s['name']=='training' and s['status']!='pending' for s in state['stages']):
                raise ValueError('Automatic split repair is only available before model fitting. Keep evaluated test patients fixed.')
            source=state.get('prepared_job')
        from src.training.data import load_prepared_metadata
        from src.training.readiness import repair_splits
        return repair_splits(saved,load_prepared_metadata(saved))

    def prepare(self,config,folder,prepared_job=None,prepared_directory=None):
        c=copy.deepcopy(config)
        raw=self.raw_directory(c)
        c['dataset']['raw_dir']=str(raw)
        c['dataset']['interim_dir']=(self.check_prepared_compatibility(c,prepared_job)['dataset']['interim_dir']
                                   if prepared_job else str(prepared_directory or folder/'interim'))
        c['training']['output_dir']=str(folder/'training')
        path=folder/'config.yaml'
        path.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')
        validated=load_config(path)
        candidates(validated)
        if not c['features']['enabled'] or not c['labels']['enabled']:
            raise ValueError('Dashboard experiments require features and labels enabled')
        return c

    def validate(self,config):
        import tempfile
        with tempfile.TemporaryDirectory(dir=self.directory) as tmp:
            self.prepare(config,Path(tmp))
        return {'valid':True,'message':'Configuration accepted. Data/channel and split readiness are checked when their stages run.'}

    def start(self,config,mode,name,prepared_job=None):
        if mode not in MODES:
            raise ValueError('Unknown pipeline mode')
        if prepared_job:
            if mode not in ('check','train'):
                raise ValueError('Prepared data supports readiness or training only')
            self.check_prepared_compatibility(config,prepared_job)
        if mode in ('check','train'):
            splits=config['training']['splits']
            groups=[set(splits[k]) for k in ('train','validation','test')]
            if any(not group for group in groups):
                raise ValueError('Assign train/validation/test patient lists before readiness or training. Downloading patients does not fill these lists automatically. Use Load real-data check for the small configured real-data pilot, or choose preprocessing/features only.')
            if any(groups[a]&groups[b] for a,b in [(0,1),(0,2),(1,2)]):
                raise ValueError('Training, validation and test patients must not overlap')
            if len(groups[0])<config['training']['inner_folds']:
                raise ValueError('Too few training patients for the configured grouped folds')
        with self.lock:
            if self.active:
                raise ValueError('Another experiment is running; wait or cancel it first')
            self.validate(config)
            job=uuid.uuid4().hex;folder=self.directory/job;folder.mkdir()
            shared_snapshot=self.prepared_directory/job if not prepared_job else None
            prepared=self.prepare(config,folder,prepared_job,shared_snapshot)
            self._write_phase_configs(prepared,folder,shared_snapshot)
            stages=[s for s in MODES[mode] if not prepared_job or s not in ('preprocessing','features')]
            state={'id':job,'name':str(name).strip()[:120] or 'Untitled experiment',
                'mode':mode,'status':'queued','created':now(),'finished':None,'error':None,
                'stages':[{'name':s,'status':'pending'} for s in stages],'current_stage':None,
                'prepared_job':prepared_job,
                'prepared_snapshot':str(shared_snapshot) if shared_snapshot else None}
            self._save(state);self.active=job
            threading.Thread(target=self._run,args=(job,),daemon=True).start()
            return state

    def _run(self,job):
        folder=self.directory/job
        try:
            with (folder/'pipeline.log').open('a',encoding='utf-8') as log:
                source=self.get(job).get('prepared_job')
                if source:
                    log.write(f'Using prepared feature snapshot from experiment {source}; preprocessing and feature extraction are reused.\n')
                    log.flush()
                for stage in [s['name'] for s in self.get(job)['stages']]:
                    with self.lock:
                        state=self.get(job)
                        if state['status']=='cancelling':
                            break
                        state.update(status='running',current_stage=stage)
                        for item in state['stages']:
                            if item['name']==stage:item.update(status='running',started=now())
                        self._save(state)
                        log.write(f'\n--- {stage.upper()} ---\n');log.flush()
                        args=[sys.executable,'-u','-m',SCRIPTS[stage],'--config',str(folder/'config.yaml')]
                        if stage=='training':args.append('--start-training')
                        env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'}
                        self.process=subprocess.Popen(args,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,
                            env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        if os.name=='nt':
                            from .windows_process import ProcessTree
                            try:self.tree=ProcessTree(self.process)
                            except Exception:
                                self.process.terminate();self.process.wait()
                                raise
                    code=self.process.wait()
                    with self.lock:
                        if self.tree:self.tree.close();self.tree=None
                        self.process=None;state=self.get(job)
                        cancelled=state['status']=='cancelling'
                        for item in state['stages']:
                            if item['name']==stage:item.update(status='cancelled' if cancelled else 'complete' if code==0 else 'failed',finished=now(),exit_code=code)
                        self._save(state)
                        if cancelled:break
                        if code:raise RuntimeError(f'{stage} exited with code {code}. See the live log for details.')
                with self.lock:
                    state=self.get(job)
                    state.update(status='cancelled' if state['status']=='cancelling' else 'complete',finished=now())
                    self._save(state)
        except Exception as exc:
            with self.lock:
                state=self.get(job);state.update(status='failed',error=str(exc),finished=now())
                for item in state['stages']:
                    if item['status']=='running':item['status']='failed'
                self._save(state)
        finally:
            with self.lock:self.active=None;self.process=None

    def cancel(self,job):
        with self.lock:
            state=self.get(job)
            if job!=self.active or state['status'] not in ('running','queued'):
                raise ValueError('Experiment is not running')
            state['status']='cancelling';self._save(state)
            if self.process and self.process.poll() is None:
                if os.name=='nt':
                    self.tree.terminate()
                else:self.process.terminate()
            return state

    def log(self,job):
        self.get(job)
        path=self.directory/job/'pipeline.log'
        if not path.exists():return ''
        with path.open('rb') as f:
            f.seek(max(0,path.stat().st_size-100000))
            return f.read().decode('utf-8',errors='replace')

    def artifacts(self,job):
        self.get(job);folder=self.directory/job
        paths=[folder/'config.yaml',folder/'preparation_config.yaml',folder/'training_config.yaml',folder/'pipeline.log',folder/'interim/qualification.json',folder/'training/readiness.json']
        paths+=list((folder/'training').glob('*/validation_reports/*/*'))
        paths+=list((folder/'training').glob('*/model_comparison.json'))
        paths+=list((folder/'training').glob('*/selected_features.json'))
        return [str(p.relative_to(folder)).replace('\\','/') for p in paths
                if p.is_file() and p.suffix in {'.json','.csv','.md','.png','.yaml','.log'}]

    def results(self,job):
        state=self.get(job);files=self.artifacts(job);folder=self.directory/job
        reports={}
        for file in files:
            if file.endswith('/metrics.json'):
                report=json.loads((folder/file).read_text())
                reports[report['partition']]=report
        preparation=None
        saved=yaml.safe_load((folder/'config.yaml').read_text())
        interim=Path(saved['dataset']['interim_dir'])
        manifest=interim/'feature_manifest.json'
        if manifest.exists():
            entries=json.loads(manifest.read_text())
            preparation={'runs':len(entries),'eligible_windows':sum(e['eligible'] for e in entries),
                         'positive_windows':sum(e['positive'] for e in entries)}
        qualification=interim/'qualification.json'
        readiness=folder/'training/readiness.json'
        return {'job':{'status':state['status'],'mode':state['mode'],'stages':state['stages'],
                       'error':state.get('error')},
                'reports':reports,'files':files,'preparation':preparation,
                'qualification':json.loads(qualification.read_text()) if qualification.exists() else None,
                'readiness':json.loads(readiness.read_text()) if readiness.exists() else None}
