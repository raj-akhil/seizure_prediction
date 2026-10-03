"""Loopback-only JSON API and static UI; no external service dependencies."""
import argparse
import json
import mimetypes
from pathlib import Path
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
import yaml
from .jobs import Jobs


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,root,port=8765):
        # Bind first: a second launch on the same port must not touch active job state.
        super().__init__(('127.0.0.1',port),Handler)
        self.jobs=Jobs(root);self.token=secrets.token_urlsafe(32)
        self.static=Path(__file__).parent/'static'


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass

    def send(self,value,status=200,kind='application/json'):
        body=json.dumps(value,allow_nan=False).encode() if kind=='application/json' and not isinstance(value,bytes) else value
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)

    def guard(self,mutating=False):
        port=self.server.server_port
        allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host') not in allowed:
            raise PermissionError('Only local dashboard requests are accepted')
        origin=self.headers.get('Origin')
        if origin and origin not in {'http://'+h for h in allowed}:
            raise PermissionError('Cross-origin requests are blocked')
        if mutating and self.headers.get('X-Dashboard-Token')!=self.server.token:
            raise PermissionError('Refresh the dashboard before submitting changes')

    def do_GET(self):
        try:
            self.guard();path=unquote(urlparse(self.path).path)
            if path=='/api/bootstrap':
                config=yaml.safe_load((self.server.jobs.root/'config'/'config.yaml').read_text(encoding='utf-8-sig'))
                return self.send({'config':config,'token':self.server.token,
                                  'subjects':sorted(p.name for p in (self.server.jobs.root/'data'/'raw').glob('sub-*') if p.is_dir())})
            if path=='/api/jobs':return self.send(self.server.jobs.list())
            if path=='/api/real-check':
                path=self.server.jobs.root/'config/real_data_check.yaml'
                config=yaml.safe_load(path.read_text())
                self.server.jobs.validate(config)
                from src.data.bids_loader import BIDSLoader, require_complete_recordings
                runs=BIDSLoader(self.server.jobs.root/config['dataset']['raw_dir']).selected_runs(config['dataset'])
                require_complete_recordings(runs,config)
                return self.send({'config':config})
            if path=='/api/demo':
                root=self.server.jobs.root
                pointer=root/'outputs/smoke_tests/latest.json'
                if not pointer.exists():raise ValueError('Quick-test fixtures are unavailable; run python -m tests.smoke.training once to create them')
                report=Path(json.loads(pointer.read_text())['report']).resolve()
                if not report.is_relative_to(root/'outputs/smoke_tests'):raise PermissionError('Invalid demo fixture path')
                config=yaml.safe_load((report.parent/'synthetic_config.yaml').read_text())
                self.server.jobs.validate(config)
                return self.send({'config':config})
            parts=path.strip('/').split('/')
            if len(parts)>=3 and parts[:2]==['api','jobs']:
                job=parts[2]
                if len(parts)==3:return self.send(self.server.jobs.get(job))
                if parts[3]=='log':return self.send({'text':self.server.jobs.log(job)})
                if parts[3]=='results':return self.send(self.server.jobs.results(job))
                if parts[3]=='prepared-config':return self.send(self.server.jobs.prepared_source(job))
                if parts[3]=='config':
                    self.server.jobs.get(job)
                    return self.send(yaml.safe_load((self.server.jobs.directory/job/'config.yaml').read_text()))
                if parts[3]=='files':
                    relative='/'.join(parts[4:])
                    if relative not in self.server.jobs.artifacts(job):raise FileNotFoundError('Artifact not available')
                    file=(self.server.jobs.directory/job/relative).resolve()
                    if not file.is_relative_to(self.server.jobs.directory/job):raise PermissionError('Invalid file path')
                    return self.send(file.read_bytes(),kind=mimetypes.guess_type(file)[0] or 'text/plain')
            names={'/':'index.html','/app.js':'app.js','/patients.js':'patients.js','/style.css':'style.css'}
            if path not in names:raise FileNotFoundError('Unknown route')
            file=self.server.static/names[path]
            self.send(file.read_bytes(),kind={'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css'}[file.suffix])
        except Exception as exc:self.error(exc)

    def do_POST(self):
        try:
            if self.headers.get('Content-Type')!='application/json':raise ValueError('JSON request required')
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1048576:raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(length));self.guard(True);path=urlparse(self.path).path
            if path=='/api/config/validate':return self.send(self.server.jobs.validate(body['config']))
            if path=='/api/cleanup':return self.send(self.server.jobs.cleanup_generated())
            if path in ('/api/patients','/api/patients/details','/api/patients/suggest'):
                from .patients import patient_catalog, suggest_split
                config=body['config']
                self.server.jobs.validate(config)
                # Listing a large cohort should be responsive.  Annotation and
                # timing inspection is deliberately deferred until requested.
                catalog=patient_catalog(self.server.jobs.raw_directory(config),config,
                                        include_timing=path!='/api/patients')
                if path.endswith('/suggest'):
                    catalog['suggestion']=suggest_split(catalog,config)
                return self.send(catalog)
            if path=='/api/jobs':return self.send(self.server.jobs.start(body['config'],body['mode'],body.get('name',''),body.get('prepared_job')),202)
            parts=path.strip('/').split('/')
            if len(parts)==4 and parts[:2]==['api','jobs'] and parts[3]=='repair-split':
                return self.send(self.server.jobs.repair(parts[2]))
            if len(parts)==4 and parts[:2]==['api','jobs'] and parts[3]=='cancel':
                return self.send(self.server.jobs.cancel(parts[2]))
            raise FileNotFoundError('Unknown route')
        except Exception as exc:self.error(exc)

    def error(self,exc):
        self.send({'error':str(exc)},403 if isinstance(exc,PermissionError) else 404 if isinstance(exc,FileNotFoundError) else 400)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args();server=Server(Path(__file__).resolve().parents[2],args.port)
    print(f'Dashboard: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if server.jobs.active:server.jobs.cancel(server.jobs.active)
        server.server_close()
