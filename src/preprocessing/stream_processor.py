"""Bounded-memory, timestamp-based EDF window preparation. No learned transforms."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import importlib.metadata
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, resample_poly
from .labels import read_events, label_windows

VERSION = '1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def integer_samples(seconds, rate):
    n = seconds * rate
    if abs(n-round(n)) > 1e-6:
        raise ValueError(f'{seconds}s does not fall on a {rate}Hz sample grid')
    return int(round(n))


class StreamProcessor:
    def __init__(self, config):
        self.config = config
        self.out = Path(config['dataset']['interim_dir'])
        self.out.mkdir(parents=True, exist_ok=True)

    def process_run(self, run):
        c, p = self.config, self.config['preprocessing']
        raws, inventory, fingerprints = {}, {}, {}
        for mod in c['sensors']['enabled']:
            path = run['files'].get(mod)
            if path is None:
                if c['sensors']['missing_policy'] == 'skip_run':
                    return {'run':run['id'], 'status':'skipped', 'reason':f'Missing {mod}'}
                raise ValueError(f'Missing {mod}')
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            chosen = c['sensors']['channels'].get(mod)
            if chosen is not None:
                if not chosen or len(set(chosen)) != len(chosen) or set(chosen)-set(raw.ch_names):
                    if c['sensors']['missing_policy'] == 'skip_run':
                        return {'run':run['id'], 'status':'skipped', 'reason':f'Unavailable channels: {chosen}'}
                    raise ValueError(f'{mod}: requested {chosen}; available {raw.ch_names}')
                raw.pick(chosen)
            if len(raw.annotations):
                raise ValueError('EDF annotations require review for gaps/artifacts before processing')
            rate = raw.info['sfreq']
            duration = raw.n_times / rate
            side = path.with_suffix('.json')
            metadata = json.loads(side.read_text()) if side.exists() else {}
            if abs(metadata.get('RecordingDuration',duration)-duration) > c['alignment']['duration_tolerance_sec']:
                raise ValueError(f'{mod}: EDF/JSON duration mismatch')
            if abs(metadata.get('SamplingFrequency',rate)-rate) > 1e-6:
                raise ValueError(f'{mod}: EDF/JSON sampling rate mismatch')
            inventory[mod] = {'file':str(path), 'channels':raw.ch_names,
                'units':{ch:raw._orig_units.get(ch,'unknown') for ch in raw.ch_names},
                'array_units':'MNE calibrated values: voltage channels in volts; other units follow EDF calibration',
                'native_rate':rate, 'samples':raw.n_times, 'duration_sec':duration,
                'header_start':str(raw.info['meas_date']), 'sidecar':metadata}
            fingerprints[mod] = {'path':str(path.resolve()), 'bytes':path.stat().st_size,
                'mtime_ns':path.stat().st_mtime_ns, 'metadata':metadata}
            raws[mod] = raw
        # Event origin remains the EEG recording origin, even in ECG-only experiments.
        eeg_path = run['files'].get('eeg')
        if eeg_path is None:
            raise ValueError('EEG header needed as event timeline reference (signal need not be selected)')
        reference = mne.io.read_raw_edf(eeg_path, preload=False, verbose=False)
        annotation_duration = reference.n_times/reference.info['sfreq']
        offsets = {}
        for mod, raw in raws.items():
            overrides = c['alignment'].get('run_offsets_sec',{}).get(run['id'],{})
            offset = float(overrides.get(mod,c['alignment']['offsets_sec'].get(mod,0)))
            if not np.isfinite(offset):
                raise ValueError('Offsets must be finite')
            if c['alignment']['mode'] == 'header':
                if raw.info['meas_date'] is None or reference.info['meas_date'] is None:
                    raise ValueError('Missing header timestamp')
                offset += (raw.info['meas_date']-reference.info['meas_date']).total_seconds()
            offsets[mod] = offset
            inventory[mod]['offset_sec'] = offset
        start = max(0, *offsets.values())
        end = min(annotation_duration, *(offsets[m]+r.n_times/r.info['sfreq'] for m,r in raws.items()))
        length = p['window_sec']+p['context_sec']
        # Full causal filter warm-up is required; it is not an input to the model.
        first_end = start+p['filter_warmup_sec']+length
        count = max(0, int(np.floor((end-first_end)/p['stride_sec']+1e-9))+1)
        if not count:
            raise ValueError('No full windows in the common time interval; check offsets and dates')
        ends = first_end + np.arange(count)*p['stride_sec']
        starts = ends-length
        code_hash = digest({str(f):f.read_text() for f in Path(__file__).parent.glob('*.py')})
        versions = {name:importlib.metadata.version(name) for name in ('mne','numpy','scipy','pandas')}
        signal_key = digest({'version':VERSION, 'code':code_hash, 'versions':versions, 'sources':fingerprints,
            'preprocessing':p,'sensors':c['sensors'],'alignment':c['alignment'],'interval':[start,end]})
        cache = self.out/'signals'/signal_key/run['id']
        cache.mkdir(parents=True, exist_ok=True)
        marker = cache/'complete.json'
        cached = marker.exists()
        if cached:
            try:
                saved = json.loads(marker.read_text())
                for mod in raws:
                    arr = np.load(cache/f'{mod}.npy',mmap_mode='r')
                    if list(arr.shape) != saved['inventory'][mod]['shape'] or arr.dtype != np.float32:
                        raise ValueError('Invalid cached signal shape/dtype')
                    del arr
                if np.load(cache/'quality_valid.npy').shape != (count,):
                    raise ValueError('Invalid cached quality mask')
            except (OSError,ValueError,KeyError):
                marker.unlink(missing_ok=True)
                cached = False
        if not cached:
            valid = np.ones(count,dtype=bool)
            for mod, raw in raws.items():
                fs = raw.info['sfreq']
                target = p['output_sampling_rate'] or fs
                n = integer_samples(length, target)
                cfg = p['filters'][mod]
                lo, hi = cfg['low_hz'], cfg['high_hz']
                sos = None
                if lo is not None or hi is not None:
                    cutoff = [lo,hi] if lo is not None and hi is not None else (lo if lo is not None else hi)
                    kind = 'bandpass' if lo is not None and hi is not None else ('highpass' if lo is not None else 'lowpass')
                    sos = butter(cfg['order'],cutoff,btype=kind,fs=fs,output='sos')
                ratio = Fraction(target/fs).limit_denominator(10000)
                array = np.lib.format.open_memmap(cache/f'{mod}.npy',mode='w+',dtype='float32',shape=(count,len(raw.ch_names),n))
                for i,(a,b) in enumerate(zip(starts,ends)):
                    left = integer_samples(a-offsets[mod]-p['filter_warmup_sec'],fs)
                    right = integer_samples(b-offsets[mod],fs)
                    data = raw.get_data(start=left,stop=right)
                    warm = integer_samples(p['filter_warmup_sec'],fs)
                    clean = np.isfinite(data).all()
                    if mod in p['quality']['reject_flat_modalities']:
                        clean = clean and bool((np.ptp(data[:,warm:],axis=1)>p['quality']['flat_peak_to_peak']).all())
                    valid[i] &= clean
                    if clean:
                        if sos is not None:
                            data = sosfilt(sos,data,axis=1)
                        data = data[:,warm:]
                        if target != fs:
                            # Resample only observed input, never samples after forecast time.
                            data = resample_poly(data,ratio.numerator,ratio.denominator,axis=1)
                        if data.shape[1] != n:
                            raise ValueError('Resampling shape mismatch')
                        array[i] = data.astype('float32')
                    else:
                        array[i] = np.nan
                array.flush()
                del array
                inventory[mod]['output_rate'] = target
                inventory[mod]['shape'] = [count,len(raw.ch_names),n]
            np.save(cache/'quality_valid.npy',valid)
            save_json(marker, {'inventory':inventory, 'versions':versions, 'alignment':c['alignment'],
                'common_interval_sec':[start,end], 'signal_key':signal_key})
        else:
            inventory = json.loads(marker.read_text())['inventory']
            valid = np.load(cache/'quality_valid.npy')
        event_path = run['files'].get('events')
        event_text = event_path.read_text() if event_path else ''
        label_key = digest({'signals':signal_key, 'labels':c['labels'], 'events':event_text})
        result = self.out/'experiments'/label_key/run['id']
        result.mkdir(parents=True,exist_ok=True)
        if c['labels']['enabled']:
            events = read_events(event_path,annotation_duration)
            y,reasons,matched = label_windows(starts-p['filter_warmup_sec'],ends,events,annotation_duration,c['labels'])
            events.to_csv(result/'events.tsv',sep='\t',index=False)
        else:
            y = np.full(count,-1,dtype=np.int8)
            reasons = np.full(count,'labels_disabled',dtype=object)
            matched = np.full(count,'',dtype=object)
        y[~valid], reasons[~valid], matched[~valid] = -1,'signal_quality',''
        table = pd.DataFrame({'window_index':np.arange(count),'subject':run['subject'],
            'session':run['session'],'run':run['run'],'input_start_sec':starts,
            'processing_support_start_sec':starts-p['filter_warmup_sec'],'prediction_time_sec':ends,
            'label':y,'reason':reasons,'event_row':matched,'signal_valid':valid,'eligible':y>=0})
        table.to_csv(result/'windows.csv',index=False)
        np.save(result/'y.npy',y)
        save_json(result/'config.json',c)
        report = {'run':run['id'],'status':'complete','windows':count,'eligible':int((y>=0).sum()),
            'label_counts':table.label.value_counts().to_dict(),'reason_counts':table.reason.value_counts().to_dict(),
            'signal_cache_reused':cached,'signal_directory':str(cache.resolve()),
            'result_directory':str(result.resolve()),'inventory':inventory,
            'alignment':c['alignment'],'cross_run_continuity':'not assumed; windows and labels never cross runs'}
        save_json(result/'report.json',report)
        for raw in raws.values():
            raw.close()
        reference.close()
        return report
