from pathlib import Path
import math
import yaml


def load_config(config_path='src/config/config.yaml'):
    with Path(config_path).open(encoding='utf-8-sig') as f:
        c = yaml.safe_load(f)
    root = Path(__file__).resolve().parents[2]
    for key in ('raw_dir', 'interim_dir'):
        p = Path(c['dataset'][key])
        c['dataset'][key] = str((root / p).resolve() if not p.is_absolute() else p.resolve())
    raw, out = (Path(c['dataset'][k]) for k in ('raw_dir','interim_dir'))
    recordings=c['dataset'].get('recordings')
    if recordings is not None and (not isinstance(recordings,list) or not recordings or any(not isinstance(r,str) for r in recordings) or len(set(recordings))!=len(recordings)):
        raise ValueError('dataset.recordings must be null or a nonempty list of unique full recording IDs')
    if raw == out or raw in out.parents or out in raw.parents:
        raise ValueError('Raw and interim directories must be separate, non-nested directories')
    enabled = c['sensors']['enabled']
    if not enabled or len(set(enabled)) != len(enabled) or set(enabled)-{'eeg','ecg','emg','mov'}:
        raise ValueError('Select unique known modalities')
    for mod in enabled:
        chosen = c['sensors']['channels'].get(mod)
        if chosen is not None and (not isinstance(chosen,list) or not chosen or any(not isinstance(ch,str) for ch in chosen)):
            raise ValueError('channels must be null or a nonempty list of exact EDF names')
    if c['sensors']['missing_policy'] not in ('error','skip_run'):
        raise ValueError('Invalid missing_policy')
    qualification=c.get('qualification',{})
    if qualification and (not isinstance(qualification,dict) or
                          any(not isinstance(v,bool) for v in qualification.values())):
        raise ValueError('Qualification settings must be boolean values')
    if c['alignment']['mode'] not in ('header','published_run_relative'):
        raise ValueError('Invalid alignment mode')
    if c['alignment']['mode'] == 'published_run_relative' and not c['alignment']['evidence']:
        raise ValueError('Run-relative alignment requires documented evidence')
    for section, keys, positive in [
        ('preprocessing',['window_sec','stride_sec'],True),
        ('preprocessing',['context_sec','filter_warmup_sec'],False),
        ('labels',['minimum_warning_min','postictal_exclusion_min','background_separation_min'],False),
        ('labels',['occurrence_period_min'],True)]:
        for key in keys:
            v = c[section][key]
            if not math.isfinite(v) or (v <= 0 if positive else v < 0):
                raise ValueError(f'Invalid {key}')
    rate = c['preprocessing']['output_sampling_rate']
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError('Invalid output_sampling_rate')
    if c['labels']['annotation_coverage'] != 'whole_run' or not c['labels']['seizure_prefixes']:
        raise ValueError('Specify annotation coverage and seizure prefixes')
    if any(not isinstance(x,str) or not x.startswith('sz') for x in c['labels']['seizure_prefixes']):
        raise ValueError('Seizure prefixes must start with sz')
    if c['alignment']['duration_tolerance_sec'] < 0:
        raise ValueError('Duration tolerance must be nonnegative')
    if set(c['preprocessing']['quality']['reject_flat_modalities'])-{'eeg','ecg','emg','mov'}:
        raise ValueError('Unknown quality-control modality')
    if 'features' in c:
        f=c['features']
        if not isinstance(f['batch_windows'],int) or f['batch_windows']<1 or f['welch_segment_sec']<=0:
            raise ValueError('Invalid feature batch size or spectral segment')
        allowed={'eeg':{'time','spectral','hjorth','entropy','correlation'},
                 'ecg':{'time','spectral','cardiac'},'emg':{'time','spectral','muscle'},'mov':{'time','spectral','motion'}}
        for mod in enabled:
            if not f['families'][mod] or set(f['families'][mod])-allowed[mod]:
                raise ValueError(f'Invalid feature families for {mod}')
            for bounds in f['bands'][mod].values():
                if len(bounds)!=2 or bounds[0]<0 or bounds[1]<=bounds[0]:
                    raise ValueError('Invalid frequency band')
        ecg=f['ecg']
        if ecg['history_sec']<=0 or ecg['min_hrv_sec']<=0 or ecg['min_rr_count']<2 or not 0<=ecg['max_rejected_fraction']<1:
            raise ValueError('Invalid ECG history or beat quality settings')
        if any(v<=0 for v in f['temporal']['history_sec']):
            raise ValueError('Temporal feature history must be positive')
    if 'training' in c:
        t=c['training'];s=t['selection']
        if t['inner_folds']<2 or t['min_positive_events_per_partition']<1 or not t['models']:
            raise ValueError('Invalid training folds, event count or model list')
        if s['method'] not in ('anova','mutual_info','none') or (s['k']!='all' and (not isinstance(s['k'],int) or s['k']<1)):
            raise ValueError('Invalid feature selection')
        if not 0<=s['max_missing_fraction']<1 or not 0<s['correlation_threshold']<=1:
            raise ValueError('Invalid feature cleanup thresholds')
        if t['scoring']!='average_precision':
            raise ValueError('This trainer currently selects models using average_precision')
        if t['max_feature_matrix_gib']<=0:
            raise ValueError('Feature memory limit must be positive')
        if not isinstance(t.get('native_threads',1),int) or t.get('native_threads',1)<1:
            raise ValueError('native_threads must be a positive integer')
        lstm=t.get('lstm',{})
        if not isinstance(lstm,dict):
            raise ValueError('LSTM settings must be an object')
        if lstm:
            required={'enabled','sequence_windows','feature_k','hidden_size','num_layers','dropout',
                      'learning_rate','weight_decay','batch_size','max_epochs','patience','max_training_sequences'}
            if set(lstm)!=required or not isinstance(lstm['enabled'],bool):
                raise ValueError('Invalid LSTM settings')
            integer_keys={'sequence_windows','hidden_size','num_layers','batch_size','max_epochs',
                          'patience','max_training_sequences'}
            if any(not isinstance(lstm[key],int) or lstm[key] < (2 if key=='sequence_windows' else 1)
                   for key in integer_keys):
                raise ValueError('Invalid LSTM integer settings')
            if lstm['feature_k']!='all' and (not isinstance(lstm['feature_k'],int) or lstm['feature_k']<1):
                raise ValueError('Invalid LSTM feature_k')
            if not 0<=lstm['dropout']<1 or lstm['learning_rate']<=0 or lstm['weight_decay']<0:
                raise ValueError('Invalid LSTM optimisation settings')
        alarm=t['alarms']
        if not alarm['thresholds'] or any(not 0<=v<=1 for v in alarm['thresholds']) or alarm['consecutive_windows']<1 or alarm['refractory_sec']<0:
            raise ValueError('Invalid alarm settings')
    if 'validation' in c:
        from src.validation.report import settings
        settings(c)
    return c
