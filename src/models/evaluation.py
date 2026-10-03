import numpy as np
from src.validation.metrics import window_metrics


def alarm_metrics(frame, probabilities, threshold, config, events_by_run):
    """Conditional evaluation on eligible windows only; no claim of continuous prospective FAR."""
    probabilities=np.asarray(probabilities,dtype=float)
    if probabilities.shape!=(len(frame),) or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('Alarm probabilities must match rows and lie in [0, 1]')
    if not np.isfinite(threshold) or not 0<=threshold<=1:
        raise ValueError('Invalid alarm threshold')
    data=frame.copy()
    data['probability']=probabilities
    alarms=[];eligible_events=set();warned=set();event_records=[]
    step=config['preprocessing']['stride_sec']
    exposure_step=min(step,config['preprocessing']['window_sec'])
    horizon=config['labels']['minimum_warning_min']*60
    period=config['labels']['occurrence_period_min']*60
    alarmcfg=config['training']['alarms']
    warning_cells=0;warning_seconds=0.;exposure_seconds=0.
    for run,block in data.groupby('recording_id',sort=False):
        block=block.sort_values('prediction_time_sec')
        times=block.prediction_time_sec.to_numpy(dtype=float)
        if not np.isfinite(times).all() or (np.diff(times)<=0).any():
            raise ValueError('Prediction timestamps must be finite and unique within each recording')
        if (np.diff(times)<step-1e-6).any():
            raise ValueError('Prediction decisions closer together than configured stride')
        events=events_by_run[run]
        selected=events[events.eventType.str.startswith(tuple(config['labels']['seizure_prefixes']))]
        last=None;last_alarm=-np.inf;streak=0;warning_until=-np.inf
        for row in block.itertuples():
            t=row.prediction_time_sec
            hits=selected.index[(selected.onset>=t+horizon)&(selected.onset<t+horizon+period)].tolist()
            eligible_events.update((run,str(i)) for i in hits)
            if last is None or abs(t-last-step)>1e-6:
                streak=0
            streak=streak+1 if row.probability>=threshold else 0
            if streak>=alarmcfg['consecutive_windows'] and t-last_alarm>=alarmcfg['refractory_sec']:
                last_alarm=t;warning_until=t+horizon+period
                warned.update((run,str(i)) for i in hits)
                alarms.append({'recording_id':run,'time_sec':float(t),'true_alarm':bool(hits),
                    'event_rows':[str(i) for i in hits],'lead_times_sec':[float(selected.loc[i,'onset']-t) for i in hits]})
                streak=0
            warning_cells+=int(t<warning_until)
            # Eligible decision-support cells only. Never bridge a missing decision gap.
            cell_end=t+exposure_step
            exposure_seconds+=exposure_step
            warning_seconds+=max(0.,min(cell_end,warning_until)-t)
            last=t
        for idx,event in selected.iterrows():
            key=(run,str(idx))
            leads=[lead for alarm in alarms if alarm['recording_id']==run
                   for row,lead in zip(alarm['event_rows'],alarm['lead_times_sec']) if row==str(idx)]
            event_records.append({'recording_id':run,'event_row':str(idx),'event_type':event.eventType,
                'onset_sec':float(event.onset),'eligible':key in eligible_events,'warned':key in warned,
                'first_warning_lead_sec':max(leads) if leads else None})
    hours=exposure_seconds/3600
    false=sum(not a['true_alarm'] for a in alarms)
    leads=[e['first_warning_lead_sec'] for e in event_records if e['warned']]
    return {'eligible_events':len(eligible_events),'warned_events':len(warned),
        'selected_events_in_evaluated_runs':len(event_records),
        'unevaluable_events_in_evaluated_runs':len(event_records)-len(eligible_events),
        'missed_eligible_events':len(eligible_events)-len(warned),
        'event_sensitivity':len(warned)/len(eligible_events) if eligible_events else None,
        'alarms':len(alarms),'false_alarms':false,'eligible_monitoring_hours':hours,
        'false_alarms_per_eligible_day':false/(hours/24) if hours else None,
        'false_alarms_per_eligible_hour':false/hours if hours else None,
        'alarm_precision':(len(alarms)-false)/len(alarms) if alarms else None,
        'warning_hours_on_eligible_support':warning_seconds/3600,
        'time_in_warning_fraction_eligible':warning_seconds/exposure_seconds if exposure_seconds else None,
        'median_first_warning_lead_sec':float(np.median(leads)) if leads else None,
        'fraction_eligible_decisions_in_warning':warning_cells/len(data) if len(data) else None,
        'evaluation_scope':'Only eligible forecast windows; excluded periods and run boundaries are not evaluated',
        'warning_definition':'[alarm time, alarm time + warning horizon + occurrence period), intersected with eligible decision cells',
        'event_records':event_records,'alarm_records':alarms}


def choose_threshold(frame, probabilities, config, events_by_run):
    results=[]
    for threshold in config['training']['alarms']['thresholds']:
        metrics=alarm_metrics(frame,probabilities,threshold,config,events_by_run)
        results.append({'threshold':threshold,**metrics})
    budget=config['training']['alarms']['max_false_alarms_per_day']
    feasible=[r for r in results if r['false_alarms_per_eligible_day']<=budget]
    if not feasible:
        raise ValueError('No validation alarm threshold meets the false-alarm budget; test set remains unopened')
    best=max(feasible,key=lambda r:(r['event_sensitivity'] or 0,-r['false_alarms_per_eligible_day'],r['threshold']))
    return best,results
