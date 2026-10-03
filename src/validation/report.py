"""Auditable reports for fixed held-out predictions; never select a threshold here."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from src.models.evaluation import alarm_metrics
from .metrics import window_metrics, checked_predictions, calibration_table, ratio


DEFAULTS = {'calibration_bins':10, 'bootstrap_replicates':1000,
            'confidence_level':.95, 'min_subjects_for_ci':5, 'seed':42, 'plots':True}


def settings(config):
    result = {**DEFAULTS, **config.get('validation', {})}
    for key, minimum in [('calibration_bins',2), ('bootstrap_replicates',0), ('min_subjects_for_ci',2)]:
        if not isinstance(result[key], int) or result[key] < minimum:
            raise ValueError(f'Invalid validation.{key}')
    if not 0 < result['confidence_level'] < 1:
        raise ValueError('Invalid confidence level')
    return result


def validate_frame(frame, config, events):
    required = {'subject','recording_id','prediction_time_sec','label','probability'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError('Prediction table requires complete identifiers, timestamps, labels and probabilities')
    checked_predictions(frame.label, frame.probability)
    if 'eligible' in frame and not frame.eligible.eq(True).all():
        raise ValueError('Only eligible windows may enter conditional evaluation')
    if frame.duplicated(['recording_id','prediction_time_sec']).any():
        raise ValueError('Duplicate prediction windows')
    if (frame.groupby('recording_id').subject.nunique() != 1).any():
        raise ValueError('Recording assigned to multiple patients')
    h=config['labels']['minimum_warning_min']*60
    p=config['labels']['occurrence_period_min']*60
    for run, block in frame.groupby('recording_id'):
        times=block.prediction_time_sec.to_numpy(dtype=float)
        if not np.isfinite(times).all() or (times < 0).any():
            raise ValueError('Invalid prediction times')
        selected=events[run]
        selected=selected[selected.eventType.str.startswith(tuple(config['labels']['seizure_prefixes']))]
        expected=np.zeros(len(times),dtype=bool)
        for onset in selected.onset:
            expected |= (times+h <= onset) & (onset < times+h+p)
        if not np.array_equal(expected,block.label.to_numpy()==1):
            raise ValueError('Saved labels do not match configured seizure horizon/types')


def bootstrap(frame, patient_alarms, threshold, reference, cfg):
    subjects=frame.subject.unique()
    result={'method':'Patient cluster percentile bootstrap; complete patients resampled with replacement',
            'subjects':len(subjects),'confidence_level':cfg['confidence_level'],
            'requested_replicates':cfg['bootstrap_replicates'],'intervals':{}}
    if len(subjects)<cfg['min_subjects_for_ci'] or cfg['bootstrap_replicates']==0:
        result['status']='not_computed_insufficient_patients_or_disabled'
        return result
    rng=np.random.default_rng(cfg['seed'])
    blocks=[frame.loc[frame.subject==s,['label','probability']] for s in subjects]
    values={key:[] for key in ['average_precision','roc_auc','brier_score','brier_skill_score',
                              'event_sensitivity','false_alarms_per_eligible_day','time_in_warning_fraction_eligible']}
    for _ in range(cfg['bootstrap_replicates']):
        ids=rng.integers(0,len(subjects),len(subjects))
        sample=pd.concat([blocks[i] for i in ids],ignore_index=True)
        scores=window_metrics(sample.label,sample.probability,threshold,reference)
        alarms=[patient_alarms[subjects[i]] for i in ids]
        hours=sum(a['eligible_monitoring_hours'] for a in alarms)
        scores.update(event_sensitivity=ratio(sum(a['warned_events'] for a in alarms),sum(a['eligible_events'] for a in alarms)),
            false_alarms_per_eligible_day=ratio(sum(a['false_alarms'] for a in alarms),hours/24),
            time_in_warning_fraction_eligible=ratio(sum(a['warning_hours_on_eligible_support'] for a in alarms),hours))
        for key in values:
            if scores[key] is not None:
                values[key].append(scores[key])
    alpha=(1-cfg['confidence_level'])/2
    for key, samples in values.items():
        bounds=np.quantile(samples,[alpha,1-alpha]).tolist() if samples else [None,None]
        result['intervals'][key]={'lower':bounds[0],'upper':bounds[1],'valid_replicates':len(samples)}
    result['status']='computed; exploratory with small patient cohorts; fixed-model uncertainty only'
    return result


def write_report(frame, config, events, threshold, reference_probability, output_dir, partition):
    cfg=settings(config)
    validate_frame(frame,config,events)
    dest=Path(output_dir)
    dest.mkdir(parents=True,exist_ok=False)
    scores=window_metrics(frame.label,frame.probability,threshold,reference_probability)
    alarms=alarm_metrics(frame,frame.probability,threshold,config,events)
    patients=[];patient_alarms={}
    for subject, block in frame.groupby('subject'):
        a=alarm_metrics(block,block.probability,threshold,config,events)
        patient_alarms[subject]=a
        patients.append({'subject':subject,**window_metrics(block.label,block.probability,threshold,reference_probability),
                         **{k:v for k,v in a.items() if isinstance(v,(int,float)) or v is None}})
    patient_table=pd.DataFrame(patients)
    macro={key:{'mean':float(patient_table[key].dropna().mean()) if patient_table[key].notna().any() else None,
                'defined_patients':int(patient_table[key].notna().sum())}
           for key in ['average_precision','roc_auc','brier_score','event_sensitivity','false_alarms_per_eligible_day']}
    calibration=pd.DataFrame(calibration_table(frame.label,frame.probability,cfg['calibration_bins']))
    uncertainty=bootstrap(frame,patient_alarms,threshold,reference_probability,cfg)
    result={'schema_version':1,'partition':partition,'window':scores,
        'alarm':{k:v for k,v in alarms.items() if k not in ('event_records','alarm_records')},
        'patient_macro':macro,'uncertainty':uncertainty,'settings':cfg,
        'reference_definition':'Unweighted training-window prevalence fixed before validation/test',
        'limitations':['Conditional on eligible windows; not continuous clinical monitoring.',
            'Average precision is step-weighted PR area, not trapezoidal PR-AUC.',
            'Class-weighted probabilities are not automatically calibrated.',
            'Event eligibility requires at least one eligible forecasting decision, not full preictal coverage.',
            'No surrogate chance-significance test, external validation or patient adaptation is claimed.']}
    (dest/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    patient_table.to_csv(dest/'per_patient.csv',index=False)
    pd.DataFrame(alarms['event_records']).to_csv(dest/'events.csv',index=False)
    pd.DataFrame(alarms['alarm_records'],columns=['recording_id','time_sec','true_alarm','event_rows','lead_times_sec']).to_csv(dest/'alarms.csv',index=False)
    calibration.to_csv(dest/'calibration.csv',index=False)
    flat=[{'level':level,'metric':key,'value':value} for level in ['window','alarm']
          for key,value in result[level].items() if isinstance(value,(int,float)) or value is None]
    pd.DataFrame(flat).to_csv(dest/'metrics.csv',index=False)
    curves={}
    if frame.label.nunique()==2:
        precision,recall,thresholds=precision_recall_curve(frame.label,frame.probability)
        curves['precision_recall']=pd.DataFrame({'recall':recall,'precision':precision,'threshold':np.r_[thresholds,np.nan]})
        fpr,tpr,thresholds=roc_curve(frame.label,frame.probability)
        curves['roc']=pd.DataFrame({'false_positive_rate':fpr,'true_positive_rate':tpr,'threshold':thresholds})
        for name,table in curves.items():
            table.to_csv(dest/f'{name}.csv',index=False)
    if cfg['plots']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        if curves:
            axes[0].step(curves['precision_recall'].recall,curves['precision_recall'].precision,where='post')
            axes[0].axhline(scores['prevalence'],ls='--',color='gray',label='Prevalence');axes[0].legend()
            axes[1].plot(curves['roc'].false_positive_rate,curves['roc'].true_positive_rate)
        axes[1].plot([0,1],[0,1],'--',color='gray')
        axes[2].plot([0,1],[0,1],'--',color='gray')
        nonempty=calibration[calibration['count']>0]
        axes[2].plot(nonempty.mean_probability,nonempty.observed_fraction,'o-')
        for ax,title,x,y in zip(axes,['Precision–recall','ROC','Calibration'],
            ['Recall','False positive rate','Mean predicted probability'],['Precision','True positive rate','Observed fraction']):
            ax.set(title=title,xlabel=x,ylabel=y,xlim=(0,1),ylim=(0,1));ax.grid(alpha=.2)
        fig.suptitle(f'{partition}: eligible-window evaluation');fig.tight_layout()
        fig.savefig(dest/'curves.png',dpi=160);plt.close(fig)
    lines=[f'# {partition.capitalize()} validation report','',
        'Fixed model and threshold. Eligible-window evaluation; not continuous clinical performance.','',
        '| Metric | Value |','|---|---|']
    for key in ['average_precision','prevalence','roc_auc','brier_score','brier_skill_score','precision','recall','f1']:
        lines.append(f'| {key} | {scores[key]} |')
    for key in ['eligible_events','warned_events','event_sensitivity','false_alarms_per_eligible_day','time_in_warning_fraction_eligible','median_first_warning_lead_sec']:
        lines.append(f'| {key} | {alarms[key]} |')
    lines+=['',f"Confidence intervals: {uncertainty['status']}",'',
            'See metrics.json for definitions/limitations, per_patient.csv, events.csv, alarms.csv and calibration.csv.']
    (dest/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (dest/'complete.json').write_text(json.dumps({'status':'complete','partition':partition}),encoding='utf-8')
    return result
