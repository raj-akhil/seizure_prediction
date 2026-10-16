"""Runner responsibilities for the training pipeline."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.models.evaluation import window_metrics, alarm_metrics, choose_threshold
from src.preprocessing.stream_processor import save_json, digest
from .registry import candidates
from . import lstm
from threadpoolctl import threadpool_limits
from src.validation.report import write_report


def train(config, X, meta, events, entries, indices, folds):
    with threadpool_limits(limits=config['training'].get('native_threads',1)):
        return _train(config,X,meta,events,entries,indices,folds)


def _train(config, X, meta, events, entries, indices, folds):
    """Call after explicit authorization via python -m src.commands.train --start-training."""
    t=config['training'];train_idx=indices['train'];val_idx=indices['validation'];test_idx=indices['test']
    target=Path(t['output_dir'])
    if not target.is_absolute():
        target=Path(__file__).resolve().parents[2]/target
    run_dir=target/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+digest(config))
    run_dir.mkdir(parents=True,exist_ok=False)
    save_json(run_dir/'config.json',config)
    save_json(run_dir/'feature_manifest.json',entries)
    packages=['numpy','scipy','scikit-learn','pandas','joblib']
    if lstm.enabled(config):
        packages.append('torch')
    save_json(run_dir/'versions.json',{name:importlib.metadata.version(name) for name in packages})
    save_json(run_dir/'splits.json',t['splits'])
    save_json(run_dir/'inner_folds.json',[
        {'train_subjects':sorted(meta.iloc[train_idx].iloc[a].subject.unique()),
         'validation_subjects':sorted(meta.iloc[train_idx].iloc[b].subject.unique())}
        for a,b in folds])
    board=[];best=None
    # Sequence endpoints omit initial/history-incomplete windows.  Do not rank a
    # sequence score against a tabular score over a different observation set;
    # users compare the two modes as separate, fully documented experiments.
    if not lstm.enabled(config):
        for name,(pipeline,grid) in candidates(config).items():
            print(f'Fitting candidate: {name}',flush=True)
            search=GridSearchCV(pipeline,grid,cv=folds,scoring=t['scoring'],n_jobs=t['n_jobs'],refit=True,error_score='raise',verbose=t.get('cv_verbose',2))
            search.fit(X.iloc[train_idx],meta.label.iloc[train_idx])
            pd.DataFrame(search.cv_results_).to_csv(run_dir/f'{name}_inner_cv.csv',index=False)
            board.append({'model':name,'inner_cv_average_precision':float(search.best_score_),'parameters':search.best_params_})
            if best is None or search.best_score_>best[0]:
                best=(search.best_score_,name,search.best_estimator_,'tabular')
    if lstm.enabled(config):
        print('Fitting candidate: feature_lstm (ordered feature sequences)', flush=True)
        score, model, detail = lstm.fit(config, X, meta, train_idx, folds)
        save_json(run_dir/'feature_lstm_inner_cv.json', detail)
        board.append({'model':'feature_lstm','inner_cv_average_precision':score,
                      'parameters':{'sequence_windows':config['training']['lstm']['sequence_windows'],
                                    'hidden_size':config['training']['lstm']['hidden_size'],
                                    'feature_k':config['training']['lstm']['feature_k']},
                      'sequence_input_features':detail['input_features']})
        if best is None or score>best[0]:
            best=(score,'feature_lstm',model,'sequence')
    if best is None:
        raise ValueError('No enabled training models. Enable a tabular model or training.lstm.enabled.')
    save_json(run_dir/'model_comparison.json',board)
    _,name,model,model_kind=best
    if model_kind=='sequence':
        validation_rows, val_p=model.predict_for_rows(X,meta,val_idx)
    else:
        validation_rows, val_p=val_idx,model.predict_proba(X.iloc[val_idx])[:,1]
    validation_meta=meta.iloc[validation_rows]
    threshold,thresholds=choose_threshold(validation_meta,val_p,config,events)
    save_json(run_dir/'validation_thresholds.json',thresholds)
    # No refit after threshold selection: deployed model remains the one validated above.
    selected=(model.get_feature_names_out().tolist() if model_kind=='sequence'
              else model.named_steps['features'].get_feature_names_out().tolist())
    if model_kind=='sequence':
        reference_rows,_=model.predict_for_rows(X,meta,train_idx)
    else:
        reference_rows=train_idx
    reference_probability=float(meta.label.iloc[reference_rows].mean())
    save_json(run_dir/'validation_reference.json',{'training_prevalence':reference_probability,
        'training_windows':len(reference_rows),'source':'unweighted eligible training labels only'})
    save_json(run_dir/'selected_features.json',selected)
    save_json(run_dir/'selected_channel_groups.json',sorted({name.split('::')[0]+'::'+name.split('::')[1] for name in selected}))
    joblib.dump({'model':model,'threshold':threshold['threshold'],'config':config,'selected_features':selected},run_dir/'model.joblib')
    results={'selected_model':name,'threshold':threshold['threshold'],'validation':window_metrics(validation_meta.label,val_p),
        'validation_alarms':threshold,'probability_note':'Class-weighted model probabilities are not independently calibrated'}
    # Test is evaluated once, after all model/feature/threshold choices are fixed.
    if model_kind=='sequence':
        test_rows,test_p=model.predict_for_rows(X,meta,test_idx)
    else:
        test_rows,test_p=test_idx,model.predict_proba(X.iloc[test_idx])[:,1]
    for partition,rows,prob in [('validation',validation_rows,val_p),('test',test_rows,test_p)]:
        predictions=meta.iloc[rows].copy();predictions['probability']=prob
        predictions.to_csv(run_dir/f'{partition}_predictions.csv',index=False)
        results[partition]=window_metrics(predictions.label,prob,threshold['threshold'],reference_probability)
        results[f'{partition}_alarms']=alarm_metrics(predictions,prob,threshold['threshold'],config,events)
        patients={}
        for subject in predictions.subject.unique():
            mask=predictions.subject.eq(subject).to_numpy()
            patients[subject]={'window':window_metrics(predictions.label[mask],prob[mask],threshold['threshold'],reference_probability),
                'alarm':alarm_metrics(predictions.loc[mask],prob[mask],threshold['threshold'],config,events)}
        results[f'{partition}_patients']=patients
        write_report(predictions,config,events,threshold['threshold'],reference_probability,
                     run_dir/'validation_reports'/partition,partition)
    save_json(run_dir/'evaluation.json',results)
    save_json(run_dir/'complete.json',{'status':'complete','selected_model':name})
    return run_dir
