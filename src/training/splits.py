"""Splits responsibilities for the training pipeline."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def preflight(config, X, meta):
    if len(X)!=len(meta) or not X.index.equals(meta.index):
        raise ValueError('Features and metadata must have identical row indices')
    if meta.duplicated(['recording_id','prediction_time_sec']).any() if 'prediction_time_sec' in meta else False:
        raise ValueError('Duplicate prediction windows')
    t=config['training'];splits=t['splits']
    groups={name:set(splits[name]) for name in ('train','validation','test')}
    if any(not group for group in groups.values()):
        raise ValueError('Assign explicit train/validation/test subject lists; downloading patients does not assign partitions automatically')
    if any(groups[a]&groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Subjects overlap between partitions')
    observed=set(meta.subject)
    assigned=set.union(*groups.values())
    if assigned!=observed:
        raise ValueError(f'Every eligible subject must be assigned exactly once. Unassigned: {observed-assigned}; no eligible data: {assigned-observed}')
    indices={name:np.flatnonzero(meta.subject.isin(subjects)) for name,subjects in groups.items()}
    problems=[]
    for name,idx in indices.items():
        part=meta.iloc[idx]
        if set(part.label)!={0,1}:
            problems.append(f'{name} needs both classes: {int((part.label==0).sum())} background and '
                            f'{int((part.label==1).sum())} positive windows; patients {sorted(groups[name])}')
        n=part.loc[part.label==1,['recording_id','event_row']].drop_duplicates().shape[0]
        if n<t['min_positive_events_per_partition']:
            problems.append(f'{name} has only {n} positive seizure episodes; '
                            f"requires {t['min_positive_events_per_partition']}")
    if problems:
        raise ValueError('; '.join(problems))
    train=meta.iloc[indices['train']]
    if train.subject.nunique()<t['inner_folds']:
        raise ValueError('Too few training subjects for grouped cross-validation')
    cv=StratifiedGroupKFold(n_splits=t['inner_folds'],shuffle=True,random_state=t['seed'])
    folds=list(cv.split(X.iloc[indices['train']],train.label,train.subject))
    for a,b in folds:
        if set(train.iloc[a].label)!={0,1} or set(train.iloc[b].label)!={0,1}:
            raise ValueError('A grouped inner fold lacks a class; add positive-event patients or revise partitions')
        if set(train.iloc[a].subject)&set(train.iloc[b].subject):
            raise ValueError('Inner-fold subject leakage')
    return indices,folds


