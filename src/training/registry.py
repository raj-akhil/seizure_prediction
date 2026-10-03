"""Registry responsibilities for the training pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from src.models.selection import FeatureSelection


def candidates(config):
    t=config['training'];seed=t['seed']
    registry={'logistic':LogisticRegression(random_state=seed),
        'random_forest':RandomForestClassifier(random_state=seed,n_jobs=1),
        # Disable implicit random-window early-stopping validation.
        'hist_gradient_boosting':HistGradientBoostingClassifier(random_state=seed,early_stopping=False)}
    result={}
    for name,grid in t['models'].items():
        if name not in registry:
            raise ValueError(f'Unsupported model {name}')
        selector=FeatureSelection(**t['selection'],seed=seed)
        pipeline=Pipeline([('features',selector),('model',registry[name])])
        parameters={f'model__{key}':values for key,values in grid.items()}
        parameters.update({f'features__{key}':values for key,values in t['selection_grid'].items()})
        if set(parameters)-set(pipeline.get_params()):
            raise ValueError(f'Unknown model/selection parameters for {name}')
        if any(not isinstance(values,list) or not values for values in parameters.values()):
            raise ValueError('Search grid values must be nonempty lists')
        result[name]=(pipeline,parameters)
    return result


