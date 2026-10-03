"""Training-only feature cleanup and selection, cloned inside every CV fold."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, mutual_info_classif


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, method='anova', k=100, max_missing_fraction=.5,
                 correlation_threshold=.98, add_missing_indicators=True, seed=42):
        self.method=method;self.k=k;self.max_missing_fraction=max_missing_fraction
        self.correlation_threshold=correlation_threshold
        self.add_missing_indicators=add_missing_indicators;self.seed=seed

    def fit(self, X, y):
        if not isinstance(X,pd.DataFrame):
            raise ValueError('Named feature DataFrame required')
        if any(not str(c).startswith(('eeg::','ecg::','emg::','mov::')) for c in X):
            raise ValueError('Metadata/unknown columns must not enter predictors')
        clean=X.replace([np.inf,-np.inf],np.nan)
        candidates=[c for c in clean if clean[c].isna().mean()<=self.max_missing_fraction and clean[c].nunique()>1]
        if not candidates:
            raise ValueError('No usable features in this training fold')
        medians=clean[candidates].median()
        filled=clean[candidates].fillna(medians)
        corr=filled.corr().abs().to_numpy()
        kept=[]
        for j in range(len(candidates)):
            if not kept or not np.any(corr[j,kept]>self.correlation_threshold):
                kept.append(j)
        self.columns_=[candidates[j] for j in kept]
        self.medians_=medians[self.columns_]
        self.means_=filled[self.columns_].mean()
        self.scales_=filled[self.columns_].std(ddof=0).replace(0,1)
        transformed=self._clean(X)
        self.names_=list(transformed.columns)
        all_values=transformed.to_numpy()
        # Constant missingness flags cannot inform ranking and cause ANOVA warnings.
        variable=np.flatnonzero(np.var(all_values,axis=0)>0)
        values=all_values[:,variable]
        if self.method=='none':
            scores=np.ones(values.shape[1])
        elif self.method=='anova':
            scores=f_classif(values,y)[0]
        elif self.method=='mutual_info':
            scores=mutual_info_classif(values,y,random_state=self.seed)
        else:
            raise ValueError(f'Unknown selection method {self.method}')
        scores=np.nan_to_num(scores,nan=-np.inf,posinf=np.finfo(float).max)
        n=len(scores) if self.k=='all' or self.method=='none' else min(int(self.k),len(scores))
        self.indices_=np.sort(variable[np.argsort(-scores,kind='stable')[:n]])
        self.scores_=scores
        return self

    def _clean(self, X):
        selected=X.reindex(columns=self.columns_).replace([np.inf,-np.inf],np.nan)
        standardized=(selected.fillna(self.medians_)-self.means_)/self.scales_
        if self.add_missing_indicators:
            mask=selected.isna().astype(float)
            mask.columns=[f'{c}::missing' for c in mask]
            standardized=pd.concat([standardized,mask],axis=1)
        return standardized

    def transform(self,X):
        return self._clean(X).to_numpy(dtype=np.float64)[:,self.indices_]

    def get_feature_names_out(self,input_features=None):
        return np.asarray(self.names_,dtype=object)[self.indices_]
