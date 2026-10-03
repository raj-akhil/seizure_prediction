"""Leakage-safe LSTM training over ordered prepared feature windows.

This module deliberately lives beside, rather than inside, the sklearn tabular
pipeline.  An LSTM needs the order, recording boundary and time gap for every
row.  Passing a shuffled feature table to an LSTM would create invalid
sequences and overstate its performance.
"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from src.models.selection import FeatureSelection


def enabled(config):
    """Whether the separately configured sequence model is part of this run."""
    return bool(config['training'].get('lstm', {}).get('enabled', False))


def build_sequences(values, metadata, sequence_windows, stride_sec):
    """Return contiguous trailing sequences and the local row used as each label.

    Sequence boundaries never cross recordings or missing/excluded windows.  The
    latter appears as a time gap larger than the configured prepared-data stride.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(metadata):
        raise ValueError('LSTM features and metadata must have matching two-dimensional rows')
    if sequence_windows < 2:
        raise ValueError('LSTM sequence_windows must be at least 2')
    required = {'recording_id', 'prediction_time_sec'}
    if not required.issubset(metadata):
        raise ValueError('LSTM requires recording_id and prediction_time_sec metadata')

    frame = metadata.loc[:, ['recording_id', 'prediction_time_sec']].copy()
    frame['_local_row'] = np.arange(len(frame))
    sequences, endpoints = [], []
    tolerance = max(1e-6, abs(float(stride_sec)) * 1e-5)
    for _, recording in frame.groupby('recording_id', sort=False):
        recording = recording.sort_values('prediction_time_sec', kind='stable')
        rows = recording['_local_row'].to_numpy()
        times = recording['prediction_time_sec'].to_numpy(dtype=float)
        start = 0
        for pos in range(len(rows)):
            if pos and not np.isclose(times[pos] - times[pos - 1], stride_sec,
                                      rtol=1e-5, atol=tolerance):
                start = pos
            if pos - start + 1 >= sequence_windows:
                sequence_rows = rows[pos - sequence_windows + 1:pos + 1]
                sequences.append(values[sequence_rows])
                endpoints.append(rows[pos])
    if not sequences:
        return (np.empty((0, sequence_windows, values.shape[1]), dtype=np.float32),
                np.empty(0, dtype=int))
    return np.stack(sequences).astype(np.float32, copy=False), np.asarray(endpoints, dtype=int)


def _bounded_training_sample(sequences, labels, maximum, seed):
    """Bound CPU work while retaining every class when possible.

    This is applied only to the fitting set; validation and test sequences are
    never sampled.  It is not used as a claimed population prevalence.
    """
    if len(sequences) <= maximum:
        return sequences, labels
    labels = np.asarray(labels, dtype=int)
    positive, negative = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError('LSTM training sequences need both classes')
    rng = np.random.default_rng(seed)
    positive_count = min(len(positive), maximum // 2)
    negative_count = min(len(negative), maximum - positive_count)
    # If positives are scarce, use the remaining budget for negatives.
    negative_count = min(len(negative), maximum - positive_count)
    picked = np.concatenate([rng.choice(positive, positive_count, replace=False),
                             rng.choice(negative, negative_count, replace=False)])
    rng.shuffle(picked)
    return sequences[picked], labels[picked]


class SequenceNetwork(nn.Module):
    """Pickleable network definition saved with an experiment artifact."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        output, _ = self.lstm(batch)
        return self.classifier(output[:, -1, :]).squeeze(1)


def _network(input_size, settings):
    return SequenceNetwork(input_size, settings['hidden_size'], settings['num_layers'], settings['dropout'])


def _probabilities(network, sequences, batch_size):
    network.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = torch.from_numpy(sequences[start:start + batch_size])
            values.append(torch.sigmoid(network(batch)).cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def _fit_network(train_sequences, train_labels, settings, seed, validation=None, epochs=None):
    """Fit a small CPU LSTM; validation is only an inner-CV fold."""
    torch.manual_seed(int(seed))
    if len(train_sequences) == 0 or set(np.asarray(train_labels, dtype=int)) != {0, 1}:
        raise ValueError('LSTM training sequences need both classes')
    network = _network(train_sequences.shape[2], settings)
    labels = np.asarray(train_labels, dtype=np.float32)
    positives, negatives = float(labels.sum()), float(len(labels) - labels.sum())
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, dtype=torch.float32))
    optimiser = torch.optim.AdamW(network.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    maximum_epochs = int(epochs if epochs is not None else settings['max_epochs'])
    best_score, best_epoch, stale, best_state = -math.inf, 0, 0, None
    history = []
    generator = torch.Generator().manual_seed(int(seed))
    for epoch in range(1, maximum_epochs + 1):
        network.train()
        order = torch.randperm(len(train_sequences), generator=generator).numpy()
        losses = []
        for start in range(0, len(order), settings['batch_size']):
            chosen = order[start:start + settings['batch_size']]
            inputs = torch.from_numpy(train_sequences[chosen])
            targets = torch.from_numpy(labels[chosen])
            optimiser.zero_grad()
            batch_loss = loss(network(inputs), targets)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimiser.step()
            losses.append(float(batch_loss.detach()))
        item = {'epoch': epoch, 'training_loss': float(np.mean(losses))}
        if validation is None:
            best_epoch = epoch
        else:
            validation_sequences, validation_labels = validation
            score = average_precision_score(validation_labels,
                                            _probabilities(network, validation_sequences, settings['batch_size']))
            item['validation_average_precision'] = float(score)
            if score > best_score:
                best_score, best_epoch, stale = float(score), epoch, 0
                best_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
            else:
                stale += 1
                if stale >= settings['patience']:
                    history.append(item)
                    break
        history.append(item)
    if best_state is not None:
        network.load_state_dict(best_state)
    return network, {'epochs_completed': len(history), 'best_epoch': best_epoch,
                     'best_validation_average_precision': None if best_score == -math.inf else best_score,
                     'history': history}


class FeatureLSTM:
    """Fitted selector plus recurrent classifier with metadata-aware prediction."""
    def __init__(self, selector, network, settings, input_features):
        self.selector = selector
        self.network = network
        self.settings = settings
        self.input_features = list(input_features)

    def get_feature_names_out(self):
        return np.asarray(self.input_features, dtype=object)

    def predict_for_rows(self, X, metadata, rows):
        rows = np.asarray(rows, dtype=int)
        features = self.selector.transform(X.iloc[rows])
        sequences, endpoints = build_sequences(features, metadata.iloc[rows],
                                               self.settings['sequence_windows'], self.settings['stride_sec'])
        if not len(endpoints):
            raise ValueError('No complete LSTM sequences in this partition; reduce sequence_windows or prepare longer contiguous recordings')
        return rows[endpoints], _probabilities(self.network, sequences, self.settings['batch_size'])


def _selector(config, feature_k):
    settings = deepcopy(config['training']['selection'])
    settings['k'] = feature_k
    settings['seed'] = config['training']['seed']
    return FeatureSelection(**settings)


def _sequence_partition(selector, X, metadata, rows, settings):
    values = selector.transform(X.iloc[rows])
    sequences, endpoints = build_sequences(values, metadata.iloc[rows], settings['sequence_windows'], settings['stride_sec'])
    if not len(endpoints):
        raise ValueError('No complete LSTM sequences; sequence length exceeds every contiguous recording segment')
    labels = metadata.iloc[rows].label.to_numpy(dtype=int)[endpoints]
    if set(labels) != {0, 1}:
        raise ValueError('LSTM sequence endpoints need both classes; revise partitions or sequence length')
    return sequences, labels


def fit(config, X, metadata, train_rows, folds):
    """Cross-validate and refit the LSTM on only the explicit training subjects."""
    configured = deepcopy(config['training']['lstm'])
    settings = {**configured, 'stride_sec': float(config['preprocessing']['stride_sec'])}
    feature_k = configured['feature_k']
    fold_reports, scores, selected_epochs = [], [], []
    for number, (fold_train, fold_validation) in enumerate(folds, start=1):
        fit_rows = np.asarray(train_rows)[fold_train]
        validation_rows = np.asarray(train_rows)[fold_validation]
        selector = _selector(config, feature_k).fit(X.iloc[fit_rows], metadata.iloc[fit_rows].label)
        train_sequences, train_labels = _sequence_partition(selector, X, metadata, fit_rows, settings)
        train_sequences, train_labels = _bounded_training_sample(train_sequences, train_labels,
                                                                 settings['max_training_sequences'],
                                                                 config['training']['seed'] + number)
        validation_sequences, validation_labels = _sequence_partition(selector, X, metadata, validation_rows, settings)
        network, report = _fit_network(train_sequences, train_labels, settings,
                                      config['training']['seed'] + number,
                                      validation=(validation_sequences, validation_labels))
        score = float(report['best_validation_average_precision'])
        scores.append(score); selected_epochs.append(report['best_epoch'])
        fold_reports.append({'fold': number, 'average_precision': score,
                             'best_epoch': report['best_epoch'],
                             'training_sequences': len(train_sequences),
                             'validation_sequences': len(validation_sequences),
                             'input_features': train_sequences.shape[2]})
        print(f'LSTM inner fold {number}: average_precision={score:.5f}, best_epoch={report["best_epoch"]}', flush=True)
    refit_epochs = max(1, int(round(float(np.median(selected_epochs)))))
    selector = _selector(config, feature_k).fit(X.iloc[train_rows], metadata.iloc[train_rows].label)
    all_sequences, all_labels = _sequence_partition(selector, X, metadata, train_rows, settings)
    all_sequences, all_labels = _bounded_training_sample(all_sequences, all_labels,
                                                         settings['max_training_sequences'],
                                                         config['training']['seed'])
    network, refit = _fit_network(all_sequences, all_labels, settings, config['training']['seed'],
                                  validation=None, epochs=refit_epochs)
    model = FeatureLSTM(selector, network, settings, selector.get_feature_names_out())
    detail = {'inner_folds': fold_reports, 'mean_inner_cv_average_precision': float(np.mean(scores)),
              'refit_epochs': refit_epochs, 'refit_training_sequences': len(all_sequences),
              'sequence_windows': settings['sequence_windows'],
              'input_features': len(model.input_features), 'refit': refit}
    return float(np.mean(scores)), model, detail
