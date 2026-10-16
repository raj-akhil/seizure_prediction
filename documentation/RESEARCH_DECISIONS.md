> Historical pre-training notes. Update: authorized engineering fits completed on 2026-09-16; see [current training status](TRAINING_PIPELINE.md). Full population training remains pending.

# Research and implementation decisions

## Objective and scope

Stage 1 is population-level seizure forecasting from configurable wearable signals. Stage 2 patient adaptation is deferred. The local project brief specifies at most four physical electrodes near the ears; derived channels are not equivalent to physical electrodes. This implementation uses the recorded channels and does not invent unavailable bipolar derivations or claim hardware feasibility from feature counts.

## Evidence behind the implementation

1. **SeizeIT2 (2025), primary dataset paper.** https://doi.org/10.1038/s41597-025-05580-x . Provides the wearable EEG/ECG/EMG/motion context and describes publisher synchronization. Its benchmark task is seizure detection. Its results cannot be quoted as forecasting performance or evidence of a 30-minute warning.
2. **Leal et al., Unsupervised EEG preictal interval identification (2023).** https://www.nature.com/articles/s41598-022-23902-6 . Investigates linear/nonlinear scalp-EEG features over five-second segments and finds heterogeneous preictal patterns. It motivates configurable time/spectral features and testing more than one history length. It is not a validation of our exact feature list or behind-ear model.
3. **Nasseri et al., hybrid wearable forecasting (2025).** https://doi.org/10.1111/epi.18466 . Investigates heart-rate/activity information and short/long forecasting horizons with pseudo-prospective evaluation. It motivates cardiac/activity inputs and temporal experiments. Its longitudinal patient-specific setting differs from our initial population model and discontinuous runs.
4. **ESC/NASPE HRV Task Force (1996).** https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf . Standard short-term HRV typically uses five-minute recordings. Our default 60-second RR summaries are exploratory, are not standard five-minute HRV, and omit LF/HF estimates. No normal-to-normal clinical beat annotation is claimed.
5. **scikit-learn leakage guidance.** https://scikit-learn.org/1.8/common_pitfalls.html . Learned feature selection, imputation and scaling must be fitted on training data only. We encapsulate these in the estimator pipeline inside subject-grouped CV.
6. **SciPy signal processing.** https://docs.scipy.org/doc/scipy/reference/signal.html . Welch spectra, causal SOS filters, and peak detection provide deterministic signal-processing primitives.

## Candidate models, not an asserted universal best

The first comparison uses logistic regression, random forest, and histogram gradient boosting. They provide a linear baseline and two nonlinear tabular baselines using a common feature pipeline. This is an engineering choice for interpretable feature experiments, not a claim that a publication proves histogram boosting best on SeizeIT2 forecasting. The best candidate will be selected by training-only grouped cross-validation after adequate patients arrive. No language model is required to consume these numerical signals. Deep neural encoders remain a future comparison if data support them.

## Feature families

- EEG: mean, variability, RMS, amplitude range, mean absolute amplitude, line length, zero crossings, skewness/kurtosis, Hjorth activity/mobility/complexity, Welch power and relative power in delta/theta/alpha/beta/low-gamma bands, spectral centroid/median/dominant frequency, spectral entropy, pairwise channel correlation.
- ECG: time/spectral summaries plus QRS-energy beat detection from an independently configurable trailing raw-ECG interval; estimated heart rate, mean RR, RR standard deviation, RMSSD and pNN50 with count/range checks and availability flags. Inverted polarity is handled with energy and absolute-peak refinement. This heuristic needs real beat-level validation; plausible heart rates and synthetic tests do not prove clinical accuracy.
- EMG: time/spectral summaries, integrated absolute activity, slope sign changes, median/mean frequency proxies and band powers.
- Motion: axis statistics/spectra, dynamic RMS, derivative RMS, and vector-magnitude summaries for recognized complete three-axis groups on each device. Units follow EDF calibration; these are not calibrated posture labels or physical jerk claims for all axis types.
- Optional trailing feature means use only prior/current windows and require complete history within a run. Nothing uses future signals.

Frequency analysis is capped by the original modality Nyquist frequency, even if preprocessing upsamples motion. Short segments limit frequency resolution. Fixed filter settings and feature bands are experimental choices, not universal physiological optimizers.

## Selection and evaluation

Manual feature-family/channel selection operates now. Data-dependent cleanup and selection run only later, inside each training fold: excessive missingness/constant removal, training-median imputation, deterministic correlation pruning, training scaling, missing indicators, and ANOVA or optional mutual-information ranking. This distinguishes feature selection from hardware channel selection; report selected feature groups but do not convert them into an electrode count.

Explicit disjoint train/validation/test subject lists are required. No random-window fallback exists. Train-only grouped CV compares candidates; validation determines the alarm threshold; the fixed winner is then evaluated once on test subjects. Internal random-window early stopping is disabled for histogram boosting. Class weights address imbalance, but raw probabilities are not independently calibrated; Brier score and log loss are reported.

Alarm sensitivity, false alarms per eligible monitoring day, lead times, and fraction of eligible decisions in warning accompany window metrics and patient-level results. These are **conditional on eligible windows**, not continuous prospective clinical performance. Most run tails are excluded due to unknown continuity; eight short runs lack eligible labels under the present forecast definition. Future validation must resolve coverage and continuity before reporting full-day device performance.

## Work intentionally not performed

No model fitting, fitted feature ranking, hyperparameter search, validation threshold optimization or test performance evaluation has been run. Population performance cannot be evaluated from one patient and one eligible seizure. Stage 2 patient calibration, neural-network training, full artifact annotation and independent signal synchronization are not implemented by this stage.
