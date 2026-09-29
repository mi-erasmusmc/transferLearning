# PLP develop contract used by the pilot

The runner now follows merged PLP develop behavior. CI pins commit
[`f2cef128f1bfdf69295f2d5ab0eb911d5103a93a`](https://github.com/OHDSI/PatientLevelPrediction/commit/f2cef128f1bfdf69295f2d5ab0eb911d5103a93a).
Install a develop build containing these changes, then follow
[Installation.md](Installation.md). No local PLP patch is applied in CI or needed
for production installation. The earlier patch is preserved in Git history.

## Coefficient handling

PLP matches supplied coefficients to Cyclops columns by semantic covariate ID,
avoids changing the caller's covariates, preserves fixed source coefficients in
CV refits, combines source slopes and fitted corrections for prediction, and
aligns CV predictions by rowId. Native Cyclops IDs preserve large coefficient IDs.

Develop deliberately **drops source coefficients for predictors absent from target
training**, following commit
[`8ad78b2416`](https://github.com/OHDSI/PatientLevelPrediction/commit/8ad78b2416).
The runner adopts this policy independently in every inner training fold and in
the final refit. Presence comes from the prepared training covariates, not the
covariate reference table or held-out patients. Only overlapping source slopes
are converted into the fold's units:

```
betaTarget = betaSource * maxTarget / maxSource
```

There are no fallback normalization factors for absent source predictors. If no
source predictors overlap, the target fit is target-only. If a feature is absent
from an inner training fold but present in the complete final training sample,
its source coefficient can participate in the final fit. Missing or invalid
normalization factors for an overlapping source coefficient still fail explicitly.
The source intercept is excluded; the target intercept is freely fitted.

Dropped source IDs are recorded in each tuning row (`droppedSourceIds`, comma-
separated; NA on a failed fit) and in the final fit's
`transferDetails$droppedSourceIds`, retained in `tuning.rds`. This policy applies
to `priorCoefs` adaptation. Frozen-source and recalibration comparators still
predict with the source model and its original feature set.

This supersedes the earlier runner contract requiring source-only features to
contribute at prediction time. That change in feature availability is particularly
relevant to small target training samples; results under the two contracts must
not be treated as identical experiments. Use a new output folder for this build.

## Runner-side tuning, without a new PLP API

For each variance candidate and inner fold, the runner subsets raw training data,
estimates preprocessing on that fold, converts overlapping source coefficients,
and calls public `fitPlp()` with a local all-ones fold table. PLP's existing
single-fold path fits the supplied variance without internal tuning or CV
predictions. The actual validation assignments remain separate and unchanged.
`createDefaultSplitSetting(nfold=1)` is not used.

The runner predicts held-out patients using the fold's preprocessing, selects
variance by held-out log loss, and refits on the complete final training sample.
PLP's `getCV()` does not replace this loop because it receives preprocessed data.
Every fit checks the actual variance, fit status and absence of internal CV output.
No new constructor argument, monkeypatch or vendored fitting implementation is used.

## Behavioral readiness and provenance

`validateExperiment()` announces a synthetic readiness check before database work.
It exercises fixed-variance fitting below the internal search limits, coefficient
matching for overlapping source features, dropping a source-only feature, no
prediction contribution from that dropped feature, and unchanged caller data.
Regression tests also exercise PLP's native drop behavior directly, no-overlap
agreement with target-only, fold-specific dropping and final-refit inclusion.

The manifest records versions and fingerprints of loaded PLP and runner functions.
A version string alone does not identify a develop build. The check establishes
the exercised runner contract, not correctness of every PLP feature.

## JSON precision is still unresolved

These changes do not change PLP model serialization. JSON can round coefficients
and normalization factors, so an interrupted run that reloads a source model for
unfinished jobs is not guaranteed to match an uninterrupted in-memory run exactly.
No precision sidecar is introduced. Raw-data cache and in-memory prediction tests
are distinct from model-JSON round-trip claims. See [Validation.md](Validation.md)
for validation scope and history.
