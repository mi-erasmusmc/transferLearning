# PatientLevelPrediction correctness requirements and runner-side tuning

The bundled patch captures the reviewed `fix-priorcoefs-correctness` work.
Use the portable [installation instructions](Installation.md) in a new environment.

Its base is `1d91b7de03adb2332073420f440eeefdd2e2e837`.
[PatientLevelPrediction-transfer.patch](PatientLevelPrediction-transfer.patch)
is a snapshot of that revised worktree's changes against HEAD, including staged
and unstaged changes. CI applies this snapshot to the recorded base. It replaces
the earlier `fix-priorcoefs-transfer` proposal.

## Revised PLP scope: correctness only

The patch fixes PLP's `priorCoefs` implementation:

1. **Coefficient identity:** match supplied source coefficients to Cyclops columns
   by semantic covariate ID, rather than assuming a positional column order.
2. **Input isolation:** add synthetic source columns in separate storage, without
   appending them to the caller's covariate table.
3. **Refit state:** pass fixed and starting source coefficients into every CV
   refit, so the source contribution is not reset or dropped.
4. **Prediction coefficients:** combine fixed source slopes and fitted target
   corrections by covariate ID, for both CV and final prediction.
5. **Source-only features:** retain source covariates absent from target training,
   while respecting explicit include/exclude covariate selection.
6. **Patient identity:** align CV fold assignments and predictions by `rowId`.

These are separate from the support/intercept indexing bug in the historical
transferLearning glmnet scripts. No Cyclops changes are required.

The revised patch does **not** add a `useCrossValidation` constructor argument,
change PLP's public model-settings representation, add an RDS precision sidecar,
or include the earlier prediction-resource cleanup proposal.

## Why the runner can fit a candidate without a new API

The runner calls the public `PatientLevelPrediction::fitPlp()` entry point with
prepared `trainData`. Inspection of `R/Fit.R` confirms that it passes those data
to the model fitter; it does not reconstruct the folds with a split constructor.
In `R/CyclopsModels.R`, the lasso fitting path enables variance tuning and CV
prediction generation only when `max(trainData$folds$index) > 1`.

The runner's `fitPreparedVariance()` helper therefore makes this local list copy:

```r
fitData <- trainData
fitData$folds <- data.frame(
  rowId = trainData$labels$rowId,
  index = rep(1L, nrow(trainData$labels))
)
fit <- PatientLevelPrediction::fitPlp(
  fitData, modelSettings, analysisId = "transfer", analysisPath = NULL
)
```

`modelSettings` comes from the unchanged `setLassoLogisticRegression(variance=...)`
constructor. There is no setting injection, private fitter call, monkeypatch,
or vendored fitting implementation. `createDefaultSplitSetting(nfold=1)` is not
used: that constructor requires more than one fold.

The all-ones table is only a fitting control for this local PLP call. The runner's
real held-out assignments remain separately owned by `tuneModel()` and are saved
in the experiment split artifacts. Every candidate fit is checked for:

- fitted variance equal to the supplied candidate;
- no CV prediction rows;
- no internal CV object or hyperparameter-search results.

## External validation and normalization still belong in the runner

PLP's `getCV()` operates on already-preprocessed input. It cannot replace the
runner's external loop, which must perform these steps separately for each fold:

1. Copy only inner-training patients' raw covariates.
2. Estimate normalization on that copy.
3. Convert source slopes into those training-fold units.
4. Fit the candidate variance through the single-fold PLP call above.
5. Apply that fitted preprocessing and model to held-out raw covariates.
6. Select the variance using held-out log loss.
7. Recompute preprocessing and source conversion on the complete final training
   sample, then refit at the selected variance through the same fitting path.

PLP already stores normalization factors under
`model$preprocessing$tidyCovariates$normFactors`. With multiplicative normalization,
source slopes are converted as `sourceBeta * targetMax / sourceMax`, keyed by
covariate ID. A source-only feature retains its source scale and slope. The source
intercept is excluded; the target intercept is freely fitted. The fitted model is:

```
logit(p) = targetIntercept + sum_j xTarget[j] * (sourceBetaInTargetUnits[j] + delta[j])
```

Only target corrections `delta` are penalized. This differs from merely using
source coefficients as a numerical starting point for ordinary target-only lasso.

## Behavioral readiness and backend provenance

`validateExperiment()` runs a small deterministic synthetic fit through the same
public path. It checks a variance below the internal search limits, correctly
assigned source slopes, a source-only feature's prediction contribution, and an
unchanged caller covariate table. It does not use constructor-argument presence or
a package version as a readiness test.

The experiment manifest continues to record package versions and a fingerprint
of the actual loaded PLP functions. Different uncommitted builds may share a
version. The behavioral check establishes only the exercised runner contract;
it does not replace the revised PLP regression suite or full upstream validation.

## JSON precision remains a separate unresolved issue

The revised patch does not change PLP model serialization. Existing JSON output
can round coefficients and normalization factors, so saved/reloaded models may
produce slightly different predictions. In particular, an interrupted run that
reloads a source model for unfinished jobs is not guaranteed to be numerically
equivalent to an uninterrupted run using the original in-memory source model.

This change does not introduce a sidecar or otherwise solve persistence precision.
Tests distinguish exact in-memory/raw-data-cache predictions from model JSON
round trips; the latter are not claimed to be exact. A separate persistence fix
and its compatibility review remain necessary for exact restart equivalence.

## Installation and tests

After following [Installation.md](Installation.md), run tests from the clone in
an R environment with the development dependencies installed:

```sh
Rscript --vanilla -e 'devtools::test()'
```

Runner regression tests cover candidates outside internal CV bounds, no internal
CV output, unchanged settings/covariates/validation assignments, per-fold scale
conversion, held-out log-loss selection, final full-sample preprocessing/refitting,
and source-only prediction effects. See [Validation.md](Validation.md) for results
and the remaining database/full-suite validation limits.
