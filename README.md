# TransferLearning

An R experiment runner for patient-level transfer learning across OMOP databases,
using PatientLevelPrediction (PLP) and Cyclops. This implementation is a proof of
concept and requires the [PLP changes](extras/UpstreamRequirements.md) in this
repository's upstream patch. Those changes are not yet published upstream.

## Installation and execution

Clone the `plp-cyclops-pilot` branch and follow the portable
[installation instructions](extras/Installation.md). They install dependencies,
apply the bundled PLP correctness patch to a pinned upstream commit, install both
packages and run a synthetic backend preflight. `R CMD INSTALL .` alone does not
install dependencies or correct PLP.
The active tree contains only the new runner. The original glmnet code, models,
and environment lockfile are preserved at the `legacy-glmnet` Git tag.

Copy [the study configuration](inst/examples/config.R), supply frozen cohort
sets and local database connection details, and run:

```sh
Rscript --vanilla extras/runExperiment.R /path/to/study-config.R
```

The registry maps database names to connection details, CDM and writable cohort
scratch schemas, and immutable snapshot identifiers. The runner creates tables
prefixed `tl` in the scratch schema. Credentials are not written to the experiment
manifest. Keep the registry outside version control. Prepared raw PLP data can
also be supplied through `runExperiment(..., preparedData = ...)`; see its help.

## Experiment design

- Five methods: target-only lasso, frozen source, intercept recalibration,
  intercept-and-slope recalibration, and PLP `priorCoefs` transfer. Transfer fits
  penalized target corrections around fixed source slopes and a new intercept.
- A fixed stratified 25% patient holdout per database/problem. One eligible index
  per patient; binary outcomes. The same patient holdout is excluded when a
  database serves as a source. Source models use the full remaining development
  pool, fixed across all target sample sizes and repetitions.
- Ten nested stratified training-sample sequences at 25, 50, 100, 150, 200,
  500 and 1000 outcome-positive training patients. Controls are sampled in the
  development-pool ratio (rounded); the test set stays fixed. Oversized requests and
  samples below the case/control threshold are explicitly skipped.
- Five inner folds, fold-specific normalization, and log-loss tuning over prior
  variances `10^(-6:6)`. Expand a selected boundary by three decades once; record
  any remaining boundary optimum. Target-only and transfer are tuned separately
  on identical folds. A local all-ones fold table passed directly to `fitPlp()`
  disables internal tuning through existing PLP behavior; the real validation
  folds remain separate. No new PLP constructor argument is required. Final models
  refit on the complete sampled training set.
- Multiplicative normalization only: `minFraction = 0`, `removeRedundancy = FALSE`.
  Source slopes are converted as `betaTarget = betaSource * maxTarget / maxSource`,
  matching semantic covariate IDs. A source feature absent from target training
  retains its source scale and slope. Missing source scales fail explicitly.
- Standard OMOP features (age, gender, conditions, drugs, procedures, observations)
  versus frozen phenotype cohorts plus demographics. Defaults use days -365 to -1.
  Phenotype ascertainment must be reviewed for prediction-time availability:
  an earlier cohort start alone does not rule out future information in its logic.

## Outputs and interpretation

The output folder holds configuration/backend fingerprints, patient splits,
source and target PLP models, per-fold tuning losses, predictions, job status,
metrics, and paired bootstrap intervals. Resume reuses completed jobs under the
same manifest; failed jobs are retried. A lock prevents concurrent writers.
PLP JSON model serialization still rounds numerical values; reloading source
models for unfinished jobs can change predictions slightly. Exact interrupted-run
equivalence remains unresolved and is separate from fixed-variance tuning.
Database snapshots and prepared input contents must remain immutable within a
run. Use a new output folder for changed data or settings.

`collectExperimentResults()` reads status, metrics, and intervals.
`plotLearningCurves()` plots repetitions and their mean; `plotTransferEffects()`
plots paired improvements against target training size. Metrics include AUROC,
AUPRC (average precision with ties grouped), log loss, Brier score, and calibration.
Bootstrap differences are positive for improvement: transfer minus target-only
AUROC, target-only minus transfer log loss. These intervals condition on the
trained models and shared test set; training repetitions are not independent
replications of the test population. Assess a data-capacity effect using paired
performance differences against target training size within each database/problem
and feature profile; do not pool repeated test predictions as independent data.

Execution is serial with configurable Cyclops threads. Large runs can be divided
into separate output folders by database pair/problem. All patient-level artifacts
stay local; sharing aggregate results requires the study's disclosure review.
Live database extraction needs validation in the destination environment.

## Development

```r
roxygen2::roxygenise()
devtools::test()
devtools::check()
```

Tests exercise the full synthetic two-database runner and the data-isolation,
normalization, split, persistence, and resume contracts. See
[upstream requirements](extras/UpstreamRequirements.md) for backend regression
coverage and remaining full-suite validation.

To inspect the original experiments without changing this working directory:

```sh
git worktree add ../transferLearning-legacy legacy-glmnet
```

The tag preserves the original committed state, before the new runner and local
diagnostic edits. Historical glmnet bundles are not imported into the new runner.

The AF–stroke pilot uses phenotypes plus age and sex, Optum EHR → MDCR, and
training-event budgets. See [pilot instructions](extras/AfStrokePilot.md).
