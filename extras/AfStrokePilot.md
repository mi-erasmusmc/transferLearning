# AF–stroke pilot

Use `inst/examples/afStrokePilot.R` as a standalone entrypoint. Site-specific
inputs are at the top; extraction and fitting are called at the bottom.
The initial pair is **Optum EHR → MDCR**. Both are protocol development databases;
the protocol does not mandate this pair. This choice is a pilot hypothesis, not
an assertion about actual eligible event counts. Keep the source development
population fixed while varying target training events. This isolates target-data
capacity; it is not the protocol's full federated experiment.

## Frozen problem and features

AF target and inpatient non-hemorrhagic stroke outcome JSON come from the
[FederatedLearningCurves protocol](https://github.com/ohdsi-studies/FederatedLearningCurves/blob/8f2b0866e11c717fa681e62c91dcf2bbec21ad61/docs/index.html),
commit `8f2b0866e11c717fa681e62c91dcf2bbec21ad61`. The rendered JSON used typographic
quotes, converted to JSON quotes; SQL is regenerated with CirceR. No embedded SQL
is copied. Bundled files are `inst/pilot/af-target.json` and `stroke-outcome.json`.
The pilot explicitly uses 365 days of prior observation, first AF index, excludes
stroke at offsets greater than −365 and before day 1 (PLP’s strict lookback boundary),
and predicts days 1–365. Observation through the end of the risk window is required for noncases
(`minTimeAtRisk=364`, PLP’s end-minus-start convention for days 1–365);
`includeAllOutcomes=TRUE` retains cases observed before earlier follow-up ends.
These are explicit pilot choices where the protocol does not fully specify eligibility.

The earlier PhenotypeLibrary 3.37.0 definitions, IDs 1152–1215, are frozen in
`phenotypes-original.rds`. The derived frozen set has 51 phenotypes. The conservative pilot config omits
the two stroke predictors discussed below, using 49 phenotypes plus age and sex.
Antibiotic groups 1201–1214 are removed and replaced by study-local ID **900001**:
any drug exposure in their union of concept expressions (descendant/mapping flags
retained). Unlike the old groups' first exposure plus drug-era continuation, this
feature uses **all qualifying exposure starts**, with zero-day duration, so it
means an antibiotic exposure start during days −365 through −1. No group-specific
antibiotic predictors remain. `extras/freezePilotPhenotypes.R` rebuilds the files;
`phenotypes.csv` lists the features. Rebuilding requires PhenotypeLibrary 3.37.0
installed separately; running the package or pilot does not. Non-antibiotic
definitions are unchanged.

## Predictor timing

The structural audit found no positive-day start windows or required post-index
observation in the recovered predictor definitions. However, stroke predictors
1155 and 1156 use visit **end** dates to establish that the diagnosis is within
an inpatient visit; their end windows extend forward without a bound. A pre-index
feature window alone does not prove availability of those visit records at
prediction time. The pilot omits these two predictors. To restore them, confirm the intended
completed-record retrospective interpretation or revise their definitions to use
information available at index.
Do not simply declare every phenotype safe based on its extraction window.

FeatureExtraction's cohort-based binary features use interval overlap, not only
cohort entry dates. The unchanged chronic and drug-era phenotypes retain that
meaning; they are not all “new diagnosis within the last year” indicators.
The pilot retains those established interval semantics for the other predictors.
The frozen original definitions remain available for a later study-specific review.

## Run sequence

1. Install PLP develop with the merged correctness fixes, then install this
   package using [Installation.md](Installation.md). The runner automatically
   checks backend behavior and records implementation fingerprints.
2. Copy `inst/examples/afStrokePilot.R` into the study directory. Fill in the
   `sourceName` / `targetName`, source/target CDM schemas (`catalog.schema`),
   `sourceCohortTable` / `targetCohortTable`, writable cohort/temp schema and output
   folder at the top. Snapshot IDs are not required. The example shares
   one Databricks connection between the two CDM schemas. Supply host, HTTP path,
   token and JDBC driver directory through the indicated environment variables,
   or adapt the connection block to your site's existing authentication setup.
   No site-configuration RDS file is needed. Bundled phenotype definitions still
   load from the package's frozen RDS artifact.
3. In Windows R/RStudio, set the working directory to the study folder and run
   `source("afStrokePilot.R")`. From a terminal, `Rscript --vanilla afStrokePilot.R`
   is also available if Rscript is on PATH. Do not pass this entrypoint to `extras/runExperiment.R`: it already
   calls the runner. The generic CLI remains available for configuration-only
   scripts such as `inst/examples/config.R`.

The specified cohort tables contain the target/outcome cohorts. Each database's
phenotype table defaults to its cohort table name plus `_phenotypes`; the registry
can override this with `phenotypeCohortTable`. Names are unqualified table names;
`cohortDatabaseSchema` supplies the catalog/schema. For each table, the runner
checks existence: existing tables are read without recreation or regeneration,
while missing tables are created and populated. Existing tables must already
contain the intended complete cohorts (including the correct cohort IDs); a table's
existence is not a content/version check. If a generation attempt is interrupted,
inspect its tables before retrying. Use fresh table names to regenerate changed
definitions, and a new output folder when data change. Database schemas/table
names are recorded in cache fingerprints without connection credentials.

The entrypoint performs live extraction and fitting. For an extraction-only
feasibility check, execute its configuration section, then call
`prepareExperimentData(settings, databaseRegistry)` instead of the final
`runExperiment()` call. Returned cache paths contain `population.rds` for checking
eligible case/control counts. Cohort/feature SQL executes on Databricks; extracted
patient-level caches and Cyclops fits reside on the machine running R, under the
specified output folder.

The prototype requests 25, 50, 100, 150, 200, 500, 1000 **target training events**,
three nested repetitions and three inner folds. Controls retain the development
pool's event/control ratio up to rounding. The stratified 25% test set is fixed.
Oversized requests are skipped, not silently replaced. Plots default to realized
training events; metrics also record patients, controls and training prevalence.
PLP itself recommends `trainEvents` but converts these into fractions; this runner
samples exact event counts. Events are a capacity measure, not a guarantee of
equal statistical power or equal AUPRC across populations.

After the smoke pilot, increase to 20 repetitions and five inner folds for the
main comparison. Phenotypes plus demographics deliberately differ from the
protocol's high-dimensional conditions/drugs and visit-count profile.
Known PLP model-JSON precision loss remains unresolved; avoid relying on exact
resumed-versus-uninterrupted equality. See `UpstreamRequirements.md`.

No production database extraction or model fitting has been run locally.

The priorCoefs comparator drops source coefficients absent from each target
training fold, matching PLP develop. Overlap is recalculated for the final refit.
Frozen-source/recalibration comparators keep the original source feature set.
Dropped IDs are saved per tuning fold and for the final fit; see
[UpstreamRequirements.md](UpstreamRequirements.md).

## Post-run transfer diagnostics

After installing the updated package and restarting R, run this separately from
`afStrokePilot.R` (do not rerun the experiment):

```r
outputFolder <- "C:/path/to/completed/af-stroke-pilot"
TransferLearning::summarizeTransferDiagnostics(outputFolder)
```

This reads the original manifest, saved source/transfer models, splits, predictions,
and cached target data. It uses the original cache fingerprints, so installing an
updated runner does not invalidate this read-only analysis. No database credentials,
connection, extraction or refitting is needed. Keep the complete experiment folder
in production. If the experiment used `preparedData` rather than the package cache,
supply the same prepared-data list to this function.

Only the CSVs in `outputFolder/transfer-diagnostics/` are intended as aggregate
exports; review them under local disclosure rules, including small counts:

- `coefficients.csv`: counts of eligible, nonzero, changed, zeroed and sign-reversed
  coefficients, separately for source-selected and new target predictors. Magnitudes
  are absolute coefficients/changes multiplied by target-training predictor SD,
  including implicit zeros. Medians include all eligible predictors in each group.
- `contributions.csv`: means and SDs of held-out log-odds contributions from retained
  source coefficients, adjustments to source-selected predictors, new target
  coefficients, total corrections and the final predictor (excluding the intercept).
- `correlations.csv`: correlations between these components; constant components
  have missing correlations. These are not independent contributions or percentages.
- `ablations.csv`: AUROC, log loss and Brier score for the reloaded full model and
  models with new-target, source-adjustment or all correction components removed.
  All use the final target intercept and no refitting. Positive AUROC change is an
  improvement over the full model; positive log-loss/Brier change is deterioration.
  Removing all corrections is not the frozen-source or intercept-recalibrated model.
- `verification.csv`: training/test counts, selected variance, boundary selection,
  source feature dropping, reconstruction checks, and drift versus original saved
  predictions. This does not certify convergence. Material drift should be resolved
  before interpreting ablations as representing the original fitted model.
- `provenance.csv`: original manifest hash, original/current backend fingerprints,
  current runner fingerprint, PLP version and coefficient classification tolerance.

The default tolerance is `1e-6` in target-normalized coefficient units. Classification
of target nonzero coefficients and changes can depend on this tolerance and PLP's
JSON rounding. Source selection uses exactly nonzero saved source coefficients. The
continuous prediction decomposition retains all coefficient values, including small
ones. The function checks its reconstructed probabilities against PLP prediction
from the loaded model and stops if they differ by more than `1e-6`. Drift from the
original in-memory model is reported separately; the known serialization precision
issue is not repaired by these diagnostics. No covariate identities, individual
coefficients, patient IDs or individual predictions are written to these CSVs.

### Check whether the source model is genuinely dense

From the updated repository clone, run in a fresh R session:

```r
source("extras/sourceModelDiagnostics.R")
sourceModelDiagnostics("C:/path/to/completed/af-stroke-pilot")
```

No package reinstall is required for this standalone script if the pilot's PLP
and other dependencies are already installed. It requires the original source
cache and saved source model/tuning files. It never connects to Databricks or refits.
Share only the reviewed aggregate CSVs in `source-diagnostics/`:

- `summary.csv`: training events and predictor counts; selected/fitted variance;
  stored fit status; whether selection reaches the strongest/weakest tested penalty.
  Boundary checks use a numeric tolerance, alongside the original recorded flag.
- `tuning.csv`: weighted validation log loss at every candidate variance, failed-fold
  counts, fold loss range and differences from the selected candidate. Smaller
  variance means stronger regularization. Fold loss ranges are not confidence intervals.
- `thresholds.csv` and `quantiles.csv`: distributions of absolute coefficients in
  normalized units and per source-training predictor SD (implicit zeros included).
  Values above thresholds are counts, not lists of predictor identities. Thresholds
  are descriptive, not significance tests or universally meaningful effect cutoffs.
- `provenance.csv`: original manifest/backend identifiers and script fingerprint.

The intercept is excluded from counts and magnitudes. Saved coefficients reflect
PLP JSON precision. `savedStatusOK` reports the stored Cyclops status; it is not an
independent convergence assessment. A dense model with good held-out loss at an
interior variance is compatible with plentiful source data and a curated feature
set; density alone does not establish that every predictor matters. The script
exports neither patient-level data nor individual coefficients or predictor IDs.

## Extending a completed learning curve

Update/reinstall TransferLearning and restart R. In your existing copy of
`afStrokePilot.R`, keep the same `outputFolder` and site inputs, then change:

```r
trainingEvents <- c(25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000)
repetitions <- 5L
```

Run the original script again. `runExperiment()` resumes by default. Existing
extraction caches, source models, completed target jobs, and fixed test partitions
are reused. With complete caches, cohort generation and feature extraction are
not called. Only new budget/repetition combinations and failed or missing methods
are run. Budgets exceeding the available development events are recorded as
skipped. Successful methods in a partially failed job are reused when their saved
predictions and metrics are available. Keep the entire local output folder.

The original seven budgets and three repetitions become twelve budgets and five
repetitions: 39 additional jobs, if all budgets are feasible. Repetitions 1–3 keep
their original samples; added budgets use nested samples within each repetition.
All repetitions share the original held-out test population. Their variability
therefore describes training-sample variability, not uncertainty across databases
or independent test sets.

Only additive budget/repetition changes are supported in the same folder. Changes
to features, cohorts, preprocessing, tuning, methods, seed, bootstrap settings,
database identity, dependencies or fitting implementation require a new folder.
Reuse assumes the underlying database contents have not changed; use a new folder
for a refreshed dataset. Credentials can change without invalidating saved work.

The runner recognizes the earlier pilot builds with merged PLP behavior and
migrates their job identifiers without moving or rewriting completed jobs. Unknown
older implementations are rejected. `manifest.rds` retains the original cache and
source provenance; `resume-manifest.rds` records the latest requested grid and
`runs/` retains the manifests for individual extensions. Summary CSVs are refreshed
to include all jobs. Diagnostics can be rerun after the extension.

New transfer fits load the saved source model. PLP's existing JSON coefficient
precision limitation still applies; this change does not provide lossless model
serialization or guarantee bitwise identity to one uninterrupted run.

## Comparing native PLP/Cyclops tuning with the saved experiment

After the experiment finishes, pull the repository update and source the standalone
script. It uses the installed TransferLearning package; no reinstall is needed for
this script-only addition. From the repository directory in R/Positron:

```r
source("extras/comparePlpAutoCv.R")
comparison <- comparePlpAutoCv(
  outputFolder = outputFolder,
  trainingEvents = c(25, 200, 1000),
  repetitions = 1:3
)
```

This initial selection compares nine saved jobs, each with target-only and transfer
models under two variants (36 native model-development calls). Each call includes
Cyclops automatic tuning, its final fit, and PLP CV prediction refits. Omit the
`trainingEvents` and `repetitions` filters to compare all completed jobs. Rerunning
the same call skips completed comparisons; failed comparisons are retried. The
script rejects a running experiment lock and a different PLP backend fingerprint.
No database connection, extraction, or source-model refitting occurs.

The two variants are:

- `matchedPreprocessing`: retain the pilot's normalization, `minFraction = 0`, and
  `removeRedundancy = FALSE`, but estimate preprocessing once using the complete
  training sample and let PLP/Cyclops tune internally.
- `plpDefaults`: use `createPreprocessSettings()` defaults, including frequency and
  redundancy filtering, followed by native PLP/Cyclops tuning.

Both use the original features (including age), exact outer training/test patients,
original fold count, seed and thread count, and a starting variance of 0.01. The
source model is the original saved model, held fixed to isolate changes in target
fitting. Source coefficients are matched by ID and converted once to each prepared
training sample's units; this is necessary for age and has no effect on binary
predictors with unit normalization in both models. This is a comparison of target
fitting workflows, not a complete rerun of source development using PLP defaults.

Cyclops automatic tuning constructs its own seeded folds. The saved fold labels
are supplied to PLP for its CV prediction refits, but do not force the automatic
variance search to use the runner's exact inner assignments. The comparison
therefore changes tuning search, inner assignments and preprocessing scope;
it does not isolate the search algorithm alone. PLP defaults also change feature
filtering. All evaluation uses the original untouched outer test patients.
The `lowerLimit`/`upperLimit` settings apply to grid search, not auto-search.

Results are under `outputFolder/plp-auto-cv/aggregate/`:

- `comparisons.csv`: reference and native variances, nonzero predictor counts,
  rescaled/dropped source counts, AUC, AUPRC, log loss, Brier score, calibration
  diagnostics, prediction differences and elapsed time. Every `difference_*` is
  native minus reference: positive AUC/AUPRC and negative loss/Brier differences
  indicate improvement. Prediction correlation is undefined for constant predictions.
- `provenance.csv` and `packages.csv`: source strategy, backend/script fingerprints,
  starting variance and dependency versions.

Only these aggregate CSVs are intended for review/sharing, subject to local count
rules. No patient IDs, individual predictions, coefficients or covariate IDs are
exported. Errors are printed locally; failed rows contain only a status. The
`local/` subfolder holds resume records. Original experiment artifacts are read-only.

Reference predictions are the original saved predictions. Separate reference-reload
drift columns quantify PLP JSON precision effects; the native source is also loaded
from its saved representation. Compare predictive performance and prediction
agreement, not just selected variance: several variances can produce the same
model. Agreement is empirical, not a claim of mathematical or statistical equivalence.
Calibration fields are diagnostic measurements, not additional recalibrated models.

To check sensitivity to the automatic search's starting point, use a separate folder:

```r
comparePlpAutoCv(
  outputFolder,
  comparisonFolder = file.path(outputFolder, "plp-auto-cv-start-1"),
  trainingEvents = c(25, 200, 1000), repetitions = 1L,
  variants = "matchedPreprocessing", startingVariance = 1
)
```
