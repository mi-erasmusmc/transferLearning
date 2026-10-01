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
