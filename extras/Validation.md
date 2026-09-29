# Validation of the runner with the revised PLP correctness build

Backend: the uncommitted worktree
`/home/egill/github/PatientLevelPrediction/fix-priorcoefs-correctness`, based on
`1d91b7de03adb2332073420f440eeefdd2e2e837`, installed into the ignored `.library/`.
This supersedes the earlier validation against `fix-priorcoefs-transfer`.
The PLP worktree was inspected and installed, not modified by the runner change.

Latest local results: **R CMD check passed with 0 errors, 0 warnings, and 0 notes**
(including installed-package tests and vignette rebuilding). The targeted runner
run passed **67 assertions**, with no failures, warnings, or skips. Both packages
installed successfully into `.library/`.

## Targeted coverage

`tests/testthat/test-fixed-variance.R` exercises the actual public `fitPlp()` path:

- Candidate variances 0.123456 and 50 with internal search limits [1, 2], for
  both target-only and transfer fits. Fitted variance must equal the candidate.
- No internal CV object, no CV prediction rows, and no hyperparameter-search rows.
- No changes to supplied model settings, prepared covariates, or real fold tables.
  PLP's existing settings normalizer can still add its usual modelName and
  requiresDenseMatrix fields; the runner adds no new model-settings field.
- Distinct training-fold maxima (10 and 20) and a source scale of 40, verifying
  per-fold source conversion, not merely per-fold patient selection.
- Held-out losses matching independently evaluated fold models, selection by
  weighted held-out log loss, then a refit with full-sample normalization and the
  selected variance. Final coefficients match a separate refit at that variance.
- Behavioral readiness and rejection of unexpected retuning/internal CV output.

`test-absent-feature.R` additionally checks the expected log-odds contribution of
a source feature absent from target training but present at prediction time.
Existing tests cover paired multi-database/profile orchestration, patient splits,
nested sampling, raw cache reload, reports, and completed-job resume.

The readiness probe was also run in a separate R session against the unpatched
installed PLP. It correctly rejected that backend because source coefficients were
not retained by covariate ID. The revised build passes the same behavioral probe.

The revised PLP `test-priorCoefsTransfer.R` was run independently of its shared
suite setup. Its focused tests pass. The portable patch passes `git apply --check`
against the recorded base. CI now uses this revised patch, without the earlier
API addition, precision sidecar, or prediction-resource change.

## Persistence scope and remaining validation

Exact model JSON round-trip tests from the earlier sidecar proposal are no longer
claimed to pass. The raw-data-cache reload test uses the same in-memory model and
still requires exact predictions. JSON coefficient/normalization rounding and
exact restart equivalence remain separate unresolved issues; see
[UpstreamRequirements.md](UpstreamRequirements.md).

The synthetic profile jobs reuse toy features to exercise orchestration, not
clinical phenotype definitions. Live CDM extraction and destination drivers still
need study-environment validation. Windows/macOS CI was not run locally.

The full PLP suite is not certified by this change. An earlier full-suite attempt
failed in shared `reduceData()` setup with `slice_max(NULL)`; this turn ran the
revised focused regression file, not that full suite.

Current local logs:

- `/tmp/transfer-fixed-variance.log`: targeted runner tests.
- `/tmp/transfer-backend-negative.log`: rejection of the unpatched backend.
- `/tmp/plp-correctness-focused.log`: revised upstream focused tests.
- `/tmp/transfer-correctness-check.log`: runner package check.

These temporary paths are local evidence, not portable experiment artifacts.

## Event-budget AF–stroke pilot update (2026-09-29)

- Full runner `R CMD check --no-manual`: **0 errors, 0 warnings, 0 notes**,
  4m 16s, against the same corrected PLP build.
- Nineteen focused assertions cover exact event counts, proportional controls,
  nested subsets, full-pool/oversized budgets, unchanged caller RNG, and the
  antibiotic merge's concept union and unchanged non-antibiotic JSON.
- End-to-end integration now runs 25-event jobs across two feature profiles and
  checks realized events and event-axis plots; oversized 1000-event jobs skip.
  Existing pairing, resume, input immutability and fixed-variance tests pass.
- The installed AF–stroke example compiles both frozen cohort JSON definitions,
  uses 49 predictors plus demographics with stroke predictors provisionally
  omitted, and passes the backend behavioral check. Its grid contains 21 samples.
- Installed the updated runner in `.library/`. No production database was accessed.

Logs: `/tmp/transfer-event-pilot-check.log`, `/tmp/pilot-unit.log`,
`/tmp/pilot-installed-preflight.log`, `/tmp/transfer-event-pilot-install.log`.

## Publication installation check

Cloned PLP afresh from GitHub, checked out the pinned base, applied the bundled
patch successfully and installed it and the runner into a separate temporary R
library. The new `extras/checkBackend.R` synthetic readiness check passed against
those installations. Existing system R dependencies were reused; installation of
all dependencies on a bare production host and live Databricks extraction remain
untested. The standalone pilot entrypoint also passed configuration and missing-
input smoke checks without opening a database connection.
