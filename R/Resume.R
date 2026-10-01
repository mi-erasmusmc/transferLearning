# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

# Full namespace fingerprints remain provenance. This narrower contract determines
# whether saved scientific work can be reused after reporting/orchestration changes.
experimentScienceFingerprint <- function() {
  names <- c("subsetTrain", "sourceInTargetUnits", "fitVariance", "fitPreparedVariance",
    "assertFixedVarianceFit", "predictValues", "logLoss", "tuneModel", "fitRecalibration",
    "applyRecalibration", "performance", "pairedIntervals", "makePartition", "sampleNested",
    "makeFolds", "seedFor", "withSeed", "classCounts", "openInput", "ensureCohortTable",
    "cohortTableName", "databaseIdentity", "prepareExperimentDataFromManifest")
  definitions <- lapply(names, function(name) {
    f <- get(name, envir = environment(experimentScienceFingerprint))
    list(formals = deparse(formals(f)), body = deparse(body(f)))
  })
  digest::digest(stats::setNames(definitions, names), algo = "sha256")
}

# These released builds used the same fitting, sampling and extraction semantics.
# Earlier source-only-retention builds are intentionally not supported.
legacyResumeFingerprints <- function() c(
  "07dc42ab4f8d98fcf149ec982c8b458a52a6e971360efb382b6bdbbb93b5e307", # e99408a
  "2c52bb34c79d4a0187a9db765096607b8ae4474a825210c8aa420d6b7a3cb8fe", # 2b80529
  "e63e86ee93b60862f751d125f756516a6aa51242976fc1144d01e74ad0ebf880"  # 74c1a4c/e3a6ffe
)

assertExperimentExtension <- function(previous, current) {
  fail <- function(what) stop("Manifest changed: ", what, "; use a new output folder")
  old <- previous$settings; new <- current$settings
  oldCurve <- old$learningCurve; newCurve <- new$learningCurve
  if (!all(oldCurve$trainingBudgets %in% newCurve$trainingBudgets) ||
      newCurve$repetitions < oldCurve$repetitions) fail("only additional budgets/repetitions may be resumed")
  old$learningCurve$trainingBudgets <- new$learningCurve$trainingBudgets <- NULL
  old$learningCurve$repetitions <- new$learningCurve$repetitions <- NULL
  if (!identical(old, new)) fail("scientific settings differ")
  if (!identical(previous$databases, current$databases)) fail("database identities differ")
  packagesOld <- previous$packages; packagesNew <- current$packages
  packagesOld <- packagesOld[names(packagesOld) != "TransferLearning"]
  packagesNew <- packagesNew[names(packagesNew) != "TransferLearning"]
  if (!identical(packagesOld, packagesNew)) fail("dependency versions differ")
  if (!identical(previous$backend, current$backend)) fail("PLP implementation differs")
  if (is.null(previous$reuseContract)) {
    if (!previous$implementation %in% legacyResumeFingerprints() ||
        !identical(current$reuseContract$science, legacyCompatibleScienceFingerprint())) {
      fail("unrecognized legacy implementation")
    }
  } else if (!identical(previous$reuseContract, current$reuseContract)) fail("runner fitting/extraction implementation differs")
  invisible(TRUE)
}

# Frozen migration contract, not a dynamically computed bypass. Future scientific
# changes must not silently approve legacy artifacts.
legacyCompatibleScienceFingerprint <- function() "9ef09b8843deca00c9871bbd41d166aeb2eed76345de40ede259acbbc050a305"

experimentJobKey <- function(job) {
  fields <- c("problemId", "profileId", "sourceId", "targetId", "repetition", "trainingBudget", "budgetUnit")
  if (!all(fields %in% names(job)) || nrow(job) < 1) stop("Missing job identity")
  values <- lapply(job[1, fields, drop = FALSE], as.character)
  digest::digest(values, algo = "xxhash64")
}

indexExperimentJobs <- function(folder) {
  files <- list.files(file.path(folder, "jobs"), "^result.rds$", recursive = TRUE, full.names = TRUE)
  paths <- list()
  for (file in files) {
    status <- readRDS(file)$status
    keys <- vapply(seq_len(nrow(status)), function(i) experimentJobKey(status[i, , drop = FALSE]), character(1))
    if (length(unique(keys)) != 1) stop("Inconsistent saved job identity: ", dirname(file))
    key <- keys[1]
    if (!is.null(paths[[key]])) stop("Duplicate saved job identity; resolve duplicate results before resuming")
    paths[[key]] <- dirname(file)
  }
  paths
}

jobIsComplete <- function(result, methods) {
  status <- result$status
  if (nrow(status) == 1 && status$method == "all" && status$status == "skipped") return(TRUE)
  all(methods %in% status$method[status$status == "completed"])
}

cachedExperimentInputs <- function(settings, databaseRegistry, manifest) {
  prepareExperimentDataFromManifest(settings, databaseRegistry, manifest)
}
