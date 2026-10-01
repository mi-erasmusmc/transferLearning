# Source this file after the experiment finishes, then call comparePlpAutoCv().
# Uses local caches only. Share only the CSVs in the aggregate subfolder.
comparePlpAutoCv <- function(outputFolder,
    comparisonFolder = file.path(outputFolder, "plp-auto-cv"),
    trainingEvents = NULL, repetitions = NULL,
    variants = c("matchedPreprocessing", "plpDefaults"), startingVariance = 0.01) {
  if (dir.exists(file.path(outputFolder, ".runner-lock")))
    stop("Wait for the experiment runner to finish before comparing")
  stopifnot(length(startingVariance) == 1L, is.finite(startingVariance), startingVariance > 0,
    length(variants) > 0L, !anyDuplicated(variants),
    all(variants %in% c("matchedPreprocessing", "plpDefaults")))
  tl <- function(name) getFromNamespace(name, "TransferLearning")
  manifest <- readRDS(file.path(outputFolder, "manifest.rds"))
  settings <- manifest$settings
  backend <- tl("functionFingerprint")("PatientLevelPrediction")
  if (!identical(backend, manifest$backend))
    stop("PLP differs from the original run; use the original backend for this comparison")
  dir.create(comparisonFolder, recursive = TRUE, showWarnings = FALSE)
  lock <- file.path(comparisonFolder, ".comparison-lock")
  if (!dir.create(lock, showWarnings = FALSE)) stop("Comparison is already running or its lock remains")
  on.exit(unlink(lock, recursive = TRUE))
  contract <- list(originalManifestHash = manifest$hash, plp = backend,
    packages = tl("packageVersions")(), runner = tl("functionFingerprint")("TransferLearning"),
    script = digest::digest(lapply(list(comparePlpAutoCv, autoCvFit, autoCvCompareOne),
      function(f) list(formals = deparse(formals(f)), body = deparse(body(f)))), algo = "sha256"),
    startingVariance = startingVariance)
  contractPath <- file.path(comparisonFolder, "contract.rds")
  if (file.exists(contractPath) && !identical(readRDS(contractPath), contract)) {
    previous <- readRDS(contractPath)
    changed <- names(contract)[!vapply(names(contract), function(n) identical(previous[[n]], contract[[n]]), logical(1))]
    stop("Comparison settings/backend changed (", paste(changed, collapse = ", "), "); choose a new comparisonFolder")
  }
  tl("atomicSave")(contract, contractPath)
  files <- list.files(file.path(outputFolder, "jobs"), "^result.rds$", recursive = TRUE, full.names = TRUE)
  inputs <- sources <- list()
  on.exit(for (input in inputs) Andromeda::close(input$data$covariateData), add = TRUE)
  getInput <- function(job) {
    id <- as.character(job$targetId); problem <- as.character(job$problemId)
    profile <- as.character(job$profileId)
    key <- digest::digest(list(id, manifest$databases[[id]], settings$problems[[problem]],
      settings$featureProfiles[[profile]], manifest$packages, manifest$implementation), algo = "sha256")
    if (is.null(inputs[[key]])) inputs[[key]] <<- tl("openInput")(
      file.path(outputFolder, "data", key), settings$problems[[problem]], id)
    inputs[[key]]
  }
  getSource <- function(job) {
    key <- digest::digest(list(as.character(job$sourceId), as.character(job$problemId),
      as.character(job$profileId)), algo = "xxhash64")
    if (is.null(sources[[key]])) sources[[key]] <<- PatientLevelPrediction::loadPlpModel(
      file.path(outputFolder, "sources", key, "model"))
    sources[[key]]
  }
  records <- list()
  for (file in files) {
    status <- readRDS(file)$status
    for (method in c("targetOnly", "priorCoefs")) {
      job <- status[status$method == method & status$status == "completed", , drop = FALSE]
      if (!nrow(job)) next
      stopifnot(nrow(job) == 1L)
      if (!is.null(trainingEvents) && (job$budgetUnit != "events" || !job$trainingBudget %in% trainingEvents)) next
      if (!is.null(repetitions) && !job$repetition %in% repetitions) next
      job <- job[, setdiff(names(job), c("method", "status", "reason")), drop = FALSE]
      path <- dirname(file)
      referenceFiles <- c(file, file.path(path, c("split.rds", "predictions.rds")),
        file.path(path, method, "tuning.rds"))
      if (!all(file.exists(referenceFiles))) stop("Missing saved reference artifacts: ", path)
      signature <- digest::digest(list(contract, tools::md5sum(referenceFiles)), algo = "sha256")
      for (variant in variants) {
        key <- paste(basename(path), method, variant, sep = "-")
        recordPath <- file.path(comparisonFolder, "local", paste0(key, ".rds"))
        if (file.exists(recordPath)) {
          previous <- readRDS(recordPath)
          if (!identical(previous$signature, signature)) stop("Reference artifacts changed: ", path)
          if (previous$row$status == "completed") {
            records[[key]] <- previous$row
            next
          }
        }
        message("Auto CV: ", job$targetId, " / ", job$trainingBudget, " ", job$budgetUnit,
          " / repetition ", job$repetition, " / ", method, " / ", variant)
        started <- proc.time()[["elapsed"]]
        row <- tryCatch({
          input <- getInput(job)
          source <- if (method == "priorCoefs") getSource(job) else NULL
          autoCvCompareOne(path, method, variant, input, source, settings, startingVariance)
        }, error = function(e) {
          # Keep free-text errors in the local console, outside the aggregate export.
          message("Comparison failed: ", conditionMessage(e))
          data.frame(status = "failed")
        })
        row <- cbind(job, method = method, variant = variant,
          elapsedSeconds = proc.time()[["elapsed"]] - started, row)
        tl("atomicSave")(list(signature = signature, row = row), recordPath)
        records[[key]] <- row
      }
    }
  }
  if (!length(records)) stop("No completed reference models match the requested selection")
  output <- dplyr::bind_rows(records)
  export <- file.path(comparisonFolder, "aggregate")
  dir.create(export, recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(output, file.path(export, "comparisons.csv"), row.names = FALSE)
  provenance <- data.frame(originalManifestHash = manifest$hash, plpFingerprint = backend,
    runnerFingerprint = contract$runner, scriptFingerprint = contract$script,
    plpVersion = as.character(utils::packageVersion("PatientLevelPrediction")),
    startingVariance = startingVariance,
    sourceStrategy = "reuse original saved source model",
    cvAssignments = "Cyclops automatic tuning uses its own seeded folds; same outer train/test patients")
  utils::write.csv(provenance, file.path(export, "provenance.csv"), row.names = FALSE)
  utils::write.csv(data.frame(package = names(contract$packages), version = unname(contract$packages)),
    file.path(export, "packages.csv"), row.names = FALSE)
  message("Comparison written to ", export, "; failed: ", sum(output$status == "failed"))
  invisible(output)
}

autoCvFit <- function(data, population, folds, source, settings, variant, startingVariance) {
  tl <- function(name) getFromNamespace(name, "TransferLearning")
  stopifnot(length(folds) == nrow(population), !anyNA(folds),
    identical(sort(unique(as.integer(folds))), seq_len(max(folds))), max(folds) > 1L)
  train <- tl("subsetTrain")(data, population, folds)
  on.exit(Andromeda::close(train$covariateData))
  preprocessing <- if (variant == "matchedPreprocessing")
    PatientLevelPrediction::createPreprocessSettings(normalize = TRUE, minFraction = 0, removeRedundancy = FALSE)
    else PatientLevelPrediction::createPreprocessSettings()
  train$covariateData <- PatientLevelPrediction::preprocessData(train$covariateData, preprocessing)
  prior <- NULL; dropped <- character(); rescaled <- 0L
  if (!is.null(source)) {
    converted <- tl("sourceInTargetUnits")(source, train$covariateData)
    if (nrow(converted$coefficients)) {
      prior <- converted$coefficients
      sc <- source$model$coefficients
      original <- sc$betas[match(prior$covariateIds, sc$covariateIds)]
      rescaled <- sum(abs(prior$betas - original) > 1e-12)
    }
    dropped <- converted$droppedSourceIds
  }
  modelSettings <- PatientLevelPrediction::setLassoLogisticRegression(
    variance = startingVariance, priorCoefs = prior, forceIntercept = FALSE,
    seed = settings$learningCurve$seed, threads = settings$threads)
  # Real folds enable native PLP/Cyclops CV. Do not call fitPreparedVariance(),
  # which deliberately sets every fold index to 1 for the custom-CV runner.
  model <- PatientLevelPrediction::fitPlp(train, modelSettings,
    analysisId = "autoCvComparison", analysisPath = NULL)
  if (!identical(model$model$modelStatus, "OK") || any(!is.finite(model$model$coefficients$betas)))
    stop("PLP/Cyclops fit did not return finite coefficients with status OK")
  if (!any(model$prediction$evaluationType == "CV") || NROW(model$trainDetails$hyperParamSearch) == 0L)
    stop("Native PLP CV did not produce CV predictions and tuning diagnostics")
  list(model = model, dropped = length(dropped), rescaled = rescaled,
    preprocessing = preprocessing)
}

autoCvCompareOne <- function(path, method, variant, input, source, settings, startingVariance) {
  tl <- function(name) getFromNamespace(name, "TransferLearning")
  split <- readRDS(file.path(path, "split.rds"))
  saved <- readRDS(file.path(path, "predictions.rds"))
  trainIx <- match(split$rowIds, input$population$rowId)
  testIx <- match(split$testRowIds, input$population$rowId)
  if (anyNA(c(trainIx, testIx)) || anyDuplicated(c(split$rowIds, split$testRowIds)))
    stop("Invalid or overlapping saved training/test rows")
  train <- input$population[trainIx, , drop = FALSE]
  test <- input$population[testIx, , drop = FALSE]
  ix <- match(test$rowId, saved$population$rowId)
  if (anyNA(ix) || anyDuplicated(saved$population$rowId) ||
      !identical(as.integer(test$outcomeCount), as.integer(saved$population$outcomeCount[ix])))
    stop("Saved test outcomes or row IDs differ")
  reference <- saved$predictions[[method]][ix]
  if (length(reference) != nrow(test) || any(!is.finite(reference))) stop("Missing saved predictions")
  oldTuning <- readRDS(file.path(path, method, "tuning.rds"))
  oldModel <- PatientLevelPrediction::loadPlpModel(file.path(path, method, "model"))
  reloaded <- tl("predictValues")(oldModel, input$data, test)
  auto <- autoCvFit(input$data, train, split$folds, source, settings, variant, startingVariance)
  p <- tl("predictValues")(auto$model, input$data, test)
  y <- as.integer(test$outcomeCount > 0)
  baseline <- tl("performance")(y, reference)
  current <- tl("performance")(y, p)
  differences <- current - baseline
  coefficients <- auto$model$model$coefficients
  nNonzero <- sum(coefficients$covariateIds != "(Intercept)" & abs(coefficients$betas) > 1e-6)
  row <- data.frame(status = "completed", nTrain = nrow(train), eventsTrain = sum(train$outcomeCount > 0),
    nTest = nrow(test), eventsTest = sum(y), folds = max(split$folds),
    referenceVariance = oldTuning$selectedVariance, autoVariance = auto$model$model$priorVariance,
    autoNonzeroPredictors = nNonzero, droppedSourcePredictors = auto$dropped,
    rescaledSourcePredictors = auto$rescaled,
    minFraction = auto$preprocessing$minFraction, normalize = auto$preprocessing$normalize,
    removeRedundancy = auto$preprocessing$removeRedundancy,
    meanAbsPredictionDifference = mean(abs(p - reference)), maxAbsPredictionDifference = max(abs(p - reference)),
    predictionCorrelation = if (stats::sd(p) > 0 && stats::sd(reference) > 0) stats::cor(p, reference) else NA_real_,
    meanAbsReferenceReloadDrift = mean(abs(reloaded - reference)),
    maxAbsReferenceReloadDrift = max(abs(reloaded - reference)))
  for (name in names(baseline)) {
    row[[paste0("reference_", name)]] <- baseline[[name]]
    row[[paste0("auto_", name)]] <- current[[name]]
    row[[paste0("difference_", name)]] <- differences[[name]]
  }
  row
}
