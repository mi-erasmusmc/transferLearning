# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

#' Summarize source contributions and target corrections in saved transfer models
#' @param outputFolder Completed experiment output folder, including cached data and models.
#' @param exportFolder Destination for aggregate CSVs. Defaults to a new
#'   transfer-diagnostics subdirectory. Existing CSVs are replaced.
#' @param preparedData Optional original prepared-data list, as for runExperiment.
#'   When omitted, caches are located using the saved manifest, not current fingerprints.
#' @param tolerance Absolute tolerance in target-normalized coefficient units for
#'   classifying nonzero coefficients and changes. Default 1e-6.
#' @return Invisibly, a list of aggregate data frames written as CSV files.
#' @details No database connection or fitting is performed. Patient-level data remain
#'   local. Exports contain no patient IDs, individual predictions, covariate IDs,
#'   coefficient tables, or free-text errors. Apply local disclosure rules to counts.
#'   Standardized magnitudes use predictor SD in the original target training sample,
#'   including implicit zeros. Source selection refers to nonzero saved source
#'   coefficients; target-only and source-zero predictors form the new-coefficient group.
#'   Ablations hold the final target intercept fixed and do not refit any coefficients.
#'   Component SDs and correlations describe held-out linear predictors, not additive
#'   percentages of performance. Prediction drift from saved-model serialization is
#'   reported separately. Failed diagnostics stop rather than publish incomplete CSVs.
#' @export
summarizeTransferDiagnostics <- function(outputFolder,
    exportFolder = file.path(outputFolder, "transfer-diagnostics"),
    preparedData = NULL, tolerance = 1e-6) {
  stopifnot(length(tolerance) == 1, is.finite(tolerance), tolerance > 0)
  manifest <- readRDS(file.path(outputFolder, "manifest.rds"))
  files <- list.files(file.path(outputFolder, "jobs"), "^result.rds$", recursive = TRUE, full.names = TRUE)
  output <- list(coefficients = list(), contributions = list(), correlations = list(),
    ablations = list(), verification = list())
  for (file in files) {
    result <- readRDS(file)
    job <- result$status[result$status$method == "priorCoefs" & result$status$status == "completed", , drop = FALSE]
    if (!nrow(job)) next
    job <- job[, setdiff(names(job), c("method", "status", "reason")), drop = FALSE]
    stopifnot(nrow(job) == 1)
    ParallelLogger::logInfo("Transfer diagnostics: ", job$problemId, " / ", job$trainingBudget,
      " / repetition ", job$repetition)
    id <- as.character(job$targetId); problemId <- as.character(job$problemId)
    profileId <- as.character(job$profileId)
    problem <- manifest$settings$problems[[problemId]]
    input <- if (!is.null(preparedData)) preparedData[[id]][[problemId]][[profileId]] else {
      key <- digest::digest(list(id, manifest$databases[[id]], problem,
        manifest$settings$featureProfiles[[profileId]], manifest$packages,
        manifest$implementation), algo = "sha256")
      file.path(outputFolder, "data", key)
    }
    localResult <- diagnoseTransferJob(dirname(file), outputFolder, job, input, problem, tolerance)
    for (name in names(output)) output[[name]][[length(output[[name]]) + 1L]] <- cbind(job[rep(1L, nrow(localResult[[name]])), , drop = FALSE], localResult[[name]])
  }
  if (!length(output$coefficients)) stop("No completed priorCoefs jobs found")
  output <- lapply(output, dplyr::bind_rows)
  output$provenance <- data.frame(originalManifestHash = manifest$hash,
    originalPlpFingerprint = manifest$backend,
    diagnosticsPlpFingerprint = functionFingerprint("PatientLevelPrediction"),
    diagnosticsRunnerFingerprint = functionFingerprint("TransferLearning"),
    plpVersion = as.character(utils::packageVersion("PatientLevelPrediction")),
    tolerance = tolerance)
  dir.create(exportFolder, recursive = TRUE, showWarnings = FALSE)
  for (name in names(output)) utils::write.csv(output[[name]], file.path(exportFolder,
    paste0(name, ".csv")), row.names = FALSE)
  invisible(output)
}

# Coefficients in common target-normalized units. All target predictors remain
# eligible, including those whose final coefficient is zero.
transferCoefficientVectors <- function(source, target) {
  factors <- target$preprocessing$tidyCovariates$normFactors
  ids <- as.character(factors$covariateId)
  if (is.null(factors) || anyDuplicated(ids) || any(!is.finite(factors$maxValue) | factors$maxValue <= 0))
    stop("Invalid target normalization factors")
  extract <- function(model) {
    x <- model$model$coefficients
    x$covariateIds <- as.character(x$covariateIds)
    if (anyDuplicated(x$covariateIds) || any(!is.finite(x$betas))) stop("Invalid model coefficients")
    x
  }
  sc <- extract(source); tc <- extract(target)
  if (any(!tc$covariateIds %in% c(ids, "(Intercept)"))) stop("Target coefficient lacks normalization factor")
  b <- tc$betas[match(ids, tc$covariateIds)]; b[is.na(b)] <- 0
  s <- sc$betas[match(ids, sc$covariateIds)]; s[is.na(s)] <- 0
  sn <- source$preprocessing$tidyCovariates$normFactors
  sm <- sn$maxValue[match(ids, as.character(sn$covariateId))]
  if (any(s != 0 & (!is.finite(sm) | sm <= 0))) stop("Source coefficient lacks normalization factor")
  sm[s == 0] <- 1
  s <- s * factors$maxValue / sm
  intercept <- tc$betas[tc$covariateIds == "(Intercept)"]
  if (length(intercept) != 1) stop("Expected one target intercept")
  list(ids = ids, scale = factors$maxValue, source = s, target = b,
    intercept = intercept, sourceAll = sc[sc$covariateIds != "(Intercept)", ])
}

# Sparse matrix products, including implicit zero values, in population row order.
transferLinearComponents <- function(raw, rowIds, vectors, tolerance) {
  ri <- match(raw$rowId, rowIds); ci <- match(as.character(raw$covariateId), vectors$ids)
  keep <- !is.na(ri) & !is.na(ci)
  ri <- ri[keep]; ci <- ci[keep]; x <- raw$covariateValue[keep] / vectors$scale[ci]
  selected <- vectors$source != 0
  correction <- vectors$target - vectors$source
  # Keep sub-tolerance source terms in source so the decomposition remains exact.
  weights <- cbind(source = vectors$source,
    sourceAdjustments = correction * selected, newTarget = correction * !selected)
  ans <- matrix(0, nrow = length(rowIds), ncol = 3, dimnames = list(NULL, colnames(weights)))
  if (length(x)) for (j in seq_len(ncol(weights))) {
    sums <- rowsum(x * weights[ci, j], ri, reorder = FALSE)
    ans[as.integer(rownames(sums)), j] <- sums[, 1]
  }
  ans
}

transferTrainingSd <- function(raw, rowIds, vectors) {
  ix <- match(as.character(raw$covariateId), vectors$ids)
  keep <- raw$rowId %in% rowIds & !is.na(ix)
  ix <- ix[keep]; x <- raw$covariateValue[keep] / vectors$scale[ix]
  total <- square <- numeric(length(vectors$ids))
  if (length(ix)) {
    a <- rowsum(cbind(x, x^2), ix, reorder = FALSE)
    total[as.integer(rownames(a))] <- a[, 1]; square[as.integer(rownames(a))] <- a[, 2]
  }
  n <- length(rowIds)
  if (n < 2) stop("Training sample too small")
  sqrt(pmax(0, (square - total^2 / n) / (n - 1)))
}

transferMetrics <- function(y, p) {
  data.frame(auroc = if (length(unique(y)) == 2) as.numeric(pROC::auc(y, p,
    direction = "<", quiet = TRUE)) else NA_real_, logLoss = logLoss(y, p),
    brier = mean((y - p)^2))
}

diagnoseTransferJob <- function(path, folder, job, prepared, problem, tolerance) {
  input <- openInput(prepared, problem, as.character(job$targetId))
  if (input$owned) on.exit(Andromeda::close(input$data$covariateData))
  split <- readRDS(file.path(path, "split.rds"))
  train <- input$population[match(split$rowIds, input$population$rowId), , drop = FALSE]
  test <- input$population[match(split$testRowIds, input$population$rowId), , drop = FALSE]
  if (anyNA(train$rowId) || anyNA(test$rowId) || anyDuplicated(train$rowId) ||
      anyDuplicated(test$rowId) || length(intersect(train$rowId, test$rowId))) stop("Invalid saved split")
  saved <- readRDS(file.path(path, "predictions.rds"))
  order <- match(test$rowId, saved$population$rowId)
  if (anyNA(order) || !identical(as.integer(test$outcomeCount),
      as.integer(saved$population$outcomeCount[order]))) stop("Saved test outcomes differ from cache")
  sourceKey <- digest::digest(list(as.character(job$sourceId), as.character(job$problemId),
    as.character(job$profileId)), algo = "xxhash64")
  source <- PatientLevelPrediction::loadPlpModel(file.path(folder, "sources", sourceKey, "model"))
  target <- PatientLevelPrediction::loadPlpModel(file.path(path, "priorCoefs", "model"))
  vectors <- transferCoefficientVectors(source, target)
  wanted <- c(train$rowId, test$rowId)
  raw <- dplyr::collect(dplyr::filter(input$data$covariateData$covariates, .data$rowId %in% wanted))
  if (any(!is.finite(raw$covariateValue))) stop("Nonfinite covariate values")
  sd <- transferTrainingSd(raw, train$rowId, vectors)
  parts <- transferLinearComponents(raw, test$rowId, vectors, tolerance)
  full <- stats::plogis(vectors$intercept + rowSums(parts))
  plp <- predictValues(target, input$data, test)
  mismatch <- max(abs(full - plp))
  if (mismatch > 1e-6) stop("Linear predictor reconstruction disagrees with loaded PLP model: ", mismatch)
  original <- saved$predictions$priorCoefs[order]
  if (length(original) != length(full) || any(!is.finite(original))) stop("Invalid saved transfer predictions")
  selected <- vectors$source != 0
  nonzero <- abs(vectors$target) > tolerance
  delta <- vectors$target - vectors$source
  group <- ifelse(selected, "sourceSelected", "newTarget")
  coef <- dplyr::bind_rows(lapply(c("sourceSelected", "newTarget"), function(g) {
    ix <- group == g
    mag <- abs(vectors$target[ix] * sd[ix]); change <- abs(delta[ix] * sd[ix])
    data.frame(group = g, eligible = sum(ix), nonzero = sum(nonzero & ix),
      changed = sum(abs(delta) > tolerance & ix), zeroed = sum(selected & !nonzero & ix),
      signReversed = sum(selected & nonzero & vectors$source * vectors$target < 0 & ix),
      maxAbsStandardizedCoefficient = if (length(mag)) max(mag) else NA_real_,
      medianAbsStandardizedCoefficient = if (length(mag)) stats::median(mag) else NA_real_,
      maxAbsStandardizedChange = if (length(change)) max(change) else NA_real_,
      medianAbsStandardizedChange = if (length(change)) stats::median(change) else NA_real_)
  }))
  components <- cbind(parts, targetCorrection = parts[, 2] + parts[, 3], total = rowSums(parts))
  contributions <- data.frame(component = colnames(components),
    mean = colMeans(components), sd = apply(components, 2, stats::sd), row.names = NULL)
  pairs <- utils::combn(colnames(components), 2)
  correlations <- data.frame(component1 = pairs[1, ], component2 = pairs[2, ],
    correlation = apply(pairs, 2, function(p) {
      if (stats::sd(components[, p[1]]) == 0 || stats::sd(components[, p[2]]) == 0) return(NA_real_)
      stats::cor(components[, p[1]], components[, p[2]])
    }))
  predictors <- list(full = rowSums(parts), withoutNewTarget = parts[, 1] + parts[, 2],
    withoutSourceAdjustments = parts[, 1] + parts[, 3], withoutAllCorrections = parts[, 1])
  ablations <- dplyr::bind_rows(lapply(names(predictors), function(name) cbind(ablation = name,
    transferMetrics(test$outcomeCount, stats::plogis(vectors$intercept + predictors[[name]])))))
  ablations$aurocChangeFromFull <- ablations$auroc - ablations$auroc[1]
  ablations$logLossChangeFromFull <- ablations$logLoss - ablations$logLoss[1]
  ablations$brierChangeFromFull <- ablations$brier - ablations$brier[1]
  tuning <- readRDS(file.path(path, "priorCoefs", "tuning.rds"))
  originalMetrics <- transferMetrics(test$outcomeCount, original)
  verification <- data.frame(nTrain = nrow(train), eventsTrain = sum(train$outcomeCount),
    nTest = nrow(test), eventsTest = sum(test$outcomeCount), selectedVariance = tuning$selectedVariance,
    boundary = tuning$boundary, sourceNonzero = sum(vectors$sourceAll$betas != 0),
    droppedSourceNonzero = sum(vectors$sourceAll$betas != 0 &
      !vectors$sourceAll$covariateIds %in% vectors$ids),
    maxPredictionReconstructionError = mismatch, maxSavedPredictionDrift = max(abs(full - original)),
    meanSavedPredictionDrift = mean(abs(full - original)),
    aurocDrift = ablations$auroc[1] - originalMetrics$auroc,
    logLossDrift = ablations$logLoss[1] - originalMetrics$logLoss)
  list(coefficients = coef, contributions = contributions, correlations = correlations,
    ablations = ablations, verification = verification)
}
