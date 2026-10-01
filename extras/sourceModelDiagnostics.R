# Source this file, then call sourceModelDiagnostics(outputFolder).
# No database access, extraction or fitting. Only aggregate CSVs are exported.
sourceModelDiagnostics <- function(outputFolder,
    exportFolder = file.path(outputFolder, "source-diagnostics")) {
  manifest <- readRDS(file.path(outputFolder, "manifest.rds"))
  settings <- manifest$settings
  summaries <- thresholds <- quantiles <- tuning <- list()
  for (id in unique(as.character(settings$pairs$sourceId))) {
    for (problemId in names(settings$problems)) for (profileId in names(settings$featureProfiles)) {
      modelKey <- digest::digest(list(id, problemId, profileId), algo = "xxhash64")
      folder <- file.path(outputFolder, "sources", modelKey)
      if (!file.exists(file.path(folder, "tuning.rds"))) next
      message("Source diagnostics: ", id, " / ", problemId, " / ", profileId)
      cacheKey <- digest::digest(list(id, manifest$databases[[id]], settings$problems[[problemId]],
        settings$featureProfiles[[profileId]], manifest$packages, manifest$implementation), algo = "sha256")
      cache <- file.path(outputFolder, "data", cacheKey)
      result <- sourceDiagnosticsOne(folder, cache)
      key <- data.frame(sourceId = id, problemId = problemId, profileId = profileId)
      attachKey <- function(x) cbind(key[rep(1L, nrow(x)), , drop = FALSE], x)
      summaries[[modelKey]] <- attachKey(result$summary)
      thresholds[[modelKey]] <- attachKey(result$thresholds)
      quantiles[[modelKey]] <- attachKey(result$quantiles)
      tuning[[modelKey]] <- attachKey(result$tuning)
    }
  }
  if (!length(summaries)) stop("No saved source models found")
  output <- list(summary = dplyr::bind_rows(summaries), thresholds = dplyr::bind_rows(thresholds),
    quantiles = dplyr::bind_rows(quantiles), tuning = dplyr::bind_rows(tuning))
  output$provenance <- data.frame(originalManifestHash = manifest$hash,
    originalPlpFingerprint = manifest$backend,
    currentPlpFingerprint = getFromNamespace("functionFingerprint", "TransferLearning")("PatientLevelPrediction"),
    currentPlpVersion = as.character(utils::packageVersion("PatientLevelPrediction")),
    scriptFingerprint = digest::digest(list(body(sourceModelDiagnostics), body(sourceDiagnosticsOne),
      body(sourceDiagnosticsTuning)), algo = "sha256"))
  dir.create(exportFolder, recursive = TRUE, showWarnings = FALSE)
  for (name in names(output)) utils::write.csv(output[[name]], file.path(exportFolder,
    paste0(name, ".csv")), row.names = FALSE)
  message("Aggregate source diagnostics written to ", exportFolder)
  invisible(output)
}

sourceDiagnosticsTuning <- function(scores, selected) {
  stopifnot(all(c("variance", "fold", "loss", "n", "error") %in% names(scores)),
    all(is.finite(scores$variance) & scores$variance > 0),
    all(is.finite(scores$n) & scores$n > 0), is.finite(selected), selected > 0)
  same <- function(a, b) abs(log(a) - log(b)) < 1e-10
  candidates <- sort(unique(scores$variance))
  result <- dplyr::bind_rows(lapply(candidates, function(v) {
    rows <- scores[scores$variance == v, ]
    valid <- is.finite(rows$loss) & !is.na(rows$error) & rows$error == ""
    data.frame(variance = v, folds = nrow(rows), failedFolds = sum(!valid),
      weightedLogLoss = if (all(valid)) stats::weighted.mean(rows$loss, rows$n) else NA_real_,
      minFoldLogLoss = if (all(valid)) min(rows$loss) else NA_real_,
      maxFoldLogLoss = if (all(valid)) max(rows$loss) else NA_real_,
      selected = same(v, selected))
  }))
  if (sum(result$selected) != 1 || !is.finite(result$weightedLogLoss[result$selected]))
    stop("Selected variance does not identify one successful tuning candidate")
  result$lossAboveSelected <- result$weightedLogLoss - result$weightedLogLoss[result$selected]
  result
}

sourceDiagnosticsOne <- function(folder, cache) {
  fit <- readRDS(file.path(folder, "tuning.rds"))
  model <- PatientLevelPrediction::loadPlpModel(file.path(folder, "model"))
  population <- readRDS(file.path(cache, "population.rds"))
  rows <- match(fit$rowIds, population$rowId)
  if (anyNA(rows) || anyDuplicated(fit$rowIds) || length(rows) < 2) stop("Invalid source training row IDs")
  y <- population$outcomeCount[rows]
  if (anyNA(y) || any(!y %in% c(0, 1))) stop("Expected binary source outcomes")
  factors <- model$preprocessing$tidyCovariates$normFactors
  ids <- as.character(factors$covariateId)
  if (is.null(factors) || anyDuplicated(ids) || any(!is.finite(factors$maxValue) | factors$maxValue <= 0))
    stop("Invalid source normalization factors")
  coefs <- model$model$coefficients
  coefs <- coefs[as.character(coefs$covariateIds) != "(Intercept)", , drop = FALSE]
  if (anyDuplicated(as.character(coefs$covariateIds)) || any(!is.finite(coefs$betas)) ||
      any(!as.character(coefs$covariateIds) %in% ids)) stop("Invalid source coefficients")
  beta <- coefs$betas[match(ids, as.character(coefs$covariateIds))]; beta[is.na(beta)] <- 0
  data <- PatientLevelPrediction::loadPlpData(file.path(cache, "plpData"))
  on.exit(Andromeda::close(data$covariateData))
  trainIds <- fit$rowIds
  # Aggregate in the local Andromeda database; do not materialize all covariate rows in R.
  sums <- data$covariateData$covariates |>
    dplyr::filter(.data$rowId %in% trainIds) |>
    dplyr::group_by(.data$covariateId) |>
    dplyr::summarise(total = sum(.data$covariateValue, na.rm = TRUE),
      missing = sum(dplyr::if_else(is.na(.data$covariateValue), 1L, 0L), na.rm = TRUE),
      square = sum(.data$covariateValue * .data$covariateValue, na.rm = TRUE), .groups = "drop") |>
    dplyr::collect()
  if (any(sums$missing > 0) || any(!is.finite(sums$total)) || any(!is.finite(sums$square)))
    stop("Invalid source training covariate values")
  ix <- match(ids, as.character(sums$covariateId))
  total <- sums$total[ix]; square <- sums$square[ix]
  total[is.na(ix)] <- 0; square[is.na(ix)] <- 0
  n <- length(rows)
  sdRaw <- sqrt(pmax(0, (square - total^2 / n) / (n - 1)))
  magnitude <- list(normalizedCoefficient = abs(beta),
    perTrainingSD = abs(beta) * sdRaw / factors$maxValue)
  thresholds <- dplyr::bind_rows(lapply(names(magnitude), function(scale) {
    cuts <- c(0, 1e-6, 1e-4, .001, .01, .05, .1, .25, .5, 1)
    data.frame(scale = scale, threshold = cuts,
      countAbove = vapply(cuts, function(cut) sum(magnitude[[scale]] > cut), integer(1)),
      eligiblePredictors = length(ids))
  }))
  quantiles <- dplyr::bind_rows(lapply(names(magnitude), function(scale) {
    probs <- c(0, .25, .5, .75, .9, .95, 1)
    data.frame(scale = scale, quantile = probs,
      value = if (length(ids)) as.numeric(stats::quantile(magnitude[[scale]], probs)) else NA_real_)
  }))
  tuning <- sourceDiagnosticsTuning(fit$tuning, fit$selectedVariance)
  same <- function(a, b) length(a) == 1 && is.finite(a) && abs(log(a) - log(b)) < 1e-10
  status <- model$model$modelStatus
  # Do not export arbitrary status strings that might contain environment details.
  summary <- data.frame(nTrain = n, eventsTrain = sum(y), controlsTrain = sum(y == 0),
    eligiblePredictors = length(ids), storedCoefficientRows = nrow(coefs),
    exactlyNonzero = sum(beta != 0), above1eMinus6 = sum(abs(beta) > 1e-6),
    nonzeroConstantPredictors = sum(beta != 0 & sdRaw == 0),
    selectedVariance = fit$selectedVariance, fittedVariance = model$model$priorVariance,
    varianceMatches = same(model$model$priorVariance, fit$selectedVariance),
    atStrongestPenalty = same(fit$selectedVariance, min(tuning$variance)),
    atWeakestPenalty = same(fit$selectedVariance, max(tuning$variance)),
    gridExpanded = isTRUE(fit$expanded), recordedBoundary = isTRUE(fit$boundary),
    savedStatusAvailable = !is.null(status), savedStatusOK = if (is.null(status)) NA else identical(status, "OK"),
    selectedCvLogLoss = tuning$weightedLogLoss[tuning$selected],
    selectedIsCvMinimum = tuning$weightedLogLoss[tuning$selected] <= min(tuning$weightedLogLoss, na.rm = TRUE) + 1e-12)
  list(summary = summary, thresholds = thresholds, quantiles = quantiles, tuning = tuning)
}
