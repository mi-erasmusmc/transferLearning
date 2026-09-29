# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

subsetTrain <- function(data, population, folds = rep(1L, nrow(population))) {
	ids <- population$rowId
	original <- data$covariateData
	covariates <- dplyr::filter(original$covariates, .data$rowId %in% ids)
	copy <- Andromeda::andromeda(covariates = covariates,
		covariateRef = original$covariateRef, analysisRef = original$analysisRef)
	class(copy) <- "CovariateData"
	meta <- attr(original, "metaData")
	meta$populationSize <- nrow(population)
	attr(copy, "metaData") <- meta
	train <- list(covariateData = copy, labels = population,
		folds = data.frame(rowId = population$rowId, index = folds))
	class(train) <- "plpData"
	attr(train, "metaData") <- attr(data, "metaData")
	train
}

sourceInTargetUnits <- function(source, targetCovariates) {
	coefficients <- source$model$coefficients
	coefficients <- coefficients[coefficients$covariateIds != "(Intercept)" & coefficients$betas != 0, ]
	# Presence is determined only from this preprocessed training fold, not from
	# covariateRef or validation data. PLP develop drops non-overlapping sources.
	present <- dplyr::collect(dplyr::distinct(targetCovariates$covariates, .data$covariateId))$covariateId
	keep <- as.character(coefficients$covariateIds) %in% as.character(present)
	dropped <- as.character(coefficients$covariateIds[!keep])
	coefficients <- coefficients[keep, , drop = FALSE]
	ids <- as.character(coefficients$covariateIds)
	sourceNorm <- source$preprocessing$tidyCovariates$normFactors
	targetNorm <- attr(targetCovariates, "metaData")$tidyCovariateDataSettings$normFactors
	scaleSource <- sourceNorm$maxValue[match(ids, as.character(sourceNorm$covariateId))]
	if (length(scaleSource) != length(ids) || any(!is.finite(scaleSource) | scaleSource <= 0)) stop("Missing source normalization factors")
	scaleTarget <- targetNorm$maxValue[match(ids, as.character(targetNorm$covariateId))]
	if (length(scaleTarget) != length(ids) || any(!is.finite(scaleTarget) | scaleTarget <= 0)) stop("Invalid target normalization factors")
	coefficients$betas <- coefficients$betas * scaleTarget / scaleSource
	list(coefficients = coefficients, covariateData = targetCovariates,
		droppedSourceIds = dropped)
}

fitVariance <- function(data, population, variance, settings, source = NULL) {
	train <- subsetTrain(data, population)
	on.exit(Andromeda::close(train$covariateData))
	train$covariateData <- PatientLevelPrediction::preprocessData(train$covariateData,
		PatientLevelPrediction::createPreprocessSettings(normalize = TRUE,
			minFraction = 0, removeRedundancy = FALSE))
	prior <- NULL
	dropped <- character()
	if (!is.null(source)) {
		converted <- sourceInTargetUnits(source, train$covariateData)
		prior <- if (nrow(converted$coefficients)) converted$coefficients else NULL
		train$covariateData <- converted$covariateData
		dropped <- converted$droppedSourceIds
	}
	modelSettings <- PatientLevelPrediction::setLassoLogisticRegression(
		variance = variance, priorCoefs = prior,
		threads = settings$threads, seed = settings$learningCurve$seed)
	fit <- fitPreparedVariance(train, modelSettings)
	fit$transferDetails <- list(droppedSourceIds = dropped)
	fit
}

# Keep real validation assignments outside the local data supplied to PLP.
# fitPlp accepts prepared trainData and does not invoke a split constructor.
fitPreparedVariance <- function(trainData, modelSettings) {
	fitData <- trainData
	fitData$folds <- data.frame(rowId = trainData$labels$rowId,
		index = rep(1L, nrow(trainData$labels)))
	fit <- PatientLevelPrediction::fitPlp(fitData, modelSettings,
		analysisId = "transfer", analysisPath = NULL)
	if (!identical(fit$model$modelStatus, "OK") || any(!is.finite(fit$model$coefficients$betas))) {
		stop("Cyclops fit failed: ", fit$model$modelStatus)
	}
	assertFixedVarianceFit(fit, modelSettings$param$priorParams$variance)
	fit
}

assertFixedVarianceFit <- function(fit, variance) {
	if (length(fit$model$priorVariance) != 1L ||
			!isTRUE(all.equal(as.numeric(fit$model$priorVariance), variance, tolerance = 1e-12))) {
		stop("PLP did not fit the supplied candidate variance")
	}
	if (!is.null(fit$model$cv) || any(fit$prediction$evaluationType != "Train") ||
			NROW(fit$trainDetails$hyperParamSearch) > 0L) {
		stop("PLP unexpectedly produced internal CV predictions or tuning results")
	}
	invisible(TRUE)
}

predictValues <- function(model, data, population) {
	# Prediction preprocessing operates on a disposable copy, never the raw cache.
	copy <- subsetTrain(data, population)
	on.exit(Andromeda::close(copy$covariateData))
	prediction <- PatientLevelPrediction::predictPlp(model, copy, population)
	values <- prediction$value[match(population$rowId, prediction$rowId)]
	if (any(!is.finite(values)) || any(values < 0 | values > 1)) stop("Invalid predictions")
	values
}

logLoss <- function(y, p) {
	p <- pmax(1e-15, pmin(1 - 1e-15, p))
	mean(-y * log(p) - (1 - y) * log1p(-p))
}

tuneModel <- function(data, population, folds, settings, source = NULL) {
	scores <- data.frame()
	evaluate <- function(variances) {
		for (variance in variances) {
			for (fold in sort(unique(folds))) {
				validation <- population[folds == fold, , drop = FALSE]
				result <- tryCatch({
					model <- fitVariance(data, population[folds != fold, , drop = FALSE], variance, settings, source)
					p <- predictValues(model, data, validation)
					list(loss = logLoss(as.integer(validation$outcomeCount > 0), p), error = "",
						droppedSourceIds = paste(model$transferDetails$droppedSourceIds, collapse = ","))
				}, error = function(e) list(loss = Inf, error = conditionMessage(e), droppedSourceIds = NA_character_))
				scores <<- rbind(scores, data.frame(variance = variance, fold = fold,
					n = nrow(validation), loss = result$loss, error = result$error,
					droppedSourceIds = result$droppedSourceIds))
			}
		}
	}
	select <- function() {
		means <- vapply(split(scores, scores$variance), function(x) stats::weighted.mean(x$loss, x$n), numeric(1))
		if (!any(is.finite(means))) stop("All tuning candidates failed: ", paste(unique(scores$error), collapse = "; "))
		as.numeric(names(means)[order(means, as.numeric(names(means)))][1])
	}
	evaluate(settings$variances)
	best <- select()
	expanded <- FALSE
	if (best == min(settings$variances) || best == max(settings$variances)) {
		extra <- if (best == min(settings$variances)) best / 10^(1:3) else best * 10^(1:3)
		evaluate(setdiff(extra, scores$variance))
		best <- select()
		expanded <- TRUE
	}
	model <- fitVariance(data, population, best, settings, source)
	list(model = model, tuning = scores, selectedVariance = best,
		transferDetails = model$transferDetails,
		expanded = expanded, boundary = best %in% range(scores$variance))
}

fitRecalibration <- function(y, p, slope = FALSE) {
	eta <- stats::qlogis(pmax(1e-15, pmin(1 - 1e-15, p)))
	frame <- data.frame(y = y, eta = eta)
	formula <- if (slope) y ~ eta else y ~ offset(eta)
	data <- Cyclops::createCyclopsData(formula, data = frame, modelType = "lr")
	fit <- Cyclops::fitCyclopsModel(data, prior = Cyclops::createPrior("none"))
	coefficients <- stats::coef(fit)
	if (!identical(fit$return_flag, "SUCCESS") || any(!is.finite(coefficients))) {
		stop("Recalibration failed: ", fit$return_flag)
	}
	list(intercept = unname(coefficients[1]), slope = if (slope) unname(coefficients[2]) else 1)
}
applyRecalibration <- function(model, p) {
	stats::plogis(model$intercept + model$slope * stats::qlogis(pmax(1e-15, pmin(1 - 1e-15, p))))
}
