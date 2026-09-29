# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

#' Create paired learning-curve settings
#' @param trainingSizes Optional positive patient counts, instead of trainingEvents.
#' @param trainingEvents Outcome-positive training patients. Defaults to
#' 25, 50, 100, 150, 200, 500, 1000. Inf uses the full development pool.
#' @param repetitions Number of nested subsample sequences.
#' @param testFraction Fraction reserved for the fixed test set.
#' @param folds Number of inner stratified folds.
#' @param seed Master random seed.
#' @param minClassCount Minimum cases and controls in training and test sets.
#' @return A settings list.
#' @export
createLearningCurveSettings <- function(trainingSizes = NULL, repetitions = 10L,
		testFraction = 0.25, folds = 5L, seed = 42L, minClassCount = 20L, trainingEvents = NULL) {
	if (!is.null(trainingSizes) && !is.null(trainingEvents)) stop("Supply trainingEvents or trainingSizes, not both")
	if (is.null(trainingSizes) && is.null(trainingEvents)) trainingEvents <- c(25, 50, 100, 150, 200, 500, 1000)
	budgetUnit <- if (is.null(trainingEvents)) "patients" else "events"
	trainingSizes <- if (is.null(trainingEvents)) trainingSizes else trainingEvents
	stopifnot(is.numeric(trainingSizes), length(trainingSizes) > 0,
		!anyNA(trainingSizes), all(trainingSizes > 0),
		all(is.infinite(trainingSizes) | trainingSizes == floor(trainingSizes)),
		length(repetitions) == 1, is.finite(repetitions), repetitions >= 1, repetitions == as.integer(repetitions),
		length(testFraction) == 1, testFraction > 0, testFraction < 1,
		length(folds) == 1, is.finite(folds), folds >= 2, folds == as.integer(folds),
		length(minClassCount) == 1, is.finite(minClassCount), minClassCount >= folds,
		minClassCount == floor(minClassCount), length(seed) == 1, is.finite(seed),
		seed >= 0, seed <= .Machine$integer.max, seed == floor(seed))
	list(trainingBudgets = sort(unique(trainingSizes)), budgetUnit = budgetUnit, repetitions = as.integer(repetitions),
		testFraction = testFraction, folds = as.integer(folds), seed = as.integer(seed),
		minClassCount = as.integer(minClassCount))
}

#' Create an experiment specification
#' @param problems Named problem definitions. Each has cohortDefinitionSet, targetId,
#' outcomeId, populationSettings, and a descriptive name.
#' @param pairs Data frame with sourceId and targetId, matching database registry names.
#' @param featureProfiles Named profiles. Standard profiles use type="standard";
#' phenotype profiles additionally supply frozen cohortDefinitionSet and
#' ascertainmentReviewed=TRUE after checking prediction-time availability.
#' @param outputFolder Explicit location for all generated artifacts.
#' @param learningCurve Settings from createLearningCurveSettings.
#' @param variances Positive finite Cyclops prior variance candidates.
#' @param bootstrapReplicates Number of paired test-patient bootstrap replicates.
#' @param threads Number of Cyclops threads per fit.
#' @return Experiment settings, suitable for saveRDS; contains no connections.
#' @export
createExperimentSettings <- function(problems, pairs, featureProfiles,
		outputFolder, learningCurve = createLearningCurveSettings(),
		variances = 10^seq(-6, 6), bootstrapReplicates = 500L, threads = 1L) {
	stopifnot(is.list(problems), length(problems) > 0, !is.null(names(problems)),
		is.data.frame(pairs), all(c("sourceId", "targetId") %in% names(pairs)),
		nrow(pairs) > 0, all(pairs$sourceId != pairs$targetId),
		is.list(featureProfiles), length(featureProfiles) > 0,
		!is.null(names(featureProfiles)), length(outputFolder) == 1,
		is.character(outputFolder), nzchar(outputFolder),
		length(variances) > 0, all(is.finite(variances)), all(variances > 0),
		length(bootstrapReplicates) == 1, is.finite(bootstrapReplicates),
		bootstrapReplicates >= 0, bootstrapReplicates == floor(bootstrapReplicates),
		length(threads) == 1, is.finite(threads), threads >= 1, threads == floor(threads))
	pairs$sourceId <- as.character(pairs$sourceId)
	pairs$targetId <- as.character(pairs$targetId)
	identifiers <- c(names(problems), names(featureProfiles), pairs$sourceId, pairs$targetId)
	if (anyNA(identifiers) || any(!nzchar(identifiers))) stop("Experiment identifiers must be nonempty")
	if (anyDuplicated(names(problems)) || anyDuplicated(names(featureProfiles)) ||
			anyDuplicated(pairs[c("sourceId", "targetId")])) stop("Duplicate experiment identifiers")
	for (problem in problems) {
		if (!all(c("targetId", "outcomeId", "populationSettings") %in% names(problem))) {
			stop("Each problem needs targetId, outcomeId, and explicit populationSettings")
		}
		if (!isTRUE(problem$populationSettings$binary) ||
				!isTRUE(problem$populationSettings$firstExposureOnly)) {
			stop("v1 requires binary problems with firstExposureOnly=TRUE")
		}
	}
	for (profile in featureProfiles) {
		if (length(profile$type) != 1 || !profile$type %in% c("standard", "phenotype")) stop("Unknown feature profile")
		if (profile$type == "phenotype" &&
				(!isTRUE(profile$ascertainmentReviewed) || is.null(profile$cohortDefinitionSet))) {
			stop("Phenotype profiles need frozen definitions and ascertainmentReviewed=TRUE")
		}
	}
	structure(list(problems = problems, pairs = pairs, featureProfiles = featureProfiles,
		outputFolder = outputFolder, learningCurve = learningCurve,
		variances = sort(unique(variances)), bootstrapReplicates = as.integer(bootstrapReplicates),
		threads = as.integer(threads), methods = c("targetOnly", "frozenSource",
			"interceptRecalibration", "slopeRecalibration", "priorCoefs")),
		class = "transferExperimentSettings")
}

#' Validate experiment requirements
#' @param settings Experiment settings.
#' @param databaseRegistry Named runtime database settings. snapshotId is optional.
#' @param checkBackend Run the optional synthetic PLP readiness check. Default FALSE.
#' @return A data frame describing the job grid, invisibly.
#' @export
validateExperiment <- function(settings, databaseRegistry, checkBackend = FALSE) {
	stopifnot(inherits(settings, "transferExperimentSettings"))
	ids <- unique(c(as.character(settings$pairs$sourceId), as.character(settings$pairs$targetId)))
	if (!all(ids %in% names(databaseRegistry))) stop("Missing database registry entries")
	for (id in ids) {
		for (field in c("cohortTable", "phenotypeCohortTable")) {
			table <- databaseRegistry[[id]][[field]]
			if (!is.null(table) && (!is.character(table) || length(table) != 1 || is.na(table) ||
					!grepl("^[A-Za-z][A-Za-z0-9_]*$", table))) {
				stop(field, " must be an unqualified table name containing letters, digits or underscores")
			}
		}
		snapshot <- databaseRegistry[[id]]$snapshotId
		if (!is.null(snapshot) && (!is.character(snapshot) || length(snapshot) != 1 || is.na(snapshot) || !nzchar(snapshot))) {
			stop("snapshotId must be a nonempty string when supplied")
		}
	}
	if (isTRUE(checkBackend)) checkPlpBackend()
	grid <- expand.grid(problemId = names(settings$problems),
		profileId = names(settings$featureProfiles), pairIndex = seq_len(nrow(settings$pairs)),
		repetition = seq_len(settings$learningCurve$repetitions),
		trainingBudget = settings$learningCurve$trainingBudgets, stringsAsFactors = FALSE)
	grid$budgetUnit <- settings$learningCurve$budgetUnit
	grid$sourceId <- settings$pairs$sourceId[grid$pairIndex]
	grid$targetId <- settings$pairs$targetId[grid$pairIndex]
	invisible(grid)
}
