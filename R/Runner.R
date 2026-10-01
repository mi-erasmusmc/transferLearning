# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

#' Run paired transfer-learning experiments
#' @param settings Experiment settings.
#' @param databaseRegistry Named runtime database settings, optionally including snapshotId.
#' @param preparedData Optional nested list indexed by database, problem, profile.
#' Entries are cache paths or lists with raw plpData and population. When omitted,
#' prepareExperimentData generates and extracts the cohorts.
#' @param resume Reuse compatible completed work. Additional training budgets and
#' repetitions are allowed; other scientific settings must remain unchanged.
#' @return Collected status, metrics, and paired bootstrap intervals.
#' @details Runs serially, with bounded Cyclops threads. Stores patient-level
#' artifacts locally. A failed method is recorded and does not stop other jobs.
#' Bootstrap intervals condition on the trained models and fixed test population;
#' repetitions describe training-sample variability separately.
#' @export
runExperiment <- function(settings, databaseRegistry, preparedData = NULL, resume = TRUE) {
	grid <- validateExperiment(settings, databaseRegistry)
	folder <- settings$outputFolder
	dir.create(folder, recursive = TRUE, showWarnings = FALSE)
	lock <- file.path(folder, ".runner-lock")
	if (!dir.create(lock, showWarnings = FALSE)) stop("Another runner owns this output folder (or a stale .runner-lock remains)")
	on.exit(unlink(lock, recursive = TRUE))
	fingerprintSettings <- settings
	fingerprintSettings$outputFolder <- NULL
	manifest <- list(settings = fingerprintSettings,
		databases = lapply(databaseRegistry, databaseIdentity), packages = packageVersions(),
		implementation = functionFingerprint("TransferLearning"),
		backend = functionFingerprint("PatientLevelPrediction"),
		reuseContract = list(version = 1L, science = experimentScienceFingerprint()))
	manifest$hash <- digest::digest(manifest, algo = "sha256")
	manifestPath <- file.path(folder, "manifest.rds")
	cacheManifest <- manifest
	if (file.exists(manifestPath)) {
		if (!resume) stop("Existing experiment: use resume=TRUE or a new output folder")
		cacheManifest <- readRDS(manifestPath)
		activePath <- file.path(folder, "resume-manifest.rds")
		previous <- if (file.exists(activePath)) readRDS(activePath) else cacheManifest
		assertExperimentExtension(previous, manifest)
		ParallelLogger::logInfo("Reusing compatible experiment; only missing work will run")
	}
	jobPaths <- indexExperimentJobs(folder)
	if (!file.exists(manifestPath)) atomicSave(manifest, manifestPath)
	# Preserve the extraction/source provenance; record each extension separately.
	atomicSave(manifest, file.path(folder, "runs", paste0(manifest$hash, ".rds")))
	atomicSave(manifest, file.path(folder, "resume-manifest.rds"))
	if (is.null(preparedData)) preparedData <- cachedExperimentInputs(settings, databaseRegistry, cacheManifest)
	inputs <- list()
	on.exit(for (input in inputs) if (input$owned) Andromeda::close(input$data$covariateData), add = TRUE)
	getInput <- function(id, problemId, profileId) {
		key <- digest::digest(list(id, problemId, profileId), algo = "xxhash64")
		if (is.null(inputs[[key]])) {
			input <- openInput(preparedData[[id]][[problemId]][[profileId]], settings$problems[[problemId]], id)
			population <- input$population
			partition <- makePartition(population, settings$learningCurve, id, problemId)
			input$partition <- partition
			# Profile comparisons must refer to precisely the same eligible patients.
			populationKey <- digest::digest(list(id, problemId), algo = "xxhash64")
			path <- file.path(folder, "splits", paste0(populationKey, ".rds"))
			saved <- list(population = data.frame(subjectId = as.character(population$subjectId),
				outcomeCount = as.integer(population$outcomeCount),
				cohortStartDate = as.character(population$cohortStartDate)), partition = partition)
			if (file.exists(path) && !identical(readRDS(path), saved)) {
				if (input$owned) Andromeda::close(input$data$covariateData)
				stop("Populations differ across feature profiles or cached inputs")
			}
			if (!file.exists(path)) atomicSave(saved, path)
			inputs[[key]] <<- input
		}
		inputs[[key]]
	}
	sources <- list()
	getSource <- function(id, problemId, profileId) {
		key <- digest::digest(list(id, problemId, profileId), algo = "xxhash64")
		if (!is.null(sources[[key]])) return(sources[[key]])
		path <- file.path(folder, "sources", key)
		if (resume && file.exists(file.path(path, "tuning.rds"))) {
			source <- PatientLevelPrediction::loadPlpModel(file.path(path, "model"))
		} else {
			input <- getInput(id, problemId, profileId)
			population <- input$population[input$partition$development, , drop = FALSE]
			if (any(classCounts(population) < settings$learningCurve$minClassCount)) stop("Insufficient source cases or controls")
			folds <- makeFolds(population, settings$learningCurve$folds,
				seedFor(settings$learningCurve$seed, id, problemId, "source"))
			fit <- tuneModel(input$data, population, folds, settings)
			source <- fit$model
			PatientLevelPrediction::savePlpModel(source, file.path(path, "model"))
			fit$model <- NULL
			fit$rowIds <- population$rowId
			fit$folds <- folds
			atomicSave(fit, file.path(path, "tuning.rds"))
		}
		sources[[key]] <<- source
		source
	}
	for (i in seq_len(nrow(grid))) {
		job <- grid[i, , drop = FALSE]
		identity <- experimentJobKey(job)
		path <- jobPaths[[identity]]
		if (is.null(path)) path <- file.path(folder, "jobs", identity)
		key <- basename(path)
		resultPath <- file.path(path, "result.rds")
		old <- if (file.exists(resultPath)) readRDS(resultPath) else NULL
		if (!is.null(old) && jobIsComplete(old, settings$methods)) next
		ParallelLogger::logInfo("Job ", i, "/", nrow(grid), ": ", job$sourceId, " -> ", job$targetId,
			" / ", job$problemId, " / ", job$profileId, " / ", job$budgetUnit, "=", job$trainingBudget)
		warnings <- character(); reused <- character()
		result <- withCallingHandlers(tryCatch({
			input <- getInput(job$targetId, job$problemId, job$profileId)
			development <- input$population[input$partition$development, , drop = FALSE]
			test <- input$population[input$partition$test, , drop = FALSE]
			available <- if (job$budgetUnit == "events") sum(development$outcomeCount > 0) else nrow(development)
			if (is.finite(job$trainingBudget) && job$trainingBudget > available) {
				list(status = cbind(job, method = "all", status = "skipped", reason = "Training budget exceeds development pool"))
			} else {
				ix <- sampleNested(development, job$trainingBudget,
					seedFor(settings$learningCurve$seed, job$targetId, job$problemId, job$repetition, "sample"), unit = job$budgetUnit)
				train <- development[ix, , drop = FALSE]
				if (any(classCounts(train) < settings$learningCurve$minClassCount) ||
						any(classCounts(test) < settings$learningCurve$minClassCount)) {
					list(status = cbind(job, method = "all", status = "skipped", reason = "Insufficient cases or controls"))
				} else {
					folds <- makeFolds(train, settings$learningCurve$folds,
						seedFor(settings$learningCurve$seed, job$targetId, job$problemId, job$repetition, job$trainingBudget, "folds"))
					splitPath <- file.path(path, "split.rds")
					split <- list(rowIds = train$rowId, folds = folds, testRowIds = test$rowId)
					if (file.exists(splitPath) && !identical(readRDS(splitPath), split)) stop("Saved job split differs")
					if (!file.exists(splitPath)) atomicSave(split, splitPath)
					predictions <- list(); statuses <- list(); metrics <- list(); intervals <- list()
					# Recover successful methods from a partially failed job without refitting.
					predictionPath <- file.path(path, "predictions.rds")
					if (!is.null(old) && file.exists(predictionPath)) {
						prior <- readRDS(predictionPath)
						if (!identical(lapply(prior$population, function(x) x), lapply(test, function(x) x))) stop("Saved predictions use a different test population")
						for (method in settings$methods) {
							success <- old$status[old$status$method == method & old$status$status == "completed", , drop = FALSE]
							metric <- old$metrics[old$metrics$method == method, , drop = FALSE]
							p <- prior$predictions[[method]]
							if (nrow(success) == 1 && NROW(metric) == 1 && length(p) == nrow(test) && all(is.finite(p))) {
								predictions[[method]] <- p; statuses[[method]] <- success; metrics[[method]] <- metric
							}
						}
					}
					reused <- names(predictions)
					for (method in settings$methods) {
						if (method %in% reused) next
						outcome <- tryCatch({
							if (method == "targetOnly" || method == "priorCoefs") {
								source <- if (method == "priorCoefs") getSource(job$sourceId, job$problemId, job$profileId) else NULL
								fit <- tuneModel(input$data, train, folds, settings, source)
								p <- predictValues(fit$model, input$data, test)
								PatientLevelPrediction::savePlpModel(fit$model, file.path(path, method, "model"))
								fit$model <- NULL
								atomicSave(fit, file.path(path, method, "tuning.rds"))
							} else {
								source <- getSource(job$sourceId, job$problemId, job$profileId)
								p <- predictValues(source, input$data, test)
								if (method != "frozenSource") {
									model <- fitRecalibration(as.integer(train$outcomeCount > 0),
										predictValues(source, input$data, train), method == "slopeRecalibration")
									p <- applyRecalibration(model, p)
									atomicSave(model, file.path(path, method, "recalibration.rds"))
								}
							}
							predictions[[method]] <- p
							metrics[[method]] <- cbind(job, method = method, nTrain = nrow(train),
								eventsTrain = sum(train$outcomeCount > 0), controlsTrain = sum(train$outcomeCount == 0),
								prevalenceTrain = mean(train$outcomeCount > 0), nTest = nrow(test), eventsTest = sum(test$outcomeCount),
								as.data.frame(as.list(performance(as.integer(test$outcomeCount > 0), p))))
							cbind(job, method = method, status = "completed", reason = "")
						}, error = function(e) cbind(job, method = method, status = "failed", reason = conditionMessage(e)))
						statuses[[method]] <- outcome
					}
					for (method in setdiff(names(predictions), "targetOnly")) {
						if (!is.null(predictions$targetOnly) && settings$bootstrapReplicates > 0) {
							priorInterval <- if (is.null(old$intervals)) NULL else old$intervals[old$intervals$method == method, , drop = FALSE]
							if (all(c("targetOnly", method) %in% reused) && NROW(priorInterval) == 2) {
								intervals[[method]] <- priorInterval
								next
							}
							ci <- pairedIntervals(as.integer(test$outcomeCount > 0), predictions$targetOnly,
								predictions[[method]], settings$bootstrapReplicates,
								seedFor(settings$learningCurve$seed, key, "bootstrap"))
							intervals[[method]] <- cbind(job[rep(1L, nrow(ci)), , drop = FALSE], method = method, ci)
						}
					}
					atomicSave(list(population = test, predictions = predictions), file.path(path, "predictions.rds"))
					list(status = dplyr::bind_rows(statuses), metrics = dplyr::bind_rows(metrics), intervals = dplyr::bind_rows(intervals))
				}
			}
		}, error = function(e) list(status = cbind(job, method = "all", status = "failed", reason = conditionMessage(e)))), warning = function(w) {
			warnings <<- c(warnings, conditionMessage(w))
			invokeRestart("muffleWarning")
		})
		result$warnings <- unique(c(old$warnings, warnings))
		result$provenance <- list(runManifestHash = manifest$hash, reusedMethods = reused)
		atomicSave(result, resultPath)
	}
	results <- collectExperimentResults(folder)
	for (name in names(results)) utils::write.csv(results[[name]], file.path(folder, paste0(name, ".csv")), row.names = FALSE)
	results
}
