# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

#' Extract and cache data for an experiment
#' @param settings Experiment settings.
#' @param databaseRegistry Named database settings. Each entry contains
#' connectionDetails, cdmDatabaseSchema, and cohortDatabaseSchema. cohortTable names
#' the target/outcome table; phenotypeCohortTable optionally names the predictor table
#' (defaults to cohortTable plus _phenotypes). snapshotId is optional.
#' @return Nested list of cache directories, indexed by database, problem, and profile.
#' @details Reuses existing cohort tables without regenerating their contents.
#' Missing tables are created and populated from the frozen definitions. When no
#' cohortTable is supplied, names are generated from the extraction fingerprint.
#' Existing tables must contain the intended complete cohort definitions.
#' Cohort definitions must be frozen CohortGenerator definition sets. Existing
#' caches are reused only under the same database configuration and optional snapshot.
#' Use a new output folder when underlying data change.
#' @export
prepareExperimentData <- function(settings, databaseRegistry) {
	validateExperiment(settings, databaseRegistry)
	prepareExperimentDataFromManifest(settings, databaseRegistry,
		list(packages = packageVersions(), implementation = functionFingerprint("TransferLearning")))
}

prepareExperimentDataFromManifest <- function(settings, databaseRegistry, cacheManifest) {
	paths <- list()
	for (id in unique(c(settings$pairs$sourceId, settings$pairs$targetId))) {
		database <- databaseRegistry[[id]]
		for (problemId in names(settings$problems)) {
			problem <- settings$problems[[problemId]]
			if (is.null(problem$cohortDefinitionSet)) stop("Live extraction requires frozen problem cohortDefinitionSet")
			for (profileId in names(settings$featureProfiles)) {
				profile <- settings$featureProfiles[[profileId]]
				key <- digest::digest(list(id, databaseIdentity(database), problem, profile, cacheManifest$packages,
					cacheManifest$implementation), algo = "sha256")
				folder <- file.path(settings$outputFolder, "data", key)
				paths[[id]][[problemId]][[profileId]] <- folder
				if (file.exists(file.path(folder, "complete.rds"))) next
				dir.create(folder, recursive = TRUE, showWarnings = FALSE)
				cohortTable <- ensureCohortTable(database, problem$cohortDefinitionSet,
					cohortTableName(database, key))
				covariateSettings <- profile$covariateSettings
				if (is.null(covariateSettings)) {
					covariateSettings <- FeatureExtraction::createCovariateSettings(
						useDemographicsAge = TRUE, useDemographicsGender = TRUE,
						useConditionOccurrenceLongTerm = profile$type == "standard",
						useDrugExposureLongTerm = profile$type == "standard",
						useProcedureOccurrenceLongTerm = profile$type == "standard",
						useObservationLongTerm = profile$type == "standard",
						longTermStartDays = -365, endDays = -1)
				}
				if (profile$type == "phenotype") {
					phenotypeTable <- ensureCohortTable(database, profile$cohortDefinitionSet,
						cohortTableName(database, key, phenotype = TRUE))
					phenotypes <- data.frame(cohortId = profile$cohortDefinitionSet$cohortId,
						cohortName = profile$cohortDefinitionSet$cohortName)
					covariateSettings <- list(covariateSettings,
						FeatureExtraction::createCohortBasedCovariateSettings(analysisId = 49,
							covariateCohortDatabaseSchema = database$cohortDatabaseSchema,
							covariateCohortTable = phenotypeTable, covariateCohorts = phenotypes,
							startDay = -365, endDay = -1))
				}
				details <- PatientLevelPrediction::createDatabaseDetails(
					connectionDetails = database$connectionDetails, cdmDatabaseSchema = database$cdmDatabaseSchema,
					cdmDatabaseId = id, cdmDatabaseName = id,
					tempEmulationSchema = if (is.null(database$tempEmulationSchema)) database$cohortDatabaseSchema else database$tempEmulationSchema,
					cohortDatabaseSchema = database$cohortDatabaseSchema, cohortTable = cohortTable,
					targetId = problem$targetId, outcomeIds = problem$outcomeId)
				data <- PatientLevelPrediction::getPlpData(details, covariateSettings)
				tryCatch({
					population <- PatientLevelPrediction::createStudyPopulation(data,
						outcomeId = problem$outcomeId, populationSettings = problem$populationSettings)
					unlink(file.path(folder, "plpData"), recursive = TRUE)
					PatientLevelPrediction::savePlpData(data, file.path(folder, "plpData"), overwrite = TRUE)
					atomicSave(population, file.path(folder, "population.rds"))
					atomicSave(list(key = key, snapshotId = database$snapshotId), file.path(folder, "complete.rds"))
				}, finally = Andromeda::close(data$covariateData))
			}
		}
	}
	paths
}

openInput <- function(input, problem, databaseId) {
	owned <- is.character(input)
	if (owned) input <- list(plpData = PatientLevelPrediction::loadPlpData(file.path(input, "plpData")),
		population = readRDS(file.path(input, "population.rds")))
	if (is.null(input$plpData$covariateData) || is.null(input$population)) stop("Invalid prepared data")
	population <- input$population
	population <- population[order(population$subjectId), , drop = FALSE]
	if (anyNA(population$outcomeCount) || any(!population$outcomeCount %in% c(0, 1))) stop("Expected binary outcomes")
	meta <- input$plpData$metaData
	if (is.null(meta)) meta <- attr(input$plpData, "metaData")
	meta$targetId <- problem$targetId
	meta$outcomeId <- problem$outcomeId
	meta$populationSettings <- problem$populationSettings
	meta$cdmDatabaseId <- databaseId
	meta$cdmDatabaseName <- databaseId
	if (is.null(meta$covariateSettings)) meta$covariateSettings <- FeatureExtraction::createCovariateSettings(useDemographicsAge = TRUE)
	meta$splitSettings <- PatientLevelPrediction::createDefaultSplitSetting(splitSeed = 42)
	attr(input$plpData, "metaData") <- meta
	attr(population, "metaData") <- meta
	list(data = input$plpData, population = population, owned = owned)
}

cohortTableName <- function(database, key, phenotype = FALSE) {
	if (phenotype && !is.null(database$phenotypeCohortTable)) return(database$phenotypeCohortTable)
	if (!is.null(database$cohortTable)) {
		return(paste0(database$cohortTable, if (phenotype) "_phenotypes" else ""))
	}
	paste0("tl", substr(key, 1, 10), if (phenotype) "p" else "o")
}

ensureCohortTable <- function(database, definitions, tableName) {
	connection <- DatabaseConnector::connect(database$connectionDetails)
	on.exit(DatabaseConnector::disconnect(connection))
	if (DatabaseConnector::existsTable(connection, database$cohortDatabaseSchema, tableName)) {
		ParallelLogger::logInfo("Reusing cohort table ", database$cohortDatabaseSchema, ".", tableName)
		return(tableName)
	}
	tables <- CohortGenerator::getCohortTableNames(tableName)
	CohortGenerator::createCohortTables(connection = connection,
		cohortDatabaseSchema = database$cohortDatabaseSchema, cohortTableNames = tables,
		incremental = TRUE)
	CohortGenerator::generateCohortSet(connection = connection,
		cdmDatabaseSchema = database$cdmDatabaseSchema,
		cohortDatabaseSchema = database$cohortDatabaseSchema,
		tempEmulationSchema = if (is.null(database$tempEmulationSchema)) database$cohortDatabaseSchema else database$tempEmulationSchema,
		cohortTableNames = tables, cohortDefinitionSet = definitions)
	tableName
}
