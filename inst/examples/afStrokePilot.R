# AF–stroke pilot entrypoint. Copy this file into your study environment,
# edit the site inputs below, then run: Rscript --vanilla afStrokePilot.R
# Requires TransferLearning, the corrected PLP build, and a Databricks JDBC driver.

# ---- Site-specific inputs ----
# Host only (no https://); HTTP path from your Databricks connection details.
databricksHost <- Sys.getenv("DATABRICKS_HOST")
databricksHttpPath <- Sys.getenv("DATABRICKS_HTTP_PATH")
databricksToken <- Sys.getenv("DATABRICKS_TOKEN")
pathToDriver <- Sys.getenv("DATABASECONNECTOR_JAR_FOLDER")

sourceId <- "optumEhr"
sourceCdmSchema <- ""       # e.g. catalog.optum_ehr_cdm
sourceSnapshotId <- ""      # actual data release identifier

targetId <- "mdcr"
targetCdmSchema <- ""       # e.g. catalog.mdcr_cdm
targetSnapshotId <- ""      # actual data release identifier

cohortDatabaseSchema <- ""  # writable catalog.schema for generated cohort tables
tempEmulationSchema <- cohortDatabaseSchema
outputFolder <- "af-stroke-pilot"  # storage on the machine running R

# ---- Pilot settings ----
trainingEvents <- c(25, 50, 100, 150, 200, 500, 1000)
repetitions <- 3L
folds <- 3L
threads <- 1L
bootstrapReplicates <- 200L

# ---- Construct runtime database registry ----
requiredInputs <- c("databricksHost", "databricksHttpPath", "databricksToken",
  "pathToDriver", "sourceCdmSchema", "sourceSnapshotId", "targetCdmSchema",
  "targetSnapshotId", "cohortDatabaseSchema", "tempEmulationSchema")
missingInputs <- requiredInputs[!vapply(mget(requiredInputs), function(value) {
  is.character(value) && length(value) == 1L && !is.na(value) && nzchar(trimws(value))
}, logical(1))]
if (length(missingInputs)) {
  stop("Set the site inputs at the top of this script: ",
    paste(missingInputs, collapse = ", "), call. = FALSE)
}

# Token-based JDBC example. Replace this with your site's existing
# createConnectionDetails() call if it uses another authentication method.
connectionDetails <- DatabaseConnector::createConnectionDetails(
  dbms = "spark",
  connectionString = paste0("jdbc:databricks://", databricksHost,
    ":443/default;transportMode=http;ssl=1;AuthMech=3;httpPath=",
    databricksHttpPath, ";"),
  user = "token", password = databricksToken, pathToDriver = pathToDriver
)
databaseRegistry <- setNames(list(
  list(connectionDetails = connectionDetails,
    cdmDatabaseSchema = sourceCdmSchema, snapshotId = sourceSnapshotId,
    cohortDatabaseSchema = cohortDatabaseSchema,
    tempEmulationSchema = tempEmulationSchema),
  list(connectionDetails = connectionDetails,
    cdmDatabaseSchema = targetCdmSchema, snapshotId = targetSnapshotId,
    cohortDatabaseSchema = cohortDatabaseSchema,
    tempEmulationSchema = tempEmulationSchema)
), c(sourceId, targetId))

# ---- Frozen problem and feature definitions bundled with the package ----
pilotPath <- system.file("pilot", package = "TransferLearning", mustWork = TRUE)
names <- c("af-target", "stroke-outcome")
json <- vapply(names, function(name) paste(readLines(file.path(pilotPath,
  paste0(name, ".json"))), collapse = "\n"), character(1))
cohorts <- data.frame(cohortId = c(1L, 2L), cohortName = names, json = unname(json),
  sql = vapply(json, function(j) as.character(CirceR::buildCohortQuery(
    CirceR::cohortExpressionFromJson(j), CirceR::createGenerateOptions())), character(1)))
phenotypes <- readRDS(file.path(pilotPath, "phenotypes.rds"))
# Conservative pilot default: omit visit-end-dependent stroke predictors pending
# prediction-time review; the frozen 51-feature source set remains intact.
phenotypes <- phenotypes[!phenotypes$cohortId %in% c(1155L, 1156L), ]
settings <- TransferLearning::createExperimentSettings(
  problems = list(afStroke = list(targetId = 1L, outcomeId = 2L,
    cohortDefinitionSet = cohorts,
    protocolCommit = "8f2b0866e11c717fa681e62c91dcf2bbec21ad61",
    populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
      binary = TRUE, firstExposureOnly = TRUE, washoutPeriod = 365,
      removeSubjectsWithPriorOutcome = TRUE, priorOutcomeLookback = 365,
      includeAllOutcomes = TRUE, requireTimeAtRisk = TRUE, minTimeAtRisk = 364,
      riskWindowStart = 1, riskWindowEnd = 365,
      startAnchor = "cohort start", endAnchor = "cohort start",
      restrictTarToCohortEnd = FALSE))),
  pairs = data.frame(sourceId = sourceId, targetId = targetId),
  featureProfiles = list(phenotypesDemographics = list(type = "phenotype",
    cohortDefinitionSet = phenotypes, ascertainmentReviewed = TRUE,
    phenotypeLibraryVersion = "3.37.0", antibioticGroupIds = 1201:1214,
    ascertainmentNotes = "Stroke predictors 1155/1156 omitted pending visit-end review; remaining original interval semantics retained")),
  learningCurve = TransferLearning::createLearningCurveSettings(
    trainingEvents = trainingEvents,
    repetitions = repetitions, testFraction = .25, folds = folds),
  variances = 10^seq(-6, 6, by = 2),
  outputFolder = outputFolder, threads = threads,
  bootstrapReplicates = bootstrapReplicates
)
# ---- Run extraction, fitting and reporting ----
results <- TransferLearning::runExperiment(settings, databaseRegistry)
if (any(results$status$status == "failed")) {
  stop("Some pilot jobs failed; inspect status.csv in ", outputFolder, call. = FALSE)
}
