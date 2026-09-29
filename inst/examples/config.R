# Run from a study directory containing frozen CohortGenerator definition sets.
# Definitions and registry are supplied by the study environment, not this repo.
problems <- list(
  example = list(
    targetId = 1L,
    outcomeId = 2L,
    cohortDefinitionSet = readRDS("problem-cohorts.rds"),
    populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
      binary = TRUE, firstExposureOnly = TRUE,
      washoutPeriod = 365, riskWindowStart = 1, riskWindowEnd = 365
    )
  )
)
profiles <- list(
  standard = list(type = "standard"),
  phenotype = list(type = "phenotype",
    cohortDefinitionSet = readRDS("reviewed-phenotype-cohorts.rds"),
    ascertainmentReviewed = TRUE)
)
settings <- TransferLearning::createExperimentSettings(
  problems = problems,
  pairs = data.frame(sourceId = c("databaseA", "databaseB"),
    targetId = c("databaseB", "databaseA")),
  featureProfiles = profiles,
  outputFolder = "experiment-results"
)
# Each registry entry needs connectionDetails, cdmDatabaseSchema,
# cohortDatabaseSchema (writable experiment scratch schema), and optionally
# tempEmulationSchema. Set cohortTable to reuse an existing table or generate
# it if missing; snapshotId is optional. Populate connections from environment secrets.
databaseRegistry <- readRDS("local-database-registry.rds")
