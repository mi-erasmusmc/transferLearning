# Synthetic readiness check only: no connection details and no database access.
settings <- TransferLearning::createExperimentSettings(
  problems = list(preflight = list(targetId = 1L, outcomeId = 2L,
    populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
      binary = TRUE, firstExposureOnly = TRUE))),
  pairs = data.frame(sourceId = "source", targetId = "target"),
  featureProfiles = list(preflight = list(type = "standard")),
  outputFolder = tempfile("transfer-preflight-")
)
invisible(TransferLearning::validateExperiment(settings,
  list(source = list(snapshotId = "synthetic"),
    target = list(snapshotId = "synthetic"))))
message("TransferLearning backend preflight passed; no database was accessed.")
