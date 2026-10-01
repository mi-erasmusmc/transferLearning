test_that("extensions reuse legacy caches, models and jobs across reordered grids", {
  folder <- tempfile(); on.exit(unlink(folder, recursive = TRUE))
  source <- makeFixture(42); target <- makeFixture(43)
  on.exit(Andromeda::close(source$plpData$covariateData), add = TRUE)
  on.exit(Andromeda::close(target$plpData$covariateData), add = TRUE)
  settings <- fixtureSettings(folder)
  settings$methods <- c("targetOnly", "priorCoefs")
  settings$learningCurve <- createLearningCurveSettings(trainingEvents = 25,
    repetitions = 1, folds = 2, minClassCount = 5)
  settings$problems$example$cohortDefinitionSet <- data.frame(cohortId = 1:2)
  registry <- list(source = list(), target = list())
  inputs <- list(source = list(example = list(standard = source)),
    target = list(example = list(standard = target)))
  original <- runExperiment(settings, registry, inputs)
  expect_true(all(original$status$status == "completed"))
  # Recreate the old manifest, cache keys and row-dependent job directory.
  manifestPath <- file.path(folder, "manifest.rds")
  manifest <- readRDS(manifestPath)
  manifest$reuseContract <- NULL
  manifest$implementation <- legacyResumeFingerprints()[3]
  saveRDS(manifest, manifestPath)
  unlink(file.path(folder, "resume-manifest.rds"))
  for (id in names(inputs)) {
    key <- digest::digest(list(id, manifest$databases[[id]], settings$problems$example,
      settings$featureProfiles$standard, manifest$packages, manifest$implementation), algo = "sha256")
    cache <- file.path(folder, "data", key)
    data <- inputs[[id]]$example$standard$plpData; class(data) <- "plpData"
    PatientLevelPrediction::savePlpData(data, file.path(cache, "plpData"))
    saveRDS(inputs[[id]]$example$standard$population, file.path(cache, "population.rds"))
    saveRDS(list(key = key), file.path(cache, "complete.rds"))
  }
  oldPath <- dirname(list.files(file.path(folder, "jobs"), "^result.rds$", recursive = TRUE, full.names = TRUE))
  legacyPath <- file.path(folder, "jobs", "legacy-row-hash")
  expect_true(file.rename(oldPath, legacyPath))
  files <- c(manifestPath, list.files(file.path(folder, "sources"), recursive = TRUE, full.names = TRUE),
    list.files(legacyPath, recursive = TRUE, full.names = TRUE),
    list.files(file.path(folder, "data"), recursive = TRUE, full.names = TRUE),
    list.files(file.path(folder, "splits"), recursive = TRUE, full.names = TRUE))
  checksums <- tools::md5sum(files); times <- file.info(files)$mtime
  settings$learningCurve$trainingBudgets <- c(20, 25)
  settings$learningCurve$repetitions <- 3L
  # No connection details: this fails if cached extraction is not reused.
  extended <- runExperiment(settings, registry)
  expect_equal(nrow(extended$status), 12)
  expect_true(all(extended$status$status == "completed"), info = paste(extended$status$reason, collapse = "\n"))
  expect_identical(tools::md5sum(files), checksums)
  expect_identical(file.info(files)$mtime, times)
  expect_equal(length(indexExperimentJobs(folder)), 6)
  expect_equal(readRDS(file.path(folder, "resume-manifest.rds"))$settings$learningCurve$repetitions, 3L)
  expect_equal(runExperiment(settings, registry), extended)
  # A failed method is retried without refitting the successful companion.
  resultPath <- file.path(legacyPath, "result.rds")
  result <- readRDS(resultPath)
  result$status$status[result$status$method == "priorCoefs"] <- "failed"
  saveRDS(result, resultPath)
  fitPath <- file.path(legacyPath, "targetOnly", "tuning.rds")
  fitTime <- file.info(fitPath)$mtime
  retried <- runExperiment(settings, registry)
  expect_true(all(retried$status$status == "completed"), info = paste(retried$status$reason, collapse = "\n"))
  expect_identical(file.info(fitPath)$mtime, fitTime)
  expect_identical(readRDS(resultPath)$provenance$reusedMethods, "targetOnly")
  smaller <- settings; smaller$learningCurve$repetitions <- 2L
  expect_error(runExperiment(smaller, registry), "only additional")
  expect_error(runExperiment(settings, registry, resume = FALSE), "Existing experiment")
})

test_that("resume rejects scientific changes and unknown legacy builds", {
  settings <- fixtureSettings(tempfile())
  base <- list(settings = settings, databases = list(), packages = c(TransferLearning = "1", Cyclops = "1"),
    implementation = legacyResumeFingerprints()[3], backend = "plp",
    reuseContract = list(version = 1L, science = experimentScienceFingerprint()))
  expect_identical(base$reuseContract$science, legacyCompatibleScienceFingerprint())
  legacy <- base; legacy$reuseContract <- NULL
  expect_silent(assertExperimentExtension(legacy, base))
  legacy$implementation <- "unknown"
  expect_error(assertExperimentExtension(legacy, base), "legacy")
  changed <- base; changed$packages["Cyclops"] <- "2"
  expect_error(assertExperimentExtension(base, changed), "dependency")
  changed <- base; changed$backend <- "other"
  expect_error(assertExperimentExtension(base, changed), "PLP implementation")
  changed <- base; changed$settings$learningCurve$seed <- 999
  expect_error(assertExperimentExtension(base, changed), "scientific settings")
  changed <- base; changed$reuseContract$science <- "other"
  expect_error(assertExperimentExtension(base, changed), "implementation")
  changed <- base; changed$implementation <- "reporting-only update"
  expect_silent(assertExperimentExtension(base, changed))
})
