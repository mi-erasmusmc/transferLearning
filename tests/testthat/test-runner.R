test_that("end-to-end jobs are paired, restartable, and preserve input", {
	folder <- tempfile()
	on.exit(unlink(folder, recursive = TRUE))
	source <- makeFixture(42); target <- makeFixture(43)
	on.exit(Andromeda::close(source$plpData$covariateData), add = TRUE)
	on.exit(Andromeda::close(target$plpData$covariateData), add = TRUE)
	before <- dplyr::collect(target$plpData$covariateData$covariates)
	settings <- fixtureSettings(folder)
	settings$learningCurve <- createLearningCurveSettings(trainingEvents = c(25, 1000),
		repetitions = 1, folds = 2, minClassCount = 5)
	settings$featureProfiles$phenotype <- list(type = "phenotype", ascertainmentReviewed = TRUE,
		cohortDefinitionSet = data.frame(cohortId = 1, cohortName = "test"))
	registry <- list(source = list(snapshotId = "s1"), target = list(snapshotId = "t1"))
	inputs <- list(source = list(example = list(standard = source, phenotype = source)), target = list(example = list(standard = target, phenotype = target)))
	result <- runExperiment(settings, registry, inputs)
	expect_equal(sort(result$status$status), c(rep("completed", 10), rep("skipped", 2)), info = paste(result$status$reason, collapse = "\n"))
	expect_equal(nrow(result$metrics), 10)
	expect_true(all(result$metrics$eventsTrain == 25))
	expect_true(all(result$metrics$budgetUnit == "events"))
	expect_s3_class(plotLearningCurves(result), "ggplot")
	expect_equal(nrow(result$intervals), 16)
	expect_equal(nrow(result$differences), 8)
	expect_s3_class(plotTransferEffects(result), "ggplot")
	expect_equal(dplyr::collect(target$plpData$covariateData$covariates), before)
	resumed <- runExperiment(settings, registry, inputs)
	expect_equal(result, resumed)
	settings$variances <- .2
	expect_error(runExperiment(settings, registry, inputs), "Manifest changed")
})

test_that("sampling is nested and patient holdouts are reproducible", {
	population <- data.frame(rowId = 1:200, subjectId = 1:200, outcomeCount = rep(c(0, 1), c(150, 50)))
	settings <- createLearningCurveSettings()
	split <- makePartition(population, settings, "db", "problem")
	expect_length(intersect(split$test, split$development), 0)
	expect_equal(split, makePartition(population, settings, "db", "problem"))
	expect_true(all(sampleNested(population, 40, 4) %in% sampleNested(population, 100, 4)))
	set.seed(5); before <- .Random.seed
	makeFolds(population, 5, 123)
	expect_identical(.Random.seed, before)
})

test_that("source units and absent features are mapped by semantic IDs", {
	fixture <- makeFixture()
	on.exit(Andromeda::close(fixture$plpData$covariateData))
	data <- fixture$plpData$covariateData
	attr(data, "metaData")$tidyCovariateDataSettings <- list(normFactors = data.frame(covariateId = c(20, 10), maxValue = c(5, 1)))
	source <- list(model = list(coefficients = data.frame(covariateIds = c("30", "(Intercept)", "20"), betas = c(3, -2, 4))),
		preprocessing = list(tidyCovariates = list(normFactors = data.frame(covariateId = c(20, 30), maxValue = c(10, 7)))))
	converted <- sourceInTargetUnits(source, data)
	expect_equal(converted$coefficients$betas, c(3, 2))
	expect_equal(converted$fallbackIds, "30")
	expect_equal(attr(converted$covariateData, "metaData")$tidyCovariateDataSettings$normFactors$maxValue, c(5, 1, 7))
	source$preprocessing <- NULL
	expect_error(sourceInTargetUnits(source, data), "Missing source normalization")
})

test_that("normalization learns only from fitting patients and raw cache reload preserves predictions", {
	fixture <- makeFixture(51)
	on.exit(Andromeda::close(fixture$plpData$covariateData))
	settings <- fixtureSettings(tempfile())
	input <- openInput(fixture, settings$problems$example, "target")
	training <- input$population[1:100, ]
	validation <- input$population[101:160, ]
	model <- fitVariance(input$data, training, 1, settings)
	raw <- dplyr::collect(input$data$covariateData$covariates)
	maximum <- max(raw$covariateValue[raw$covariateId == 20 & raw$rowId %in% training$rowId])
	factors <- model$preprocessing$tidyCovariates$normFactors
	expect_equal(factors$maxValue[factors$covariateId == 20], maximum)
	p <- predictValues(model, input$data, validation)
	# Raw PLP cache load follows the same route used by live database extraction.
	cache <- tempfile(); on.exit(unlink(cache, recursive = TRUE), add = TRUE)
	data <- fixture$plpData; class(data) <- "plpData"
	PatientLevelPrediction::savePlpData(data, file.path(cache, "plpData"))
	saveRDS(fixture$population, file.path(cache, "population.rds"))
	loaded <- openInput(cache, settings$problems$example, "target")
	on.exit(Andromeda::close(loaded$data$covariateData), add = TRUE)
	expect_equal(predictValues(model, loaded$data, validation), p, tolerance = 1e-12)
})

test_that("performance handles tied predictions and paired directions", {
	y <- c(0, 1, 0, 1)
	metrics <- performance(y, rep(.5, 4))
	expect_equal(unname(metrics["auroc"]), .5)
	expect_equal(unname(metrics["auprc"]), .5)
	expect_equal(unname(metrics["logLoss"]), log(2))
	intervals <- pairedIntervals(y, rep(.5, 4), c(.1, .9, .1, .9), 20, 2)
	expect_true(all(intervals$lower > 0))
})
