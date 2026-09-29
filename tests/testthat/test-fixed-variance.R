test_that("prepared PLP fits use the candidate outside CV limits without changing folds", {
	fixture <- makeFixture(61)
	on.exit(Andromeda::close(fixture$plpData$covariateData))
	settings <- fixtureSettings(tempfile())
	input <- openInput(fixture, settings$problems$example, "target")
	folds <- makeFolds(input$population, 2L, 71L)
	train <- subsetTrain(input$data, input$population, folds)
	on.exit(Andromeda::close(train$covariateData), add = TRUE)
	train$covariateData <- PatientLevelPrediction::preprocessData(train$covariateData,
		PatientLevelPrediction::createPreprocessSettings(normalize = TRUE, minFraction = 0, removeRedundancy = FALSE))
	before <- dplyr::collect(train$covariateData$covariates)
	beforeFolds <- train$folds
	prior <- data.frame(covariateIds = c("20", "10", "30"), betas = c(-2, 1, .7))
	for (source in list(NULL, prior)) {
		for (candidate in c(.123456, 50)) {
			modelSettings <- PatientLevelPrediction::setLassoLogisticRegression(
				variance = candidate, lowerLimit = 1, upperLimit = 2,
				priorCoefs = source, threads = 1, seed = 42)
			originalSettings <- modelSettings
			fit <- fitPreparedVariance(train, modelSettings)
			expect_equal(as.numeric(fit$model$priorVariance), candidate, tolerance = 1e-12)
			expect_equal(as.numeric(fit$trainDetails$finalModelParameters$variance), candidate, tolerance = 1e-12)
			expect_null(fit$model$cv)
			expect_true(all(fit$prediction$evaluationType == "Train"))
			expect_equal(NROW(fit$trainDetails$hyperParamSearch), 0L)
			expect_identical(modelSettings, originalSettings)
			# fitPlp's existing normalizer adds modelName/requiresDenseMatrix.
			expect_identical(fit$modelDesign$modelSettings$param, originalSettings$param)
			expect_identical(fit$modelDesign$modelSettings$settings[names(originalSettings$settings)], originalSettings$settings)
			expect_false("useCrossValidation" %in% names(fit$modelDesign$modelSettings$settings))
			expect_identical(train$folds, beforeFolds)
			expect_equal(dplyr::collect(train$covariateData$covariates), before)
			if (!is.null(source)) {
				coefs <- fit$model$coefficients
				expect_equal(coefs$betas[coefs$covariateIds == "30"], .7)
			}
		}
	}
})

test_that("outer tuning uses fold-specific units and final refit uses the full sample", {
	fixture <- makeFixture(63)
	on.exit(Andromeda::close(fixture$plpData$covariateData))
	raw <- dplyr::collect(fixture$plpData$covariateData$covariates)
	# Distinct fold maxima make accidental full-sample preprocessing detectable.
	raw$covariateValue[raw$covariateId == 20] <- seq_len(160) / 8
	fixture$plpData$covariateData$covariates <- raw
	settings <- fixtureSettings(tempfile())
	settings$variances <- c(.01, .1)
	input <- openInput(fixture, settings$problems$example, "target")
	folds <- rep(1:2, each = 80)
	input$data$folds <- data.frame(rowId = rev(input$population$rowId), index = rev(folds))
	assignments <- input$data$folds
	source <- list(model = list(coefficients = data.frame(covariateIds = "20", betas = 3)),
		preprocessing = list(tidyCovariates = list(normFactors = data.frame(covariateId = 20, maxValue = 40))))
	foldLoss <- numeric(2)
	for (fold in 1:2) {
		train <- input$population[folds != fold, ]
		validation <- input$population[folds == fold, ]
		fit <- fitVariance(input$data, train, .01, settings, source)
		maximum <- max(raw$covariateValue[raw$covariateId == 20 & raw$rowId %in% train$rowId])
		factors <- fit$preprocessing$tidyCovariates$normFactors
		expect_equal(factors$maxValue[factors$covariateId == 20], maximum)
		prior <- fit$modelDesign$modelSettings$param$priorCoefs
		expect_equal(prior$betas[prior$covariateIds == "20"], 3 * maximum / 40)
		foldLoss[fold] <- logLoss(validation$outcomeCount, predictValues(fit, input$data, validation))
	}
	fit <- tuneModel(input$data, input$population, folds, settings, source)
	expect_equal(fit$tuning$loss[fit$tuning$variance == .01], foldLoss)
	means <- vapply(split(fit$tuning, fit$tuning$variance), function(x) stats::weighted.mean(x$loss, x$n), numeric(1))
	expected <- as.numeric(names(means)[order(means, as.numeric(names(means)))][1])
	expect_equal(fit$selectedVariance, expected)
	expect_equal(as.numeric(fit$model$model$priorVariance), expected, tolerance = 1e-12)
	factors <- fit$model$preprocessing$tidyCovariates$normFactors
	expect_equal(factors$maxValue[factors$covariateId == 20], 20)
	prior <- fit$model$modelDesign$modelSettings$param$priorCoefs
	expect_equal(prior$betas[prior$covariateIds == "20"], 1.5)
	expect_setequal(fit$model$prediction$rowId, input$population$rowId)
	expect_identical(folds, rep(1:2, each = 80))
	expect_identical(input$data$folds, assignments)
	expect_equal(dplyr::collect(input$data$covariateData$covariates), raw)
	# The final fit should agree with a separate refit at the selected candidate.
	refit <- fitVariance(input$data, input$population, expected, settings, source)
	expect_equal(fit$model$model$coefficients, refit$model$coefficients)
})

test_that("backend readiness is behavioral and unexpected internal CV is rejected", {
	expect_true(checkPlpBackend())
	fit <- list(model = list(priorVariance = .5), prediction = data.frame(evaluationType = "Train"),
		trainDetails = list(hyperParamSearch = data.frame()))
	expect_true(assertFixedVarianceFit(fit, .5))
	expect_error(assertFixedVarianceFit(fit, .1), "candidate variance")
	fit$prediction$evaluationType <- "CV"
	expect_error(assertFixedVarianceFit(fit, .5), "internal CV")
	fit$prediction$evaluationType <- "Train"
	fit$trainDetails$hyperParamSearch <- data.frame(value = .5)
	expect_error(assertFixedVarianceFit(fit, .5), "tuning results")
})
