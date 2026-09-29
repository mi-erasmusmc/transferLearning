test_that("a source-only feature is dropped and no overlap matches target-only", {
	fixture <- makeFixture(53)
	on.exit(Andromeda::close(fixture$plpData$covariateData))
	data <- fixture$plpData$covariateData
	covariates <- dplyr::collect(data$covariates)
	covariates$covariateValue[covariates$rowId %in% c(159, 160)] <- 0
	data$covariates <- rbind(covariates, data.frame(rowId = 160, covariateId = 30, covariateValue = 7))
	data$covariateRef <- rbind(dplyr::collect(data$covariateRef),
		data.frame(covariateId = 30, covariateName = "source-only", analysisId = 1L, conceptId = 30))
	settings <- fixtureSettings(tempfile())
	input <- openInput(fixture, settings$problems$example, "target")
	# A complete source slope/scale contract; the target never observes feature 30.
	source <- list(model = list(coefficients = data.frame(covariateIds = c("(Intercept)", "30"), betas = c(-6, 2))),
		preprocessing = list(tidyCovariates = list(normFactors = data.frame(covariateId = 30, maxValue = 7))))
	fit <- fitVariance(input$data, input$population[1:100, ], 1, settings, source)
	expect_equal(fit$transferDetails$droppedSourceIds, "30")
	p <- predictValues(fit, input$data, input$population[159:160, ])
	expect_equal(diff(stats::qlogis(p)), 0, tolerance = 1e-8)
	expect_false("30" %in% fit$model$coefficients$covariateIds)
	expect_false(30 %in% fit$preprocessing$tidyCovariates$normFactors$covariateId)
	baseline <- fitVariance(input$data, input$population[1:100, ], 1, settings)
	expect_equal(fit$model$coefficients, baseline$model$coefficients)
	expect_equal(predictValues(fit, input$data, input$population),
		predictValues(baseline, input$data, input$population), tolerance = 1e-8)
})


test_that("presence and source conversion are fold-specific and final refit restores observed features", {
	fixture <- makeFixture(54)
	on.exit(Andromeda::close(fixture$plpData$covariateData))
	data <- fixture$plpData$covariateData
	raw <- dplyr::collect(data$covariates)
	data$covariates <- rbind(raw, data.frame(rowId = 101:160, covariateId = 30, covariateValue = 7))
	data$covariateRef <- rbind(dplyr::collect(data$covariateRef),
		data.frame(covariateId = 30, covariateName = "rare", analysisId = 1L, conceptId = 30))
	before <- dplyr::collect(data$covariates)
	settings <- fixtureSettings(tempfile())
	settings$variances <- 1e-8
	input <- openInput(fixture, settings$problems$example, "target")
	source <- list(model = list(coefficients = data.frame(covariateIds = c("30", "10"), betas = c(2, .5))),
		preprocessing = list(tidyCovariates = list(normFactors = data.frame(covariateId = c(30, 10), maxValue = c(14, 1)))))
	folds <- c(rep(1L, 100), rep(2L, 60))
	fit <- tuneModel(input$data, input$population, folds, settings, source)
	expect_true(all(fit$tuning$droppedSourceIds[fit$tuning$fold == 2] == "30"))
	expect_true(all(fit$tuning$droppedSourceIds[fit$tuning$fold == 1] == ""))
	expect_length(fit$transferDetails$droppedSourceIds, 0)
	expect_true("30" %in% fit$model$model$coefficients$covariateIds)
	prior <- fit$model$modelDesign$modelSettings$param$priorCoefs
	expect_equal(prior$betas[prior$covariateIds == "30"], 1)
	expect_equal(dplyr::collect(data$covariates), before)
	expect_identical(folds, c(rep(1L, 100), rep(2L, 60)))
})
