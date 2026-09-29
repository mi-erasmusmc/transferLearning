test_that("a source-only feature contributes when it first appears at prediction time", {
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
	expect_equal(fit$transferDetails$fallbackIds, "30")
	p <- predictValues(fit, input$data, input$population[159:160, ])
	expect_equal(diff(stats::qlogis(p)), 2, tolerance = 1e-8)
})
