# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.

# Exercise the public fitting path rather than recognizing a version or argument.
# No persistence check: JSON numerical precision is a separate unresolved issue.
checkPlpBackend <- function() {
	tryCatch({
		n <- 120L
		x1 <- rep(c(0, 0, 1, 1), length.out = n)
		x2 <- rep(c(0, 1, 0, 1), length.out = n)
		covariates <- data.frame(rowId = rep(seq_len(n), 2),
			covariateId = rep(c(10, 20), each = n), covariateValue = c(x1, x2))
		# Rows 121 and 122 differ only in a feature absent from training.
		covariates <- rbind(covariates, data.frame(rowId = c(121, 122, 122),
			covariateId = c(10, 10, 30), covariateValue = c(0, 0, 2)))
		data <- Andromeda::andromeda(covariates = covariates,
			covariateRef = data.frame(covariateId = c(10, 20, 30),
				covariateName = c("a", "b", "absent"), analysisId = 1L, conceptId = c(10, 20, 30)),
			analysisRef = data.frame(analysisId = 1L, analysisName = "probe", domainId = "Condition", isBinary = "N"))
		class(data) <- "CovariateData"
		attr(data, "metaData") <- list(populationSize = n + 2L)
		on.exit(Andromeda::close(data))
		population <- data.frame(rowId = seq_len(n + 2L), subjectId = seq_len(n + 2L),
			outcomeCount = rep(c(0L, 1L), each = 4, length.out = n + 2L),
			survivalTime = 365, cohortStartDate = as.Date("2000-01-01"),
			daysToCohortEnd = 365, daysToObsEnd = 365, ageYear = 40, gender = 1)
		problem <- list(targetId = 1L, outcomeId = 2L,
			populationSettings = PatientLevelPrediction::createStudyPopulationSettings(firstExposureOnly = TRUE))
		input <- openInput(list(plpData = list(covariateData = data), population = population), problem, "backendProbe")
		source <- list(model = list(coefficients = data.frame(
			covariateIds = c("20", "(Intercept)", "30", "10"), betas = c(-2, -6, .7, 1))),
			preprocessing = list(tidyCovariates = list(normFactors = data.frame(
				covariateId = c(10, 20, 30), maxValue = 1))))
		settings <- list(threads = 1L, learningCurve = list(seed = 42L))
		fit <- fitVariance(input$data, input$population[seq_len(n), ], 1e-8, settings, source)
		coefficients <- fit$model$coefficients
		actual <- coefficients$betas[match(c("10", "20", "30"), coefficients$covariateIds)]
		if (!isTRUE(all.equal(actual, c(1, -2, .7), tolerance = 1e-7))) {
			stop("Source coefficients were not retained by covariate ID")
		}
		p <- predictValues(fit, input$data, input$population[n + 1:2, ])
		if (!isTRUE(all.equal(unname(diff(stats::qlogis(p))), 1.4, tolerance = 1e-7))) {
			stop("Source-only covariate did not contribute at prediction time")
		}
		if (!isTRUE(all.equal(dplyr::collect(data$covariates), covariates, check.attributes = FALSE))) {
			stop("PLP changed the caller's covariates")
		}
		invisible(TRUE)
	}, error = function(e) {
		stop("PLP backend behavioral check failed: ", conditionMessage(e),
			". Install the reviewed priorCoefs correctness build; see extras/UpstreamRequirements.md", call. = FALSE)
	})
}
