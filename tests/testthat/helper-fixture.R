makeFixture <- function(seed = 42, n = 160L) {
	withSeed(seed, {
		x <- cbind(stats::rbinom(n, 1, .5), stats::runif(n, 0, 10))
		y <- stats::rbinom(n, 1, stats::plogis(-1 + x[, 1] + .07 * x[, 2]))
		covariateData <- Andromeda::andromeda(
			covariates = data.frame(rowId = rep(seq_len(n), 2), covariateId = rep(c(10, 20), each = n), covariateValue = as.vector(x)),
			covariateRef = data.frame(covariateId = c(10, 20), covariateName = c("binary", "continuous"), analysisId = 1L, conceptId = c(10, 20)),
			analysisRef = data.frame(analysisId = 1L, analysisName = "test", domainId = "Condition", isBinary = "N"))
		class(covariateData) <- "CovariateData"
		attr(covariateData, "metaData") <- list(populationSize = n)
		population <- data.frame(rowId = seq_len(n), subjectId = seq_len(n), outcomeCount = y,
			survivalTime = 365, cohortStartDate = as.Date("2000-01-01"), daysToCohortEnd = 365,
			daysToObsEnd = 365, ageYear = 40, gender = 1)
		list(plpData = list(covariateData = covariateData), population = population)
	})
}
fixtureSettings <- function(folder) {
	createExperimentSettings(
		problems = list(example = list(targetId = 1L, outcomeId = 2L,
			populationSettings = PatientLevelPrediction::createStudyPopulationSettings(binary = TRUE, firstExposureOnly = TRUE))),
		pairs = data.frame(sourceId = "source", targetId = "target"),
		featureProfiles = list(standard = list(type = "standard")), outputFolder = folder,
		learningCurve = createLearningCurveSettings(trainingSizes = c(60, 1000), repetitions = 1,
			folds = 2, minClassCount = 5), variances = c(.01, 1), bootstrapReplicates = 3)
}
