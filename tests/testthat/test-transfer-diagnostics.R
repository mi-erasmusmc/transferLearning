test_that("diagnostic decomposition handles new, revised and source-only predictors by ID", {
  source <- list(model = list(coefficients = data.frame(covariateIds = c("30", "10", "(Intercept)"),
    betas = c(7, 4, -9))), preprocessing = list(tidyCovariates = list(normFactors =
    data.frame(covariateId = c(30, 10), maxValue = c(1, 2)))))
  target <- list(model = list(coefficients = data.frame(covariateIds = c("20", "(Intercept)", "10"),
    betas = c(3, -1, -2))), preprocessing = list(tidyCovariates = list(normFactors =
    data.frame(covariateId = c(10, 20), maxValue = c(4, 1)))))
  v <- transferCoefficientVectors(source, target)
  expect_equal(v$source, c(8, 0)); expect_equal(v$target, c(-2, 3))
  raw <- data.frame(rowId = c(1, 2, 1, 3, 1), covariateId = c(10, 10, 20, 20, 30),
    covariateValue = c(2, 4, 1, 1, 50))
  parts <- transferLinearComponents(raw, c(3, 1, 2), v, 1e-6)
  expect_equal(unname(parts), cbind(c(0, 4, 8), c(0, -5, -10), c(3, 3, 0)))
  expect_equal(rowSums(parts), c(3, 2, -2))
  expect_equal(transferTrainingSd(raw, 1:3, v), c(stats::sd(c(.5, 1, 0)), stats::sd(c(1, 0, 1))))
  expect_equal(stats::plogis(v$intercept + rowSums(parts)), stats::plogis(c(2, 1, -3)))
})

test_that("diagnostic decomposition accepts intercept-only sparse input", {
  v <- list(ids = c("10"), scale = 1, source = 0, target = 0)
  raw <- data.frame(rowId = numeric(), covariateId = numeric(), covariateValue = numeric())
  expect_equal(unname(transferLinearComponents(raw, 1:3, v, 1e-6)), matrix(0, 3, 3))
  expect_equal(transferTrainingSd(raw, 1:3, v), 0)
})
