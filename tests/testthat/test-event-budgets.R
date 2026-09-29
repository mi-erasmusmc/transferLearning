test_that("event budgets are exact, nested, proportional, and leave RNG untouched", {
  population <- data.frame(outcomeCount = c(rep(1, 43), rep(0, 957)))
  set.seed(42); before <- .Random.seed
  small <- sampleNested(population, 7, 5, "events")
  large <- sampleNested(population, 20, 5, "events")
  expect_equal(sum(population$outcomeCount[small]), 7)
  expect_equal(sum(population$outcomeCount[large]), 20)
  expect_equal(length(small) - 7, round(7 * 957 / 43))
  expect_true(all(small %in% large))
  expect_equal(sampleNested(population, Inf, 5, "events"), 1:1000)
  expect_equal(sort(sampleNested(population, 43, 5, "events")), 1:1000)
  expect_error(sampleNested(population, 44, 5, "events"), "exceeds")
  expect_identical(.Random.seed, before)
  expect_equal(createLearningCurveSettings()$budgetUnit, "events")
  expect_equal(createLearningCurveSettings(trainingSizes = 100)$budgetUnit, "patients")
  expect_error(createLearningCurveSettings(trainingSizes = 100, trainingEvents = 25), "not both")
  expect_error(createLearningCurveSettings(trainingEvents = 2.5))
})

test_that("pilot replaces all antibiotic groups with one exposure feature", {
  path <- system.file("pilot", package = "TransferLearning")
  x <- readRDS(file.path(path, "phenotypes.rds"))
  original <- readRDS(file.path(path, "phenotypes-original.rds"))
  expect_equal(nrow(x), 51)
  expect_false(any(x$cohortId %in% 1201:1214))
  expect_equal(sum(x$cohortId == 900001), 1)
  expect_equal(x$json[x$cohortId != 900001], original$json[!original$cohortId %in% 1201:1214])
  skip_if_not_installed("jsonlite")
  merged <- jsonlite::fromJSON(x$json[x$cohortId == 900001], simplifyVector = FALSE)
  expect_equal(merged$QualifiedLimit$Type, "All")
  expect_equal(merged$EndStrategy$DateOffset$Offset, 0)
  ids <- function(z) {
    id <- z$PrimaryCriteria$CriteriaList[[1]]$DrugExposure$CodesetId
    cs <- z$ConceptSets[[which(vapply(z$ConceptSets, function(v) v$id == id, logical(1)))]]
    vapply(cs$expression$items, function(i) i$concept$CONCEPT_ID, numeric(1))
  }
  earlier <- unlist(lapply(original$json[original$cohortId %in% 1201:1214], function(j) ids(jsonlite::fromJSON(j, simplifyVector = FALSE))))
  expect_setequal(ids(merged), earlier)
})
