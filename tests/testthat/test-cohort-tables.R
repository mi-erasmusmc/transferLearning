test_that("explicit cohort tables are reused and missing tables are generated", {
  calls <- list(); present <- FALSE
  local_mocked_bindings(
    connect = function(...) "connection",
    disconnect = function(...) calls$disconnect <<- TRUE,
    existsTable = function(connection, databaseSchema, tableName) {
      expect_equal(connection, "connection")
      expect_equal(databaseSchema, "catalog.scratch")
      expect_equal(tableName, "pilot_cohorts")
      present
    }, .package = "DatabaseConnector")
  local_mocked_bindings(
    createCohortTables = function(...) calls$create <<- list(...),
    generateCohortSet = function(...) calls$generate <<- list(...),
    .package = "CohortGenerator")
  db <- list(connectionDetails = "details", cdmDatabaseSchema = "catalog.cdm",
    cohortDatabaseSchema = "catalog.scratch", cohortTable = "pilot_cohorts")
  definitions <- data.frame(cohortId = c(1, 2))
  expect_equal(ensureCohortTable(db, definitions, db$cohortTable), "pilot_cohorts")
  expect_true(calls$create$incremental)
  expect_equal(calls$generate$cohortDefinitionSet, definitions)
  expect_equal(calls$generate$cdmDatabaseSchema, "catalog.cdm")
  expect_equal(calls$generate$tempEmulationSchema, "catalog.scratch")
  expect_equal(calls$create$cohortTableNames$cohortTable, "pilot_cohorts")
  expect_true(calls$disconnect)
  present <- TRUE; calls <- list()
  expect_equal(ensureCohortTable(db, definitions, db$cohortTable), "pilot_cohorts")
  expect_null(calls$create)
  expect_null(calls$generate)
  expect_true(calls$disconnect)
})

test_that("table names and data locations define credential-free cache identity", {
  db <- list(cohortTable = "pilot", cdmDatabaseSchema = "catalog.cdm",
    connectionDetails = list(password = "not-for-manifests"))
  expect_equal(cohortTableName(db, "abcdef"), "pilot")
  expect_equal(cohortTableName(db, "abcdef", TRUE), "pilot_phenotypes")
  db$phenotypeCohortTable <- "custom_predictors"
  expect_equal(cohortTableName(db, "abcdef", TRUE), "custom_predictors")
  expect_equal(cohortTableName(list(), "abcdef"), "tlabcdefo")
  identity <- databaseIdentity(db)
  expect_false("connectionDetails" %in% names(identity))
  db$cohortTable <- "other_table"
  expect_false(identical(identity, databaseIdentity(db)))
})

test_that("snapshot labels are optional but supplied table names are validated", {
  local_mocked_bindings(checkPlpBackend = function() invisible(TRUE))
  settings <- fixtureSettings(tempfile())
  registry <- list(source = list(), target = list())
  expect_s3_class(validateExperiment(settings, registry), "data.frame")
  registry$source$cohortTable <- "catalog.schema.table"
  expect_error(validateExperiment(settings, registry), "unqualified table name")
  registry$source$cohortTable <- "source_cohorts"
  registry$source$snapshotId <- ""
  expect_error(validateExperiment(settings, registry), "when supplied")
})
