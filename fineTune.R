library(glmnet)


database <- "ducky"
cdmDatabaseSchema <- "main"
cohortDatabaseSchema <- "cohorts"
# fill in your connection details and path to driver
connectionDetails <- DatabaseConnector::createConnectionDetails(
  dbms = "duckdb",
  server = "~/database/database-1M_filtered.duckdb"
)
cohortsToCreate <- CohortGenerator::createEmptyCohortDefinitionSet()
cohortJsonFiles <- list.files(path = "./cohorts/", full.names = TRUE)
for (i in 1:length(cohortJsonFiles)) {
  cohortJsonFileName <- cohortJsonFiles[i]
  cohortName <- tools::file_path_sans_ext(basename(cohortJsonFileName))
  cohortJson <- readChar(cohortJsonFileName, file.info(cohortJsonFileName)$size)
  cohortExpression <- CirceR::cohortExpressionFromJson(cohortJson)
  cohortSql <- CirceR::buildCohortQuery(cohortExpression, options = CirceR::createGenerateOptions(generateStats = FALSE))
  if (grepl("target", cohortJsonFiles[[i]])) {
    cohortId <- 1
  } else {
    cohortId <- 2
  }
  cohortsToCreate <- rbind(cohortsToCreate, data.frame(
    cohortId = cohortId,
    cohortName = cohortName,
    sql = cohortSql,
    stringsAsFactors = FALSE
  ))
}

# Create the cohort tables to hold the cohort generation results
cohortTableNames <- CohortGenerator::getCohortTableNames(cohortTable = "transfer_learning")
CohortGenerator::createCohortTables(
  connectionDetails = connectionDetails,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTableNames = cohortTableNames
)
# Generate the cohorts
cohortsGenerated <- CohortGenerator::generateCohortSet(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTableNames = cohortTableNames,
  cohortDefinitionSet = cohortsToCreate
)

# extractData
covariateSettings <- FeatureExtraction::createCovariateSettings(
  useDemographicsAge = TRUE,
  useDemographicsGender = TRUE,
  useConditionOccurrenceLongTerm = TRUE,
  useDrugExposureLongTerm = TRUE,
  useProcedureOccurrenceLongTerm = TRUE,
  useObservationLongTerm = TRUE
)

databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cdmDatabaseName = "ducky",
  cdmDatabaseId = paste0("ducky"),
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTable = "transfer_learning",
  targetId = 1,
  outcomeIds = 2
)

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = covariateSettings
)

population <- PatientLevelPrediction::createStudyPopulation(
  plpData,
  populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
    requireTimeAtRisk = FALSE,
    riskWindowStart = 1,
    riskWindowEnd = 3 * 365
  )
)

allAucs <- c()
allAlphas <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
for (alpha in allAlphas) {
  splittedData <- PatientLevelPrediction::splitData(
    plpData,
    population = population,
    splitSettings = PatientLevelPrediction::createDefaultSplitSetting(
      nfold = 4,
      splitSeed = 42
    )
  )

  trainData <- splittedData$Train
  testData <- splittedData$Test

  trainData$covariateData <- FeatureExtraction::tidyCovariateData(
    removeRedundancy = FALSE,
    plpData$covariateData
  )

  preTrained <- readRDS("pretrainedLungCancer.rds")
  sparseData <- PatientLevelPrediction::toSparseM(trainData,
    cohort = trainData$labels,
    map = preTrained$covariateMap
  )

  XTrain <- sparseData$dataMatrix
  yTrain <- sparseData$labels$outcomeCount

  # compute linear predictor of using source model on target data
  eta <- predict(
    preTrained$fitObject,
    newx = XTrain,
    s = preTrained$lambda,
    type = "link"
  )
  offset <- (1 - alpha) * eta

  p <- ncol(XTrain)
  penaltyFactor <- rep(1 / alpha, p)
  penaltyFactor[preTrained$support] <- 1

  doParallel::registerDoParallel(cores = 4)
  cvFit <- glmnet::cv.glmnet(
    x = XTrain,
    y = yTrain,
    family = "binomial",
    alpha = 1,
    foldid = trainData$folds$index,
    offset = offset,
    penalty.factor = penaltyFactor,
    trace.it = 1,
    parallel = TRUE,
    type.measure = "auc"
  )
  tidyCovariates <- attr(trainData$covariateData, "metaData")
  testData$covariateData <- PatientLevelPrediction:::applyTidyCovariateData(
    testData$covariateData,
    preprocessSettings = tidyCovariates
  )


  sparseTestData <- PatientLevelPrediction::toSparseM(testData,
    cohort = testData$labels,
    map = preTrained$covariateMap
  )
  XTest <- sparseTestData$dataMatrix
  yTest <- sparseTestData$labels$outcomeCount

  etaBig <- predict(
    preTrained$fitObject,
    newx = XTest,
    s = preTrained$lambda,
    type = "link"
  )

  offsetTest <- (1 - alpha) * etaBig

  etaTest <- predict(
    cvFit,
    newx = XTest,
    newoffset = offsetTest,
    s = cvFit$lambda.min,
    type = "link"
  )


  propTest <- plogis(etaTest)

  rocObj <- pROC::roc(
    response = yTest,
    predictor = as.numeric(propTest),
    quiet = TRUE
  )
  auc <- pROC::auc(rocObj)
  allAucs <- c(allAucs, auc)
  print(paste0("Alpha: ", alpha, "AUC: ", auc))
}

df <- data.frame(
  alpha = allAlphas,
  auc = allAucs
)

library(ggplot2)
ggplot(df, aes(alpha, auc)) +
  geom_point() +
  labs(x = expression(alpha), y = "AUC", title = "Test AUC vs α") 

ggsave("aucVsAlpha.png", width = 8, height = 6, dpi = 300)
