database <- "optum_ehr"
version <- "v3323"
cdmDatabaseSchema <- paste0("cdm_", database, "_", version)
cohortDatabaseSchema <- "scratch_efridgei"
# fill in your connection details and path to driver
connectionDetails <- DatabaseConnector::createConnectionDetails(
  dbms = "redshift",
  user = "efridgei",
  password = keyring::key_get("database", "password"),
  server = paste0("ohda-prod-1.cldcoxyrkflo.us-east-1.redshift.amazonaws.com/", database),
  port = 5439,
  extraSettings = "ssl=true&sslfactory=com.amazon.redshift.ssl.NonValidatingFactory",
  
  pathToDriver = Sys.getenv("REDSHIFT_DRIVER")
)
cohortsToCreate <- CohortGenerator::createEmptyCohortDefinitionSet()
cohortJsonFiles <- list.files(path = "./cohorts/", full.names=TRUE)
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
  cohortsToCreate <- rbind(cohortsToCreate, data.frame(cohortId = cohortId,
                                                       cohortName = cohortName, 
                                                       sql = cohortSql,
                                                       stringsAsFactors = FALSE))
}

# Create the cohort tables to hold the cohort generation results
cohortTableNames <- CohortGenerator::getCohortTableNames(cohortTable = "transfer_learning")
CohortGenerator::createCohortTables(connectionDetails = connectionDetails,
                                    cohortDatabaseSchema = cohortDatabaseSchema,
                                    cohortTableNames = cohortTableNames)
# Generate the cohorts
cohortsGenerated <- CohortGenerator::generateCohortSet(connectionDetails = connectionDetails,
                                                       cdmDatabaseSchema = cdmDatabaseSchema,
                                                       cohortDatabaseSchema = cohortDatabaseSchema,
                                                       cohortTableNames = cohortTableNames,
                                                       cohortDefinitionSet = cohortsToCreate)

# extractData 
covariateSettings <- FeatureExtraction::createCovariateSettings(
  useDemographicsAge = TRUE,
  useDemographicsGender = TRUE,
  useConditionOccurrenceLongTerm = TRUE,
  useDrugExposureLongTerm = TRUE,
  useProcedureOccurrenceLongTerm = TRUE,
  useObservationLongTerm = TRUE)

databaseDetails <- PatientLevelPrediction::createDatabaseDetails(
  connectionDetails = connectionDetails,
  cdmDatabaseSchema = cdmDatabaseSchema,
  cdmDatabaseName = "optum-ehr",
  cdmDatabaseId = paste0("optum-ehr-", version),
  cohortDatabaseSchema = cohortDatabaseSchema,
  cohortTable = "transfer_learning",
  targetId = 1,
  outcomeIds = 2)

plpData <- PatientLevelPrediction::getPlpData(
  databaseDetails = databaseDetails,
  covariateSettings = covariateSettings
)
  
population <- PatientLevelPrediction::createStudyPopulation(
  plpData,
  populationSettings = PatientLevelPrediction::createStudyPopulationSettings(
    requireTimeAtRisk = FALSE,
    riskWindowStart = 1,
    riskWindowEnd = 3*365))

plpData$covariateData <- FeatureExtraction::tidyCovariateData(
  removeRedundancy = FALSE,
  plpData$covariateData
)

sparseData <- PatientLevelPrediction::toSparseM(plpData,
                                  cohort = population)

X <- sparseData$dataMatrix
y <- sparseData$labels$outcomeCount

doParallel::registerDoParallel(cores = 5)
cvFit <- glmnet::cv.glmnet(
  x = X,
  y = y,
  family = "binomial",
  alpha = 1,
  nfolds = 5,
  trace.it = 1,
  parallel = TRUE
)

bestLambda <- cvFit$lambda.min
beta <- as.numeric(coef(cvFit, s = bestLambda))
support <- which(beta != 0)
covariateRef <- sparseData$covariateRef
covariateMap <- sparseData$covariateMap
saveRDS(list(
  fitObject = cvFit,
  lambda = bestLambda,
  support = support,
  covariateMap = covariateMap,
  covariateRef = covariateRef),
  file = "pretrainedLungCancer.rds")
