# Run from repository root: Rscript extras/tests/comparePlpAutoCv.R
.libPaths(c('.library', .libPaths()))
pkgload::load_all('.', quiet = TRUE)
source('tests/testthat/helper-fixture.R')
source('extras/comparePlpAutoCv.R')
root <- tempfile(); dir.create(root)
fixture <- makeFixture(42)
settings <- fixtureSettings(root)
input <- TransferLearning:::openInput(fixture, settings$problems$example, 'target')
train <- input$population[1:120, ]; test <- input$population[121:160, ]
folds <- TransferLearning:::makeFolds(train, 3L, 42L)
raw <- dplyr::collect(fixture$plpData$covariateData$covariates)
sourceModel <- TransferLearning:::fitVariance(input$data, input$population[1:100, ], 1, settings)
sourceKey <- digest::digest(list('source', 'example', 'standard'), algo = 'xxhash64')
# Use a known different source age scale to verify one-time unit conversion.
sourceModel$preprocessing$tidyCovariates$normFactors$maxValue[
  sourceModel$preprocessing$tidyCovariates$normFactors$covariateId == 20] <- 20
sourceModel$model$coefficients$betas[sourceModel$model$coefficients$covariateIds == '20'] <- 1
PatientLevelPrediction::savePlpModel(sourceModel, file.path(root, 'sources', sourceKey, 'model'))
manifest <- list(settings = settings, databases = list(target = list()),
  packages = TransferLearning:::packageVersions(), implementation = TransferLearning:::functionFingerprint('TransferLearning'),
  backend = TransferLearning:::functionFingerprint('PatientLevelPrediction'), hash = 'fixture')
saveRDS(manifest, file.path(root, 'manifest.rds'))
cacheKey <- digest::digest(list('target', list(), settings$problems$example,
  settings$featureProfiles$standard, manifest$packages, manifest$implementation), algo = 'sha256')
cache <- file.path(root, 'data', cacheKey)
data <- fixture$plpData; class(data) <- 'plpData'
PatientLevelPrediction::savePlpData(data, file.path(cache, 'plpData'))
saveRDS(fixture$population, file.path(cache, 'population.rds'))
path <- file.path(root, 'jobs', 'fixture'); dir.create(path, recursive = TRUE)
saveRDS(list(rowIds = train$rowId, testRowIds = test$rowId, folds = folds), file.path(path, 'split.rds'))
job <- data.frame(problemId = 'example', profileId = 'standard', pairIndex = 1L,
  repetition = 1L, trainingBudget = sum(train$outcomeCount), budgetUnit = 'events', sourceId = 'source', targetId = 'target')
saveRDS(list(status = cbind(job[c(1,1), ], method = c('targetOnly', 'priorCoefs'), status = 'completed', reason = '')),
  file.path(path, 'result.rds'))
predictions <- list()
for (method in c('targetOnly', 'priorCoefs')) {
  model <- TransferLearning:::fitVariance(input$data, train, .01, settings,
    if (method == 'priorCoefs') sourceModel else NULL)
  PatientLevelPrediction::savePlpModel(model, file.path(path, method, 'model'))
  saveRDS(list(selectedVariance = .01), file.path(path, method, 'tuning.rds'))
  predictions[[method]] <- rev(TransferLearning:::predictValues(model, input$data, test))
}
saveRDS(list(population = test[nrow(test):1, ], predictions = predictions), file.path(path, 'predictions.rds'))
files <- list.files(root, recursive = TRUE, full.names = TRUE)
before <- tools::md5sum(files)
out <- comparePlpAutoCv(root)
stopifnot(nrow(out) == 4L, all(out$status == 'completed'), all(out$nTrain == 120), all(out$nTest == 40),
  all(out$rescaledSourcePredictors[out$method == 'priorCoefs'] == 1),
  all(is.finite(out$autoVariance)), all(out$autoVariance > 0),
  all(is.finite(out$auto_auroc)), identical(tools::md5sum(files), before),
  identical(dplyr::collect(fixture$plpData$covariateData$covariates), raw),
  !any(c('rowId', 'subjectId', 'covariateId', 'betas', 'error') %in% names(out)))
localFiles <- list.files(file.path(root, 'plp-auto-cv', 'local'), full.names = TRUE)
times <- file.info(localFiles)$mtime
resumed <- comparePlpAutoCv(root)
stopifnot(identical(out, resumed), identical(times, file.info(localFiles)$mtime))
# Fold assignments supplied to PLP retain multiple folds; no fixed-variance shortcut.
fit <- autoCvFit(input$data, train, folds, sourceModel, settings, 'matchedPreprocessing', .01)
prior <- fit$model$modelDesign$modelSettings$param$priorCoefs
maxTarget <- max(raw$covariateValue[raw$covariateId == 20 & raw$rowId %in% train$rowId])
stopifnot(any(fit$model$prediction$evaluationType == 'CV'),
  NROW(fit$model$trainDetails$hyperParamSearch) > 0,
  abs(prior$betas[prior$covariateIds == '20'] - maxTarget / 20) < 1e-10)
# Binary feature normalization does not change a transferred binary coefficient.
priorBinary <- prior$betas[prior$covariateIds == '10']
sourceBinary <- sourceModel$model$coefficients$betas[sourceModel$model$coefficients$covariateIds == '10']
if (length(priorBinary)) stopifnot(identical(priorBinary, sourceBinary))
# Reference changes and live runner locks are rejected, without refitting.
saveRDS(list(selectedVariance = .02), file.path(path, 'targetOnly', 'tuning.rds'))
stopifnot(inherits(try(comparePlpAutoCv(root), silent = TRUE), 'try-error'))
dir.create(file.path(root, '.runner-lock'))
stopifnot(inherits(try(comparePlpAutoCv(root), silent = TRUE), 'try-error'))
Andromeda::close(fixture$plpData$covariateData)
unlink(root, recursive = TRUE)
cat('AUTO CV COMPARISON TESTS PASSED\n')
