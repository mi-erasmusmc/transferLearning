# Run from the repository root: Rscript extras/tests/sourceModelDiagnostics.R
.libPaths(c('.library', .libPaths()))
pkgload::load_all('.', quiet=TRUE)
source('tests/testthat/helper-fixture.R')
source('extras/sourceModelDiagnostics.R')
x <- makeFixture()
settings <- fixtureSettings(tempfile())
input <- TransferLearning:::openInput(x, settings$problems$example, 'source')
train <- input$population[1:100, ]
model <- TransferLearning:::fitVariance(input$data, train, .1, settings)
root <- tempfile(); dir.create(root)
folder <- file.path(root,'model'); cache <- file.path(root,'cache'); dir.create(folder); dir.create(cache)
PatientLevelPrediction::savePlpModel(model,file.path(folder,'model'))
saveRDS(list(rowIds=train$rowId,selectedVariance=.1,expanded=FALSE,boundary=FALSE,
 tuning=data.frame(variance=c(.01,.01,.1,.1,1,1),fold=rep(1:2,3),loss=c(.6,.6,.5,.5,.55,.55),n=50,error='')),file.path(folder,'tuning.rds'))
data <- x$plpData; class(data) <- 'plpData'
PatientLevelPrediction::savePlpData(data,file.path(cache,'plpData'))
saveRDS(x$population,file.path(cache,'population.rds'))
z <- sourceDiagnosticsOne(folder,cache)
stopifnot(z$summary$nTrain==100,z$summary$eventsTrain==sum(train$outcomeCount),
 z$summary$varianceMatches,z$summary$savedStatusOK,z$summary$selectedIsCvMinimum,
 !z$summary$atWeakestPenalty,!z$summary$atStrongestPenalty,nrow(z$tuning)==3)
# Compare sparse aggregation with dense calculations in exactly the training rows.
raw <- dplyr::collect(x$plpData$covariateData$covariates)
factors <- model$preprocessing$tidyCovariates$normFactors
coefs <- model$model$coefficients
b <- coefs$betas[match(as.character(factors$covariateId),as.character(coefs$covariateIds))]; b[is.na(b)]<-0
sd <- vapply(factors$covariateId,function(id) stats::sd(raw$covariateValue[raw$covariateId==id & raw$rowId %in% train$rowId]),numeric(1))
expected <- abs(b)*sd/factors$maxValue
stopifnot(abs(z$quantiles$value[z$quantiles$scale=='perTrainingSD' & z$quantiles$quantile==1]-max(expected))<1e-3)
# Test top-level export using original manifest cache keys.
manifest <- list(settings=settings,databases=list(source=list()),packages=list(old='version'),implementation='old-fingerprint',backend='old-plp',hash='original')
saveRDS(manifest,file.path(root,'manifest.rds'))
key <- digest::digest(list('source','example','standard'),algo='xxhash64')
dir.create(file.path(root,'sources'),recursive=TRUE)
file.rename(folder,file.path(root,'sources',key))
key <- digest::digest(list('source',list(),settings$problems$example,settings$featureProfiles$standard,manifest$packages,manifest$implementation),algo='sha256')
dir.create(file.path(root,'data'));file.rename(cache,file.path(root,'data',key))
out <- sourceModelDiagnostics(root)
stopifnot(length(list.files(file.path(root,'source-diagnostics'),pattern='csv$'))==5,
 !any(c('rowId','subjectId','covariateId','betas','error') %in% unlist(lapply(out,names))))
scores <- data.frame(variance=c(1e-9,1e-8),fold=1,loss=c(.5,Inf),n=50,error=c('','private path'))
t <- sourceDiagnosticsTuning(scores,1.000000000000001e-9)
stopifnot(t$selected[1],t$failedFolds[2]==1,is.na(t$weightedLogLoss[2]))
Andromeda::close(x$plpData$covariateData);unlink(root,recursive=TRUE)
cat('SOURCE DIAGNOSTICS TESTS PASSED\n')
