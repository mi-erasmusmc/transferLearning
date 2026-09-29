args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript --vanilla extras/runExperiment.R study-config.R")
configuration <- new.env(parent = globalenv())
sys.source(args[1], envir = configuration)
results <- TransferLearning::runExperiment(configuration$settings,
  configuration$databaseRegistry,
  preparedData = configuration$preparedData)
if (any(results$status$status == "failed")) quit(status = 1L)
