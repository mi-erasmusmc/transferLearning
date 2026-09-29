# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.
performance <- function(y, p) {
	auc <- if (length(unique(y)) == 2) as.numeric(pROC::auc(y, p, direction = "<", quiet = TRUE)) else NA_real_
	thresholds <- sort(unique(p), decreasing = TRUE)
	groups <- match(p, thresholds)
	tp <- cumsum(tabulate(groups[y == 1], nbins = length(thresholds)))
	count <- cumsum(tabulate(groups, nbins = length(thresholds)))
	ap <- if (sum(y) > 0) sum(diff(c(0, tp)) * tp / count) / sum(y) else NA_real_
	calibration <- tryCatch(fitRecalibration(y, p, slope = TRUE), error = function(e) list(intercept = NA_real_, slope = NA_real_))
	c(auroc = auc, auprc = ap, logLoss = logLoss(y, p), brier = mean((y - p)^2),
		calibrationIntercept = calibration$intercept, calibrationSlope = calibration$slope)
}
pairedIntervals <- function(y, baseline, transfer, repetitions, seed) {
	if (repetitions == 0) return(data.frame())
	values <- withSeed(seed, replicate(repetitions, {
		ix <- sample.int(length(y), replace = TRUE)
		auc <- function(p) if (length(unique(y[ix])) == 2)
			as.numeric(pROC::auc(y[ix], p[ix], direction = "<", quiet = TRUE)) else NA_real_
		c(auroc = auc(transfer) - auc(baseline),
			logLoss = logLoss(y[ix], baseline[ix]) - logLoss(y[ix], transfer[ix]))
	}))
	dplyr::bind_rows(lapply(seq_len(nrow(values)), function(i) {
		ci <- stats::quantile(values[i, ], c(.025, .975), na.rm = TRUE)
		data.frame(metric = rownames(values)[i], lower = unname(ci[1]), upper = unname(ci[2]),
			validReplicates = sum(is.finite(values[i, ])))
	}))
}

#' Collect completed experiment results
#' @param outputFolder Experiment output folder.
#' @return List containing job status, metrics, and paired confidence intervals.
#' @export
collectExperimentResults <- function(outputFolder) {
	files <- list.files(file.path(outputFolder, "jobs"), pattern = "result.rds$", recursive = TRUE, full.names = TRUE)
	results <- lapply(files, readRDS)
	metrics <- dplyr::bind_rows(lapply(results, `[[`, "metrics"))
	differences <- data.frame()
	if (nrow(metrics)) {
		keys <- c("problemId", "profileId", "pairIndex", "repetition", "trainingBudget", "budgetUnit", "sourceId", "targetId")
		baseline <- metrics[metrics$method == "targetOnly", c(keys, "auroc", "auprc", "logLoss", "brier")]
		comparisons <- merge(metrics[metrics$method != "targetOnly", ], baseline, by = keys, suffixes = c("", "Baseline"))
		if (nrow(comparisons)) {
			differences <- comparisons[c(keys, "method", "nTrain", "eventsTrain")]
			for (metric in c("auroc", "auprc", "logLoss", "brier")) {
				differences[[metric]] <- (comparisons[[metric]] - comparisons[[paste0(metric, "Baseline")]]) *
					if (metric %in% c("logLoss", "brier")) -1 else 1
			}
		}
	}
	list(status = dplyr::bind_rows(lapply(results, `[[`, "status")),
		metrics = metrics, differences = differences,
		intervals = dplyr::bind_rows(lapply(results, `[[`, "intervals")))
}

#' Plot paired learning curves
#' @param results Output of collectExperimentResults.
#' @param metric One of auroc, auprc, logLoss, brier, calibrationIntercept, calibrationSlope.
#' @param abscissa Training events (default) or patients.
#' @return A ggplot showing individual repetitions and their mean.
#' @export
plotLearningCurves <- function(results, metric = "auroc", abscissa = c("events", "patients")) {
	stopifnot(metric %in% names(results$metrics))
	data <- results$metrics
	abscissa <- match.arg(abscissa)
	data$trainingCount <- data[[if (abscissa == "events") "eventsTrain" else "nTrain"]]
	data$value <- data[[metric]]
	ggplot2::ggplot(data, ggplot2::aes(x = .data$trainingCount, y = .data$value, color = .data$method)) +
		ggplot2::geom_point(alpha = .25) +
		ggplot2::stat_summary(fun = mean, geom = "line") +
		ggplot2::scale_x_log10() +
		ggplot2::facet_wrap(~ problemId + sourceId + targetId + profileId, scales = "free_y") +
		ggplot2::labs(x = paste("Target training", abscissa), y = metric) + ggplot2::theme_bw()
}

#' Plot paired transfer improvements by target data size
#' @param results Output of collectExperimentResults.
#' @param metric One of auroc, auprc, logLoss, or brier. Positive values favor transfer.
#' @param abscissa Training events (default) or patients.
#' @return A ggplot showing paired differences and their mean across repetitions.
#' @export
plotTransferEffects <- function(results, metric = "auroc", abscissa = c("events", "patients")) {
	stopifnot(metric %in% c("auroc", "auprc", "logLoss", "brier"))
	data <- results$differences
	abscissa <- match.arg(abscissa)
	data$trainingCount <- data[[if (abscissa == "events") "eventsTrain" else "nTrain"]]
	data$value <- data[[metric]]
	ggplot2::ggplot(data, ggplot2::aes(x = .data$trainingCount, y = .data$value, color = .data$method)) +
		ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
		ggplot2::geom_point(alpha = .25) + ggplot2::stat_summary(fun = mean, geom = "line") +
		ggplot2::scale_x_log10() +
		ggplot2::facet_wrap(~ problemId + sourceId + targetId + profileId, scales = "free_y") +
		ggplot2::labs(x = paste("Target training", abscissa), y = paste(metric, "improvement over target-only")) +
		ggplot2::theme_bw()
}
