# Copyright 2026 Observational Health Data Sciences and Informatics
# Licensed under the Apache License, Version 2.0.
withSeed <- function(seed, code) {
	hadSeed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
	if (hadSeed) oldSeed <- get(".Random.seed", envir = .GlobalEnv)
	on.exit(if (hadSeed) assign(".Random.seed", oldSeed, envir = .GlobalEnv)
		else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
			rm(".Random.seed", envir = .GlobalEnv))
	set.seed(seed)
	force(code)
}
seedFor <- function(seed, ...) {
	as.integer(strtoi(substr(digest::digest(list(seed, ...), algo = "xxhash32"), 1, 7), 16L))
}
atomicSave <- function(object, path) {
	dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
	temporary <- tempfile(tmpdir = dirname(path))
	on.exit(unlink(temporary))
	saveRDS(object, temporary)
	if (!file.rename(temporary, path)) stop("Could not publish artifact: ", path)
}
classCounts <- function(population) table(factor(population$outcomeCount > 0, levels = c(FALSE, TRUE)))
makePartition <- function(population, settings, databaseId, problemId) {
	stopifnot(!anyDuplicated(as.character(population$subjectId)),
		!anyDuplicated(as.character(population$rowId)))
	test <- withSeed(seedFor(settings$seed, databaseId, problemId, "holdout"), {
		unlist(lapply(split(seq_len(nrow(population)), population$outcomeCount > 0), function(ix) {
			ix[sample.int(length(ix), max(1L, floor(length(ix) * settings$testFraction)))]
		}), use.names = FALSE)
	})
	list(test = test, development = setdiff(seq_len(nrow(population)), test))
}
sampleNested <- function(population, n, seed, unit = "patients") {
	stopifnot(unit %in% c("patients", "events"), length(n) == 1, !is.na(n), n > 0,
		is.infinite(n) || n == floor(n))
	available <- if (unit == "events") sum(population$outcomeCount > 0) else nrow(population)
	if (is.finite(n) && n > available) stop("Training budget exceeds development pool")
	if (is.infinite(n)) return(seq_len(nrow(population)))
	withSeed(seed, {
		byClass <- split(seq_len(nrow(population)), factor(population$outcomeCount > 0, levels = c(FALSE, TRUE)))
		nPositive <- floor(n * length(byClass[[2]]) / nrow(population))
		counts <- if (unit == "events") {
			c(round(n * length(byClass[[1]]) / length(byClass[[2]])), n)
		} else c(n - nPositive, nPositive)
		unlist(lapply(seq_along(byClass), function(i) {
			ix <- byClass[[i]]
			if (!length(ix)) return(integer())
			ix[sample.int(length(ix))][seq_len(counts[i])]
		}), use.names = FALSE)
	})
}
makeFolds <- function(population, folds, seed) {
	withSeed(seed, {
		result <- integer(nrow(population))
		for (ix in split(seq_len(nrow(population)), population$outcomeCount > 0)) {
			result[ix[sample.int(length(ix))]] <- rep(seq_len(folds), length.out = length(ix))
		}
		result
	})
}
packageVersions <- function() {
	packages <- c("TransferLearning", "PatientLevelPrediction", "Cyclops", "FeatureExtraction",
		"Andromeda", "DatabaseConnector", "CohortGenerator", "ParallelLogger")
	stats::setNames(vapply(packages, function(p) as.character(utils::packageVersion(p)), character(1)), packages)
}
functionFingerprint <- function(package) {
	namespace <- asNamespace(package)
	names <- sort(ls(namespace, all.names = TRUE))
	definitions <- lapply(names, function(name) {
		object <- get(name, namespace)
		if (is.function(object)) list(formals = deparse(formals(object)), body = deparse(body(object))) else NULL
	})
	digest::digest(stats::setNames(definitions, names), algo = "sha256")
}
