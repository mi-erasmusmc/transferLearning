# Run from the repository root. Rebuilds the frozen pilot definitions.
# Maintenance-only dependency: install PhenotypeLibrary 3.37.0 separately.
# Running the package/pilot uses bundled definitions and does not need it.
stopifnot(as.character(utils::packageVersion('PhenotypeLibrary')) == '3.37.0')
x <- PhenotypeLibrary::getPlCohortDefinitionSet(1152:1215)
# Preserve the exact recovered input alongside the derived pilot set.
saveRDS(x, 'inst/pilot/phenotypes-original.rds', compress = 'xz')
antibioticIds <- 1201:1214
parsed <- lapply(x$json[match(antibioticIds, x$cohortId)], jsonlite::fromJSON, simplifyVector = FALSE)
items <- unlist(lapply(parsed, function(z) {
  id <- z$PrimaryCriteria$CriteriaList[[1]]$DrugExposure$CodesetId
  z$ConceptSets[[which(vapply(z$ConceptSets, function(cs) cs$id == id, logical(1)))]]$expression$items
}), recursive = FALSE)
stopifnot(all(vapply(items, function(i) !isTRUE(i$isExcluded), logical(1))))
items <- items[!duplicated(vapply(items, function(i) jsonlite::toJSON(i, auto_unbox = TRUE), character(1)))]
merged <- parsed[[1]]
merged$ConceptSets <- list(list(id = 0L, name = 'Any antibiotic (union of 14 earlier groups)', expression = list(items = items)))
merged$QualifiedLimit$Type <- 'All'
merged$EndStrategy <- list(DateOffset = list(DateField = 'StartDate', Offset = 0L))
merged$CollapseSettings$EraPad <- 0L
mergedJson <- as.character(jsonlite::toJSON(merged, auto_unbox = TRUE, null = 'null'))
mergedSql <- CirceR::buildCohortQuery(CirceR::cohortExpressionFromJson(mergedJson), CirceR::createGenerateOptions())
mergedRow <- data.frame(cohortId = 900001L, cohortName = 'Any antibiotic exposure', json = mergedJson, sql = as.character(mergedSql))
y <- rbind(as.data.frame(x[!x$cohortId %in% antibioticIds, ]), mergedRow)
saveRDS(y, 'inst/pilot/phenotypes.rds', compress = 'xz')
write.csv(y[c('cohortId','cohortName')], 'inst/pilot/phenotypes.csv', row.names = FALSE)
writeLines(mergedJson, 'inst/pilot/any-antibiotic.json')
