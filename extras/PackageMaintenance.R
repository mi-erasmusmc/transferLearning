# Use an R library containing the merged PLP develop implementation.
roxygen2::roxygenise()
devtools::test()
devtools::check()
# For a release, also run on Windows and macOS with the same pinned PLP develop backend.
# Generate a fresh environment lockfile after upstream changes are published;
# the historical glmnet environment is preserved at the legacy-glmnet Git tag.
