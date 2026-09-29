# Use an R library containing the corrected upstream PLP implementation.
roxygen2::roxygenise()
devtools::test()
devtools::check()
# For a release, also run on Windows and macOS with the same patched PLP backend.
# Generate a fresh environment lockfile after upstream changes are published;
# the historical glmnet environment is preserved at the legacy-glmnet Git tag.
