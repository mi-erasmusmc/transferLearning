# Run from the root of the TransferLearning clone, before installing patched PLP.
if (!file.exists("DESCRIPTION") ||
    read.dcf("DESCRIPTION", fields = "Package")[[1]] != "TransferLearning") {
  stop("Run this script from the TransferLearning repository root")
}
options(repos = c(OHDSI = "https://ohdsi.r-universe.dev",
  CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_deps(dependencies = NA, upgrade = "never")
# CirceR compiles the bundled AF/stroke JSON in the pilot entrypoint.
remotes::install_cran("CirceR", dependencies = NA, upgrade = "never")
message("Dependencies installed. Next install the pinned, patched PLP build; see extras/Installation.md.")
