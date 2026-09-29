# Install and run on Windows

Assume you have already installed **PatientLevelPrediction develop after the
required correctness fixes merge**. Use a fresh R/RStudio session, with your
working directory set to the `transferLearning` clone on `plp-cyclops-pilot`.

```r
options(repos = c(
  OHDSI = "https://ohdsi.r-universe.dev",
  CRAN = "https://cloud.r-project.org"
))
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_local(".", dependencies = NA, upgrade = "never")
```

This builds and installs the package and installs missing required dependencies
from `DESCRIPTION`, including CirceR and DatabaseConnector for the pilot. It
retains your installed PLP develop build. `dependencies = NA` includes
Depends/Imports/LinkingTo; optional testing and development packages in Suggests
are not required to run the pilot. See the
[remotes documentation](https://remotes.r-lib.org/reference/install_local.html).

`R CMD build` creates a package archive; `R CMD INSTALL` installs a package but
does not fetch missing dependencies. The R command above handles both dependency
installation and package installation. No separate dependency or patch script is
needed. The runner automatically checks required PLP behavior before running,
since the PLP version alone does not identify its implementation.

Copy `inst/examples/afStrokePilot.R` into your study directory and fill in the
site inputs at the top. In R/RStudio:

```r
setwd("C:/your/study")  # change this path
source("afStrokePilot.R")
```

Use forward slashes for Windows paths, including your JDBC driver directory.
The environment needs working Java/rJava and the Databricks JDBC driver; these
are system prerequisites, not R package dependencies. Rtools matching your R
version is needed if dependencies must be compiled from source. Extracted data
and Cyclops fitting reside on the machine running R.

See [AfStrokePilot.md](AfStrokePilot.md) for the experiment settings and an
extraction-only feasibility check. Windows execution and live Databricks extraction
still need validation in the production environment. The known PLP JSON precision
issue remains separate from the correctness fixes; see
[UpstreamRequirements.md](UpstreamRequirements.md).
