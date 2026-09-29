# Install the pilot on Windows

Use the `plp-cyclops-pilot` branch. No dependency-installation script is needed:
normal R package tools install the dependencies declared in DESCRIPTION. The extra
setup below is for the **unmerged PLP correctness patch**, not for the runner's
ordinary dependencies.

Use Git for Windows, R >= 4.1 and RStudio (optional). Install the
[Rtools version matching your R](https://cran.r-project.org/bin/windows/Rtools/)
if dependencies need compilation from source. Java must be configured for
rJava/CirceR and DatabaseConnector. Databricks JDBC drivers and authentication
remain site-specific inputs. Cyclops fitting and extracted patient-level caches
reside on the machine running R, not on Spark.

## 1. Clone and apply the PLP patch (PowerShell or Git Bash)

Run these commands one at a time, stopping if a command fails. Start in the parent
folder where you want the two repositories. If you already cloned the runner,
enter that clone and start at the second `git clone` command instead.

```powershell
git clone --branch plp-cyclops-pilot https://github.com/mi-erasmusmc/transferLearning.git
cd transferLearning
git clone https://github.com/OHDSI/PatientLevelPrediction.git ../PatientLevelPrediction-pilot
git -C ../PatientLevelPrediction-pilot checkout --detach 1d91b7de03adb2332073420f440eeefdd2e2e837
git -C ../PatientLevelPrediction-pilot apply --check ../transferLearning/extras/PatientLevelPrediction-transfer.patch
git -C ../PatientLevelPrediction-pilot apply ../transferLearning/extras/PatientLevelPrediction-transfer.patch
```

These paths assume the runner clone is named `transferLearning` and the PLP clone
is beside it. Use a fresh PLP clone for this step; do not apply the patch twice.
The pinned commit and patch match the runner's CI configuration.

## 2. Install from a fresh R/RStudio session

Set the working directory to your runner clone; use forward slashes in R paths.
Use the same R library for installation and execution. A writable personal library
is sufficient; no administrator installation is required.

```r
setwd("C:/your/work/transferLearning")  # change this path
options(repos = c(
  OHDSI = "https://ohdsi.r-universe.dev",
  CRAN = "https://cloud.r-project.org"
))

if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")

# Required runner dependencies, without every optional PLP modelling backend.
remotes::install_deps(".", dependencies = NA, upgrade = "never")
# CirceR is needed by the pilot entrypoint to compile its frozen cohort JSON.
remotes::install_cran("CirceR", dependencies = NA, upgrade = "never")

# Install corrected PLP AFTER ordinary dependencies, then install the runner.
remotes::install_local("../PatientLevelPrediction-pilot",
  dependencies = NA, upgrade = "never", build = FALSE, force = TRUE)
remotes::install_local(".",
  dependencies = FALSE, build = FALSE, force = TRUE)

# Synthetic fitting check only: no database connection.
source("extras/checkBackend.R")
```

`dependencies = NA` installs Depends/Imports/LinkingTo rather than all Suggests;
see [remotes installation documentation](https://remotes.r-lib.org/reference/install_local.html).
A released PLP may be installed in the dependency step; the explicit local install
replaces it with the corrected build. The final runner installation does not
modify dependencies. Restart R before reinstalling loaded packages, particularly
on Windows. Keep corrected PLP in the library used by the study process: its
version alone cannot identify the patch. The readiness check verifies exercised
source-coefficient and fixed-variance behavior.

This is not a complete dependency lockfile. Record the resulting environment
before the study; the runner records package versions and implementation
fingerprints. The bundled patch SHA-256 is:

```
c38786ca22a36f9d46ecbfb5abad5f2ab99e7cda8116d20b9e8a4e1715cbe1ba
```

Once a suitable upstream PLP build incorporates these fixes and passes the
behavioral check, the special patch installation can be retired. See
[UpstreamRequirements.md](UpstreamRequirements.md).

## 3. Configure and run in R/RStudio

Copy `inst/examples/afStrokePilot.R` into your study folder. Fill in the site inputs
at the top: Databricks connection details, two CDM schemas, data-release IDs,
writable scratch schema and output folder. Use forward slashes for Windows paths,
including the JDBC driver folder. Credentials can come from environment variables.
No site-configuration RDS is required.

```r
setwd("C:/your/study")  # change this path
source("afStrokePilot.R")
```

The entrypoint runs extraction and fitting. For an extraction-only feasibility
check, follow [AfStrokePilot.md](AfStrokePilot.md). The patched-build installation
and preflight have been verified on Linux; Windows execution and live Databricks
extraction still need validation in the production environment. Known PLP JSON
precision loss remains unresolved; see the upstream requirements before relying
on exact interrupted-run equivalence.
