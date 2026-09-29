# Install the pilot from a clone

Use the `plp-cyclops-pilot` branch. Installation does not require access to the
original developer worktree. Run the commands below in a Bash-compatible shell.
Install into the R library used by your production R process; if using a dedicated
library, set `R_LIBS_USER` to its existing directory before starting these commands.

Prerequisites: Git, R >= 4.1, the system build tools needed by your R dependencies,
and a Java installation configured for rJava/CirceR and DatabaseConnector. The
Databricks JDBC driver and network/authentication configuration are site inputs;
R package installation does not configure those. Use a study environment that can
store the extracted patient-level data, since Cyclops fitting runs in R, not Spark.

## Clone and install dependencies

```sh
git clone --branch plp-cyclops-pilot https://github.com/mi-erasmusmc/transferLearning.git
cd transferLearning
pilot_repo="$PWD"
Rscript --vanilla extras/installDependencies.R
```

The dependency helper uses CRAN and the OHDSI R-universe. It installs required
runtime dependencies and CirceR for the pilot, without all optional PLP machine
learning backends. A released PLP may be installed as a dependency here; the next
step replaces it with the corrected build. This is not a complete dependency
lockfile. Record the resulting environment before the study; the runner also
records versions and implementation fingerprints in each experiment manifest.

## Install the pinned PLP correctness build, then the runner

The base commit and bundled patch are the same as the runner's CI configuration.
The patch adds correctness fixes, not a new PLP API or the JSON precision sidecar.

```sh
plp_build=$(mktemp -d)
git clone https://github.com/OHDSI/PatientLevelPrediction.git "$plp_build/PatientLevelPrediction"
git -C "$plp_build/PatientLevelPrediction" checkout --detach 1d91b7de03adb2332073420f440eeefdd2e2e837
git -C "$plp_build/PatientLevelPrediction" apply --check "$pilot_repo/extras/PatientLevelPrediction-transfer.patch"
git -C "$plp_build/PatientLevelPrediction" apply "$pilot_repo/extras/PatientLevelPrediction-transfer.patch"
Rscript --vanilla -e 'options(repos = c(OHDSI="https://ohdsi.r-universe.dev", CRAN="https://cloud.r-project.org")); remotes::install_deps(commandArgs(TRUE)[1], dependencies=NA, upgrade="never")' "$plp_build/PatientLevelPrediction"
R CMD INSTALL "$plp_build/PatientLevelPrediction"
R CMD INSTALL "$pilot_repo"
Rscript --vanilla "$pilot_repo/extras/checkBackend.R"
```

Stop if any command fails. Keep the corrected PLP installed in the library loaded
by the study process. Do not subsequently replace it with an unpatched release
merely because its version matches. The preflight performs synthetic fits and
checks source-coefficient behavior and fixed-variance fitting; it does not connect
to a database. Patch SHA-256:

```
c38786ca22a36f9d46ecbfb5abad5f2ab99e7cda8116d20b9e8a4e1715cbe1ba
```

`R CMD INSTALL .` alone does not install dependencies or apply the PLP patch.
Until an upstream build incorporates and passes the required behavior, all steps
above are needed. See [UpstreamRequirements.md](UpstreamRequirements.md).

## Configure and run

Copy `inst/examples/afStrokePilot.R` from the clone to your study directory. Fill
in the site inputs at the top: Databricks connection details, the two CDM schemas,
data-release IDs, writable scratch schema and output folder. Credentials can come
from environment variables. No site-configuration RDS is required.

```sh
Rscript --vanilla /path/to/study/afStrokePilot.R
```

The entrypoint runs extraction and fitting. For an extraction-only feasibility
check, follow [AfStrokePilot.md](AfStrokePilot.md). Live Databricks extraction has
not yet been validated locally. Known PLP JSON precision loss remains unresolved;
see the upstream requirements before relying on exact interrupted-run equivalence.
