# TransferLearning 0.1.0

* Add a PLP/Cyclops runner for five transfer-learning comparators, paired patient
  learning curves, fold-specific normalization, and log-loss tuning.
* Add frozen cohort extraction, resumable artifacts, and paired performance reports.
* Supply the required upstream PLP correctness patch and regression tests.
* Use PLP's existing prepared-data single-fold path for fixed-variance fits;
  validate backend behavior without extending PLP model settings.
* Use the reviewed correctness-only upstream patch. JSON model persistence
  precision remains unresolved and is documented separately.

* Default to exact target training-event budgets with proportional controls,
  fixed holdouts, and event-axis plots; retain optional patient budgets.
* Prepare the AF–stroke pilot with frozen protocol cohorts and earlier phenotype
  definitions, merging 14 antibiotic groups into one exposure-start feature.
* Simplify pilot inputs to source/target names and cohort tables, with optional
  snapshot labels. Reuse existing tables and generate missing ones; include data
  locations in extraction-cache and experiment-manifest fingerprints.
