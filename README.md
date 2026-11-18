# Robust prediction of HFNO failure before and after therapy initiation

Analysis code for a retrospective cohort study predicting high-flow nasal oxygen (HFNO)
failure in adult intensive care patients, developed in MIMIC-IV and externally validated in
the Dutch Data Warehouse (DDW).

HFNO failure is defined as escalation to invasive mechanical ventilation, or death, during
HFNO or within 4 hours of its discontinuation.

## Data availability

**This repository contains code only. No patient data and no analysis outputs are included,
and none may be added to it.** Results are reported in the paper.

MIMIC-IV is available to credentialed researchers through
[PhysioNet](https://physionet.org/content/mimiciv/) on completion of human subjects research
training and a signed data use agreement. Access to the Dutch Data Warehouse is governed by
the DDW consortium. The DUA does not permit redistribution of individual-level data, so
`filtered_data/`, `processed_data/` and any per-patient output are excluded by `.gitignore`.

All notebooks are committed **with their outputs cleared**.

The DDW analysis itself ran inside the DDW environment and is not part of this repository;
only aggregate results left that environment.

## Pipeline

Run in order. Notebooks 1 and 1b need a local PostgreSQL MIMIC-IV build; the rest read files
produced by earlier steps.

| Notebook | Purpose | Needs the database |
|---|---|---|
| `1_MIMICIV_HFNO_data_extraction.ipynb` | Cohort and pre-HFNO predictors | yes |
| `1b_HFNO_post_initiation_extraction.ipynb` | On-HFNO predictors (4/12/24 h windows) | yes |
| `2_HFNO_data_processing.ipynb` | Filtering, collinearity pruning, imputation, Table 1 | no |
| `3_HFNO_predictor_analysis.ipynb` | Five prediction models, SHAP, DeLong, ROX benchmark, Figure 2 | no |
| `4_HFNO_sepsis3_contribution_analysis.ipynb` | Decomposition of sepsis-3 into SOFA subscores | no |
| `5_HFNO_flow_rate_analysis.ipynb` | Propensity score matching on the 2-hour flow rate | no |
| `6_manuscript_exports.ipynb` | Rebuilds every manuscript figure and table | no |
| `7_export_model_for_ddw.ipynb` | Exports trained models (no patient rows) for the DDW | no |

`v2_1_cohort.ipynb` and `v2_2_features.ipynb` are earlier exploratory versions, kept for
provenance; they are not part of the pipeline.

### Database connection

Notebooks 1 and 1b read credentials from the environment; nothing is committed:

```bash
export MIMIC_DB=mimiciv MIMIC_USER=postgres MIMIC_PASSWORD=... \
       MIMIC_HOST=localhost MIMIC_PORT=5432
```

## Prediction models

All five include demographics, comorbidities and admission severity scores.

| Model | Added predictors | n predictors |
|---|---|---|
| 1 | 4 h on-HFNO only | 36 |
| 2 | 24 h pre-HFNO + 4 h on-HFNO | 60 |
| 3 | 24 h pre-HFNO + 12 h on-HFNO | 62 |
| 4 | 24 h pre-HFNO + 24 h on-HFNO | 72 |
| 5 | 24 h pre-HFNO only | 49 |

Each model using on-therapy data is applied only to patients still receiving HFNO at the end
of its window, so the test sets differ between models (reported in Table 2).

**Note on numbering.** Inside notebook 3 the scenarios are `s1`–`s5` in the order they were
developed, which is *not* the manuscript order: `s1` is Model 5 (pre-HFNO only) and `s2` is
Model 1. The mapping lives in one place, `PAPER_ORDER = [2, 3, 4, 5, 1]`, and
`Table2_model_performance.csv` is written in manuscript numbering. Notebook 6 asserts that the
variable counts in that CSV match `model_feature_counts.json`, which fails loudly if the two
numberings ever drift apart.

## Reproducibility

Every train/test split and estimator uses `random_state=42`, and the confidence intervals in
Table 2 come from `utils.test_set_ci` (stratified percentile bootstrap over the test patients,
`N_CI_BOOTSTRAP=2000`, `seed=42`). Re-running notebook 3 reproduces
`Table2_model_performance.csv`, `Model_comparisons_DeLong.csv` and `TableS_ROX_benchmark.csv`
exactly.

`utils.mc_bootstrap` additionally refits each model on 10 resampled training sets to show the
spread attributable to training data. That loop draws with an **unseeded** RNG, so those
numbers vary between runs — they are a diagnostic only, and nothing reported in the manuscript
derives from them.

## Citation

Lijović L, Hilders P, Elbers P. *Robust prediction of HFNO failure before and after therapy
initiation.* Manuscript submitted.

## License

Not yet chosen. Without a license file, default copyright applies and others have no right to
reuse the code, so add one before making the repository public. MIT is the common choice for
analysis code of this kind.
