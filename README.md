# BIT 4131 – Statistical Computing and Reporting (Group Work)

Each of the 23 assigned topics is implemented in **both Python and R**, and follows the required structure:
1. Theoretical explanation · 2. Mathematical formulation · 3. Worked example · 4. Dataset analysis (Python/R implementation) · 5. Expected statistical output · 6. Interpretation and visualisation

Every topic folder contains a `README.md` covering all six items, with `python/`, `r/` and `outputs/` (printed results + figures).

## Topics
| # | Topic | Code | Results |
|---|---|---|---|
| 1 | [Measures of central tendency and dispersion](topics/01_central_tendency_dispersion/README.md) | [py](topics/01_central_tendency_dispersion/python/analysis.py) · [R](topics/01_central_tendency_dispersion/r/analysis.R) | [outputs](topics/01_central_tendency_dispersion/outputs) |
| 2 | [Probability and conditional probability](topics/02_probability_conditional/README.md) | [py](topics/02_probability_conditional/python/analysis.py) · [R](topics/02_probability_conditional/r/analysis.R) | [outputs](topics/02_probability_conditional/outputs) |
| 3 | [Probability distributions](topics/03_probability_distributions/README.md) | [py](topics/03_probability_distributions/python/analysis.py) · [R](topics/03_probability_distributions/r/analysis.R) | [outputs](topics/03_probability_distributions/outputs) |
| 4 | [Sampling distributions and the Central Limit Theorem](topics/04_sampling_clt/README.md) | [py](topics/04_sampling_clt/python/analysis.py) · [R](topics/04_sampling_clt/r/analysis.R) | [outputs](topics/04_sampling_clt/outputs) |
| 5 | [Confidence intervals](topics/05_confidence_intervals/README.md) | [py](topics/05_confidence_intervals/python/analysis.py) · [R](topics/05_confidence_intervals/r/analysis.R) | [outputs](topics/05_confidence_intervals/outputs) |
| 6 | [Hypothesis testing](topics/06_hypothesis_testing/README.md) | [py](topics/06_hypothesis_testing/python/analysis.py) · [R](topics/06_hypothesis_testing/r/analysis.R) | [outputs](topics/06_hypothesis_testing/outputs) |
| 7 | [Correlation and covariance](topics/07_correlation_covariance/README.md) | [py](topics/07_correlation_covariance/python/analysis.py) · [R](topics/07_correlation_covariance/r/analysis.R) | [outputs](topics/07_correlation_covariance/outputs) |
| 8 | [Linear regression and least squares](topics/08_linear_regression/README.md) | [py](topics/08_linear_regression/python/analysis.py) · [R](topics/08_linear_regression/r/analysis.R) | [outputs](topics/08_linear_regression/outputs) |
| 9 | [Matrix algebra in statistics](topics/09_matrix_algebra/README.md) | [py](topics/09_matrix_algebra/python/analysis.py) · [R](topics/09_matrix_algebra/r/analysis.R) | [outputs](topics/09_matrix_algebra/outputs) |
| 10 | [Eigenvalues, eigenvectors and PCA](topics/10_eigen_pca/README.md) | [py](topics/10_eigen_pca/python/analysis.py) · [R](topics/10_eigen_pca/r/analysis.R) | [outputs](topics/10_eigen_pca/outputs) |
| 11 | [Numerical optimization](topics/11_numerical_optimization/README.md) | [py](topics/11_numerical_optimization/python/analysis.py) · [R](topics/11_numerical_optimization/r/analysis.R) | [outputs](topics/11_numerical_optimization/outputs) |
| 12 | [Random numbers and Monte Carlo methods](topics/12_monte_carlo/README.md) | [py](topics/12_monte_carlo/python/analysis.py) · [R](topics/12_monte_carlo/r/analysis.R) | [outputs](topics/12_monte_carlo/outputs) |
| 13 | [Time-series analysis](topics/13_time_series/README.md) | [py](topics/13_time_series/python/analysis.py) · [R](topics/13_time_series/r/analysis.R) | [outputs](topics/13_time_series/outputs) |
| 14 | [Statistical data visualization](topics/14_statistical_visualization/README.md) | [py](topics/14_statistical_visualization/python/analysis.py) · [R](topics/14_statistical_visualization/r/analysis.R) | [outputs](topics/14_statistical_visualization/outputs) |
| 15 | [Data transformation and normalization](topics/15_transformation_normalization/README.md) | [py](topics/15_transformation_normalization/python/analysis.py) · [R](topics/15_transformation_normalization/r/analysis.R) | [outputs](topics/15_transformation_normalization/outputs) |
| 16 | [ANOVA and experimental design](topics/16_anova/README.md) | [py](topics/16_anova/python/analysis.py) · [R](topics/16_anova/r/analysis.R) | [outputs](topics/16_anova/outputs) |
| 17 | [Chi-square and categorical data analysis](topics/17_chi_square/README.md) | [py](topics/17_chi_square/python/analysis.py) · [R](topics/17_chi_square/r/analysis.R) | [outputs](topics/17_chi_square/outputs) |
| 18 | [Logistic regression](topics/18_logistic_regression/README.md) | [py](topics/18_logistic_regression/python/analysis.py) · [R](topics/18_logistic_regression/r/analysis.R) | [outputs](topics/18_logistic_regression/outputs) |
| 19 | [Non-parametric statistical methods](topics/19_nonparametric/README.md) | [py](topics/19_nonparametric/python/analysis.py) · [R](topics/19_nonparametric/r/analysis.R) | [outputs](topics/19_nonparametric/outputs) |
| 20 | [Bootstrap and resampling methods](topics/20_bootstrap/README.md) | [py](topics/20_bootstrap/python/analysis.py) · [R](topics/20_bootstrap/r/analysis.R) | [outputs](topics/20_bootstrap/outputs) |
| 21 | [Multivariate statistical analysis](topics/21_multivariate/README.md) | [py](topics/21_multivariate/python/analysis.py) · [R](topics/21_multivariate/r/analysis.R) | [outputs](topics/21_multivariate/outputs) |
| 22 | [Missing-data analysis and imputation](topics/22_missing_data/README.md) | [py](topics/22_missing_data/python/analysis.py) · [R](topics/22_missing_data/r/analysis.R) | [outputs](topics/22_missing_data/outputs) |
| 23 | [Statistical reporting and reproducible analysis](topics/23_reproducible_reporting/README.md) | [py](topics/23_reproducible_reporting/python/analysis.py) · [R](topics/23_reproducible_reporting/r/analysis.R) | [outputs](topics/23_reproducible_reporting/outputs) |

## Repository layout
```
data/        reproducible synthetic datasets (data/generate_datasets.py, seed 4131)
common/      shared helpers (helpers.py, helpers.R)
topics/NN_name/{README.md, python/analysis.py, r/analysis.R, outputs/}
tools/       README generator
run_all.sh   re-runs every script
```

## Reproduce
```bash
pip install -r requirements.txt
Rscript install_r_packages.R
python data/generate_datasets.py      # optional: CSVs are already included
bash run_all.sh                       # or run one topic, from the repo root:
python topics/06_hypothesis_testing/python/analysis.py
Rscript topics/06_hypothesis_testing/r/analysis.R
```
Run all scripts **from the repository root**.

## Datasets
Synthetic data (allowed by the brief where no suitable dataset exists), generated with a fixed seed: `students`, `students_missing`, `tutoring_scores`, `monthly_sales`, `support_tickets`, `survey`. See the docstring in `data/generate_datasets.py`.

## Group members
Add names / student IDs here.
