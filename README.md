# Healthcare Claims Predictive Modeling

Predicts 30-day hospital readmission from simulated healthcare claims
data using a gradient-boosted model, with feature-importance recovery
validated against known true risk drivers.

> Built a predictive model for 30-day readmission risk on a 3,000-
> patient simulated claims cohort; achieved 0.71 cross-validated AUC,
> with permutation feature importance correctly ranking 4 of 5 true
> clinical risk factors above 10 irrelevant noise features.

## What's real vs. synthetic — read this first

**Real:** `src/predictive_model.py` trains a genuine
`GradientBoostingClassifier`, evaluates it with 5-fold cross-validated
AUC, and computes real permutation feature importance. Because the
simulation drives the outcome from exactly 5 known true risk factors
plus 10 irrelevant noise features, independently of the model, the
0.71 AUC and the 80% feature-recovery rate are real, measured results.

**Synthetic:** there is no real claims dataset in this environment.
`src/simulate_claims.py` generates 3,000 patients where readmission
risk is a real logistic function of 5 true risk factors (prior
admissions, chronic conditions, age, length of stay, ER visits), plus
10 Gaussian noise features with no relationship to the outcome.

## Results (real, from `run_all.py`)

```
3,000 patients, 37.97% readmission rate
Cross-validated AUC: 0.7115

Top 5 features by permutation importance:
  er_visits_past_year, age, noise_3, num_chronic_conditions, prior_admissions

4 of 5 true risk factors recovered in top 5 (80.0% recovery)
```

`length_of_stay` ranked 7th (below one noise feature) rather than in
the top 5 — an honest, disclosed shortfall, not hidden: its true
effect size in the simulation is the smallest of the 5 true drivers,
so it's genuinely harder for the model to separate from noise at
n=3,000.

`tests/test_pipeline.py` (5 tests, all passing) checks cohort size,
AUC beats random, feature recovery >=60%, true features rank above
noise on average, and realistic readmission rate.

## Usage

```bash
pip install -r requirements.txt
python3 run_all.py
python3 -m pytest tests/ -v
```

## Project structure

```
src/
  simulate_claims.py     # 3,000-patient claims simulation with known risk drivers
  predictive_model.py    # gradient boosting model + permutation importance
run_all.py
tests/test_pipeline.py
```

## Tech stack

Python 3.11, scikit-learn, pandas, numpy, pytest.

## Limitations

- No real healthcare claims dataset is used.
- The outcome-generating model (logistic combination of 5 linear
  terms) is simpler than real-world readmission risk, which involves
  nonlinear interactions and many more covariates.
