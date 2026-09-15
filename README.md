# ml_prediction

Feature selection and regression models on data read from an existing
PostgreSQL instance.


## Layout

```
config/ml_config.yaml   every tunable setting -- experiments are config edits
src/config.py           loads the YAML and the environment
src/db_link.py          database engine + loading a query into a DataFrame
src/data.py             the feature table, typed, validated and role-labelled
src/splitting.py        month-wise holdout and expanding-window CV folds
src/preprocessing.py    imputation and scaling, fitted on training rows only
src/base_class.py       the interface every model implements
src/baseline.py         the trivial predictors: 3-month average, persistence
src/metrics.py          scoring, and the two framings it has to be read in
src/evaluate.py         holdout scoring plus the per-month and per-user breakdowns
src/save_model.py       sealing a fitted model: artefact, metadata, checksum
src/mlr.py              linear regression, level and change target modes
src/xgboost_model.py    boosted trees, deliberately small
src/features_selection.py  selection (fitted on train), importance, ablation
src/train.py            the entry point the container runs
notebooks/              local exploration
models/                 trained artefacts + per-run JSON summaries (gitignored)
```

## Resources

The following resources were consulted during this task:
- https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
- https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088