# ml_prediction

Feature selection and regression models on data read from an existing
PostgreSQL instance.


## Layout

```
config/ml_config.yaml        every tunable setting -- experiments are config edits
src/config/config.py         loads the YAML and the environment
src/data/db_link.py          database engine + loading a query into a DataFrame
src/data/data.py             the feature table, typed, validated and role-labelled
src/features.py              derived ratios, momentum and calendar flags
src/window.py                month-wise holdout and expanding-window CV folds
src/metrics.py               scoring, and the two framings it has to be read in
src/evaluate.py              the report: breakdowns by month, user and account tier
src/tuning.py                the randomised search and the folds it runs over
src/models_code/base_class.py    the interface every model implements, and save/load
src/models_code/entity_scaler.py per-entity robust scale for the scaled target
src/models_code/baseline.py      the trivial predictors: 3-month average, persistence
src/models_code/mlr.py           ridge regression, level and change target modes
src/models_code/xgboost_model.py boosted trees, deliberately small
src/features_selection.py    four-stage selection, on train only, with a noise test
src/train.py                 fit every model on the training region and save it
src/test.py                  score the saved models on months none of them has seen
notebooks/                   local exploration
models/                      one .joblib per model, overwritten each run (gitignored)
```


## The three entry points

Run in this order. They are separate on purpose: choosing the columns, fitting
the models and reporting the headline number are different questions, and a
script that does two of them lets the second one's score quietly pick the
first one's answer.

```bash
python -m src.features_selection   # which columns -- gain ranking + parsimony sweep
python -m src.train                # fit + save   -- CV inside the training region
python -m src.test                 # the answer   -- one table, all models compared
```

or in the container:

```bash
docker compose run --rm train
docker compose run --rm test
```

**Selection never edits the config.** It prints the set it chose as a
`features:` block to paste into `config/ml_config.yaml`, so narrowing the
feature table stays a reviewed edit with its measurement beside it. Training
reads that block; it does not re-select. Two runs over the same data therefore
use the same columns, and the booster gets no extra attempts at the folds that
the baseline never got.

**Training never reads the holdout.** It cuts it off in the first few lines and
works only on the training region from there.

**Testing never recomputes the split.** Each saved model carries the months it
was fitted on, and `src/test.py` scores everything strictly after the last of
them. Editing `split.test_months` between the two runs moves nothing: the
models still say where they stopped. This is the one thing the structure is
built around, because the alternative -- both scripts deriving "the last N
months" from config -- fails silently in both directions, by scoring a model on
months it trained on, or by sliding the boundary back into the training region
when new data lands.


## Reading the comparison

`src/test.py` prints one row per configured model, ranked by the configured
`headline_metric`:

```
model                       rmse        mae  median_ae   wape  r2_change  skill
xgboost_scaled            30,112     21,004      9,880  0.061      0.194  0.212
ridge_change             190,455    140,223     78,110  0.402      0.041  0.043
three_month_average      180,900    138,004     74,551  0.398      0.000    NaN
```

Three things to read, in this order:

- **`skill`** is measured against `three_month_average`. Positive means better
  than doing nothing clever. This is the column that says whether a model
  earned its complexity.
- **`r2_change`**, never `r2_level`. Scored against the movement. Every model
  here clears 0.8 on the *level*, the 3-month average included, because the
  level barely moves -- quoting that number would make the trivial baseline
  look excellent.
- **the gap between the baseline row and the others.** A model that does not
  clearly separate from the top row has not been shown to work.

The baseline is not a competitor with a handicap. It averages three raw balance
lags and reads no selected features at all, which is what makes it a floor. The
ridge and the booster are given the identical frozen feature set, so the gap
between them is the model and not the columns.


## What is predicted, and on what scale

The target is `target_closing_balance_usd`, a monthly closing balance. Three
quarters of these balances are negative -- they are liability accounts -- and
they span three orders of magnitude, which is the fact that shapes everything
below.

Models do not fit the balance directly. `target_mode` picks the axis:

| mode | fits | notes |
|---|---|---|
| `level` | the balance | almost entirely last month's balance; little left to learn |
| `change` | the movement, in dollars | honest, but a dollar of error on a small account and on a whale count the same |
| `scaled_change` | movement / that user's own typical movement | the default |
| `signed_log_change` | movement in `sign(x)·log1p(\|x\|)` space | the log framing, made usable on a target that goes negative |

`scaled_change` divides each movement by a robust per-entity scale (MAD, in
`src/models_code/entity_scaler.py`). That scale is **fitted inside `_fit`**, on
the rows the model was handed, and refitted for every CV fold -- computing it in
`build_dataset` would estimate it over the holdout too. Predictions are
multiplied back and the anchor added, so every mode is scored in dollars and
the results table stays one table.

XGBoost's objective is set explicitly to `reg:absoluteerror`. It was previously
unset, which meant the library default `reg:squarederror` -- quadratic weighting
on a panel where a handful of accounts move by six figures.

## Features

All 21 source columns arrive from `feature_store_monthly` and 18 of them are
absolute dollar amounts, which teach a model account size rather than account
behaviour. `src/features.py` derives ratios, momentum, spending shares,
transaction-frequency features and calendar flags from them, and the raw dollar
columns are then dropped in `data.drop_columns`.

Every function there is row-wise over already-lagged inputs -- no `shift`, no
`rolling`, no groupby along time -- so a feature for month t uses only what
existed on day one of month t, which is what `split.gap_months: 0` relies on.

## How a feature set is chosen

Four stages, training months only, in `src/features_selection.py`:

1. **Redundancy.** Cluster columns correlated at or above
   `redundancy_threshold` under either Pearson or Spearman, keep one per
   cluster -- the member with the highest rank correlation against the
   *movement*. Run before ranking, because gain splits the credit between two
   near-identical columns and leaves both looking mediocre.
2. **Temporal cross-validation.** Rank by `gain` -- never `weight` or `cover`;
   gain is the loss actually reduced by the splits a column appears in, so it
   sees interactions a correlation cannot. Then take the top k for every k,
   cross-validate each on the expanding folds, and keep the smallest k still
   within `tolerance` of the full-feature model.
3. **Out-of-fold stability.** Permutation importance computed on each fold's
   *held-out* months, never on training rows. A feature is kept only if its
   mean importance is positive and its coefficient of variation across folds is
   at most `stability_max_cv`. Gain is a training statistic and a column
   memorising noise scores well on it; this is the check that does not.
4. **Noise test.** Re-run the whole thing against a shuffled target and require
   that nothing is selected.

R squared validates in stage 2, it does not select: candidate sets come from the
gain ranking, which never reads a score. Selecting on whatever raises R squared
is a wrapper method, and a wrapper method run to convergence fits the folds
rather than the problem.

### The gate, and why stage 4 needs it

The tolerance rule in stage 2 is purely relative -- it returns the smallest k
within `tolerance` of the full set. If the full set is worthless then every k
matches it, every k passes, and k=1 is returned looking like a finding. So
`min_signal` is checked first: the full-feature model must clear that much
cross-validated `r2_change` before any set is returned at all. Zero is a real
threshold for that metric, not an arbitrary one -- `r2_change = 0` is exactly
the accuracy of predicting that the balance does not move.

Without the gate the noise test cannot fail, and a test that cannot fail is not
a test.

### The noise test shuffles the movement, not the level

`shuffle_target` permutes `target - anchor` and rebuilds `target = anchor +
shuffled_movement`, leaving every anchor in place.

Permuting the target column outright -- the obvious first implementation -- is
wrong, and wrong in a way worth recording. It breaks the pairing between a row's
target and its own anchor, so the movement becomes `other_row_balance -
my_anchor`, whose variance is dominated by the anchor; a model then scores well
on `r2_change` just by tracking the anchor. Measured here it reached +0.253 on
shuffled data against +0.076 on the real thing. A noise run that beats the
genuine one is a broken shuffle, not a leak.

## Metrics

`wape` -- total absolute error over total absolute truth -- is the headline.
MAPE divides each row by its own truth and these balances pass through zero, so
it needs a floor and a coverage figure to be quotable at all. RMSE ranks models
by how well they fit the largest handful of accounts; `median_ae`, the previous
default, is blind to the tail entirely.

Every report also breaks down by **account-size tier** (`smb` / `mid` /
`enterprise`, cut at `evaluation.tier_quantiles` on training rows only). A good
global number on a panel this skewed can be a good number on the whales and a
bad one on everybody else, and only the tier table shows which.

## Resources

The following resources were consulted during this task:
- https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
- https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088