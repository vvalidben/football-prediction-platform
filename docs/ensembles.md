# Ensembles

Sprint 8 combines the prediction models into ensembles and measures whether that
helps. Code: `backend/app/ml/ensemble.py`. Full tables:
[model-comparison.md](model-comparison.md#ensembles).

**Result on real data:** the best ensemble is a weighted average (86% Elo, 14%
Poisson). It is not measurably better than Elo alone: forward-validation log loss
−0.0002, 95% interval [−0.0008, +0.0004]. The recommended model therefore stays
**Elo**. The ensemble is tracked and registered as its own model version.

## What is combined

Default members, as asked: logistic regression, random forest, XGBoost, LightGBM,
the Poisson goal model and Elo, each with the hyperparameters tuned in Sprint 6
(`--ensemble-members` to change them).

Every model outputs a full score grid ([prediction-engine.md](prediction-engine.md)),
so ensembles combine **grids**, not just 1X2 probabilities. Expected goals,
over/under and both teams to score are then derived from one grid and stay
consistent. This is tested with the Sprint 7 consistency checker.

| Method | How | Fitted |
| --- | --- | --- |
| `simple_average` | mean of the members' grids | nothing |
| `weighted_average` | weighted mean of the grids; weights ≥ 0, summing to 1 | weights minimising the 1X2 log loss of the mixture. The loss is convex in the weights, so the optimum is unique in value. A member that does not help gets weight 0 |
| `stacking` | multinomial logistic regression on the members' 1X2 log-odds (2 per member, standardised, C = 1); the average grid is rescaled to its 1X2 | the meta-model's coefficients |

## Weights come from validation data only

The weights and the stacking meta-model are fitted on **out-of-fold predictions**
from the Sprint 6 rolling-origin folds:

| Block | Predicted by members fitted on |
| --- | --- |
| 2021/22 | 2018/19 – 2020/21 |
| 2022/23 | 2018/19 – 2021/22 |
| 2023/24 | 2018/19 – 2022/23 |
| 2024/25 (validation) | 2018/19 – 2023/24 |

No member ever predicts a match it was trained on, and the test season (2025/26)
is not involved. For the final ensembles, the members are the models refitted on
train + validation (as in Sprint 6). The weights stay as fitted on the blocks
above.

One caveat: the members' hyperparameters were tuned on the same blocks (Sprint 6
cross-validation). That tuning has a small optimistic effect, which applies to
individual models and ensembles alike.

## Measured, not assumed: forward-chained evaluation

The weights cannot be judged on the matches they were fitted to. So every approach
is scored **forward**: block k is predicted by combiners fitted on blocks 1 to k−1
only. The first block only trains the combiners. Individual models are scored on the
same 1,140 matches (2022/23 to 2024/25).

Then the rule that decides:

> Recommend an ensemble only if its 95% paired bootstrap interval of the per-match
> log-loss difference from the best individual model lies **entirely below 0**.
> Otherwise recommend the single model: it is simpler, and the gain has not been
> demonstrated.

## Results (Premier League, 8 seasons)

Forward validation, 1,140 matches (lower is better):

| Entry | Log loss | vs Elo, 95% interval | O/U 2.5 Brier | BTTS Brier |
| --- | --- | --- | --- | --- |
| weighted average (ensemble) | 0.9684 | −0.0002 [−0.0008, +0.0004] | 0.2414 | 0.2482 |
| **elo** (best individual) | 0.9686 | – | 0.2419 | 0.2486 |
| simple average (ensemble) | 0.9717 | +0.0031 [−0.0034, +0.0094] | 0.2405 | 0.2460 |
| random forest | 0.9737 | +0.0051 [−0.0016, +0.0118] | 0.2410 | 0.2471 |
| poisson | 0.9748 | +0.0062 [−0.0028, +0.0152] | 0.2441 | 0.2473 |
| xgboost | 0.9780 | +0.0093 [−0.0000, +0.0185] | 0.2415 | 0.2465 |
| lightgbm | 0.9783 | +0.0097 [+0.0007, +0.0187] | 0.2414 | 0.2469 |
| logistic regression | 0.9835 | +0.0149 [+0.0033, +0.0263] | 0.2434 | 0.2472 |
| stacking (ensemble) | 0.9871 | +0.0185 [+0.0058, +0.0316] | 0.2409 | 0.2458 |

Weighted-average weights, as fitted at each forward step and finally:

| Fitted on | Elo | Poisson | LightGBM | Logistic | Random forest, XGBoost |
| --- | --- | --- | --- | --- | --- |
| 2021/22 | 1.000 | 0 | 0 | 0 | 0 |
| 2021/22 – 2022/23 | 0.910 | 0.090 | 0 | 0 | 0 |
| 2021/22 – 2023/24 | 0.918 | 0.068 | 0.011 | 0.003 | 0 |
| all four blocks (final) | 0.861 | 0.139 | 0 | 0 | 0 |

What the numbers say:

- **Adding models did not help measurably.** The optimiser gave the logistic
  regression and tree models (almost) zero weight at every step. Their
  probabilities add nothing that Elo and Poisson do not already carry.
- **The simple average is worse than Elo on 1X2** (+0.0031). Weaker members dilute
  the best one. It is slightly better on BTTS and over 2.5 (Brier 0.2460 and
  0.2405 vs 0.2486 and 0.2419): the averaged goal models are better than Elo's.
- **Stacking is not justified here.** It is the worst entry on 1X2 (+0.0185, with
  an interval that excludes 0). It has 39 parameters (3 outcomes × 12 features, plus
  intercepts), fitted on as few as 380 matches in the first forward step. It was implemented and measured; the
  measurement rejects it.
- **The weights are stable:** Elo 86–100% at every step, Poisson the only other
  member with a consistent share.

Test season 2025/26, reported only (not used for any choice):

| Model | Test log loss | vs Elo, 95% interval |
| --- | --- | --- |
| stacking | 1.0267 | −0.0028 [−0.0123, +0.0067] |
| weighted average | 1.0282 | −0.0013 [−0.0038, +0.0010] |
| elo | 1.0295 | – |
| simple average | 1.0311 | +0.0016 [−0.0075, +0.0105] |

On test, stacking has the lowest point estimate, but its interval is wide, and
validation had rejected it clearly. Choosing it now because of the test figure would
be selecting on the test set. The weighted average is again indistinguishable from
Elo. The test season agrees with the recommendation: nothing clearly beats Elo.

## Tracking and versions

- Each ensemble method is an MLflow run with:
  - tags `stage=ensemble`, `model_type=EnsembleModel`, `best_ensemble`,
    `best_overall` and `recommended`;
  - params: members, one `weight_<member>` per member, the period the combiner
    was fitted on, the members' fit period and the test period;
  - metrics: `validation_*` (forward-chained) and `test_*`;
  - artifacts: `ensemble.json` (combiner parameters, each member's hyperparameters
    and MLflow run ID, fold periods, the weights at every forward step) and the
    saved `EnsembleModel`.
- `--register` records the best ensemble as its **own model version**, named
  `ensemble_<method>` (here `ensemble_weighted_average`) and inactive. The row
  holds the weights, the members' hyperparameters and run IDs, the forward-validation
  ranking and the recommendation. The individually selected model is registered as
  before. Neither row overwrites or deactivates anything.
- `EnsembleModel` has the common interface. `train(data)` refits the members and
  keeps the validation-fitted weights. `predict` and `predict_proba` work as for any
  model, as do `save` and `load`, and the Sprint 7 engine accepts it like any other
  model.

```bash
python -m app.ml train --validation-start 2024-07-01 --test-start 2025-07-01 \
    --cv-folds 4 --ensemble --report ../docs/model-comparison.md --register
```

`--ensemble` needs `--cv-folds 2` or more: the weights must be fitted on earlier
validation blocks and checked on later ones.

## Tests (`backend/tests/ml/test_ensemble.py`, 27 tests; 2 integration tests)

- **Weight optimiser.**
  - Weights stay on the simplex.
  - A dominant member takes all the weight.
  - Two complementary members get 50/50.
  - The optimum is never worse than any single member or the plain average.
- **Combiners.**
  - Every combiner's output passes the Sprint 7 consistency checker.
  - The simple average is the mean grid, and a weight of 1 reproduces that member.
  - Stacking's 1X2 equals its meta-model's output.
  - A deliberately wrong member (Elo with home and away swapped) gets a weight
    below 0.02.
- **Validation only.**
  - Folds hold out only train/validation matches, and never a match the member
    was trained on.
  - Changing the results of a later block leaves every earlier step's weights
    unchanged. A mutation check confirmed this test fails if a step is allowed to
    see its own block.
  - Inverting every test result changes nothing except the test figures.
- **Measured comparison.**
  - All entries are scored on the same matches.
  - The ranking and the recommendation follow the rule.
  - An ensemble with a harmful member loses to its best member and is reported so.
  - A near-tie recommends the single model; a clear gain recommends the ensemble.
- **Tracking and integration.**
  - MLflow runs, the saved ensemble loaded from its artifact, and the report
    section.
  - The real CLI on PostgreSQL, from `--ensemble` through `--register`, gives a
    separate `ensemble_*` model version.

## Limitations

- One league, one set of eight seasons. The 1,140 forward-validation matches can
  only detect differences of roughly ±0.0006 log loss between highly correlated
  models (weighted average vs Elo), or ±0.006–0.013 between less similar ones.
- Stacking uses a fixed regularisation (C = 1); it was not tuned, to avoid giving
  it extra selection freedom that the other methods do not have.
- Weights are chosen on 1X2 log loss. The goal markets inherit them; weights tuned
  for goal markets could differ (the simple average is better there).
