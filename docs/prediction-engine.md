# Goal prediction engine

The engine turns a trained model into a complete score prediction for a match:
expected goals, the probability of every exact score, and the markets derived from
it. Code: `backend/app/ml/engine.py` (object and engine) and
`backend/app/ml/probability.py` (the score-grid maths).

## One distribution, everything derived

Every model (baseline or machine learning, see [models.md](models.md)) produces one
**score probability matrix** per match: `matrix[h][a]` = P(home scores h, away
scores a). **Every other number is a sum over that matrix.** Nothing is estimated
separately, so the outputs cannot contradict each other.

| Output | Computed as |
| --- | --- |
| `expected_home_goals`, `expected_away_goals` | Σ h·P(h, a) and Σ a·P(h, a) (marginal means) |
| `most_probable_score` | the largest cell; ties go to the fewest home goals, then the fewest away goals |
| `home_win_probability` / `draw_probability` / `away_win_probability` | sums over h > a, h = a, h < a |
| over / under line L | sums over h + a > L and h + a < L |
| `btts_probability` yes / no | sums over (h ≥ 1 and a ≥ 1) and (h = 0 or a = 0) |

Both halves of each pair (over and under, yes and no, and the three 1X2 outcomes) are
summed independently from the matrix, not computed as 1 minus the other. That their
totals come to 1 is therefore a real check on the matrix, and the tests enforce it.

## Maximum score: 10 goals per side

The matrix covers **0–10 goals per side: 121 scores, 0-0 to 10-10**, and is
normalised to sum to 1. Scores above 10 are not represented. The mass they would
carry is spread over the 121 cells when the matrix is normalised.

Why 10 is enough (P(a side scores more than 10) for Poisson goals):

| Expected goals of one side | 1.5 | 2.5 | 3.0 | 3.5 | 5.0 |
| --- | --- | --- | --- | --- | --- |
| P(more than 10) | 5.5e-7 | 6e-5 | 3e-4 | 1e-3 | 1.4e-2 |

On eight real Premier League seasons, the highest expected goals any model predicted
for one side in the test season was **3.16** (P(more than 10) = 0.045%). Typical
values are 0.8–2.2. A smaller maximum would lose visible mass: at 3.16 expected
goals, 1.6% of the probability lies beyond 7 goals. A larger one would only add
near-empty cells. The model goal rates are clipped to at most 8 as a guardrail;
rates anywhere near that would need a larger matrix, but none has occurred.

The full 11 × 11 matrix is returned. A display can show a corner such as 0-0 to 5-5.
The corner is a subset of the matrix, not a separate distribution.

## Output object

`ScorePrediction` (immutable; the matrix is a read-only private copy). `to_dict()`
gives JSON-ready data:

```json
{
  "expected_home_goals": 1.361,
  "expected_away_goals": 1.351,
  "most_probable_score": {"home_goals": 1, "away_goals": 1, "probability": 0.124},
  "max_goals": 10,
  "score_probability_matrix": [[0.067, 0.093, "... 11 x 11"]],
  "home_win_probability": 0.365,
  "draw_probability": 0.262,
  "away_win_probability": 0.373,
  "over_under_probabilities": {
    "0.5": {"over": 0.933, "under": 0.067},
    "1.5": {"over": 0.754, "under": 0.246},
    "2.5": {"over": 0.509, "under": 0.491},
    "3.5": {"over": 0.289, "under": 0.711}
  },
  "btts_probability": {"yes": 0.553, "no": 0.447}
}
```

This is a real prediction (rounded here) from the registered Elo model for Brighton v
Manchester United, 24 May 2026, loaded from its MLflow artifact. The actual result
was 0-3.

Over/under lines default to 0.5, 1.5, 2.5 and 3.5. Any half-goal line from 0.5 to
19.5 can be requested. Whole-goal lines (where a push is possible) and lines outside
that range are rejected, and a line that was not computed raises an error.

## Usage

```python
from app.ml.engine import GoalPredictionEngine, ScorePrediction

engine = GoalPredictionEngine.load(model_dir, feature_set_version="v1-079714a734")
predictions = engine.predict(dataset)          # one ScorePrediction per match
predictions[0].over_under(2.5).over
predictions[0].top_scores(5)

ScorePrediction.from_rates(1.6, 1.1, rho=-0.05)  # directly from expected goals (Poisson)
ScorePrediction.from_grid(matrix)                # from any valid matrix
```

`load` refuses a model trained on a different feature-set version. Invalid matrices
are rejected: wrong shape, negative, NaN or infinite values, or a sum other than 1.

## Tests (`backend/tests/ml/test_score_engine.py`, 45 tests)

- **Independent checker.** `assert_consistent` recomputes every field with plain
  loops over the 121 scores and checks the identities between fields:
  - the matrix is non-negative and sums to 1;
  - 1X2 sums to 1;
  - over + under = 1 for every line, and over never increases as the line rises;
  - P(under 0.5) = P(0-0);
  - P(under 1.5) = P(0-0) + P(1-0) + P(0-1);
  - P(over 0.5) ≥ P(home win) and ≥ P(away win);
  - yes + no = 1, P(BTTS) ≤ P(over 1.5), and P(BTTS no) ≥ P(0-0);
  - P(draw) ≥ P(0-0);
  - expected goals lie within 0–10;
  - the most probable score is the largest cell.
- **Property-based testing** (hypothesis, 200 cases per property) applies it to:
  - Poisson grids, with and without Dixon–Coles;
  - grids rescaled to a classifier's 1X2;
  - arbitrary non-negative matrices;
  - sparse matrices,
  with all 20 half-goal lines. It also checks the total-goals distribution and
  expected total goals, and that swapping home and away mirrors the prediction.
- **Closed forms.** For independent Poisson goals:
  - total goals are Poisson(λh + λa), which fixes over/under;
  - home minus away is Skellam-distributed, which fixes 1X2;
  - P(BTTS) = (1 − e^−λh)(1 − e^−λa).
  These match to within the truncation effect.
- **Edge cases.**
  - Certain scores give certain markets, including 10-10 and 10-0.
  - A uniform matrix resolves ties to 0-0.
  - Invalid matrices and lines are rejected, and the matrix cannot be modified.
- **Every model** (7) passes the checker through the engine, and the engine returns
  the model's matrix unchanged.
- **Mutation check.** Four deliberate bugs are each caught:
  - under computed as 1 − over with a tiny error;
  - BTTS "no" counting 0-0 twice;
  - draws counted as home wins;
  - a transposed most probable score.
- **Real data.** The verification script runs the checker on 3,040 real
  predictions: 8 models × 380 test matches, with 20 lines each. The largest
  deviation of any sum from 1 was 7e-16.

## How good are the derived markets?

`evaluate()` now also scores over 1.5 / 2.5 / 3.5 and BTTS (Brier score and log
loss) for every model. The scores are tracked in MLflow as `test_over_2_5_brier`,
`test_btts_log_loss` and so on, and shown in
[model-comparison.md](model-comparison.md). The test season 2025/26 (380 matches;
55.0% over 2.5, 56.1% both teams scored):

| Model | Over 2.5 Brier | BTTS Brier |
| --- | --- | --- |
| historical baseline (constant) | 0.2475 | 0.2471 |
| elo (selected on 1X2) | 0.2638 | 0.2523 |
| logistic regression | 0.2478 | 0.2453 |
| poisson | 0.2496 | 0.2457 |
| random forest / xgboost / lightgbm | 0.2516–0.2523 | 0.2471–0.2475 |
| logistic regression + odds | 0.2473 | 0.2454 |
| *bookmakers, closing over 2.5 (reference)* | *0.2452* | |

- **Goal totals are close to unpredictable from these features.** No model clearly
  beats a constant forecast on over 2.5. Even the market is only 0.002 better.
- **The model selected for 1X2 (Elo) is the worst for goal markets.** Its goal
  model is a scaled rating, not a fitted regression. For goal markets, the Poisson
  and logistic models' goal regressions are better.
- The engine's maths does not depend on which model supplies the matrix. Which
  model should back the goal markets is a separate choice. It should be made on
  validation goal metrics, not on the test figures above; this sprint does not
  make that choice.
