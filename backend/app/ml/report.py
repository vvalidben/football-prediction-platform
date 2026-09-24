"""Model comparison report (Markdown), generated from an experiment result.

Besides the metric tables, it gives a paired bootstrap interval for each model's
test log-loss difference from the selected model: with a few hundred test matches,
differences of a few thousandths are often within noise, and the report says so
instead of ranking by the point estimate alone.
"""

from collections.abc import Sequence
from typing import Any

from app.ml.ensemble import EnsembleResult
from app.ml.experiment import CandidateResult, ExperimentResult
from app.ml.metrics import BOOTSTRAP_SAMPLES, paired_bootstrap
from app.ml.trees import TreeEnsembleModel

DASH = chr(0x2013)  # en dash, for empty cells and date ranges

#: (metric key, column label, decimals, is a percentage)
COLUMNS: tuple[tuple[str, str, int, bool], ...] = (
    ("log_loss", "Log loss", 4, False),
    ("brier", "Brier", 4, False),
    ("rps", "RPS", 4, False),
    ("accuracy", "Accuracy", 1, True),
    ("total_goals_mae", "Goals MAE", 3, False),
    ("total_goals_rmse", "Goals RMSE", 3, False),
    ("exact_score_accuracy", "Exact score", 1, True),
    ("top3_score_coverage", "Top-3", 1, True),
    ("top5_score_coverage", "Top-5", 1, True),
    ("actual_score_log_prob", "log P(score)", 3, False),
)
#: Markets derived from each model's score grid (Brier: single-probability form, 0-1).
MARKET_COLUMNS: tuple[tuple[str, str, int, bool], ...] = (
    ("over_1_5_brier", "O/U 1.5 Brier", 4, False),
    ("over_2_5_brier", "O/U 2.5 Brier", 4, False),
    ("over_2_5_log_loss", "O/U 2.5 log loss", 4, False),
    ("over_3_5_brier", "O/U 3.5 Brier", 4, False),
    ("btts_brier", "BTTS Brier", 4, False),
    ("btts_log_loss", "BTTS log loss", 4, False),
)


def _cell(metrics: dict[str, float] | None, key: str, decimals: int, percent: bool) -> str:
    if not metrics or key not in metrics:
        return DASH
    value = metrics[key] * (100 if percent else 1)
    return f"{value:.{decimals}f}{'%' if percent else ''}"


def _metric_table(
    candidates: Sequence[CandidateResult],
    metrics_of: str,
    selected: str,
    columns: Sequence[tuple[str, str, int, bool]] = COLUMNS,
) -> list[str]:
    lines = [
        "| Model | " + " | ".join(label for _, label, _, _ in columns) + " |",
        "| --- |" + " ---: |" * len(columns),
    ]
    for c in candidates:
        metrics = getattr(c, metrics_of)
        name = f"**{c.name}** (selected)" if c.name == selected else c.name
        lines.append(
            f"| {name} | " + " | ".join(_cell(metrics, k, d, p) for k, _, d, p in columns) + " |"
        )
    return lines


def _period(info: dict[str, Any]) -> str:
    return f"{info['start'][:10]} {DASH} {info['end'][:10]} ({info['matches']} matches)"


def comparisons(result: ExperimentResult) -> dict[str, dict[str, float]]:
    """Paired test log-loss differences from the selected model."""
    reference = result.candidate(result.selected).test_losses
    if not reference:
        return {}
    return {
        c.name: paired_bootstrap(c.test_losses, reference)
        for c in result.candidates
        if c.name != result.selected and c.test_losses
    }


def render_report(
    result: ExperimentResult,
    *,
    title: str = "Model comparison",
    ensemble: EnsembleResult | None = None,
) -> str:
    split = result.split
    cv = result.selection_metric.startswith("cv")
    periods = f"Train: {_period(split['train'])}; validation: {_period(split['validation'])}; "
    periods += f"test: {_period(split['test'])}." if split["test"]["matches"] else "no test period."
    lines = [
        f"# {title}",
        "",
        f"Generated {result.finished_at:%Y-%m-%d %H:%M} UTC by `python -m app.ml train`. "
        f"Feature set `{result.feature_set_version}`.",
        "",
        "## Data and protocol",
        "",
        f"- Split: {split['split']}.",
        f"- {periods}",
        f"- Tuning and selection metric: `{result.selection_metric}` (lower is better). "
        + (
            f"Time-aware cross-validation over {len(result.cv_folds)} rolling-origin folds; "
            "each fold is predicted by a model fitted only on earlier matches:"
            if cv
            else "Hyperparameters and the model are chosen on the validation period only."
        ),
    ]
    if cv:
        lines += [
            f"  {i}. fit {_period(f['train'])} → predict {_period(f['validation'])}"
            for i, f in enumerate(result.cv_folds, 1)
        ]
    lines += [
        "- The test period is used once, after selection, to report the tuned models "
        "refitted on train + validation. It influenced no choice.",
        "",
        "## Selection (no test data)",
        "",
        "| Model | Type | Selection score | Validation log loss | CV log loss "
        "| Grid points | Chosen hyperparameters |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in sorted(result.candidates, key=lambda c: c.selection_score):
        name = f"**{c.name}** (selected)" if c.name == result.selected else c.name
        cv_loss = f"{c.cv_metrics['log_loss']:.4f}" if c.cv_metrics else DASH
        params = ", ".join(f"{k}={v}" for k, v in c.params.items()) or DASH
        lines.append(
            f"| {name} | {c.model_type} | {c.selection_score:.4f} "
            f"| {c.validation_metrics['log_loss']:.4f} | {cv_loss} | {c.tuning_trials} "
            f"| {params} |"
        )
    lines += ["", "## Validation metrics", ""]
    lines += _metric_table(result.candidates, "validation_metrics", result.selected)
    if cv:
        lines += ["", "## Cross-validation metrics (pooled over folds)", ""]
        lines += _metric_table(result.candidates, "cv_metrics", result.selected)
    if split["test"]["matches"]:
        lines += ["", "## Test metrics (reported only)", ""]
        lines += _metric_table(result.candidates, "test_metrics", result.selected)
        lines += [
            "",
            "## Derived markets on test (from each model's score grid)",
            "",
            "Over/under and both-teams-to-score probabilities are sums over the same score "
            "grid as the 1X2 and exact-score figures; lower is better.",
            "",
        ]
        lines += _metric_table(result.candidates, "test_metrics", result.selected, MARKET_COLUMNS)
        lines += [
            "",
            f"## Test log loss compared with the selected model ({result.selected})",
            "",
            "Mean per-match difference (negative = better than the selected model), "
            f"with a 95% paired bootstrap interval over test matches ({BOOTSTRAP_SAMPLES} "
            "resamples). An interval that contains 0 means the test set cannot "
            "distinguish the two models.",
            "",
            "| Model | Difference | 95% interval | Resamples where better |",
            "| --- | ---: | --- | ---: |",
        ]
        for name, row in sorted(comparisons(result).items(), key=lambda kv: kv[1]["difference"]):
            lines.append(
                f"| {name} | {row['difference']:+.4f} "
                f"| [{row['low']:+.4f}, {row['high']:+.4f}] | {row['better_share']:.0%} |"
            )
    importances = [
        (c.name, c.final_model.feature_importances())
        for c in result.candidates
        if isinstance(c.final_model, TreeEnsembleModel)
    ]
    if importances:
        lines += [
            "",
            "## Most important features (tree models, 1X2 classifier of the final model)",
            "",
            "| Rank | " + " | ".join(name for name, _ in importances) + " |",
            "| ---: |" + " --- |" * len(importances),
        ]
        ranked = [list(values.items())[:10] for _, values in importances]
        for rank in range(max(len(r) for r in ranked)):
            cells = [f"{r[rank][0]} ({r[rank][1]:.3f})" if rank < len(r) else "" for r in ranked]
            lines.append(f"| {rank + 1} | " + " | ".join(cells) + " |")
    if ensemble is not None:
        lines += render_ensembles(ensemble, result)
    return "\n".join(lines) + "\n"


#: Columns of the ensemble tables: 1X2, goals and derived markets.
ENSEMBLE_COLUMNS: tuple[tuple[str, str, int, bool], ...] = (
    ("log_loss", "Log loss", 4, False),
    ("brier", "Brier", 4, False),
    ("rps", "RPS", 4, False),
    ("accuracy", "Accuracy", 1, True),
    ("total_goals_mae", "Goals MAE", 3, False),
    ("actual_score_log_prob", "log P(score)", 3, False),
    ("over_2_5_brier", "O/U 2.5 Brier", 4, False),
    ("btts_brier", "BTTS Brier", 4, False),
)


def _row(name: str, metrics: dict[str, float] | None) -> str:
    cells = " | ".join(_cell(metrics, k, d, p) for k, _, d, p in ENSEMBLE_COLUMNS)
    return f"| {name} | {cells} |"


def _header(first: str) -> list[str]:
    return [
        f"| {first} | " + " | ".join(label for _, label, _, _ in ENSEMBLE_COLUMNS) + " |",
        "| --- |" + " ---: |" * len(ENSEMBLE_COLUMNS),
    ]


def forward_comparisons(ensemble: EnsembleResult) -> dict[str, dict[str, float]]:
    """Paired forward-validation log-loss differences from the best individual model,
    in ranking order."""
    reference = ensemble.forward.losses[ensemble.best_individual]
    return {
        name: paired_bootstrap(ensemble.forward.losses[name], reference)
        for name, _ in ensemble.ranking
        if name != ensemble.best_individual
    }


def render_ensembles(ensemble: EnsembleResult, result: ExperimentResult) -> list[str]:
    methods = [c.method for c in ensemble.candidates]
    evaluated = ensemble.folds[1:]
    lines = [
        "",
        "## Ensembles",
        "",
        f"Members: {', '.join(ensemble.members)}. Each ensemble combines the members' score "
        "grids, so every derived market stays consistent.",
        "",
        "- **simple_average**: the mean of the grids (nothing fitted).",
        "- **weighted_average**: weights >= 0 summing to 1 that minimise the 1X2 log loss "
        "of the mixture.",
        "- **stacking**: a multinomial logistic regression on the members' 1X2 log-odds; "
        "the average grid is rescaled to its 1X2.",
        "",
        "Weights and meta-models are fitted only on **out-of-fold validation predictions**: "
        "every match below was predicted by a member fitted only on earlier matches. The "
        "test period is not used.",
        "",
        "### Forward-chained validation (the basis for every choice)",
        "",
        "Each block is predicted with combiners fitted on the earlier blocks only; the first "
        "block only trains the combiners. Individual models are scored on the same matches.",
        "",
    ]
    lines += [
        f"{i}. {_period(f['validation'])}" + (" (combiner training only)" if i == 1 else "")
        for i, f in enumerate(ensemble.folds, 1)
    ]
    pooled_matches = sum(f["validation"]["matches"] for f in evaluated)
    lines += ["", f"Pooled over {pooled_matches} matches:", ""]
    lines += _header("Model")
    for name, _ in ensemble.ranking:
        label = f"{name} ({'ensemble' if name in methods else 'individual'})"
        if name == ensemble.best_overall:
            label = f"**{label}** (best)"
        lines.append(_row(label, ensemble.forward.metrics[name]))
    reference = ensemble.best_individual
    lines += [
        "",
        f"Forward-validation log loss minus the best individual model ({reference}), "
        f"with a 95% paired bootstrap interval over the {pooled_matches} matches. An interval "
        "containing 0 means validation cannot tell the entry apart from that model:",
        "",
        "| Model | Difference | 95% interval | Resamples where better |",
        "| --- | ---: | --- | ---: |",
    ]
    for name, row in forward_comparisons(ensemble).items():
        lines.append(
            f"| {name} | {row['difference']:+.4f} "
            f"| [{row['low']:+.4f}, {row['high']:+.4f}] | {row['better_share']:.0%} |"
        )
    advice = ensemble.recommendation
    lines += [
        "",
        f"**Recommended: {advice['recommended']}.** Rule: {advice['rule']}. "
        f"Reason: {advice['reason']}.",
    ]
    lines += [
        "",
        "### Weights",
        "",
        "Fitted on all validation blocks (used by the final ensembles):",
        "",
        "| Member | " + " | ".join(methods) + " |",
        "| --- |" + " ---: |" * len(methods),
    ]
    for member in ensemble.members:
        shares = []
        for option in ensemble.candidates:
            weights = option.combiner.params().get("weights")
            shares.append(f"{weights[member]:.3f}" if weights else "(meta-model)")
        lines.append(f"| {member} | " + " | ".join(shares) + " |")
    if "weighted_average" in methods:
        lines += [
            "",
            "Weighted-average weights as fitted at each forward step (stability check):",
            "",
            "| Fitted on blocks | " + " | ".join(ensemble.members) + " |",
            "| --- |" + " ---: |" * len(ensemble.members),
        ]
        for k, fold in enumerate(ensemble.forward.fitted, 1):
            weights = fold["weighted_average"]["weights"]
            step = " | ".join(f"{weights[m]:.3f}" for m in ensemble.members)
            lines.append(f"| 1-{k} | {step} |")
    if result.split["test"]["matches"]:
        lines += ["", "### Test (reported only)", ""]
        lines += _header("Model")
        for single in result.candidates:
            if single.name in ensemble.members:
                lines.append(_row(f"{single.name} (individual)", single.test_metrics))
        for entry in ensemble.candidates:
            lines.append(_row(f"{entry.method} (ensemble)", entry.test_metrics))
        references = list(dict.fromkeys([ensemble.best_overall, result.selected]))
        for reference in references:
            if reference in methods or not result.candidate(reference).test_losses:
                continue
            base = result.candidate(reference).test_losses
            lines += [
                "",
                f"Test log loss of each ensemble minus {reference} "
                "(95% paired bootstrap interval):",
                "",
                "| Ensemble | Difference | 95% interval | Resamples where better |",
                "| --- | ---: | --- | ---: |",
            ]
            for entry in ensemble.candidates:
                row = paired_bootstrap(entry.test_losses, base)
                lines.append(
                    f"| {entry.method} | {row['difference']:+.4f} "
                    f"| [{row['low']:+.4f}, {row['high']:+.4f}] | {row['better_share']:.0%} |"
                )
    return lines
