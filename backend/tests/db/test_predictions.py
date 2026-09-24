"""Storing predictions, their probability distributions and their evaluation."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import (
    FeatureSnapshot,
    Prediction,
    PredictionOverUnderProbability,
    PredictionResult,
    PredictionScoreProbability,
)
from tests.db import factories
from tests.db.factories import assert_violates, violated_constraint

pytestmark = pytest.mark.integration

SCORE_GRID = {
    (0, 0): "0.082000",
    (1, 0): "0.117000",
    (0, 1): "0.093000",
    (1, 1): "0.132000",
    (2, 1): "0.094000",
}


def test_full_prediction_round_trip(session: Session) -> None:
    match = factories.match(session)
    version = factories.model_version(session, name="dixon_coles")
    snapshot = FeatureSnapshot(
        match=match,
        feature_set_version="v1",
        data_cutoff_at=datetime(2025, 8, 15, tzinfo=UTC),
        features={"home_attack": 1.21, "away_defence": 0.93},
    )
    prediction = factories.prediction(
        session,
        match,
        version,
        feature_snapshot=snapshot,
        score_probabilities=[
            PredictionScoreProbability(home_goals=h, away_goals=a, probability=Decimal(p))
            for (h, a), p in SCORE_GRID.items()
        ],
        over_under_probabilities=[
            PredictionOverUnderProbability(
                line=Decimal("2.5"),
                over_probability=Decimal("0.512000"),
                under_probability=Decimal("0.488000"),
            ),
            PredictionOverUnderProbability(
                line=Decimal("1.5"),
                over_probability=Decimal("0.743000"),
                under_probability=Decimal("0.257000"),
            ),
        ],
    )
    session.commit()
    session.expire_all()

    loaded = session.get_one(Prediction, prediction.id)
    assert loaded.match_id == match.id
    assert loaded.model_version.name == "dixon_coles"
    assert loaded.feature_snapshot is not None
    assert loaded.feature_snapshot.features["home_attack"] == 1.21
    assert (loaded.predicted_home_goals, loaded.predicted_away_goals) == (1, 1)
    assert (loaded.expected_home_goals, loaded.expected_away_goals) == (
        Decimal("1.420"),
        Decimal("1.130"),
    )
    assert (
        loaded.home_win_probability + loaded.draw_probability + loaded.away_win_probability
        == Decimal("1")
    )
    assert loaded.btts_probability == Decimal("0.520000")
    assert {
        (cell.home_goals, cell.away_goals): str(cell.probability)
        for cell in loaded.score_probabilities
    } == SCORE_GRID
    # Ordered by line via the relationship definition.
    assert [row.line for row in loaded.over_under_probabilities] == [
        Decimal("1.50"),
        Decimal("2.50"),
    ]


def test_most_likely_score_is_queryable(session: Session) -> None:
    prediction = factories.prediction(
        session,
        score_probabilities=[
            PredictionScoreProbability(home_goals=h, away_goals=a, probability=Decimal(p))
            for (h, a), p in SCORE_GRID.items()
        ],
    )

    top = session.execute(
        select(PredictionScoreProbability.home_goals, PredictionScoreProbability.away_goals)
        .where(PredictionScoreProbability.prediction_id == prediction.id)
        .order_by(PredictionScoreProbability.probability.desc())
        .limit(1)
    ).one()

    assert tuple(top) == (1, 1)


def test_outcome_probabilities_must_sum_to_one(session: Session) -> None:
    with pytest.raises(IntegrityError, match="ck_predictions_outcome_probabilities_sum_to_one"):
        factories.prediction(session, home_win_probability=Decimal("0.500000"))


def test_probabilities_must_be_within_zero_and_one(session: Session) -> None:
    with pytest.raises(IntegrityError, match="ck_predictions_probabilities_in_range"):
        factories.prediction(session, btts_probability=Decimal("1.200000"))


def test_score_cell_is_unique_per_prediction(session: Session) -> None:
    prediction = factories.prediction(session)
    session.add(
        PredictionScoreProbability(
            prediction=prediction, home_goals=1, away_goals=0, probability=Decimal("0.1")
        )
    )
    session.flush()

    def add_duplicate() -> None:
        # Bypass the identity map, which would otherwise merge the two objects.
        session.execute(
            insert(PredictionScoreProbability).values(
                prediction_id=prediction.id, home_goals=1, away_goals=0, probability=0.2
            )
        )

    assert violated_constraint(session, add_duplicate) == "pk_prediction_score_probabilities"


def test_over_under_allows_push_but_not_excess(session: Session) -> None:
    prediction = factories.prediction(session)
    # On a whole-number line the total can land exactly on it, so over + under < 1.
    prediction.over_under_probabilities.append(
        PredictionOverUnderProbability(
            line=Decimal("3"),
            over_probability=Decimal("0.300000"),
            under_probability=Decimal("0.450000"),
        )
    )
    session.flush()

    assert_violates(
        session,
        "ck_prediction_over_under_probabilities_at_most_one",
        lambda: PredictionOverUnderProbability(
            prediction=prediction,
            line=Decimal("2.5"),
            over_probability=Decimal("0.600000"),
            under_probability=Decimal("0.600000"),
        ),
    )


def test_prediction_result_is_one_per_prediction(session: Session) -> None:
    match = factories.match(session, status="finished", home_score=1, away_score=1)
    prediction = factories.prediction(session, match)
    prediction.result = PredictionResult(
        evaluated_at=datetime(2025, 8, 17, tzinfo=UTC),
        outcome_correct=False,
        exact_score_correct=True,
        btts_correct=True,
        brier_score=Decimal("0.726600"),
        log_loss=Decimal("1.309333"),
        ranked_probability_score=Decimal("0.112150"),
        actual_score_probability=Decimal("0.132000"),
        metrics={"over_2_5_correct": True},
    )
    session.commit()
    session.expire_all()

    stored = session.get_one(Prediction, prediction.id).result
    assert stored is not None
    assert stored.exact_score_correct is True
    assert stored.metrics == {"over_2_5_correct": True}

    def add_second_result() -> None:
        session.execute(
            insert(PredictionResult).values(
                prediction_id=prediction.id,
                evaluated_at=datetime(2025, 8, 18, tzinfo=UTC),
                outcome_correct=True,
                exact_score_correct=False,
                btts_correct=False,
                brier_score=0,
                log_loss=0,
                ranked_probability_score=0,
            )
        )

    assert violated_constraint(session, add_second_result) == "pk_prediction_results"


def test_same_model_can_predict_a_match_again_later(session: Session) -> None:
    first = factories.prediction(session)

    later = factories.prediction(session, first.match, first.model_version)

    assert later.id != first.id
    assert_violates(
        session,
        "uq_predictions_match_id_model_version_id_predicted_at",
        lambda: Prediction(
            match=first.match,
            model_version=first.model_version,
            predicted_at=first.predicted_at,
            predicted_home_goals=0,
            predicted_away_goals=0,
            expected_home_goals=Decimal("1"),
            expected_away_goals=Decimal("1"),
            home_win_probability=Decimal("0.4"),
            draw_probability=Decimal("0.3"),
            away_win_probability=Decimal("0.3"),
            btts_probability=Decimal("0.5"),
        ),
    )


def test_deleting_a_prediction_removes_its_distributions(session: Session) -> None:
    prediction = factories.prediction(
        session,
        score_probabilities=[
            PredictionScoreProbability(home_goals=0, away_goals=0, probability=Decimal("0.1"))
        ],
    )
    prediction_id = prediction.id

    session.delete(prediction)
    session.flush()

    remaining = session.scalar(
        select(func.count()).where(PredictionScoreProbability.prediction_id == prediction_id)
    )
    assert remaining == 0


def test_predicted_match_and_model_version_cannot_be_deleted(session: Session) -> None:
    prediction = factories.prediction(session)

    assert (
        violated_constraint(session, lambda: session.delete(prediction.match))
        == "fk_predictions_match_id_matches"
    )
    assert (
        violated_constraint(session, lambda: session.delete(prediction.model_version))
        == "fk_predictions_model_version_id_model_versions"
    )


def test_only_one_active_version_per_model(session: Session) -> None:
    factories.model_version(session, name="elo").is_active = True
    session.flush()

    second = factories.model_version(session, name="elo")
    second.is_active = True

    assert violated_constraint(session, lambda: None) == "uq_model_versions_one_active_per_name"
