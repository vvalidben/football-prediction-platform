"""Chronological splitting: never random, never overlapping."""

from datetime import timedelta

import pytest

from app.ml.dataset import Dataset
from app.ml.splits import ChronologicalSplit, SplitError, split_by_dates, split_by_seasons
from tests.ml.synthetic import START, make_dataset


@pytest.fixture
def data() -> Dataset:
    return make_dataset(seasons=4, teams=6)


def assert_chronological(split: ChronologicalSplit) -> None:
    assert split.train.examples[-1].kickoff_at < split.validation.examples[0].kickoff_at
    if len(split.test):
        assert split.validation.examples[-1].kickoff_at < split.test.examples[0].kickoff_at
    for part in (split.train, split.validation, split.test):
        kickoffs = [e.kickoff_at for e in part.examples]
        assert kickoffs == sorted(kickoffs)  # original order kept, nothing shuffled


def test_split_by_seasons_uses_last_season_for_test(data: Dataset) -> None:
    split = split_by_seasons(data)

    assert {e.season_id for e in split.train.examples} == {1, 2}
    assert {e.season_id for e in split.validation.examples} == {3}
    assert {e.season_id for e in split.test.examples} == {4}
    assert len(split.train) + len(split.validation) + len(split.test) == len(data)
    assert_chronological(split)


def test_split_by_dates(data: Dataset) -> None:
    validation_start = START + timedelta(days=365 * 2)
    test_start = START + timedelta(days=365 * 3)

    split = split_by_dates(data, validation_start, test_start)

    assert all(e.kickoff_at < validation_start for e in split.train.examples)
    assert all(validation_start <= e.kickoff_at < test_start for e in split.validation.examples)
    assert all(e.kickoff_at >= test_start for e in split.test.examples)
    assert_chronological(split)


def test_test_end_limits_the_test_period(data: Dataset) -> None:
    test_start = START + timedelta(days=365 * 3)
    split = split_by_dates(
        data, START + timedelta(days=365 * 2), test_start, test_start + timedelta(days=2)
    )

    assert 0 < len(split.test) < len(data.where(lambda e: e.season_id == 4))


def test_every_match_lands_in_exactly_one_part(data: Dataset) -> None:
    split = split_by_seasons(data)

    ids = [
        e.match_id for part in (split.train, split.validation, split.test) for e in part.examples
    ]
    assert sorted(ids) == sorted(e.match_id for e in data.examples)


@pytest.mark.parametrize(
    ("validation_offset", "test_offset", "message"),
    [
        (3, 2, "before the test period"),
        (0, 2, "train set is empty"),
        (10, 11, "validation set is empty"),
    ],
)
def test_invalid_date_splits(
    data: Dataset, validation_offset: int, test_offset: int, message: str
) -> None:
    with pytest.raises(SplitError, match=message):
        split_by_dates(
            data,
            START + timedelta(days=365 * validation_offset),
            START + timedelta(days=365 * test_offset),
        )


def test_overlapping_parts_are_rejected(data: Dataset) -> None:
    season_one = data.where(lambda e: e.season_id == 1)

    with pytest.raises(SplitError, match="overlap"):
        ChronologicalSplit(season_one, season_one, data.where(lambda e: False), "bad")


def test_too_few_seasons_for_a_season_split() -> None:
    with pytest.raises(SplitError, match="at least 3 seasons"):
        split_by_seasons(make_dataset(seasons=2, teams=4))


def test_datasets_must_be_chronological(data: Dataset) -> None:
    with pytest.raises(ValueError, match="chronological"):
        Dataset(tuple(reversed(data.examples)), data.feature_set_version)
