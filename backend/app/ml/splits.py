"""Chronological train / validation / test splits. Never random.

Football data is a time series: shuffling would let a model learn from matches played
after the ones it is evaluated on. Every split here is by kickoff time, and
`ChronologicalSplit` verifies that the latest training match precedes the earliest
validation match, which precedes the earliest test match.
"""

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

from app.ml.dataset import Dataset


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class ChronologicalSplit:
    train: Dataset
    validation: Dataset
    test: Dataset
    description: str

    def __post_init__(self) -> None:
        for name, part in (("train", self.train), ("validation", self.validation)):
            if len(part) == 0:
                raise SplitError(f"{name} set is empty")
        ordered = [p for p in (self.train, self.validation, self.test) if len(p)]
        for earlier, later in pairwise(ordered):
            if earlier.examples[-1].kickoff_at >= later.examples[0].kickoff_at:
                raise SplitError("split periods overlap or are out of order")

    def describe(self) -> dict[str, object]:
        return {
            "split": self.description,
            "train": self.train.describe(),
            "validation": self.validation.describe(),
            "test": self.test.describe(),
        }


def split_by_dates(
    data: Dataset,
    validation_start: datetime,
    test_start: datetime,
    test_end: datetime | None = None,
) -> ChronologicalSplit:
    """train < validation_start ≤ validation < test_start ≤ test (< test_end)."""
    if not validation_start < test_start:
        raise SplitError("validation must start before the test period")
    if test_end is not None and not test_start < test_end:
        raise SplitError("test period must end after it starts")
    return ChronologicalSplit(
        train=data.where(lambda e: e.kickoff_at < validation_start),
        validation=data.where(lambda e: validation_start <= e.kickoff_at < test_start),
        test=data.where(
            lambda e: e.kickoff_at >= test_start and (test_end is None or e.kickoff_at < test_end)
        ),
        description=(
            f"dates: validation from {validation_start.date()}, test from {test_start.date()}"
            + (f" to {test_end.date()}" if test_end else "")
        ),
    )


def split_by_seasons(data: Dataset) -> ChronologicalSplit:
    """Last season → test, the one before → validation, all earlier → train.

    Seasons are ordered by their first kickoff in the data. Seasons of different
    competitions overlap in time, so this is meant for single-competition data;
    use `split_by_dates` otherwise.
    """
    first_kickoff: dict[int, datetime] = {}
    for example in data.examples:
        first_kickoff.setdefault(example.season_id, example.kickoff_at)
    seasons = sorted(first_kickoff, key=first_kickoff.__getitem__)
    if len(seasons) < 3:
        raise SplitError(f"need at least 3 seasons for a season split, have {len(seasons)}")
    validation_season, test_season = seasons[-2], seasons[-1]
    return ChronologicalSplit(
        train=data.where(lambda e: e.season_id not in (validation_season, test_season)),
        validation=data.where(lambda e: e.season_id == validation_season),
        test=data.where(lambda e: e.season_id == test_season),
        description=f"seasons: {len(seasons) - 2} train, 1 validation, 1 test",
    )
