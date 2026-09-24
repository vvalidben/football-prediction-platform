"""Explanation responses: what moved the model, and in which direction.

Every response carries `caveat`, and every field name is about the model: a
`contribution` is what a feature added to the model's own score, never a claim
about what made a result happen. See docs/explainability.md.
"""

import datetime as dt

from pydantic import BaseModel, ConfigDict


class ContributionRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    label: str
    group: str
    #: The feature's value for this match; null when the data was missing.
    value: float | None
    #: Signed, on the target's `unit` scale: positive raises the model's output.
    contribution: float
    #: "positive" or "negative", so a caller need not compare floats.
    direction: str


class TargetExplanationRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    unit: str
    baseline: float
    predicted: float
    output_unit: str | None
    baseline_output: float | None
    predicted_output: float | None
    #: The largest contributions, biggest absolute value first.
    contributions: list[ContributionRead]
    #: Everything outside `contributions`, summed, so the parts still add up.
    other_contribution: float
    #: How many features are behind `other_contribution`.
    other_features: int


class ExplanationRead(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    prediction_id: int
    match_id: int
    kickoff_at: dt.datetime
    home_team: str
    away_team: str
    model_id: int
    model_name: str
    model_version: str
    algorithm: str
    method: str
    method_detail: str
    baseline_description: str
    features_used: int
    targets: list[TargetExplanationRead]
    #: Plain-language statement that this describes the model, not causes.
    caveat: str


class ImportanceEntryRead(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    label: str
    group: str
    #: Share of the model's total reliance, between 0 and 1.
    importance: float


class ImportanceRead(BaseModel):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_id: int
    model_name: str
    model_version: str
    algorithm: str
    method: str
    method_detail: str
    entries: list[ImportanceEntryRead]
    #: Features outside `entries`, and their summed share.
    other_features: int
    other_importance: float
    caveat: str
