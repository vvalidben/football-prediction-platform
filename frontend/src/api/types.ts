/** The shapes the backend returns, as the pages read them. */

export interface Named {
  id: number
  name: string
}

export interface MatchRead {
  id: number
  kickoff_at: string
  status: string
  competition: Named
  season: string | null
  matchday: number | null
  home_team: Named
  away_team: Named
  home_score: number | null
  away_score: number | null
}

export interface MatchDetail extends MatchRead {
  venue: string | null
  referee: string | null
  half_time_score: { home: number; away: number } | null
  predictions: {
    id: number
    model: string
    version: string
    predicted_at: string
    home_win_probability: number
    draw_probability: number
    away_win_probability: number
  }[]
}

export interface ScoreProbability {
  home_goals: number
  away_goals: number
  probability: number
}

export interface OverUnder {
  line: number
  over: number
  under: number
}

export interface Prediction {
  id: number
  match_id: number
  kickoff_at: string
  competition: string | null
  home_team: string
  away_team: string
  expected_home_goals: number
  expected_away_goals: number
  most_probable_score: ScoreProbability
  top_scores: ScoreProbability[]
  score_probability_matrix: number[][]
  max_goals: number
  home_win_probability: number
  draw_probability: number
  away_win_probability: number
  over_under_probabilities: OverUnder[]
  btts_probability: { yes: number; no: number }
  model: { id: number; name: string; version: string; algorithm: string }
  predicted_at: string
  data_freshness: {
    feature_cutoff_at: string
    latest_result_at: string | null
    last_ingestion_at: string | null
    age_seconds: number
    match_started: boolean
  }
  reused: boolean
}

export interface StoredPrediction {
  id: number
  match_id: number
  kickoff_at: string
  competition: string | null
  home_team: string
  away_team: string
  home_win_probability: number
  draw_probability: number
  away_win_probability: number
  most_probable_score: string
  expected_home_goals: number
  expected_away_goals: number
  model_name: string
  model_version: string
  predicted_at: string
  actual_score: string | null
  outcome_correct: boolean | null
}

export interface SideRecord {
  side: string
  played: number
  wins: number
  draws: number
  losses: number
  scored: number
  conceded: number
  clean_sheets: number
  goals_for_per_match: number | null
  goals_against_per_match: number | null
}

export interface TeamDetail {
  team: { id: number; name: string; short_name: string | null; country: string | null }
  record: {
    played: number
    wins: number
    draws: number
    losses: number
    goals_for: number
    goals_against: number
    goals_for_per_match: number | null
    goals_against_per_match: number | null
    clean_sheets: number
    home: SideRecord
    away: SideRecord
    form: string[]
    expected_goals_for_per_match: number | null
    expected_goals_against_per_match: number | null
    matches_with_expected_goals: number
  }
  recent_matches: MatchRead[]
  trend: {
    match_id: number
    kickoff_at: string
    opponent: string
    is_home: boolean
    goals_for: number
    goals_against: number
    result: string
    expected_goals_for: number | null
    rolling_goals_for: number
    rolling_goals_against: number
  }[]
}

/** How a model did in one competition, on matches it was not trained on. */
export interface CompetitionReliability {
  matches: number
  accuracy: number
  log_loss: number
  exact_score_accuracy: number
  home_win_rate: number
  start: string
  end: string
}

export interface ModelVersion {
  id: number
  name: string
  version: string
  algorithm: string
  feature_set_version: string | null
  dataset_version: string
  training_period: { start: string | null; end: string | null }
  competitions: { id: number; name: string | null }[]
  hyperparameters: Record<string, unknown>
  metrics: Record<string, unknown>
  created_at: string
  trained_at: string | null
  is_active: boolean
  description: string | null
}

export interface ModelPerformance {
  model: { id: number; name: string; version: string; algorithm: string; is_active: boolean }
  evaluated: number
  accuracy: number | null
  log_loss: number | null
  brier: number | null
  ranked_probability_score: number | null
  exact_score_accuracy: number | null
  calibration: {
    low: number
    high: number
    count: number
    mean_predicted: number
    observed: number
  }[]
  monthly: { month: string; matches: number; accuracy: number; log_loss: number }[]
  first_match: string | null
  last_match: string | null
}

export interface DataStatus {
  generated_at: string
  feature_set_version: string
  ingestion: { job: string; status: string; last_run_at: string | null; records_inserted: number }[]
  jobs: { task: string; status: string; last_run_at: string | null }[]
  coverage: {
    competition_id: number
    competition: string
    seasons: number
    matches: number
    finished: number
    with_statistics: number
    with_features: number
  }[]
  issues: { severity: string; issue_type: string; count: number }[]
  open_error_issues: number
  last_successful_ingestion_at: string | null
  latest_finished_match_at: string | null
  latest_feature_snapshot_at: string | null
  matches_missing_features: number
  providers: {
    code: string
    name: string
    is_active: boolean
    last_run_at: string | null
    runs: number
  }[]
}

export interface Freshness {
  generated_at: string
  feature_set_version: string
  last_successful_ingestion_at: string | null
  last_successful_job_at: string | null
  latest_finished_match_at: string | null
  latest_feature_snapshot_at: string | null
  upcoming_matches: number
  predictions_last_24h: number
  active_models: number
}

export interface Contribution {
  feature: string
  label: string
  group: string
  value: number | null
  contribution: number
  direction: 'positive' | 'negative'
}

export interface ExplanationTarget {
  key: string
  label: string
  unit: string
  baseline: number
  predicted: number
  output_unit: string | null
  baseline_output: number | null
  predicted_output: number | null
  contributions: Contribution[]
  other_contribution: number
  other_features: number
}

export interface Explanation {
  prediction_id: number
  match_id: number
  kickoff_at: string
  home_team: string
  away_team: string
  model_id: number
  model_name: string
  model_version: string
  algorithm: string
  method: string
  method_detail: string
  baseline_description: string
  features_used: number
  targets: ExplanationTarget[]
  caveat: string
}

export interface Importance {
  model_id: number
  model_name: string
  model_version: string
  algorithm: string
  method: string
  method_detail: string
  entries: { feature: string; label: string; group: string; importance: number }[]
  other_features: number
  other_importance: number
  caveat: string
}

export interface HistoryRow {
  prediction_id: number
  match_id: number
  kickoff_at: string
  competition_id: number | null
  competition: string | null
  season_id: number | null
  season: string | null
  home_team_id: number
  away_team_id: number
  home_team: string
  away_team: string
  predicted_home_goals: number
  predicted_away_goals: number
  expected_home_goals: number
  expected_away_goals: number
  home_win_probability: number
  draw_probability: number
  away_win_probability: number
  btts_probability: number
  predicted_outcome: 'home' | 'draw' | 'away'
  status: string
  home_score: number | null
  away_score: number | null
  actual_outcome: 'home' | 'draw' | 'away' | null
  evaluated_at: string | null
  outcome_correct: boolean | null
  exact_score_correct: boolean | null
  btts_correct: boolean | null
  brier_score: number | null
  log_loss: number | null
  ranked_probability_score: number | null
  actual_score_probability: number | null
  home_goals_error: number | null
  away_goals_error: number | null
  actual_outcome_probability: number | null
  model_id: number
  model_name: string
  model_version: string
  predicted_at: string
}

export interface HistorySummary {
  predictions: number
  evaluated: number
  pending: number
  accuracy: number | null
  exact_score_accuracy: number | null
  btts_accuracy: number | null
  log_loss: number | null
  brier_score: number | null
  ranked_probability_score: number | null
  mean_home_goals_error: number | null
  mean_away_goals_error: number | null
  mean_absolute_goals_error: number | null
  first_kickoff_at: string | null
  last_kickoff_at: string | null
}

export interface HistoryPage {
  total: number
  limit: number
  offset: number
  items: HistoryRow[]
  summary: HistorySummary
}

export interface SeasonRead {
  id: number
  competition_id: number
  competition: string | null
  name: string
  start_date: string | null
  end_date: string | null
  is_current: boolean
}
