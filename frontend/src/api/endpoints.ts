/** One function per endpoint the pages use, so a URL is written in exactly one place. */
import { apiGet, apiPost, type Page } from './client.ts'
import type {
  DataStatus,
  Explanation,
  HistoryPage,
  Freshness,
  MatchDetail,
  MatchRead,
  Importance,
  ModelPerformance,
  ModelVersion,
  Named,
  Prediction,
  SeasonRead,
  StoredPrediction,
  TeamDetail,
} from './types.ts'

export const getUpcomingMatches = (
  params: { competition_id?: number; team_id?: number; limit?: number },
  signal?: AbortSignal,
) => apiGet<Page<MatchRead>>('/matches/upcoming', params, signal)

export const getMatches = (
  params: { competition_id?: number; team_id?: number; status?: string; limit?: number },
  signal?: AbortSignal,
) => apiGet<Page<MatchRead>>('/matches', params, signal)

export const getMatch = (matchId: number, signal?: AbortSignal) =>
  apiGet<MatchDetail>(`/matches/${String(matchId)}`, {}, signal)

export const getTeam = (teamId: number, matches: number, signal?: AbortSignal) =>
  apiGet<TeamDetail>(`/teams/${String(teamId)}`, { matches }, signal)

export const getTeams = (
  params: { search?: string; competition_id?: number; limit?: number },
  signal?: AbortSignal,
) =>
  apiGet<Page<Named & { short_name: string | null; country: string | null }>>(
    '/teams',
    params,
    signal,
  )

export const getCompetitions = (signal?: AbortSignal) =>
  apiGet<Page<{ id: number; name: string; country: string | null }>>(
    '/competitions',
    { limit: 100 },
    signal,
  )

export const getStoredPredictions = (
  params: { match_id?: number; model_name?: string; limit?: number },
  signal?: AbortSignal,
) => apiGet<Page<StoredPrediction>>('/predictions', params, signal)

export const predictMatch = (
  body: { match_id: number; model_name?: string; refresh?: boolean },
  signal?: AbortSignal,
) => apiPost<Prediction>('/predictions/match', body, signal)

export const getModels = (params: { active_only?: boolean } = {}, signal?: AbortSignal) =>
  apiGet<ModelVersion[]>('/models', params, signal)

export const getModelPerformance = (modelId: number, signal?: AbortSignal) =>
  apiGet<ModelPerformance>(`/models/${String(modelId)}/performance`, {}, signal)

export const getModelMetrics = (modelId: number, signal?: AbortSignal) =>
  apiGet<{ model: { name: string }; metrics: Record<string, unknown> }>(
    `/models/${String(modelId)}/metrics`,
    {},
    signal,
  )

export const getDataStatus = (signal?: AbortSignal) =>
  apiGet<DataStatus>('/data/status', {}, signal)

export const getFreshness = (signal?: AbortSignal) =>
  apiGet<Freshness>('/data/freshness', {}, signal)

export const getExplanation = (predictionId: number, top: number, signal?: AbortSignal) =>
  apiGet<Explanation>(`/predictions/${String(predictionId)}/explanation`, { top }, signal)

export const getImportance = (modelId: number, top: number, signal?: AbortSignal) =>
  apiGet<Importance>(`/models/${String(modelId)}/importance`, { top }, signal)

export interface HistoryQuery {
  competition_id?: number
  season_id?: number
  team_id?: number
  model_name?: string
  from_date?: string
  to_date?: string
  evaluated?: boolean
  outcome_correct?: boolean
  limit?: number
  offset?: number
}

export const getHistory = (params: HistoryQuery, signal?: AbortSignal) =>
  apiGet<HistoryPage>('/predictions/history', { ...params }, signal)

export const getSeasons = (competition_id: number | undefined, signal?: AbortSignal) =>
  apiGet<Page<SeasonRead>>('/seasons', { competition_id, limit: 100 }, signal)
