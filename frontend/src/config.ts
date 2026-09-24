/**
 * Runtime configuration derived from Vite environment variables.
 *
 * `VITE_API_BASE_URL` defaults to `/api`, which is proxied to the backend by the
 * Vite dev server locally and by nginx in the Docker image.
 */
export const config = {
  apiBaseUrl: (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/+$/, ''),
  healthPollIntervalMs: 30_000,
} as const
