import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'
import { Layout } from './components/Layout.tsx'
import { Dashboard } from './pages/Dashboard.tsx'
import { DataStatus } from './pages/DataStatus.tsx'
import { History } from './pages/History.tsx'
import { Matches } from './pages/Matches.tsx'
import { MatchPrediction } from './pages/MatchPrediction.tsx'
import { ModelPerformance } from './pages/ModelPerformance.tsx'
import { Team } from './pages/Team.tsx'
import { Teams } from './pages/Teams.tsx'

export default function App() {
  const location = useLocation()
  return (
    <Layout>
      {/* Keyed by path: navigating away from a broken view clears the error. */}
      <ErrorBoundary resetKey={location.pathname}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/matches" element={<Matches />} />
          <Route path="/matches/:matchId" element={<MatchPrediction />} />
          <Route path="/history" element={<History />} />
          <Route path="/teams" element={<Teams />} />
          <Route path="/teams/:teamId" element={<Team />} />
          <Route path="/models" element={<ModelPerformance />} />
          <Route path="/data" element={<DataStatus />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ErrorBoundary>
    </Layout>
  )
}
