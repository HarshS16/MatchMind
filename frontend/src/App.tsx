import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Jobs from './pages/Jobs'
import Candidates from './pages/Candidates'
import Comparison from './pages/Comparison'
import Evaluation from './pages/Evaluation'

const nav = [
  { to: '/', label: 'Overview' },
  { to: '/jobs', label: 'Jobs' },
  { to: '/candidates', label: 'Candidates' },
  { to: '/compare', label: 'Ranking Comparison' },
  { to: '/evaluation', label: 'Evaluation' },
]

export default function App() {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <header className="bg-slate-900 text-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <h1 className="text-lg font-semibold">Elip — Job Matching & Candidate Ranking</h1>
          <nav className="flex gap-1">
            {nav.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.to === '/'}
                className={({ isActive }) =>
                  `rounded px-3 py-1.5 text-sm ${isActive ? 'bg-slate-700' : 'hover:bg-slate-800'}`
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-6 py-6">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/candidates" element={<Candidates />} />
          <Route path="/compare" element={<Comparison />} />
          <Route path="/evaluation" element={<Evaluation />} />
        </Routes>
      </main>
    </div>
  )
}
