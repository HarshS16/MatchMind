import { useEffect, useState } from 'react'
import { api, type Health, type Job } from '../services/api'

export default function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [candidates, setCandidates] = useState(0)
  const [experiments, setExperiments] = useState(0)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
    api.listJobs().then(setJobs).catch(() => {})
    api.listCandidates(1).then((r) => setCandidates(r.total)).catch(() => {})
    api.listExperiments().then((e) => setExperiments(e.length)).catch(() => {})
  }, [])

  const cards = [
    { label: 'Resumes indexed', value: candidates },
    { label: 'Job descriptions', value: jobs.length },
    { label: 'Experiments', value: experiments },
    { label: 'API status', value: health?.status ?? 'unreachable' },
  ]

  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">Dashboard Overview</h2>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {cards.map((c) => (
          <div key={c.label} className="rounded-lg border bg-white p-4 shadow-sm">
            <div className="text-sm text-slate-500">{c.label}</div>
            <div className="mt-1 text-2xl font-bold">{c.value}</div>
          </div>
        ))}
      </div>

      {health && (
        <div className="mt-6 rounded-lg border bg-white p-4 text-sm">
          <h3 className="mb-2 font-semibold">System Status</h3>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
            <div>PostgreSQL: <span className={health.postgres === 'ok' ? 'text-green-600' : 'text-red-600'}>{health.postgres}</span></div>
            <div>Qdrant: <span className={health.qdrant === 'ok' ? 'text-green-600' : 'text-red-600'}>{health.qdrant}</span></div>
            <div>Embedding model: {health.embedding_model}</div>
            <div>Embedding loaded: {health.embedding_loaded ? 'yes' : 'no'}</div>
          </div>
        </div>
      )}
    </div>
  )
}
