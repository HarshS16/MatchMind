import { useEffect, useState } from 'react'
import { api, type Experiment, type Job } from '../services/api'

const METHODS = ['vector', 'cross_encoder', 'llm'] as const
const METRIC_KEYS = ['precision@5', 'recall@5', 'ndcg@10', 'mrr', 'mean_latency_ms']

export default function Evaluation() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedJobs, setSelectedJobs] = useState<string[]>([])
  const [methods, setMethods] = useState<string[]>([...METHODS])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const load = () => {
    api.listExperiments().then(setExperiments).catch(() => {})
    api.listJobs().then(setJobs).catch(() => {})
  }
  useEffect(load, [])

  const toggle = (list: string[], set: (v: string[]) => void, val: string) =>
    set(list.includes(val) ? list.filter((x) => x !== val) : [...list, val])

  const run = async () => {
    setBusy(true); setError('')
    try {
      await api.runEvaluation(selectedJobs, methods)
      load()
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }

  const latest = experiments[0]

  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">Evaluation & Benchmarks</h2>

      <div className="mb-6 rounded-lg border bg-white p-4">
        <h3 className="mb-3 font-semibold">Run Evaluation</h3>
        <div className="mb-3">
          <div className="mb-1 text-sm text-slate-600">Jobs (empty = all):</div>
          <div className="flex max-h-32 flex-wrap gap-2 overflow-y-auto">
            {jobs.map((j) => (
              <label key={j.id} className="flex items-center gap-1 text-sm">
                <input type="checkbox" checked={selectedJobs.includes(j.id)}
                  onChange={() => toggle(selectedJobs, setSelectedJobs, j.id)} />
                {j.title}
              </label>
            ))}
          </div>
        </div>
        <div className="mb-3 flex gap-4 text-sm">
          {METHODS.map((m) => (
            <label key={m} className="flex items-center gap-1">
              <input type="checkbox" checked={methods.includes(m)} onChange={() => toggle(methods, setMethods, m)} />
              {m}
            </label>
          ))}
        </div>
        <button onClick={run} disabled={busy || methods.length === 0}
          className="rounded bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700 disabled:opacity-50">
          {busy ? 'Running (this can take a while)…' : 'Run Evaluation'}
        </button>
        {error && <div className="mt-3 rounded bg-red-50 p-3 text-sm text-red-700">{error}</div>}
      </div>

      {latest && (
        <div className="rounded-lg border bg-white p-4">
          <h3 className="mb-3 font-semibold">Latest: {latest.experiment_name}</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-slate-500">
                <th className="py-2">Model</th>{METRIC_KEYS.map((k) => <th key={k}>{k}</th>)}
              </tr>
            </thead>
            <tbody>
              {(latest.metrics.queries ?? []).map((q) => (
                <tr key={q.ranking_method} className="border-b">
                  <td className="py-2 font-medium">{q.ranking_method}</td>
                  {METRIC_KEYS.map((k) => {
                    const v = q.metrics[k]
                    return <td key={k}>{typeof v === 'number' ? v.toFixed(3) : '—'}</td>
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-6">
        <h3 className="mb-2 font-semibold">Experiment History</h3>
        <div className="overflow-hidden rounded-lg border bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-100 text-left">
              <tr><th className="px-4 py-2">Name</th><th>Dataset</th><th>Created</th></tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.id} className="border-t">
                  <td className="px-4 py-2 font-mono text-xs">{e.experiment_name}</td>
                  <td className="px-4 py-2">{String(e.configuration.dataset_version ?? '—')}</td>
                  <td className="px-4 py-2 text-xs text-slate-500">{new Date(e.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {experiments.length === 0 && <div className="p-4 text-sm text-slate-500">No experiments yet.</div>}
        </div>
      </div>
    </div>
  )
}
