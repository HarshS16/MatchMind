import { useEffect, useState } from 'react'
import { api, type Job, type MatchRun, type MatchResultsResponse } from '../services/api'

const METHODS = ['vector', 'cross_encoder', 'llm'] as const

export default function Jobs() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [form, setForm] = useState({ title: '', description: '', required_skills: '', preferred_skills: '', experience_min: '', experience_max: '' })
  const [runs, setRuns] = useState<Record<string, MatchRun>>({})
  const [results, setResults] = useState<MatchResultsResponse | null>(null)
  const [method, setMethod] = useState<string>('cross_encoder')
  const [error, setError] = useState('')

  const load = () => api.listJobs().then(setJobs).catch((e) => setError(String(e)))
  useEffect(() => { load() }, [])

  const createJob = async () => {
    setError('')
    try {
      await api.createJob({
        title: form.title,
        description: form.description,
        required_skills: form.required_skills.split(',').map((s) => s.trim()).filter(Boolean),
        preferred_skills: form.preferred_skills.split(',').map((s) => s.trim()).filter(Boolean),
        experience_min: form.experience_min ? Number(form.experience_min) : null,
        experience_max: form.experience_max ? Number(form.experience_max) : null,
      })
      setForm({ title: '', description: '', required_skills: '', preferred_skills: '', experience_min: '', experience_max: '' })
      load()
    } catch (e) { setError(String(e)) }
  }

  const runMatch = async (jobId: string) => {
    setError('')
    try {
      const run = await api.runMatch(jobId, 20, method)
      setRuns((r) => ({ ...r, [jobId]: run }))
      setResults(await api.matchResults(run.id))
    } catch (e) { setError(String(e)) }
  }

  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">Job Management</h2>

      <div className="mb-6 rounded-lg border bg-white p-4">
        <h3 className="mb-3 font-semibold">Create Job</h3>
        <div className="grid gap-3 md:grid-cols-2">
          <input className="rounded border px-3 py-2" placeholder="Job title" value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })} />
          <input className="rounded border px-3 py-2" placeholder="Required skills (comma-separated)" value={form.required_skills}
            onChange={(e) => setForm({ ...form, required_skills: e.target.value })} />
          <input className="rounded border px-3 py-2" placeholder="Preferred skills" value={form.preferred_skills}
            onChange={(e) => setForm({ ...form, preferred_skills: e.target.value })} />
          <div className="flex gap-2">
            <input className="w-1/2 rounded border px-3 py-2" placeholder="Min yrs" type="number" value={form.experience_min}
              onChange={(e) => setForm({ ...form, experience_min: e.target.value })} />
            <input className="w-1/2 rounded border px-3 py-2" placeholder="Max yrs" type="number" value={form.experience_max}
              onChange={(e) => setForm({ ...form, experience_max: e.target.value })} />
          </div>
        </div>
        <textarea className="mt-3 w-full rounded border px-3 py-2" rows={4} placeholder="Job description"
          value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
        <button onClick={createJob} className="mt-3 rounded bg-slate-900 px-4 py-2 text-sm text-white hover:bg-slate-700">
          Create Job
        </button>
      </div>

      {error && <div className="mb-4 rounded bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      <div className="mb-4 flex items-center gap-2">
        <label className="text-sm">Ranking method:</label>
        <select className="rounded border px-2 py-1 text-sm" value={method} onChange={(e) => setMethod(e.target.value)}>
          {METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      <div className="space-y-3">
        {jobs.map((job) => (
          <div key={job.id} className="rounded-lg border bg-white p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="font-semibold">{job.title}</div>
                <div className="mt-1 text-sm text-slate-600">{job.description.slice(0, 160)}…</div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {job.required_skills.map((s) => (
                    <span key={s} className="rounded bg-blue-50 px-2 py-0.5 text-xs text-blue-700">{s}</span>
                  ))}
                </div>
              </div>
              <button onClick={() => runMatch(job.id)}
                className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white hover:bg-slate-700">
                Run Match
              </button>
            </div>
            {runs[job.id] && (
              <div className="mt-2 text-xs text-slate-500">
                Run {runs[job.id].id.slice(0, 8)} — {runs[job.id].status} — {runs[job.id].latency_ms}ms
              </div>
            )}
          </div>
        ))}
        {jobs.length === 0 && <div className="text-sm text-slate-500">No jobs yet.</div>}
      </div>

      {results && (
        <div className="mt-6 rounded-lg border bg-white p-4">
          <h3 className="mb-3 font-semibold">Results — {results.ranking_method} ({results.total_candidates} candidates)</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-slate-500">
                <th className="py-2">Rank</th><th>Retrieval Rank</th><th>Retrieval Score</th><th>Ranking Score</th><th>Matched Skills</th>
              </tr>
            </thead>
            <tbody>
              {results.results.slice(0, 20).map((r) => (
                <tr key={r.candidate_id} className="border-b">
                  <td className="py-2 font-medium">{r.final_rank}</td>
                  <td>{r.retrieval_rank}</td>
                  <td>{r.retrieval_score?.toFixed(3)}</td>
                  <td>{r.ranking_score?.toFixed(2)}</td>
                  <td className="text-xs">{r.matched_skills.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
