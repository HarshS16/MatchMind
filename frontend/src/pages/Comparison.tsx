import { useEffect, useState } from 'react'
import { api, type Job, type MatchResultsResponse } from '../services/api'

const METHODS = ['vector', 'cross_encoder', 'llm'] as const

type Row = { candidateId: string; ranks: Record<string, number | undefined> }

export default function Comparison() {
  const [jobs, setJobs] = useState<Job[]>([])
  const [jobId, setJobId] = useState('')
  const [runs, setRuns] = useState<Record<string, MatchResultsResponse>>({})
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')

  useEffect(() => { api.listJobs().then(setJobs).catch(() => {}) }, [])

  const runMethod = async (method: string) => {
    if (!jobId) return
    setBusy(method); setError('')
    try {
      const run = await api.runMatch(jobId, 20, method)
      const results = await api.matchResults(run.id)
      setRuns((r) => ({ ...r, [method]: results }))
    } catch (e) { setError(String(e)) } finally { setBusy('') }
  }

  const allIds = new Set<string>()
  METHODS.forEach((m) => runs[m]?.results.forEach((r) => allIds.add(r.candidate_id)))

  const rows: Row[] = [...allIds].map((cid) => {
    const ranks: Record<string, number | undefined> = {}
    METHODS.forEach((m) => {
      ranks[m] = runs[m]?.results.find((r) => r.candidate_id === cid)?.final_rank ?? undefined
    })
    return { candidateId: cid, ranks }
  })

  const bestRank = (row: Row) => Math.min(...METHODS.map((m) => row.ranks[m] ?? 999))

  return (
    <div>
      <h2 className="mb-4 text-xl font-semibold">Ranking Comparison</h2>

      <div className="mb-4 flex items-center gap-3 rounded-lg border bg-white p-4">
        <select className="flex-1 rounded border px-3 py-2 text-sm" value={jobId} onChange={(e) => setJobId(e.target.value)}>
          <option value="">Select a job…</option>
          {jobs.map((j) => <option key={j.id} value={j.id}>{j.title}</option>)}
        </select>
        {METHODS.map((m) => (
          <button key={m} onClick={() => runMethod(m)} disabled={busy === m || !jobId}
            className="rounded bg-slate-900 px-3 py-2 text-sm text-white hover:bg-slate-700 disabled:opacity-50">
            {busy === m ? 'Running…' : `Run ${m}`}
          </button>
        ))}
      </div>
      {error && <div className="mb-4 rounded bg-red-50 p-3 text-sm text-red-700">{error}</div>}

      {rows.length > 0 && (
        <div className="overflow-hidden rounded-lg border bg-white">
          <table className="w-full text-sm">
            <thead className="bg-slate-100 text-left">
              <tr>
                <th className="px-4 py-2">Candidate</th>
                {METHODS.map((m) => <th key={m} className="px-4 py-2">{m}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.sort((a, b) => bestRank(a) - bestRank(b)).map((row) => (
                <tr key={row.candidateId} className="border-t">
                  <td className="px-4 py-2 font-mono text-xs">{row.candidateId.slice(0, 8)}</td>
                  {METHODS.map((m) => (
                    <td key={m} className="px-4 py-2">{row.ranks[m] ?? '—'}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
