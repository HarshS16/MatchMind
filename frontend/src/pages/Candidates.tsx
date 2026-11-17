import { useEffect, useState, type ChangeEvent } from 'react'
import { api, type Candidate, type CandidateDetail } from '../services/api'

export default function Candidates() {
  const [items, setItems] = useState<Candidate[]>([])
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [skill, setSkill] = useState('')
  const [detail, setDetail] = useState<CandidateDetail | null>(null)
  const [uploading, setUploading] = useState(false)

  const load = () =>
    api.listCandidates(1, search, skill).then((r) => { setItems(r.items); setTotal(r.total) }).catch(() => {})

  useEffect(() => { load() }, [])

  const onUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await api.uploadResume(file)
      load()
    } finally {
      setUploading(false)
      e.target.value = ''
    }
  }

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-semibold">Candidates ({total})</h2>
        <div className="flex gap-2">
          <input className="rounded border px-3 py-1.5 text-sm" placeholder="Search resume text"
            value={search} onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && load()} />
          <input className="w-36 rounded border px-3 py-1.5 text-sm" placeholder="Skill"
            value={skill} onChange={(e) => setSkill(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && load()} />
          <button onClick={load} className="rounded bg-slate-900 px-3 py-1.5 text-sm text-white">Search</button>
          <label className="cursor-pointer rounded bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-500">
            {uploading ? 'Uploading…' : 'Upload Resume'}
            <input type="file" accept=".pdf,.docx" className="hidden" onChange={onUpload} />
          </label>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg border bg-white">
        <table className="w-full text-sm">
          <thead className="bg-slate-100 text-left">
            <tr>
              <th className="px-4 py-2">Name</th><th>Skills</th><th>Status</th><th>Created</th>
            </tr>
          </thead>
          <tbody>
            {items.map((c) => (
              <tr key={c.id} className="cursor-pointer border-t hover:bg-slate-50"
                onClick={() => api.getCandidate(c.id).then(setDetail)}>
                <td className="px-4 py-2 font-medium">{c.candidate_name ?? c.id.slice(0, 8)}</td>
                <td className="px-4 py-2 text-xs">{c.skills.slice(0, 6).join(', ')}</td>
                <td className="px-4 py-2">
                  <span className={`rounded px-2 py-0.5 text-xs ${c.processing_status === 'processed' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                    {c.processing_status}
                  </span>
                </td>
                <td className="px-4 py-2 text-xs text-slate-500">{new Date(c.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {items.length === 0 && <div className="p-6 text-center text-sm text-slate-500">No candidates indexed yet.</div>}
      </div>

      {detail && (
        <div className="fixed inset-0 z-10 flex items-center justify-center bg-black/40 p-4" onClick={() => setDetail(null)}>
          <div className="max-h-[80vh] w-full max-w-3xl overflow-y-auto rounded-lg bg-white p-6" onClick={(e) => e.stopPropagation()}>
            <div className="mb-3 flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold">{detail.candidate_name ?? 'Candidate'}</h3>
                <div className="text-sm text-slate-500">{detail.original_filename}</div>
              </div>
              <button onClick={() => setDetail(null)} className="text-slate-400 hover:text-slate-600">✕</button>
            </div>
            <div className="mb-3 flex flex-wrap gap-1">
              {detail.skills.map((s) => <span key={s} className="rounded bg-blue-50 px-2 py-0.5 text-xs text-blue-700">{s}</span>)}
            </div>
            <pre className="whitespace-pre-wrap rounded bg-slate-50 p-4 text-xs">{detail.resume_text}</pre>
          </div>
        </div>
      )}
    </div>
  )
}
