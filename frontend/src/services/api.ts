const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api/v1'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export interface Candidate {
  id: string
  candidate_name: string | null
  skills: string[]
  processing_status: string
  created_at: string
}

export interface CandidateDetail extends Candidate {
  resume_text: string
  original_filename: string | null
  experience: Record<string, unknown>
  education: Record<string, unknown>
}

export interface Job {
  id: string
  title: string
  description: string
  required_skills: string[]
  preferred_skills: string[]
  experience_min: number | null
  experience_max: number | null
  created_at: string
}

export interface MatchRun {
  id: string
  job_id: string
  retrieval_model: string
  ranking_model: string
  top_k: number
  status: string
  latency_ms: number | null
}

export interface MatchResult {
  candidate_id: string
  retrieval_rank: number | null
  final_rank: number | null
  retrieval_score: number | null
  ranking_score: number | null
  matched_skills: string[]
  explanation: Record<string, unknown>
}

export interface MatchResultsResponse {
  match_run_id: string
  job_id: string
  ranking_method: string
  total_candidates: number
  results: MatchResult[]
}

export interface Experiment {
  id: string
  experiment_name: string
  configuration: Record<string, unknown>
  metrics: { queries?: { ranking_method: string; metrics: Record<string, number> }[] }
  created_at: string
}

export interface Health {
  status: string
  postgres: string
  qdrant: string
  embedding_model: string
  embedding_loaded: boolean
}

export const api = {
  health: () => request<Health>('/health'),
  listCandidates: (page = 1, search = '', skill = '') =>
    request<{ total: number; items: Candidate[] }>(
      `/resumes?page=${page}&limit=50${search ? `&search=${encodeURIComponent(search)}` : ''}${skill ? `&skill=${encodeURIComponent(skill)}` : ''}`,
    ),
  getCandidate: (id: string) => request<CandidateDetail>(`/resumes/${id}`),
  uploadResume: async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${BASE}/resumes/upload`, { method: 'POST', body: form })
    if (!res.ok) throw new Error(await res.text())
    return res.json()
  },
  listJobs: () => request<Job[]>('/jobs'),
  createJob: (job: Partial<Job>) =>
    request<Job>('/jobs', { method: 'POST', body: JSON.stringify(job) }),
  runMatch: (jobId: string, topK = 20, rankingMethod = 'cross_encoder') =>
    request<MatchRun>(`/jobs/${jobId}/match`, {
      method: 'POST',
      body: JSON.stringify({ top_k: topK, ranking_method: rankingMethod }),
    }),
  matchResults: (runId: string) => request<MatchResultsResponse>(`/match-runs/${runId}/results`),
  listExperiments: () => request<Experiment[]>('/evaluation'),
  runEvaluation: (jobIds: string[], methods: string[]) =>
    request<Experiment>('/evaluation/run', {
      method: 'POST',
      body: JSON.stringify({ job_ids: jobIds, ranking_methods: methods }),
    }),
}
