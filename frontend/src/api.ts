// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Row = Record<string, any>

export async function api<T = Row[]>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api/v1${path}`, init)
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`)
  return r.json()
}

export const post = (path: string, body?: unknown) =>
  api<Row>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined })

export const fmt = (v: unknown, d = 2) =>
  v === null || v === undefined ? '-' : typeof v === 'number' ? v.toFixed(d) : String(v)

export const short = (id: string) => id?.replace(/^SYN-AC-\d+-/, '')
