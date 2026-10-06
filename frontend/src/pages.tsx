import { useEffect, useState, type ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Area, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, fmt, post, short, type Row } from './api'

function useApi<T = Row[]>(path: string | null, deps: unknown[] = []): [T | undefined, string, () => void] {
  const [data, setData] = useState<T>()
  const [err, setErr] = useState('')
  const [n, setN] = useState(0)
  useEffect(() => {
    if (!path) return
    api<T>(path).then(d => { setData(d); setErr('') }).catch(e => setErr(String(e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, n, ...deps])
  return [data, err, () => setN(n + 1)]
}

const Card = ({ title, children, right }: { title: string; children: ReactNode; right?: ReactNode }) => (
  <div className="card"><div className="row" style={{ justifyContent: 'space-between' }}><h2>{title}</h2>{right}</div>{children}</div>
)
const Err = ({ e }: { e: string }) => (e ? <div className="err">{e}</div> : null)
const health = (h?: number) => (h === undefined || h === null ? '' : h < 0.5 ? 'bad' : h < 0.8 ? 'warn' : 'ok')

function Table({ rows, cols }: { rows?: Row[]; cols: [string, string, ((r: Row) => ReactNode)?][] }) {
  if (!rows) return <div className="muted">loading...</div>
  if (!rows.length) return <div className="muted">no rows</div>
  return (
    <table>
      <thead><tr>{cols.map(([k, l]) => <th key={k}>{l}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map(([k, , f]) => <td key={k}>{f ? f(r) : fmt(r[k])}</td>)}</tr>)}</tbody>
    </table>
  )
}

// ------------------------------------------------------------------ fleet
export function Fleet() {
  const [fleet, e1] = useApi('/fleet/health')
  const [aircraft, e2] = useApi('/aircraft')
  const [alerts] = useApi('/alerts?status=OPEN')
  const byId = Object.fromEntries((fleet || []).map(f => [f.aircraft_id, f]))
  return (
    <>
      <div className="grid">
        <div className="kpi">Aircraft<b>{aircraft?.length ?? '-'}</b></div>
        <div className="kpi">Flights ingested<b>{aircraft?.reduce((s, a) => s + Number(a.flights), 0) ?? '-'}</b></div>
        <div className="kpi">Open alerts<b className={alerts?.length ? 'warn' : ''}>{alerts?.length ?? '-'}</b></div>
        <div className="kpi">Critical<b className="bad">{alerts?.filter(a => a.severity === 'CRITICAL').length ?? '-'}</b></div>
      </div>
      <Card title="Fleet health (latest flight per LRU)">
        <Err e={e1 || e2} />
        <Table rows={aircraft?.map(a => ({ ...a, ...byId[a.aircraft_id] }))} cols={[
          ['aircraft_id', 'Aircraft', r => <Link to={`/aircraft/${r.aircraft_id}`}>{r.aircraft_id}</Link>],
          ['aircraft_type', 'Type'], ['base', 'Base'], ['flights', 'Flights'],
          ['min_health', 'Worst LRU health', r => <span className={health(r.min_health)}>{fmt(r.min_health)}</span>],
          ['max_failure_probability', 'Max P(fail)'], ['min_rul_hours', 'Min RUL (h)'],
          ['open_alerts', 'Open alerts', r => <span className={r.open_alerts > 0 ? 'warn' : ''}>{fmt(r.open_alerts, 0)}</span>],
          ['mean_quality', 'Data quality'], ['last_flight', 'Last flight', r => String(r.last_flight ?? '').slice(0, 16)],
        ]} />
      </Card>
    </>
  )
}

export function Aircraft() {
  const { id } = useParams()
  const [a, err] = useApi<Row>(`/aircraft/${id}`)
  const [preds] = useApi(`/predictions?aircraft_id=${id}`)
  const [alerts] = useApi(`/alerts?status=&aircraft_id=${id}`)
  const latest: Record<string, Row> = {}
  preds?.forEach(p => { latest[p.lru_id] = p })
  const lrus = Object.keys(latest).sort((x, y) => latest[y].failure_probability - latest[x].failure_probability).slice(0, 5)
  const trend = a?.flights.map((f: Row) => {
    const o: Row = { flight: short(f.flight_id) }
    preds?.filter(p => p.flight_id === f.flight_id && lrus.includes(p.lru_id)).forEach(p => { o[short(p.lru_id)] = p.failure_probability })
    return o
  })
  return (
    <>
      <h2>{id} <span className="muted">{a?.aircraft_type} - {a?.base}</span></h2>
      <Err e={err} />
      <Card title="Failure probability trend (top 5 LRUs)">
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={trend}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="flight" /><YAxis domain={[0, 1]} /><Tooltip /><Legend />
            {lrus.map((l, i) => <Line key={l} dataKey={short(l)} stroke={['#b42318', '#c4560b', '#1f5fbf', '#18794e', '#7a3db8'][i]} dot={false} />)}
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <Card title="LRUs (latest prediction)">
        <Table rows={a?.lrus.map((l: Row) => ({ ...l, ...latest[l.lru_id] }))} cols={[
          ['lru_id', 'LRU', r => <Link to={`/lrus/${r.lru_id}`}>{short(r.lru_id)}</Link>], ['lru_name', 'Name'], ['subsystem', 'Subsystem'],
          ['serial_number', 'Serial'], ['health_index', 'Health', r => <span className={health(r.health_index)}>{fmt(r.health_index)}</span>],
          ['failure_probability', 'P(fail)'], ['failure_mode', 'Mode'], ['rul_hours', 'RUL h'],
          ['rul_lower', 'RUL 80% band', r => r.rul_lower == null ? '-' : `${fmt(r.rul_lower)} - ${fmt(r.rul_upper)}`], ['anomaly_score', 'Anomaly'],
        ]} />
      </Card>
      <Card title="Alerts"><AlertTable rows={alerts} /></Card>
      <Card title="Flights">
        <Table rows={a?.flights} cols={[
          ['flight_id', 'Flight', r => <Link to={`/flights/${r.flight_id}`}>{r.flight_id}</Link>], ['departure_time', 'Departure', r => String(r.departure_time).slice(0, 16)],
          ['mission_type', 'Mission'], ['flight_hours', 'Hours'], ['cumulative_hours', 'Cum. hours'], ['quality_score', 'Quality'], ['ingest_status', 'Ingest'], ['analysis_status', 'Analysis'],
        ]} />
      </Card>
    </>
  )
}

const PHASE_COLORS: Record<string, string> = {
  PARKED: '#8394a8', TAXI: '#5b8def', TAKEOFF: '#e8a33d', CLIMB: '#e07a3d', CRUISE: '#18794e', MANEUVER: '#b42318',
  DESCENT: '#7a3db8', APPROACH: '#a35bd6', LANDING: '#c4560b', POST_FLIGHT: '#5f6f82',
}

export function Flight() {
  const { id } = useParams()
  const [f, err] = useApi<Row>(`/flights/${id}`)
  const [params] = useApi('/parameters')
  const [sel, setSel] = useState('ENGINE_RPM')
  const [tel] = useApi(`/flights/${id}/telemetry?parameters=${sel}`, [sel])
  const [quality] = useApi(`/flights/${id}/quality`)
  const [preds] = useApi(`/predictions?flight_id=${id}`)
  const total = f?.phases?.length ? f.phases[f.phases.length - 1].end_s : 1
  const s = tel?.[0]
  const data = s?.raw_values.map((v: number | null, i: number) => ({ t: i, raw: v, clean: s.clean_values[i] }))
  return (
    <>
      <h2>{id}</h2>
      <Err e={err} />
      <Card title="Flight phases">
        <div style={{ display: 'flex', width: '100%' }}>
          {f?.phases.map((p: Row, i: number) => (
            <div key={i} className="phase" title={`${p.phase} ${p.start_s}-${p.end_s}s`}
              style={{ width: `${(100 * (p.end_s - p.start_s)) / total}%`, background: PHASE_COLORS[p.phase] }}>{p.phase}</div>
          ))}
        </div>
      </Card>
      <Card title="Telemetry (raw decoded vs cleaned, 1 Hz)" right={
        <select value={sel} onChange={e => setSel(e.target.value)}>
          {params?.map(p => <option key={p.parameter_id}>{p.parameter_id}</option>)}
        </select>}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="t" unit="s" /><YAxis /><Tooltip /><Legend />
            <Line dataKey="raw" stroke="#c5ced8" dot={false} isAnimationActive={false} />
            <Line dataKey="clean" stroke="#1f5fbf" dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <Card title="LRU predictions for this flight">
        <Table rows={preds} cols={[
          ['lru_id', 'LRU', r => <Link to={`/lrus/${r.lru_id}`}>{short(r.lru_id)}</Link>], ['health_index', 'Health', r => <span className={health(r.health_index)}>{fmt(r.health_index)}</span>],
          ['failure_probability', 'P(fail)'], ['failure_mode', 'Mode'], ['anomaly_score', 'Anomaly'], ['changepoint_probability', 'Changepoint'], ['rul_hours', 'RUL h'],
        ]} />
      </Card>
      <Card title="Source files & clock synchronisation">
        <Table rows={f?.files} cols={[['path', 'File'], ['file_type', 'Type'], ['sha256', 'SHA-256', r => <span className="muted">{r.sha256.slice(0, 16)}...</span>], ['status', 'Status']]} />
        <br />
        <Table rows={f?.clock_sync} cols={[['lru_id', 'LRU', r => short(r.lru_id)], ['offset_s', 'Offset (s)'], ['drift_ppm', 'Drift ppm'], ['dtw_cost', 'DTW cost'], ['method', 'Method']]} />
        <div className="muted">Bus message errors: {f?.bus_error_counts.map((b: Row) => `${b.error_type} ${b.n}`).join(', ') || 'none'}</div>
      </Card>
      <Card title="Data quality (worst first)">
        <Table rows={quality?.slice(0, 25)} cols={[
          ['parameter_id', 'Parameter'], ['lru_type', 'LRU'], ['completeness', 'Completeness'], ['validity', 'Validity'], ['outlier_fraction', 'Outliers', r => fmt(r.outlier_fraction, 4)],
          ['stuck_fraction', 'Stuck'], ['gap_count', 'Gaps', r => fmt(r.gap_count, 0)], ['score', 'Score'], ['issues', 'Issues', r => r.issues.map((i: string) => <span key={i} className="tag">{i}</span>)],
        ]} />
      </Card>
    </>
  )
}

export function Lru() {
  const { id } = useParams()
  const [l, err] = useApi<Row>(`/lrus/${id}`)
  const [preds] = useApi(`/predictions?lru_id=${id}`)
  const [flight, setFlight] = useState<string>()
  const last = flight ?? preds?.[preds.length - 1]?.flight_id
  const [ex] = useApi<Row>(last ? `/explanations/${last}/${id}` : null, [last])
  const data = preds?.map(p => ({ flight: short(p.flight_id), id: p.flight_id, pf: p.failure_probability, anomaly: p.anomaly_score, cp: p.changepoint_probability,
    rul: p.rul_hours, band: [p.rul_lower, p.rul_upper] }))
  return (
    <>
      <h2>{id} <span className="muted">{l?.lru_name} - S/N {l?.serial_number}</span></h2>
      <Err e={err} />
      <Card title="Health trend (click a point to explain that flight)">
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={data} onClick={(e: Row) => e?.activePayload && setFlight(e.activePayload[0].payload.id)}>
            <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="flight" /><YAxis yAxisId="p" domain={[0, 1]} /><YAxis yAxisId="a" orientation="right" /><Tooltip /><Legend />
            <Line yAxisId="p" dataKey="pf" name="P(fail)" stroke="#b42318" />
            <Line yAxisId="p" dataKey="cp" name="Changepoint" stroke="#7a3db8" strokeDasharray="4 2" />
            <Line yAxisId="a" dataKey="anomaly" name="Anomaly score" stroke="#1f5fbf" />
          </ComposedChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={data}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="flight" /><YAxis unit="h" /><Tooltip /><Legend />
            <Area dataKey="band" name="RUL 80% interval" fill="#cfe0f7" stroke="none" />
            <Line dataKey="rul" name="RUL (median)" stroke="#1f5fbf" />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>
      {ex && (
        <Card title={`Explanation - ${last}`}>
          <div className="row">
            <span>P(fail) <b>{fmt(ex.failure_probability)}</b></span><span>Mode <b>{ex.failure_mode}</b></span>
            <span>RUL <b>{fmt(ex.rul_hours)} h</b> [{fmt(ex.rul_lower)}-{fmt(ex.rul_upper)}]</span>
            <button onClick={() => post(`/diagnoses/${last}/${id}`).then(d => (window.location.href = `/diagnoses/${d.diagnosis_id}`))}>Diagnose</button>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div><b>SHAP contributions (classifier)</b>
              <Table rows={ex.shap?.shap} cols={[['feature', 'Feature'], ['contribution', 'Contribution', r => fmt(r.contribution, 3)], ['value', 'Value']]} /></div>
            <div><b>Context-normalised residuals</b>
              <Table rows={ex.shap?.top_residuals} cols={[['feature', 'Feature'], ['z', 'z-score']]} /></div>
          </div>
          <b>Anomalous windows & raw-data traceability</b>
          <Table rows={ex.windows?.filter((w: Row) => w.is_anomaly).map((w: Row) => ({ ...w, ...ex.source.find((s: Row) => s.window_id === w.window_id) }))} cols={[
            ['window_id', 'Window'], ['phase', 'Phase'], ['start_s', 'Start s'], ['score', 'Score'],
            ['top_features', 'Top features', r => r.top_features.slice(0, 3).map((t: Row) => `${t.feature} (${t.z})`).join(', ')],
            ['source_ref', 'Source', r => <span className="muted">{r.source_ref?.bus_file} msgs {r.source_ref?.first_message_id}-{r.source_ref?.last_message_id}</span>],
          ]} />
        </Card>
      )}
      <Card title="Maintenance history">
        <Table rows={l?.maintenance} cols={[['maintenance_date', 'Date', r => String(r.maintenance_date).slice(0, 10)], ['maintenance_type', 'Type'], ['finding', 'Finding'],
          ['action_taken', 'Action'], ['removed_serial_number', 'Removed S/N'], ['installed_serial_number', 'Installed S/N'], ['maintenance_reference', 'Ref']]} />
      </Card>
    </>
  )
}

// ------------------------------------------------------------------ alerts & diagnosis
function AlertTable({ rows, reload }: { rows?: Row[]; reload?: () => void }) {
  const nav = useNavigate()
  const act = (id: number, status: string) => api(`/alerts/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, actor: 'engineer' }) }).then(reload)
  return <Table rows={rows} cols={[
    ['severity', 'Severity', r => <span className={`sev-${r.severity}`}>{r.severity}</span>], ['alert_type', 'Type'], ['title', 'Alert'],
    ['flight_id', 'Flight', r => <Link to={`/flights/${r.flight_id}`}>{r.flight_id}</Link>], ['status', 'Status'],
    ['actions', '', r => <div className="row" style={{ margin: 0 }}>
      <button onClick={() => post(`/diagnoses/${r.flight_id}/${r.lru_id}`).then(d => nav(`/diagnoses/${d.diagnosis_id}`))}>Diagnose</button>
      {reload && r.status === 'OPEN' && <button className="secondary" onClick={() => act(r.alert_id, 'ACKNOWLEDGED')}>Ack</button>}
      {reload && r.status !== 'CLOSED' && <button className="secondary" onClick={() => act(r.alert_id, 'CLOSED')}>Close</button>}
    </div>],
  ]} />
}

export function Alerts() {
  const [status, setStatus] = useState('OPEN')
  const [rows, err, reload] = useApi(`/alerts?status=${status}`, [status])
  return (
    <Card title="Alerts" right={<select value={status} onChange={e => setStatus(e.target.value)}>
      {['OPEN', 'ACKNOWLEDGED', 'CLOSED', ''].map(s => <option key={s} value={s}>{s || 'ALL'}</option>)}</select>}>
      <Err e={err} /><AlertTable rows={rows} reload={reload} />
    </Card>
  )
}

export function Diagnoses() {
  const [rows, err] = useApi('/diagnoses')
  return (
    <Card title="Diagnoses"><Err e={err} />
      <Table rows={rows} cols={[['diagnosis_id', '#', r => <Link to={`/diagnoses/${r.diagnosis_id}`}>{r.diagnosis_id}</Link>], ['aircraft_id', 'Aircraft'],
        ['lru_id', 'LRU', r => short(r.lru_id)], ['flight_id', 'Flight'], ['top_candidate', 'Top candidate', r => `${r.top_candidate?.event_code} (${fmt(r.top_candidate?.probability)})`],
        ['pending', 'Pending decisions'], ['created_at', 'Created', r => String(r.created_at).slice(0, 16)]]} />
    </Card>
  )
}

function DecisionForm({ rec, done }: { rec: Row; done: () => void }) {
  const [engineer, setEngineer] = useState('')
  const [note, setNote] = useState('')
  const [override, setOverride] = useState('')
  const [err, setErr] = useState('')
  const decide = (decision: string) => post(`/recommendations/${rec.recommendation_id}/decision`,
    { decision, engineer, note: note || null, override_action: override || null }).then(done).catch(e => setErr(String(e)))
  if (rec.status !== 'PENDING') return <span className={rec.status === 'ACCEPTED' ? 'ok' : 'warn'}>{rec.status} by {rec.engineer}{rec.decision_note ? `: ${rec.decision_note}` : ''}</span>
  return (
    <div>
      <div className="row"><input placeholder="engineer id" value={engineer} onChange={e => setEngineer(e.target.value)} />
        <input placeholder="note (required to reject/override)" value={note} onChange={e => setNote(e.target.value)} style={{ flex: 1 }} /></div>
      <div className="row"><input placeholder="override action" value={override} onChange={e => setOverride(e.target.value)} style={{ flex: 1 }} />
        <button disabled={!engineer} onClick={() => decide('ACCEPT')}>Accept</button>
        <button disabled={!engineer} className="secondary" onClick={() => decide('REJECT')}>Reject</button>
        <button disabled={!engineer} className="secondary" onClick={() => decide('OVERRIDE')}>Override</button></div>
      <Err e={err} />
    </div>
  )
}

export function Diagnosis() {
  const { id } = useParams()
  const [d, err, reload] = useApi<Row>(`/diagnoses/${id}`)
  const ev = d?.evidence
  return (
    <>
      <h2>Diagnosis #{id} <span className="muted">{d?.lru_id} - {d?.flight_id}</span> <a href={`/api/v1/reports/${id}.pdf`}><button>PDF report</button></a></h2>
      <Err e={err} />
      {ev && <div className="grid">
        <div className="kpi">P(fault present)<b>{fmt(ev.p_fault_present)}</b></div>
        <div className="kpi">Model mode<b style={{ fontSize: 15 }}>{ev.failure_mode}</b></div>
        <div className="kpi">RUL (h)<b>{fmt(ev.rul_hours)}</b><span className="muted">80%: {fmt(ev.rul_interval[0])} - {fmt(ev.rul_interval[1])}</span></div>
        <div className="kpi">Anomaly / changepoint<b>{fmt(ev.anomaly_score)} / {fmt(ev.changepoint_probability)}</b></div>
      </div>}
      <Card title={`Fault candidates under ${d?.top_event} (FTA-constrained Bayesian posterior)`}>
        <Table rows={d?.candidates} cols={[['event_code', 'Event', r => <Link to={`/fta?event=${r.event_code}`}>{r.event_code}</Link>], ['description', 'Description'],
          ['probability', 'P'], ['evidence', 'Evidence', r => <span className="muted">mode p={r.evidence.model_mode_probability}, param overlap={r.evidence.indicating_parameter_overlap},
            snag/doc sim={fmt(r.evidence.snag_document_similarity)}, gate={r.evidence.gate_factor} {r.evidence.gate_notes.join('; ')}</span>]]} />
      </Card>
      <Card title="Recommendations (engineer decision required)">
        <Table rows={d?.recommendations} cols={[['rank', '#'], ['action', 'Action', r => <><b>{r.override_action || r.action}</b><div className="muted">{r.rationale}</div></>],
          ['refs', 'References', r => (r.refs || []).map((x: string) => <span key={x} className="tag">{x}</span>)], ['confidence', 'Conf.'],
          ['decision', 'Decision', r => <DecisionForm rec={r} done={reload} />]]} />
      </Card>
      {ev && <Card title="Evidence">
        <div className="muted">Top residuals: {ev.top_residuals.map((t: Row) => `${t.feature} z=${t.z}`).join(', ')}</div>
        <div className="muted">SHAP: {ev.shap.map((s: Row) => `${s.feature} ${fmt(s.contribution, 3)}`).join(', ')}</div>
        <div className="muted">Recent snags: {ev.recent_snags.map((s: Row) => `${s.snag_id} "${s.snag_title}" (${s.disposition?.slice(0, 40)})`).join('; ') || 'none'}</div>
        <div className="muted">Model versions: {JSON.stringify(d?.model_versions)}</div>
      </Card>}
    </>
  )
}

// ------------------------------------------------------------------ knowledge
export function Search() {
  const [query, setQuery] = useState('hydraulic pressure fluctuating')
  const [snags, setSnags] = useState<Row[]>()
  const [docs, setDocs] = useState<Row[]>()
  const [err, setErr] = useState('')
  const [repeated] = useApi('/snags/repeated')
  const go = () => {
    const qs = encodeURIComponent(query)
    Promise.all([api(`/snags/search?query=${qs}&k=10`), api(`/documents/search?query=${qs}&k=6`)])
      .then(([s, d]) => { setSnags(s); setDocs(d); setErr('') }).catch(e => setErr(String(e)))
  }
  return (
    <>
      <Card title="Semantic search (local SBERT embeddings, offline)">
        <div className="row"><input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && go()} style={{ flex: 1 }} /><button onClick={go}>Search</button></div>
        <Err e={err} />
      </Card>
      {snags && <Card title="Similar snags">
        <Table rows={snags} cols={[['similarity', 'Sim.'], ['snag_id', 'Snag'], ['lru_id', 'LRU', r => short(r.lru_id)], ['snag_description', 'Description'], ['disposition', 'Disposition']]} />
      </Card>}
      {docs && <Card title="Document passages (with citations)">
        {docs.map(d => <div key={d.chunk_id} style={{ marginBottom: 10 }}><b>{d.citation}</b> <span className="muted">score {fmt(d.score)}</span><div>{d.text.slice(0, 400)}</div></div>)}
      </Card>}
      <Card title="Repeat-defect candidates (similar snags on the same LRU within 90 days)">
        <Table rows={repeated} cols={[['lru_id', 'LRU'], ['first_snag', 'First'], ['repeat_snag', 'Repeat'], ['first_title', 'Title'], ['first_disposition', 'First disposition'], ['similarity', 'Sim.']]} />
      </Card>
    </>
  )
}

function TreeNode({ code, nodes, edges, depth = 0 }: { code: string; nodes: Record<string, Row>; edges: Row[]; depth?: number }) {
  const n = nodes[code]
  const kids = edges.filter(e => e.parent_event === code)
  const hl = new URLSearchParams(window.location.search).get('event') === code
  return (
    <li>
      <span style={{ background: hl ? '#fff4d6' : undefined }}><b>{code}</b> {n?.gate && <span className="tag">{n.gate}</span>}{n?.description}
        {n?.reference_document && <span className="muted"> - {n.reference_document}</span>}</span>
      {kids.length > 0 && depth < 10 && <ul>{kids.map(k => <TreeNode key={k.child_event} code={k.child_event} nodes={nodes} edges={edges} depth={depth + 1} />)}</ul>}
    </li>
  )
}

export function Fta() {
  const [trees, err] = useApi('/fta/trees')
  const ev = new URLSearchParams(window.location.search).get('event')
  const [event] = useApi<Row>(ev ? `/fta/events/${ev}` : null)
  const [sub, setSub] = useState<string>()
  const current = sub ?? trees?.find(t => event?.subsystem ? t.subsystem === event.subsystem : true)?.subsystem
  const [tree] = useApi<Row>(current ? `/fta/tree/${current}` : null, [current])
  const nodes = Object.fromEntries((tree?.nodes || []).map((n: Row) => [n.event_code, n]))
  return (
    <Card title="Fault tree analysis" right={<select value={current} onChange={e => setSub(e.target.value)}>{trees?.map(t => <option key={t.subsystem}>{t.subsystem}</option>)}</select>}>
      <Err e={err} />
      {tree && <div className="tree"><ul><TreeNode code={`TE-${current}`} nodes={nodes} edges={tree.edges} /></ul></div>}
    </Card>
  )
}

// ------------------------------------------------------------------ platform
export function Models() {
  const [rows, err] = useApi('/models')
  return (
    <Card title="Model registry"><Err e={err} />
      <Table rows={rows} cols={[['model_id', 'Model'], ['is_active', 'Active', r => (r.is_active ? 'yes' : '')], ['created_at', 'Trained', r => String(r.created_at).slice(0, 16)],
        ['metrics', 'Validation metrics', r => <span className="muted">{Object.entries(r.metrics || {}).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : JSON.stringify(v)}`).join(', ')}</span>],
        ['training_data', 'Training data', r => <span className="muted">{Object.entries(r.training_data || {}).map(([k, v]) => `${k}=${v}`).join(', ')}</span>]]} />
    </Card>
  )
}

export function Data() {
  const [summary, err, reload] = useApi<Row>('/quality/summary')
  const [jobs, , reloadJobs] = useApi('/jobs?limit=20')
  const [msg, setMsg] = useState('')
  const [file, setFile] = useState<File>()
  const [path, setPath] = useState('')
  useEffect(() => { const t = setInterval(reloadJobs, 5000); return () => clearInterval(t) })
  const upload = () => {
    if (!file) return
    const fd = new FormData()
    fd.append('relative_path', path || file.name)
    fd.append('file', file)
    api<Row>('/ingest/upload', { method: 'POST', body: fd }).then(r => { setMsg(`uploaded ${r.path}`); reload() }).catch(e => setMsg(String(e)))
  }
  return (
    <>
      <Card title="Pipeline jobs (Postgres SKIP LOCKED queue)" right={<button onClick={() => post('/pipeline/run').then(reloadJobs)}>Run pipeline</button>}>
        <Table rows={jobs} cols={[['job_id', '#'], ['job_type', 'Type'], ['status', 'Status', r => <span className={r.status === 'FAILED' ? 'bad' : r.status === 'DONE' ? 'ok' : 'warn'}>{r.status}</span>],
          ['attempts', 'Attempts'], ['created_at', 'Created', r => String(r.created_at).slice(0, 19)], ['finished_at', 'Finished', r => String(r.finished_at ?? '').slice(0, 19)],
          ['error', 'Error', r => <span className="err">{r.error?.slice(0, 200)}</span>]]} />
      </Card>
      <Card title="Upload raw file (immutable - existing files are never overwritten)">
        <div className="row"><input type="file" onChange={e => setFile(e.target.files?.[0])} />
          <input placeholder="relative path e.g. bus1553/SYN-AC-001/SYN-AC-001-F0100.bin" value={path} onChange={e => setPath(e.target.value)} style={{ flex: 1 }} />
          <button onClick={upload} disabled={!file}>Upload</button></div>
        <div className="muted">{msg}</div>
      </Card>
      <Err e={err} />
      {summary && <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Raw files"><Table rows={summary.files} cols={[['file_type', 'Type'], ['status', 'Status'], ['n', 'Count', r => fmt(r.n, 0)]]} /></Card>
        <div>
          <Card title="Flights"><Table rows={summary.flights} cols={[['ingest_status', 'Status'], ['n', 'Count', r => fmt(r.n, 0)], ['q', 'Mean quality']]} /></Card>
          <Card title="Data-quality issues"><Table rows={summary.issues} cols={[['issue', 'Issue'], ['n', 'Parameter-flights', r => fmt(r.n, 0)]]} /></Card>
          <Card title="LRU clock synchronisation"><Table rows={summary.clock_sync} cols={[['method', 'Method'], ['n', 'Logs', r => fmt(r.n, 0)], ['mean_abs_offset', 'Mean |offset| s']]} /></Card>
        </div>
      </div>}
    </>
  )
}

export function Audit() {
  const [rows, err] = useApi('/audit?limit=300')
  return (
    <Card title="Audit trail"><Err e={err} />
      <Table rows={rows} cols={[['ts', 'Time', r => String(r.ts).slice(0, 19)], ['actor', 'Actor'], ['service', 'Service'], ['action', 'Action'], ['entity', 'Entity'], ['entity_id', 'Id'],
        ['details', 'Details', r => <span className="muted">{JSON.stringify(r.details).slice(0, 160)}</span>]]} />
    </Card>
  )
}
