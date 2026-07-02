import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub, color }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const QUALITY_COLORS = { Good: '#10b981', Fair: '#f59e0b', Poor: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'recordings', label: 'Recording Inventory' },
  { id: 'features', label: 'Feature Matrix' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'seizures', label: 'Seizure Timeline' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function TimeSeriesAIDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/time-series-ai/overview`),
      axios.get(`${API_URL}/api/time-series-ai/breakdown`),
      axios.get(`${API_URL}/api/time-series-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Time-Series AI data…</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No data'}</div>

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Time-Series AI Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG spectral decomposition, complexity metrics, temporal feature extraction &amp; seizure timeline
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(overview)}
      {tab === 'recordings' && renderRecordings(breakdown)}
      {tab === 'features' && renderFeatures(breakdown)}
      {tab === 'patients' && renderPatients(breakdown)}
      {tab === 'seizures' && renderSeizures(overview, breakdown)}
      {tab === 'pipeline' && renderPipeline(breakdown)}
      {tab === 'definitions' && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(ov) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPIs */}
      {(ov.kpis || []).map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
      ))}

      {/* Band Power Chart */}
      <Card title="Relative Band Power (Mean)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.band_power_chart || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="mean_power" fill="#3b82f6" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Signal Quality Pie */}
      <Card title="Signal Quality Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={ov.quality_distribution || []} dataKey="count" nameKey="quality"
              cx="50%" cy="50%" outerRadius={80} label={({ quality, count }) => `${quality}: ${count}`}>
              {(ov.quality_distribution || []).map((d, i) => (
                <Cell key={i} fill={d.color || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Feature Category Coverage */}
      <Card title="Feature Extraction Coverage" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.category_feature_counts || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={11} />
            <YAxis dataKey="category" type="category" fontSize={11} width={80} />
            <Tooltip />
            <Bar dataKey="features_extracted" fill="#10b981" radius={[0,4,4,0]} name="Extracted" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Complexity Metrics */}
      <Card title="Complexity Metrics Summary" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Metric</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Mean</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Min</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Max</th>
                <th style={{ textAlign: 'right', padding: 8 }}>N</th>
              </tr>
            </thead>
            <tbody>
              {(ov.complexity_summary || []).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{c.metric}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{c.mean.toFixed(4)}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace', color: '#64748b' }}>{c.min.toFixed(4)}</td>
                  <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace', color: '#64748b' }}>{c.max.toFixed(4)}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{c.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Daily Activity */}
      <Card title="Daily Analysis Activity" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ov.daily_activity || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" fontSize={10} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderRecordings(bd) {
  const recordings = bd?.recording_inventory || []
  return (
    <Card title={`Recording Inventory (${recordings.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 8 }}>File</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Prediction</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Confidence</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Quality</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Channels</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Duration</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Features</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Spec. Entropy</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Hurst</th>
            </tr>
          </thead>
          <tbody>
            {recordings.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{r.patient_id}</td>
                <td style={{ padding: 8, fontSize: 12, color: '#64748b' }}>{r.file_name}</td>
                <td style={{ padding: 8 }}>{r.disease}</td>
                <td style={{ padding: 8 }}>
                  <Badge text={r.predicted_label} color={r.predicted_label === 'Epilepsy' ? '#ef4444' : '#10b981'} />
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {r.confidence ? `${(r.confidence * 100).toFixed(0)}%` : '—'}
                </td>
                <td style={{ padding: 8 }}>
                  <Badge text={r.signal_quality} color={QUALITY_COLORS[r.signal_quality] || '#6b7280'} />
                </td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.n_channels}</td>
                <td style={{ padding: 8, textAlign: 'right', fontSize: 12 }}>{r.duration_display}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.n_features}</td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {r.spectral_entropy != null ? r.spectral_entropy.toFixed(3) : '—'}
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {r.hurst_exponent != null ? r.hurst_exponent.toFixed(3) : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderFeatures(bd) {
  const matrix = bd?.feature_matrix || []
  const bands = bd?.band_power_details || []
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title="Band Power per Recording">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={bands}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="patient_id" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="delta" stackId="a" fill="#3b82f6" name="Delta" />
            <Bar dataKey="theta" stackId="a" fill="#8b5cf6" name="Theta" />
            <Bar dataKey="alpha" stackId="a" fill="#10b981" name="Alpha" />
            <Bar dataKey="beta" stackId="a" fill="#f59e0b" name="Beta" />
            <Bar dataKey="gamma" stackId="a" fill="#ef4444" name="Gamma" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Feature Matrix (${matrix.length} recordings × 4 categories)`}>
        <div style={{ overflowX: 'auto', maxHeight: 500 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 6, position: 'sticky', top: 0, background: '#fff' }}>Patient</th>
                <th style={{ textAlign: 'right', padding: 6, position: 'sticky', top: 0, background: '#fff' }}>Statistical</th>
                <th style={{ textAlign: 'right', padding: 6, position: 'sticky', top: 0, background: '#fff' }}>Spectral</th>
                <th style={{ textAlign: 'right', padding: 6, position: 'sticky', top: 0, background: '#fff' }}>Complexity</th>
                <th style={{ textAlign: 'right', padding: 6, position: 'sticky', top: 0, background: '#fff' }}>Temporal</th>
              </tr>
            </thead>
            <tbody>
              {matrix.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 6, fontWeight: 500 }}>{m.patient_id}</td>
                  <td style={{ padding: 6, textAlign: 'right', fontSize: 11, color: '#64748b' }}>
                    {Object.keys(m.statistical || {}).length} feats
                  </td>
                  <td style={{ padding: 6, textAlign: 'right', fontSize: 11, color: '#64748b' }}>
                    {Object.keys(m.spectral || {}).length} feats
                  </td>
                  <td style={{ padding: 6, textAlign: 'right', fontSize: 11, color: '#64748b' }}>
                    {Object.keys(m.complexity || {}).length} feats
                  </td>
                  <td style={{ padding: 6, textAlign: 'right', fontSize: 11, color: '#64748b' }}>
                    {Object.keys(m.temporal || {}).length} feats
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderPatients(bd) {
  const profiles = bd?.patient_profiles || []
  return (
    <Card title={`Patient Profiles (${profiles.length})`}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 12 }}>
        {profiles.map((p, i) => (
          <div key={i} style={{
            border: '1px solid #e2e8f0', borderRadius: 10, padding: 14
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontWeight: 600, fontSize: 14 }}>{p.patient_id}</span>
              <Badge text={`${p.n_recordings} recording${p.n_recordings !== 1 ? 's' : ''}`} color="#3b82f6" />
            </div>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
              {p.diseases.join(', ')} · {p.total_duration}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 12 }}>
              <div>Confidence: <strong>{(p.mean_confidence * 100).toFixed(0)}%</strong></div>
              <div>Spec. Entropy: <strong>{p.mean_spectral_entropy?.toFixed(3) || '—'}</strong></div>
              <div>Hurst: <strong>{p.mean_hurst?.toFixed(3) || '—'}</strong></div>
              <div>Sample Ent.: <strong>{p.mean_sample_entropy?.toFixed(3) || '—'}</strong></div>
            </div>
            <div style={{ marginTop: 6, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {Object.entries(p.quality_summary || {}).map(([q, cnt]) => (
                <Badge key={q} text={`${q}: ${cnt}`} color={QUALITY_COLORS[q] || '#6b7280'} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}

function renderSeizures(ov, bd) {
  const seizures = bd?.seizure_inventory || []
  const timeline = ov?.seizure_timeline || []
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title="Seizure Event Timeline">
        {timeline.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" fontSize={10} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Bar dataKey="events" fill="#ef4444" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ color: '#94a3b8', fontSize: 13 }}>No seizure events recorded.</p>
        )}
      </Card>

      <Card title={`Seizure Inventory (${seizures.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Date</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Time</th>
                <th style={{ textAlign: 'right', padding: 8 }}>Duration (s)</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Severity</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Trigger</th>
              </tr>
            </thead>
            <tbody>
              {seizures.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 500 }}>{s.patient_id}</td>
                  <td style={{ padding: 8 }}>{s.event_date || '—'}</td>
                  <td style={{ padding: 8 }}>{s.event_time || '—'}</td>
                  <td style={{ padding: 8, textAlign: 'right' }}>{s.duration_sec ?? '—'}</td>
                  <td style={{ padding: 8 }}>
                    {s.severity && <Badge text={s.severity}
                      color={s.severity === 'Severe' ? '#ef4444' : s.severity === 'Moderate' ? '#f97316' : '#f59e0b'} />}
                  </td>
                  <td style={{ padding: 8, fontSize: 12, color: '#64748b' }}>{s.trigger || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function renderPipeline(bd) {
  const events = bd?.pipeline_log || []
  return (
    <Card title={`Pipeline Event Log (last ${events.length})`}>
      <div style={{ overflowX: 'auto', maxHeight: 500 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>ID</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Component</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Action</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Actor</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Detail</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{e.id}</td>
                <td style={{ padding: 6 }}><Badge text={e.component || '?'} color="#3b82f6" /></td>
                <td style={{ padding: 6 }}>{e.action}</td>
                <td style={{ padding: 6, color: '#64748b' }}>{e.actor}</td>
                <td style={{ padding: 6, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                <td style={{ padding: 6, fontSize: 11, color: '#94a3b8' }}>{e.ts_utc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderDefinitions(defs) {
  if (!defs) return null
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Card title="Time-Series AI Concepts">
        {(defs.concepts || []).map((c, i) => (
          <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{c.term}</div>
            <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.definition}</div>
          </div>
        ))}
      </Card>

      <Card title="Quality Metrics">
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Metric</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Target</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {(defs.quality_metrics || []).map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{m.metric}</td>
                <td style={{ padding: 8 }}><Badge text={m.target} color="#3b82f6" /></td>
                <td style={{ padding: 8, color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Feature Categories">
        {(defs.feature_categories || []).map((cat, i) => (
          <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
            <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
              {cat.name} <span style={{ fontWeight: 400, fontSize: 12, color: '#64748b' }}>({cat.count} features)</span>
            </div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {(cat.features || []).map((f, j) => (
                <Badge key={j} text={f} color="#6b7280" />
              ))}
            </div>
          </div>
        ))}
      </Card>

      <Card title="Compliance References">
        {(defs.compliance || []).map((c, i) => (
          <div key={i} style={{ marginBottom: 8, fontSize: 13 }}>
            <strong>{c.standard}</strong> — <span style={{ color: '#64748b' }}>{c.reference}</span>
          </div>
        ))}
      </Card>

      <Card title="Remediation Strategies">
        {(defs.remediation || []).map((r, i) => (
          <div key={i} style={{ marginBottom: 8, fontSize: 13 }}>
            <strong style={{ color: '#ef4444' }}>{r.issue}:</strong>{' '}
            <span style={{ color: '#475569' }}>{r.strategy}</span>
          </div>
        ))}
      </Card>
    </div>
  )
}
