import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function Badge({ text, bg, fg }) {
  return (
    <span style={{
      background: bg, color: fg, padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600
    }}>{text}</span>
  )
}

function QualityBadge({ quality }) {
  const map = {
    excellent: { bg: '#dcfce7', fg: '#166534' },
    good: { bg: '#dbeafe', fg: '#1e40af' },
    fair: { bg: '#fef3c7', fg: '#92400e' },
    poor: { bg: '#fee2e2', fg: '#991b1b' },
  }
  const s = map[quality] || { bg: '#f1f5f9', fg: '#475569' }
  return <Badge text={quality} bg={s.bg} fg={s.fg} />
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v <= 1 ? (v * 100).toFixed(1) + '%' : v.toFixed(1) + '%') : String(v)
}

const TABS = ['Overview', 'Providers', 'Patients', 'Platforms', 'Reference']

export default function TelehealthDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/telehealth/overview`),
      axios.get(`${API}/api/telehealth/breakdown`),
      axios.get(`${API}/api/telehealth/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading telehealth data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const kpis = ov?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', margin: '0 0 4px' }}>
        Telehealth Sessions Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        {kpis.total_sessions || 0} sessions &middot; {kpis.unique_patients || 0} patients &middot; {kpis.unique_providers || 0} providers &middot; real clinical.db
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: 600,
            background: tab === i ? '#1e293b' : '#f1f5f9',
            color: tab === i ? '#fff' : '#475569',
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && renderOverview()}
      {tab === 1 && renderProviders()}
      {tab === 2 && renderPatients()}
      {tab === 3 && renderPlatforms()}
      {tab === 4 && renderReference()}
    </div>
  )

  function renderOverview() {
    const typeDist = ov?.session_type_distribution || []
    const qualDist = ov?.quality_distribution || []
    const satHist = ov?.satisfaction_histogram || []
    const monthly = ov?.monthly_trend || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {/* KPI row */}
        <Card title="Total Sessions"><KPI label="Sessions" value={fmt(kpis.total_sessions)} /></Card>
        <Card title="Avg Duration"><KPI label="Minutes" value={fmt(kpis.avg_duration_min)} color="#3b82f6" /></Card>
        <Card title="Avg Satisfaction"><KPI label="out of 5" value={fmt(kpis.avg_satisfaction)} color="#10b981" /></Card>
        <Card title="Tech Issue Rate"><KPI label={`${kpis.sessions_with_issues || 0} sessions`} value={fmtPct(kpis.technical_issue_rate)} color={kpis.technical_issue_rate > 0.2 ? '#ef4444' : '#f59e0b'} /></Card>

        {/* Session type bar chart */}
        <Card title="Session Types" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={typeDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Connection quality pie */}
        <Card title="Connection Quality" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={qualDist} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, pct }) => `${name} ${(pct * 100).toFixed(0)}%`}>
                {qualDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Satisfaction histogram */}
        <Card title="Patient Satisfaction Distribution" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={satHist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="rating" tick={{ fontSize: 12 }} label={{ value: 'Rating', position: 'insideBottom', offset: -2, fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Monthly trend */}
        <Card title="Monthly Session Volume" span={2}>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={monthly}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="sessions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
              <Line type="monotone" dataKey="issue_count" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>
    )
  }

  function renderProviders() {
    const providers = bd?.provider_stats || []
    const chartData = providers.map(p => ({
      name: p.provider.replace('Dr. ', ''),
      sessions: p.sessions,
      satisfaction: p.avg_satisfaction,
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        <Card title="Provider KPIs">
          <KPI label="Providers" value={providers.length} />
        </Card>
        <Card title="Most Active">
          <KPI label={providers[0]?.provider || '--'} value={providers[0]?.sessions || 0} color="#3b82f6" sub="sessions" />
        </Card>
        <Card title="Highest Rated">
          <KPI label={(() => { const best = [...providers].sort((a, b) => b.avg_satisfaction - a.avg_satisfaction)[0]; return best?.provider || '--' })()} value={(() => { const best = [...providers].sort((a, b) => b.avg_satisfaction - a.avg_satisfaction)[0]; return fmt(best?.avg_satisfaction) })()} color="#10b981" sub="avg satisfaction" />
        </Card>

        {/* Provider sessions bar chart */}
        <Card title="Sessions per Provider" span={3}>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="sessions" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Provider table */}
        <Card title="Provider Detail" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Provider', 'Sessions', 'Patients', 'Avg Duration', 'Avg Satisfaction', 'Issue Rate'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {providers.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.provider}</td>
                    <td style={{ padding: '8px 10px' }}>{p.sessions}</td>
                    <td style={{ padding: '8px 10px' }}>{p.patients}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.avg_duration)} min</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.avg_satisfaction)}/5</td>
                    <td style={{ padding: '8px 10px' }}>{fmtPct(p.issue_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const patients = bd?.patient_sessions || []
    const recent = bd?.recent_sessions || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        <Card title="Patients with Telehealth"><KPI label="Unique Patients" value={patients.length} /></Card>
        <Card title="Most Sessions">
          <KPI label={patients[0]?.patient_id || '--'} value={patients[0]?.total_sessions || 0} color="#3b82f6" sub="sessions" />
        </Card>
        <Card title="Avg Sessions / Patient">
          <KPI label="per patient" value={patients.length ? fmt(patients.reduce((s, p) => s + p.total_sessions, 0) / patients.length) : '--'} color="#8b5cf6" />
        </Card>

        {/* Patient table */}
        <Card title="Patient Session Summary" span={3}>
          <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'Sessions', 'Avg Satisfaction', 'Last Session', 'Primary Type'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontSize: 12, position: 'sticky', top: 0, background: '#fff' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 10px' }}>{p.total_sessions}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.avg_satisfaction)}/5</td>
                    <td style={{ padding: '8px 10px' }}>{p.last_session}</td>
                    <td style={{ padding: '8px 10px' }}>{p.primary_type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {/* Recent sessions */}
        <Card title="Recent Sessions" span={3}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Date', 'Patient', 'Type', 'Provider', 'Duration', 'Quality', 'Satisfaction', 'Issues'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recent.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px' }}>{s.date}</td>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{s.patient_id}</td>
                    <td style={{ padding: '6px 8px' }}>{s.type}</td>
                    <td style={{ padding: '6px 8px' }}>{s.provider}</td>
                    <td style={{ padding: '6px 8px' }}>{s.duration_min} min</td>
                    <td style={{ padding: '6px 8px' }}><QualityBadge quality={s.quality} /></td>
                    <td style={{ padding: '6px 8px' }}>{s.satisfaction}/5</td>
                    <td style={{ padding: '6px 8px' }}>{s.issues > 0 ? <Badge text={`${s.issues} issue${s.issues > 1 ? 's' : ''}`} bg="#fee2e2" fg="#991b1b" /> : <Badge text="None" bg="#dcfce7" fg="#166534" />}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderPlatforms() {
    const platforms = bd?.platform_quality || []
    const platDist = ov?.platform_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Platforms"><KPI label="Platforms in Use" value={platforms.length} /></Card>
        <Card title="Top Platform">
          <KPI label={platforms[0]?.platform || '--'} value={platforms[0]?.sessions || 0} color="#3b82f6" sub="sessions" />
        </Card>

        {/* Platform usage pie */}
        <Card title="Platform Usage Distribution">
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={platDist} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, count }) => `${name} (${count})`}>
                {platDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Card>

        {/* Platform quality comparison bar */}
        <Card title="Platform Quality Comparison">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={platforms}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="platform" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 5]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="avg_quality_score" name="Avg Quality (1-4)" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="avg_satisfaction" name="Avg Satisfaction (1-5)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Platform detail table */}
        <Card title="Platform Detail" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Platform', 'Sessions', 'Avg Quality', 'Avg Satisfaction', 'Avg Duration', 'Issue Rate'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', fontSize: 12 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {platforms.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.platform}</td>
                    <td style={{ padding: '8px 10px' }}>{p.sessions}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.avg_quality_score)}/4</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.avg_satisfaction)}/5</td>
                    <td style={{ padding: '8px 10px' }}>{fmt(p.avg_duration)} min</td>
                    <td style={{ padding: '8px 10px' }}>{fmtPct(p.issue_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderReference() {
    const types = df?.session_types || {}
    const qualities = df?.connection_quality || {}
    const scale = df?.satisfaction_scale || {}
    const kpiDefs = df?.kpi_definitions || []
    const refs = df?.clinical_references || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Session Types">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              {Object.entries(types).map(([k, v]) => (
                <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>{k}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Connection Quality Levels">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              {Object.entries(qualities).map(([k, v]) => (
                <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px' }}><QualityBadge quality={k} /></td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Satisfaction Scale">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              {Object.entries(scale).map(([k, v]) => (
                <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>{k}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="KPI Definitions">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <tbody>
              {kpiDefs.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', fontWeight: 600 }}>{d.kpi}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{d.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Clinical References" span={2}>
          <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#334155' }}>
            {refs.map((r, i) => <li key={i} style={{ marginBottom: 6 }}>{r}</li>)}
          </ul>
        </Card>
      </div>
    )
  }
}
