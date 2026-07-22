import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#1e88e5', '#7c4dff', '#00bcd4', '#e91e63', '#607d8b']
const SEV_COLORS = { P1: '#f44336', P2: '#ff9800' }
const DET_COLORS = { built: '#4caf50', partial: '#ff9800', planned: '#94a3b8', Built: '#4caf50', Partial: '#ff9800', Planned: '#94a3b8' }

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function SeverityBadge({ severity }) {
  const bg = SEV_COLORS[severity] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {severity}
    </span>
  )
}

function DetBadge({ status }) {
  const bg = DET_COLORS[status] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'capitalize'
    }}>
      {status}
    </span>
  )
}

export default function ProductionIssuesDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/production-issues/overview`),
          axios.get(`${API_URL}/api/production-issues/breakdown`),
          axios.get(`${API_URL}/api/production-issues/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Production Issues data')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Production Issues...</div>
  if (error) return <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#94a3b8' }}>Production Issues data not available.</div>

  const s = overview.summary || {}
  const tabs = ['overview', 'all issues', 'layers', 'pareto', 'definitions']

  const sevData = overview.severity_distribution || []
  const detData = overview.detection_distribution || []
  const layerDist = overview.layer_distribution || []

  // Layer bar chart data
  const layerBarData = layerDist.map(l => ({
    name: l.layer.length > 14 ? l.layer.slice(0, 12) + '..' : l.layer,
    P1: l.p1,
    P2: l.p2
  }))

  // Detection stacked bar per layer
  const layerDetData = layerDist.map(l => ({
    name: l.layer.length > 14 ? l.layer.slice(0, 12) + '..' : l.layer,
    Built: l.built,
    Partial: l.partial,
    Planned: l.planned
  }))

  return (
    <div style={{ padding: '16px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ fontSize: 20, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Production Issues Dashboard</h2>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 16 }}>Enterprise agentic AI issue catalog — 16 layers, severity tracking, detection coverage</p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '1px solid #e2e8f0', paddingBottom: 8 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            background: tab === t ? '#1e293b' : 'transparent', color: tab === t ? '#fff' : '#64748b'
          }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 8 }}>
              <KPI label="Total Issues" value={fmt(s.total_issues)} />
              <KPI label="Layers" value={fmt(s.total_layers)} />
              <KPI label="P1 (Critical)" value={fmt(s.p1)} sub="immediate risk" />
              <KPI label="P2 (Major)" value={fmt(s.p2)} sub="significant" />
              <KPI label="Built" value={fmt(s.built)} sub="fully mitigated" />
              <KPI label="Partial" value={fmt(s.partial)} sub="in progress" />
              <KPI label="Planned" value={fmt(s.planned)} sub="not started" />
              <KPI label="Coverage" value={`${fmt(s.coverage_pct)}%`} sub="built+partial" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Severity Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={sevData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {sevData.map((d, i) => <Cell key={i} fill={SEV_COLORS[d.name] || COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Detection Coverage">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={detData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`}>
                    {detData.map((d, i) => <Cell key={i} fill={DET_COLORS[d.name] || COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <Card title="Issues by Layer (Severity)">
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={layerBarData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={100} />
                <Tooltip />
                <Bar dataKey="P1" stackId="a" fill="#f44336" radius={[0, 0, 0, 0]} />
                <Bar dataKey="P2" stackId="a" fill="#ff9800" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Detection Status by Layer">
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={layerDetData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={100} />
                <Tooltip />
                <Bar dataKey="Built" stackId="a" fill="#4caf50" />
                <Bar dataKey="Partial" stackId="a" fill="#ff9800" />
                <Bar dataKey="Planned" stackId="a" fill="#94a3b8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {overview.internal_flow && (
            <Card title="Internal Request Flow">
              <div style={{ fontSize: 12, color: '#334155', lineHeight: 1.6, fontFamily: 'monospace', background: '#f8fafc', padding: 12, borderRadius: 6 }}>
                {overview.internal_flow.split(' -> ').map((step, i, arr) => (
                  <span key={i}>
                    <span style={{ background: '#e2e8f0', padding: '2px 6px', borderRadius: 4, fontWeight: 500 }}>{step}</span>
                    {i < arr.length - 1 && <span style={{ color: '#94a3b8', margin: '0 4px' }}> → </span>}
                  </span>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {/* ── All Issues Tab ── */}
      {tab === 'all issues' && (
        <Card title="All Production Issues">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Layer</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Issue</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Severity</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Root Cause</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Detection</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Solution</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.layers || []).flatMap(l =>
                  l.issues.map((iss, i) => (
                    <tr key={`${l.layer}-${i}`} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 500, color: '#334155' }}>{l.layer}</td>
                      <td style={{ padding: '8px 6px', fontWeight: 500 }}>{iss.issue}</td>
                      <td style={{ padding: '8px 6px' }}><SeverityBadge severity={iss.severity} /></td>
                      <td style={{ padding: '8px 6px', color: '#475569', fontSize: 12 }}>{iss.root_cause}</td>
                      <td style={{ padding: '8px 6px', color: '#475569', fontSize: 12 }}>{iss.detection}</td>
                      <td style={{ padding: '8px 6px', color: '#475569', fontSize: 12 }}>{iss.solution}</td>
                      <td style={{ padding: '8px 6px' }}><DetBadge status={iss.det_status} /></td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Layers Tab ── */}
      {tab === 'layers' && (
        <>
          {(breakdown?.layers || []).map((l, li) => (
            <Card key={li} title={`${l.layer} (${l.issues.length} issue${l.issues.length !== 1 ? 's' : ''})`}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Issue</th>
                      <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Sev</th>
                      <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Root Cause</th>
                      <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Solution</th>
                      <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Project Status</th>
                      <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Det.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {l.issues.map((iss, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 6px', fontWeight: 500 }}>{iss.issue}</td>
                        <td style={{ padding: '8px 6px' }}><SeverityBadge severity={iss.severity} /></td>
                        <td style={{ padding: '8px 6px', color: '#475569', fontSize: 12 }}>{iss.root_cause}</td>
                        <td style={{ padding: '8px 6px', color: '#475569', fontSize: 12 }}>{iss.solution}</td>
                        <td style={{ padding: '8px 6px', color: '#475569', fontSize: 12, maxWidth: 250 }}>{iss.detected_in_project}</td>
                        <td style={{ padding: '8px 6px' }}><DetBadge status={iss.det_status} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ))}
        </>
      )}

      {/* ── Pareto Tab ── */}
      {tab === 'pareto' && (
        <>
          <Card title="Top 20% Issues That Cause 80% of Incidents">
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              These ~10 issues represent the highest-impact failure modes in agentic AI systems. Addressing these first yields the greatest reliability improvement.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(overview.pareto_issues || []).map((issue, i) => {
                const found = (breakdown?.layers || []).flatMap(l => l.issues).find(iss => iss.issue === issue)
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px', background: '#f8fafc', borderRadius: 6, borderLeft: `4px solid ${found ? SEV_COLORS[found.severity] || '#94a3b8' : '#94a3b8'}` }}>
                    <span style={{ fontWeight: 700, fontSize: 16, color: '#94a3b8', minWidth: 28 }}>{i + 1}.</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{issue}</div>
                      {found && (
                        <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
                          Root cause: {found.root_cause} | Solution: {found.solution}
                        </div>
                      )}
                    </div>
                    {found && (
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <SeverityBadge severity={found.severity} />
                        <DetBadge status={found.det_status} />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </Card>

          <Card title="Layer Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Layer</th>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, textAlign: 'center' }}>Total</th>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, textAlign: 'center' }}>P1</th>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, textAlign: 'center' }}>P2</th>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, textAlign: 'center' }}>Built</th>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, textAlign: 'center' }}>Partial</th>
                    <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, textAlign: 'center' }}>Planned</th>
                  </tr>
                </thead>
                <tbody>
                  {layerDist.map((l, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 500 }}>{l.layer}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center' }}>{l.total}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: l.p1 > 0 ? '#f44336' : '#94a3b8', fontWeight: l.p1 > 0 ? 700 : 400 }}>{l.p1}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: l.p2 > 0 ? '#ff9800' : '#94a3b8', fontWeight: l.p2 > 0 ? 700 : 400 }}>{l.p2}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: l.built > 0 ? '#4caf50' : '#94a3b8' }}>{l.built}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: l.partial > 0 ? '#ff9800' : '#94a3b8' }}>{l.partial}</td>
                      <td style={{ padding: '8px 6px', textAlign: 'center', color: l.planned > 0 ? '#64748b' : '#94a3b8' }}>{l.planned}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs?.available && (
        <>
          <Card title="Terminology">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600, width: 180 }}>Term</th>
                  <th style={{ padding: '8px 6px', color: '#64748b', fontWeight: 600 }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.definitions || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600, color: '#334155' }}>{d.term}</td>
                    <td style={{ padding: '8px 6px', color: '#475569', lineHeight: 1.5 }}>{d.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {defs.clinical_notes && (
            <Card title="Clinical Notes">
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {defs.clinical_notes.map((n, i) => (
                  <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>{n}</li>
                ))}
              </ul>
            </Card>
          )}

          {defs.references && (
            <Card title="References">
              <ol style={{ margin: 0, paddingLeft: 20 }}>
                {defs.references.map((r, i) => (
                  <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4, lineHeight: 1.5 }}>{r}</li>
                ))}
              </ol>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
