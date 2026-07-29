import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#06b6d4']
const AGREE_COLORS = { agree: '#10b981', disagree: '#ef4444', partial: '#f59e0b' }

export default function ComponentFindingsDashboard() {
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
      axios.get(`${API}/api/component-findings/overview`),
      axios.get(`${API}/api/component-findings/breakdown`),
      axios.get(`${API}/api/component-findings/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'findings', label: 'All Findings' },
    { id: 'patients', label: 'By Patient' },
    { id: 'components', label: 'By Component' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Component Findings dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, fontFamily: "'Inter',system-ui,sans-serif", background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#0f172a', marginBottom: 4 }}>Component Findings — Doctor-AI Agreement</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>Per-component agreement between clinician review and AI EEG analysis</p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6,1fr)', gap: 12 }}>
              <KPI label="Total Findings" value={k.total_findings} />
              <KPI label="Patients" value={k.total_patients} />
              <KPI label="Reviewers" value={k.total_reviewers} />
              <KPI label="Components" value={k.total_components} />
              <KPI label="Agreement Rate" value={`${k.agreement_rate}%`} color="#10b981" />
              <KPI label="Disagreement Rate" value={`${k.disagreement_rate}%`} color="#ef4444" />
            </div>
          </Card>

          <Card title="Agreement Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.agreement_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.agreement_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Agreement by Component" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={overview.component_agreement}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="agree" stackId="a" fill="#10b981" name="Agree" />
                <Bar dataKey="partial" stackId="a" fill="#f59e0b" name="Partial" />
                <Bar dataKey="disagree" stackId="a" fill="#ef4444" name="Disagree" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Agreement by Reviewer" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={overview.reviewer_agreement} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="reviewer" type="category" tick={{ fontSize: 11 }} width={100} />
                <Tooltip />
                <Bar dataKey="agree" stackId="a" fill="#10b981" name="Agree" />
                <Bar dataKey="partial" stackId="a" fill="#f59e0b" name="Partial" />
                <Bar dataKey="disagree" stackId="a" fill="#ef4444" name="Disagree" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Monthly Trend" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={overview.monthly_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="agree" stroke="#10b981" name="Agree" strokeWidth={2} />
                <Line type="monotone" dataKey="partial" stroke="#f59e0b" name="Partial" strokeWidth={2} />
                <Line type="monotone" dataKey="disagree" stroke="#ef4444" name="Disagree" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── All Findings ── */}
      {tab === 'findings' && breakdown && (
        <Card title={`All Findings (${(breakdown.all_findings || []).length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient', 'Component', 'Doctor Finding', 'Reviewer', 'Agreement', 'Date'].map(h =>
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {(breakdown.all_findings || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{r.component}</td>
                    <td style={{ padding: '6px 10px', maxWidth: 300, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.doctor_finding}</td>
                    <td style={{ padding: '6px 10px' }}>{r.doctor}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <Badge text={r.agree_with_ai} color={AGREE_COLORS[r.agree_with_ai] || '#64748b'} />
                    </td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{(r.created_at || '').slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── By Patient ── */}
      {tab === 'patients' && breakdown && (
        <Card title={`Patient Summary (${(breakdown.patient_summary || []).length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient', 'Reviews', 'Agree', 'Partial', 'Disagree', 'Rate', 'Components', 'Reviewer', 'Flags'].map(h =>
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {(breakdown.patient_summary || []).map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>{r.total_reviews}</td>
                    <td style={{ padding: '6px 10px', color: '#10b981' }}>{r.agree}</td>
                    <td style={{ padding: '6px 10px', color: '#f59e0b' }}>{r.partial}</td>
                    <td style={{ padding: '6px 10px', color: '#ef4444' }}>{r.disagree}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={{ color: r.agree_pct >= 75 ? '#10b981' : r.agree_pct >= 50 ? '#f59e0b' : '#ef4444', fontWeight: 600 }}>
                        {r.agree_pct}%
                      </span>
                    </td>
                    <td style={{ padding: '6px 10px', fontSize: 11 }}>{(r.components_reviewed || []).join(', ')}</td>
                    <td style={{ padding: '6px 10px' }}>{r.reviewer}</td>
                    <td style={{ padding: '6px 10px' }}>
                      {(r.flags || []).map(f => <Badge key={f} text={f} color={f.includes('disagree') ? '#ef4444' : '#f59e0b'} />)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── By Component ── */}
      {tab === 'components' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
          {(breakdown.component_detail || []).map(comp => (
            <Card key={comp.component} title={comp.component}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, marginBottom: 12 }}>
                <KPI label="Total" value={comp.total} />
                <KPI label="Agree" value={comp.agree} color="#10b981" />
                <KPI label="Partial" value={comp.partial} color="#f59e0b" />
                <KPI label="Disagree" value={comp.disagree} color="#ef4444" />
              </div>
              <div style={{ fontSize: 13, marginBottom: 8 }}>
                Agreement: <strong style={{ color: comp.agree_pct >= 75 ? '#10b981' : comp.agree_pct >= 50 ? '#f59e0b' : '#ef4444' }}>{comp.agree_pct}%</strong>
              </div>
              <div style={{ fontSize: 12, color: '#475569' }}>
                <strong>Top Findings:</strong>
                <ul style={{ margin: '4px 0', paddingLeft: 16 }}>
                  {(comp.top_findings || []).map((f, i) => (
                    <li key={i}>{f.finding} ({f.count})</li>
                  ))}
                </ul>
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>
                Reviewers: {(comp.reviewers || []).join(', ')}
              </div>
            </Card>
          ))}

          <Card title="Disagreement Detail" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#fef2f2' }}>
                    {['Patient', 'Component', 'Finding', 'Reviewer', 'Status', 'Date'].map(h =>
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#991b1b' }}>{h}</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.disagreement_detail || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #fecaca' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.component}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 250, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.doctor_finding}</td>
                      <td style={{ padding: '6px 10px' }}>{r.doctor}</td>
                      <td style={{ padding: '6px 10px' }}><Badge text={r.agree_with_ai} color={AGREE_COLORS[r.agree_with_ai] || '#64748b'} /></td>
                      <td style={{ padding: '6px 10px', color: '#64748b' }}>{(r.created_at || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 16 }}>
          <Card title="Overview" span={2}>
            <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{definitions.description}</p>
          </Card>

          <Card title="EEG Components">
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(definitions.components || {}).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{k}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Agreement Levels">
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(definitions.agreement_levels || {}).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px' }}><Badge text={k} color={AGREE_COLORS[k] || '#64748b'} /></td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Metrics">
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(definitions.metrics || {}).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{k}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Relevance" span={2}>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#475569', lineHeight: 1.8 }}>
              {(definitions.clinical_relevance || []).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <tbody>
                {Object.entries(definitions.glossary || {}).map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{k}</td>
                    <td style={{ padding: '6px 10px', color: '#475569' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Related Dashboards">
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: '#3b82f6' }}>
              {(definitions.related_dashboards || []).map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
