import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#94a3b8', '#10b981', '#8b5cf6']
const SEVERITY_COLORS = { P0: '#ef4444', P1: '#f97316', P2: '#f59e0b', P3: '#94a3b8' }
const STATUS_COLORS = { open: '#ef4444', resolved: '#10b981', wontfix: '#94a3b8' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'Issues & Details' },
  { id: 'definitions', label: 'Definitions' },
]

export default function AdvisorIssuesDashboard() {
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
      axios.get(`${API_URL}/api/advisor-issues/overview`),
      axios.get(`${API_URL}/api/advisor-issues/breakdown`),
      axios.get(`${API_URL}/api/advisor-issues/definitions`),
    ])
      .then(([ov, br, df]) => {
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading advisor issues...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Advisor Issues Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        System health advisory findings from automated advisor scans
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 24, borderBottom: '2px solid #e2e8f0' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '10px 20px', border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
              color: tab === t.id ? '#2563eb' : '#64748b',
              borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
              background: 'none', marginBottom: -2,
            }}
          >{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'breakdown' && breakdown && <BreakdownTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPIs */}
      <Card>
        <KPI label="Total Issues" value={data.total_issues} />
      </Card>
      <Card>
        <KPI label="Open Issues" value={data.open_count} color="#ef4444"
             sub={`${data.open_rate}% of total`} />
      </Card>
      <Card>
        <KPI label="Critical/High Open" value={data.critical_open}
             color={data.critical_open > 0 ? '#ef4444' : '#10b981'} sub="P0 + P1 open" />
      </Card>
      <Card>
        <KPI label="Last Scan" value={data.last_scan || '--'} color="#64748b" />
      </Card>

      {/* Severity Distribution */}
      <Card title="Severity Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={data.severity_distribution} dataKey="value" nameKey="name"
                 cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {data.severity_distribution.map((e, i) => (
                <Cell key={i} fill={SEVERITY_COLORS[e.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Surface Distribution */}
      <Card title="Issues by Surface" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.surface_distribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Status Distribution */}
      <Card title="Status Distribution" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={data.status_distribution} dataKey="value" nameKey="name"
                 cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {data.status_distribution.map((e, i) => (
                <Cell key={i} fill={STATUS_COLORS[e.name] || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Scan Timeline */}
      <Card title="Scan Timeline (issues per scan date)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data.scan_timeline}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="issues" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BreakdownTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Open Issues Alert */}
      {data.open_issues.length > 0 && (
        <Card title={`Open Issues Requiring Attention (${data.open_issues.length})`} span={1}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#fef2f2', borderBottom: '2px solid #fecaca' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Sev</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Surface</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Issue</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Guidance</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left' }}>Scanned</th>
                </tr>
              </thead>
              <tbody>
                {data.open_issues.map((iss, i) => (
                  <tr key={iss.id} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                    <td style={{ padding: '8px 10px' }}>
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
                        fontWeight: 700, color: '#fff',
                        background: SEVERITY_COLORS[iss.severity] || '#94a3b8'
                      }}>{iss.severity}</span>
                    </td>
                    <td style={{ padding: '8px 10px', fontWeight: 500 }}>{iss.surface}</td>
                    <td style={{ padding: '8px 10px' }}>{iss.issue}</td>
                    <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{iss.guidance}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{iss.scanned_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* All Issues */}
      <Card title={`All Issues (${data.all_issues.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Sev</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Surface</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Status</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Issue</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Guidance</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Scanned</th>
              </tr>
            </thead>
            <tbody>
              {data.all_issues.map((iss, i) => (
                <tr key={iss.id} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{iss.id}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <span style={{
                      display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
                      fontWeight: 700, color: '#fff',
                      background: SEVERITY_COLORS[iss.severity] || '#94a3b8'
                    }}>{iss.severity}</span>
                  </td>
                  <td style={{ padding: '8px 10px', fontWeight: 500 }}>{iss.surface}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <span style={{
                      display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
                      fontWeight: 600, color: '#fff',
                      background: STATUS_COLORS[iss.status] || '#94a3b8'
                    }}>{iss.status}</span>
                  </td>
                  <td style={{ padding: '8px 10px' }}>{iss.issue}</td>
                  <td style={{ padding: '8px 10px', color: '#475569', fontSize: 12 }}>{iss.guidance}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{iss.scanned_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Surface Summary */}
      <Card title="Per-Surface Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Surface</th>
                <th style={{ padding: '8px 10px', textAlign: 'right' }}>Total</th>
                <th style={{ padding: '8px 10px', textAlign: 'right' }}>Open</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Open Rate</th>
              </tr>
            </thead>
            <tbody>
              {data.surface_summary.map((s, i) => (
                <tr key={s.surface} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{s.surface}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'right' }}>{s.total}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'right', color: s.open_cnt > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{s.open_cnt}</td>
                  <td style={{ padding: '8px 10px' }}>
                    <div style={{ background: '#f1f5f9', borderRadius: 6, height: 14, width: 100 }}>
                      <div style={{
                        background: s.open_cnt > 0 ? '#ef4444' : '#10b981',
                        borderRadius: 6, height: 14,
                        width: `${s.total ? Math.round(s.open_cnt * 100 / s.total) : 0}%`,
                      }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Surface x Severity Cross-Tab */}
      <Card title="Surface x Severity Cross-Tabulation">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Surface</th>
                <th style={{ padding: '8px 10px', textAlign: 'center' }}>P0</th>
                <th style={{ padding: '8px 10px', textAlign: 'center' }}>P1</th>
                <th style={{ padding: '8px 10px', textAlign: 'center' }}>P2</th>
                <th style={{ padding: '8px 10px', textAlign: 'center' }}>P3</th>
              </tr>
            </thead>
            <tbody>
              {data.surface_severity.map((row, i) => (
                <tr key={row.surface} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 600 }}>{row.surface}</td>
                  {['P0', 'P1', 'P2', 'P3'].map(sev => (
                    <td key={sev} style={{ padding: '8px 10px', textAlign: 'center' }}>
                      {row[sev] ? (
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 8,
                          fontWeight: 700, fontSize: 12, color: '#fff',
                          background: SEVERITY_COLORS[sev]
                        }}>{row[sev]}</span>
                      ) : <span style={{ color: '#d1d5db' }}>-</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ data }) {
  const sectionStyle = { marginBottom: 28 }
  const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
  const thStyle = { padding: '8px 10px', textAlign: 'left', background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }
  const tdStyle = { padding: '8px 10px', borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Severity Tiers */}
      <Card title="Severity Tiers">
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Tier</th>
              <th style={thStyle}>Label</th>
              <th style={thStyle}>Description</th>
            </tr>
          </thead>
          <tbody>
            {data.severity_tiers.map((t, i) => (
              <tr key={t.tier} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                <td style={tdStyle}>
                  <span style={{
                    display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
                    fontWeight: 700, color: '#fff', background: SEVERITY_COLORS[t.tier] || '#94a3b8'
                  }}>{t.tier}</span>
                </td>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{t.label}</td>
                <td style={tdStyle}>{t.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Surface Categories */}
      <Card title="Surface Categories">
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Surface</th>
              <th style={thStyle}>Description</th>
            </tr>
          </thead>
          <tbody>
            {data.surface_categories.map((s, i) => (
              <tr key={s.surface} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                <td style={{ ...tdStyle, fontWeight: 600 }}>{s.surface}</td>
                <td style={tdStyle}>{s.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Status Definitions */}
      <Card title="Status Definitions">
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Description</th>
            </tr>
          </thead>
          <tbody>
            {data.status_definitions.map((s, i) => (
              <tr key={s.status} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                <td style={tdStyle}>
                  <span style={{
                    display: 'inline-block', padding: '2px 8px', borderRadius: 8, fontSize: 11,
                    fontWeight: 600, color: '#fff', background: STATUS_COLORS[s.status] || '#94a3b8'
                  }}>{s.status}</span>
                </td>
                <td style={tdStyle}>{s.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Advisor Agent */}
      <Card title="Advisor Agent">
        <p style={{ fontSize: 13, color: '#334155', lineHeight: 1.6, margin: 0 }}>
          {data.advisor_agent.description}
        </p>
        <p style={{ fontSize: 12, color: '#64748b', marginTop: 8, marginBottom: 0 }}>
          <strong>Trigger:</strong> {data.advisor_agent.trigger}
        </p>
      </Card>

      {/* Glossary */}
      <Card title="Clinical & Technical Glossary">
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Term</th>
              <th style={thStyle}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {data.glossary.map((g, i) => (
              <tr key={g.term} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                <td style={tdStyle}>{g.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
