import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const LEVEL_COLORS = {
  mild: '#10b981',
  moderate: '#f59e0b',
  severe: '#ef4444',
  critical: '#7f1d1d',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'domains', label: 'Domain Analysis' },
  { id: 'patients', label: 'Patient Tracking' },
  { id: 'heatmap', label: 'Item Heatmap' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SeizureSeverityDashboard() {
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
      axios.get(`${API_URL}/api/seizure-severity-dashboard/overview`),
      axios.get(`${API_URL}/api/seizure-severity-dashboard/breakdown`),
      axios.get(`${API_URL}/api/seizure-severity-dashboard/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Seizure Severity Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  const severityPieData = Object.entries(ov.severity_distribution || {}).map(([level, count]) => ({
    name: level.charAt(0).toUpperCase() + level.slice(1), value: count
  }))

  const domainBarData = (bd.domain_summary || []).map(d => ({
    name: d.label, avg: d.avg_score, max: d.max_possible, pct: Math.round((d.avg_score / d.max_possible) * 100)
  }))

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Seizure Severity Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Liverpool Seizure Severity Scale (LSSS) — 20-item validated measure of seizure severity across 3 domains
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card title="Total Assessments">
            <KPI value={fmt(ov.total_assessments)} label="LSSS assessments recorded" color="#3b82f6" />
          </Card>
          <Card title="Unique Patients">
            <KPI value={fmt(ov.unique_patients)} label="Patients assessed" color="#8b5cf6" />
          </Card>
          <Card title="Average Score">
            <KPI value={ov.avg_score != null ? ov.avg_score.toFixed(1) : '--'} label="Out of 80" color="#f59e0b" />
          </Card>
          <Card title="Score Range">
            <KPI value={`${fmt(ov.min_score)}–${fmt(ov.max_score_observed)}`} label="Min – Max observed" color="#06b6d4" />
          </Card>

          {/* Severity Distribution Pie */}
          <Card title="Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={severityPieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                  {severityPieData.map((_, i) => {
                    const level = (severityPieData[i]?.name || '').toLowerCase()
                    return <Cell key={i} fill={LEVEL_COLORS[level] || COLORS[i % COLORS.length]} />
                  })}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Domain Scores Bar */}
          <Card title="Domain Scores (avg)" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={domainBarData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 44]} />
                <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v, name) => name === 'avg' ? `${v} (avg)` : `${v} (max)`} />
                <Bar dataKey="avg" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                <Bar dataKey="max" fill="#e2e8f0" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Trend over time */}
          {(bd.trend || []).length > 0 && (
            <Card title="Monthly Severity Trend" span={4}>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={bd.trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis domain={[20, 80]} />
                  <Tooltip formatter={v => v.toFixed(1)} />
                  <Line type="monotone" dataKey="avg_score" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Patient Summary Table */}
          <Card title="Patient Latest Scores" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Score</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Max</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Interpretation</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Level</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Assessed</th>
                  </tr>
                </thead>
                <tbody>
                  {(ov.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.latest_score}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', color: '#94a3b8' }}>{p.max_score}</td>
                      <td style={{ padding: '6px 10px' }}>{p.interpretation}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <Badge text={p.level} color={LEVEL_COLORS[p.level] || '#64748b'} />
                      </td>
                      <td style={{ padding: '6px 10px', fontSize: 12, color: '#94a3b8' }}>{(p.assessed_at || '').slice(0, 10)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* DOMAIN ANALYSIS TAB */}
      {tab === 'domains' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {(bd.domain_summary || []).map((d, i) => (
            <Card key={d.domain} title={d.label}>
              <KPI
                value={d.avg_score.toFixed(1)}
                label={`out of ${d.max_possible}`}
                sub={`${Math.round((d.avg_score / d.max_possible) * 100)}% of max · n=${d.n}`}
                color={COLORS[i]}
              />
              <div style={{ marginTop: 12, background: '#f1f5f9', borderRadius: 8, height: 12, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.round((d.avg_score / d.max_possible) * 100)}%`,
                  height: '100%', background: COLORS[i], borderRadius: 8
                }} />
              </div>
            </Card>
          ))}

          <Card title="Domain Comparison" span={3}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={domainBarData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="avg" name="Average Score" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="max" name="Max Possible" fill="#e2e8f0" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* PATIENT TRACKING TAB */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {Object.entries(bd.patient_history || {}).map(([pid, history]) => (
            <Card key={pid} title={`Patient ${pid} — ${history.length} assessment${history.length !== 1 ? 's' : ''}`}>
              <div style={{ display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
                <div style={{ flex: 1, minWidth: 300 }}>
                  <ResponsiveContainer width="100%" height={120}>
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => (d || '').slice(0, 10)} />
                      <YAxis domain={[20, 80]} tick={{ fontSize: 10 }} />
                      <Tooltip formatter={v => v} labelFormatter={l => (l || '').slice(0, 10)} />
                      <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ minWidth: 200 }}>
                  <table style={{ fontSize: 12, borderCollapse: 'collapse' }}>
                    <tbody>
                      {history.map((h, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{(h.date || '').slice(0, 10)}</td>
                          <td style={{ padding: '3px 8px', fontWeight: 600 }}>{h.score}</td>
                          <td style={{ padding: '3px 8px' }}>
                            <Badge text={h.level} color={LEVEL_COLORS[h.level] || '#64748b'} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ITEM HEATMAP TAB */}
      {tab === 'heatmap' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Item Severity Heatmap (ranked by average score)">
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              Each item scored 1–4. Higher average = more severe symptom across patients.
            </p>
            <ResponsiveContainer width="100%" height={500}>
              <BarChart data={bd.item_heatmap || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 4]} />
                <YAxis type="category" dataKey="label" width={220} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => v.toFixed(2)} />
                <Bar dataKey="avg" name="Avg Score" radius={[0, 4, 4, 0]}>
                  {(bd.item_heatmap || []).map((item, i) => (
                    <Cell key={i} fill={item.avg >= 2.5 ? '#ef4444' : item.avg >= 2.0 ? '#f59e0b' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={defs.title || 'Definitions'}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b', width: 180 }}>Term</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.definitions || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{d.term}</td>
                    <td style={{ padding: '8px 10px', color: '#475569' }}>{d.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
