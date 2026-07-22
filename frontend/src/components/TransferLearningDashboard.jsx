import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const STRATEGY_COLORS = ['#3b82f6', '#16a34a', '#eab308', '#ef4444', '#8b5cf6', '#ec4899', '#f59e0b', '#06b6d4']
const PIE_COLORS = ['#3b82f6', '#16a34a', '#eab308', '#ef4444', '#8b5cf6', '#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}

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

export default function TransferLearningDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [sortCol, setSortCol] = useState('improvement')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/transfer-learning/overview`),
          axios.get(`${API_URL}/api/transfer-learning/breakdown`),
          axios.get(`${API_URL}/api/transfer-learning/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Transfer Learning data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'strategy', label: 'Strategy Analysis' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'convergence', label: 'Convergence' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const strategyDist = overview?.strategy_distribution || []
  const improvementHist = overview?.improvement_histogram || []

  const strategyComparison = breakdown?.strategy_comparison || []
  const strategyDescriptions = breakdown?.strategy_descriptions || []
  const patientDetails = breakdown?.patient_details || []
  const convergenceHist = breakdown?.convergence_histogram || []
  const domainShiftScatter = breakdown?.domain_shift_scatter || []

  const concepts = defs?.concepts || []
  const strategyDefs = defs?.strategies || []
  const metricDefs = defs?.metrics || []

  const handleSort = (col) => {
    if (sortCol === col) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortCol(col)
      setSortDir('desc')
    }
  }

  const sortedPatients = [...patientDetails].sort((a, b) => {
    const aVal = a[sortCol] ?? 0
    const bVal = b[sortCol] ?? 0
    if (typeof aVal === 'string') return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal)
    return sortDir === 'asc' ? aVal - bVal : bVal - aVal
  })

  const tableStyle = { width: '100%', borderCollapse: 'collapse', fontSize: 13 }
  const thStyle = { textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12, cursor: 'pointer' }
  const tdStyle = { padding: '7px 10px', borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Transfer Learning Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Domain adaptation and transfer learning performance: strategy analysis, patient-level adaptation metrics, and convergence tracking
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview Tab ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(kpis.total_patients)} />
              <KPI label="Mean Baseline Acc" value={`${fmt(kpis.mean_baseline_accuracy)}%`} color="#64748b" />
              <KPI label="Mean Adapted Acc" value={`${fmt(kpis.mean_adapted_accuracy)}%`} color="#16a34a" />
              <KPI label="Improvement" value={`+${fmt(kpis.improvement_pct)}%`} color="#2563eb" />
              <KPI label="Success Rate" value={`${fmt(kpis.success_rate)}%`}
                   color={kpis.success_rate >= 80 ? '#16a34a' : '#eab308'} sub="Improved patients" />
              <KPI label="Avg Domain Shift" value={fmt(kpis.avg_domain_shift)} sub="Distribution distance" />
            </div>
          </Card>

          <Card title="Strategy Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={strategyDist} dataKey="count" nameKey="strategy" cx="50%" cy="50%"
                     outerRadius={85} label={({ strategy, count }) => `${strategy}: ${count}`}>
                  {strategyDist.map((d, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Improvement Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={improvementHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Strategy Analysis Tab ─── */}
      {tab === 'strategy' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Strategy Accuracy Comparison (Baseline vs Adapted)">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={strategyComparison}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="strategy" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={(v) => [`${v}%`]} />
                <Legend />
                <Bar dataKey="baseline_accuracy" name="Baseline" fill="#94a3b8" radius={[4, 4, 0, 0]} />
                <Bar dataKey="adapted_accuracy" name="Adapted" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Strategy Descriptions">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {strategyDescriptions.map((s, i) => (
                <div key={i} style={{
                  padding: 14, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0'
                }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>
                    {s.strategy}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                    {s.description}
                  </div>
                  {s.use_case && (
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 6, fontStyle: 'italic' }}>
                      Use case: {s.use_case}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Patient Transfer Learning Results">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle} onClick={() => handleSort('name')}>
                    Patient {sortCol === 'name' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('disease')}>
                    Disease {sortCol === 'disease' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('baseline_accuracy')}>
                    Baseline Acc {sortCol === 'baseline_accuracy' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('adapted_accuracy')}>
                    Adapted Acc {sortCol === 'adapted_accuracy' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('improvement')}>
                    Improvement {sortCol === 'improvement' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('strategy')}>
                    Strategy {sortCol === 'strategy' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('convergence_epochs')}>
                    Conv. Epochs {sortCol === 'convergence_epochs' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                  <th style={thStyle} onClick={() => handleSort('domain_shift')}>
                    Domain Shift {sortCol === 'domain_shift' ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : ''}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sortedPatients.map((p, i) => (
                  <tr key={p.patient_id || i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{p.name || `Patient ${p.patient_id}`}</td>
                    <td style={tdStyle}>{p.disease}</td>
                    <td style={tdStyle}>{fmt(p.baseline_accuracy)}%</td>
                    <td style={{ ...tdStyle, color: '#16a34a', fontWeight: 600 }}>{fmt(p.adapted_accuracy)}%</td>
                    <td style={{
                      ...tdStyle, fontWeight: 700,
                      color: (p.improvement || 0) > 0 ? '#16a34a' : (p.improvement || 0) < 0 ? '#ef4444' : '#64748b'
                    }}>
                      {(p.improvement || 0) > 0 ? '+' : ''}{fmt(p.improvement)}%
                    </td>
                    <td style={tdStyle}>
                      <span style={{
                        display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                        background: '#3b82f622', color: '#3b82f6'
                      }}>{p.strategy}</span>
                    </td>
                    <td style={tdStyle}>{fmt(p.convergence_epochs)}</td>
                    <td style={{ ...tdStyle, color: (p.domain_shift || 0) > 0.5 ? '#ef4444' : '#16a34a' }}>
                      {fmt(p.domain_shift)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {sortedPatients.length === 0 && <p style={{ color: '#94a3b8', textAlign: 'center', marginTop: 16 }}>No patient data available.</p>}
          </Card>
        </div>
      )}

      {/* ─── Convergence Tab ─── */}
      {tab === 'convergence' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Epochs to Convergence Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={convergenceHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Count']} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Domain Shift vs Improvement">
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain_shift" name="Domain Shift" tick={{ fontSize: 11 }} />
                <YAxis dataKey="improvement" name="Improvement %" tick={{ fontSize: 11 }} tickFormatter={v => `${v}%`} />
                <Tooltip formatter={(v, name) => [name === 'improvement' ? `${v}%` : v, name === 'improvement' ? 'Improvement' : 'Domain Shift']} />
                <Scatter data={domainShiftScatter} fill="#3b82f6" />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Transfer Learning Concepts">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {concepts.map((c, i) => (
                <div key={i} style={{
                  padding: 14, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0'
                }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>
                    {c.name || c.term}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                    {c.description || c.definition}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Strategy Definitions">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {strategyDefs.map((s, i) => (
                <div key={i} style={{
                  padding: 14, borderRadius: 8, background: '#f0fdf4', border: '1px solid #bbf7d0'
                }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#166534', marginBottom: 6 }}>
                    {s.name || s.strategy}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                    {s.description}
                  </div>
                  {s.when_to_use && (
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 6, fontStyle: 'italic' }}>
                      When to use: {s.when_to_use}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>

          <Card title="Metric Definitions">
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>Metric</th>
                  <th style={thStyle}>Description</th>
                  <th style={thStyle}>Unit</th>
                  <th style={thStyle}>Good Range</th>
                </tr>
              </thead>
              <tbody>
                {metricDefs.map((m, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{m.name || m.metric}</td>
                    <td style={tdStyle}>{m.description}</td>
                    <td style={{ ...tdStyle, color: '#94a3b8' }}>{m.unit || '--'}</td>
                    <td style={tdStyle}>{m.good_range || m.range || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 32, textAlign: 'center', color: '#94a3b8', fontSize: 11 }}>
        Transfer Learning Dashboard — Domain Adaptation and Model Personalization Analytics
      </div>
    </div>
  )
}
