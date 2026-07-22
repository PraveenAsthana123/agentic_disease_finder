import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = { excellent: '#10b981', STRONG: '#10b981', PASS: '#10b981', pass: '#10b981', good: '#3b82f6', ACCEPTABLE: '#f59e0b', DEGRADED: '#ef4444', FAIL: '#ef4444', unknown: '#64748b' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{status}</span>
  )
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function ResponsibleAIDashboard() {
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
          axios.get(`${API_URL}/api/responsible-ai-dashboard/overview`),
          axios.get(`${API_URL}/api/responsible-ai-dashboard/breakdown`),
          axios.get(`${API_URL}/api/responsible-ai-dashboard/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load responsible AI data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading responsible AI data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview || overview.available === false) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No responsible AI data available. Run responsible AI analysis first.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'frameworks', label: 'Framework Details' },
    { id: 'fairness', label: 'Fairness & Robustness' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const scoreColor = overview.overall_score >= 85 ? '#10b981' : overview.overall_score >= 70 ? '#f59e0b' : '#ef4444'

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Responsible AI Dashboard</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        Real analysis from {overview.total_frameworks} frameworks | {overview.analysis_date}
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} scoreColor={scoreColor} />}
      {tab === 'frameworks' && <FrameworksTab overview={overview} breakdown={breakdown} />}
      {tab === 'fairness' && <FairnessTab overview={overview} breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview, scoreColor }) {
  const ts = overview.test_summary || {}
  const passColor = ts.overall_status === 'PASS' ? '#10b981' : '#ef4444'

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* KPI row */}
      <Card>
        <KPI label="Overall Score" value={overview.overall_score} color={scoreColor} sub={`/ 100`} />
      </Card>
      <Card>
        <KPI label="Frameworks Assessed" value={overview.applicable_frameworks} sub={`of ${overview.total_frameworks}`} />
      </Card>
      <Card>
        <KPI label="Test Pass Rate" value={ts.pass_rate != null ? `${(ts.pass_rate * 100).toFixed(0)}%` : '--'}
              color={passColor} sub={`${ts.tests_passed}/${ts.total_tests} tests`} />
      </Card>
      <Card>
        <KPI label="Fairness Gate" value={overview.fairness?.gate || '--'}
              color={STATUS_COLORS[overview.fairness?.gate] || '#64748b'}
              sub={overview.fairness?.protected_attribute ? `by ${overview.fairness.protected_attribute}` : ''} />
      </Card>

      {/* Framework scores chart */}
      <Card title="Framework Scores" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.framework_cards} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="label" tick={{ fontSize: 11 }} width={80} />
            <Tooltip formatter={v => fmt(v)} />
            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
              {(overview.framework_cards || []).map((e, i) => (
                <Cell key={i} fill={e.score >= 85 ? '#10b981' : e.score >= 70 ? '#f59e0b' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Disease accuracy chart */}
      <Card title="Disease Accuracy (%)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.disease_accuracy}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
            <YAxis domain={[85, 100]} tick={{ fontSize: 11 }} />
            <Tooltip formatter={v => `${fmt(v)}%`} />
            <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]}>
              {(overview.disease_accuracy || []).map((e, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Reliability detail */}
      <Card title="Reliability" span={2}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
          <div><strong>Score:</strong> {fmt(overview.reliability?.score)}</div>
          <StatusBadge status={overview.reliability?.status || 'unknown'} />
        </div>
        {overview.reliability?.sla_targets && (
          <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
            <thead><tr style={{ borderBottom: '1px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 4, color: '#64748b' }}>SLA Target</th>
              <th style={{ textAlign: 'right', padding: 4, color: '#64748b' }}>Value</th>
            </tr></thead>
            <tbody>
              {Object.entries(overview.reliability.sla_targets).map(([k, v]) => (
                <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 4, fontSize: 12 }}>{k.replace(/_/g, ' ')}</td>
                  <td style={{ padding: 4, textAlign: 'right', fontFamily: 'monospace', fontSize: 12 }}>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {/* Calibration ECE */}
      <Card title="Calibration (ECE per Disease)" span={2}>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={overview.calibration_data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={v => fmt(v)} />
            <Bar dataKey="ece" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p style={{ fontSize: 11, color: '#94a3b8', margin: '4px 0 0' }}>
          Lower ECE = better calibrated probability outputs
        </p>
      </Card>

      {/* Test results */}
      <Card title="Responsible AI Tests" span={4}>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {(overview.test_cards || []).map(t => (
            <div key={t.id} style={{
              padding: '10px 16px', borderRadius: 8,
              background: t.status === 'PASS' ? '#f0fdf4' : '#fef2f2',
              border: `1px solid ${t.status === 'PASS' ? '#bbf7d0' : '#fecaca'}`,
              minWidth: 120, textAlign: 'center'
            }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{t.label}</div>
              <StatusBadge status={t.status} />
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function FrameworksTab({ overview, breakdown }) {
  const details = breakdown?.framework_details || []
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {details.map(fw => (
        <Card key={fw.id} title={`${fw.label} (${fmt(fw.score)}/100)`}>
          <StatusBadge status={fw.status} />
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12, marginTop: 12 }}>
            {(fw.analyses || []).map(a => (
              <div key={a.id} style={{ padding: 12, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{a.label}</div>
                <div style={{ fontSize: 12, color: '#3b82f6', marginBottom: 4 }}>Score: {fmt(a.score)}</div>
                {a.method && <div style={{ fontSize: 11, color: '#64748b' }}>Method: {a.method}</div>}
                {a.justification && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{a.justification}</div>}
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}

function FairnessTab({ overview, breakdown }) {
  const f = overview?.fairness || {}
  const groups = f.by_group || {}
  const groupData = Object.entries(groups).map(([name, g]) => ({
    name,
    selection_rate: g.selection_rate,
    count: g.count
  }))

  const robCurve = breakdown?.robustness_curve || []
  const consData = breakdown?.consistency_data || []
  const robLevels = breakdown?.robustness_levels || []
  const errPatterns = breakdown?.error_patterns || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {/* Fairness gate */}
      <Card title="Fairness Gate (Demographic Parity)" span={2}>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
          <div>
            <StatusBadge status={f.gate || 'unknown'} />
            <span style={{ marginLeft: 8, fontSize: 13 }}>DPD: {fmt(f.dpd)}</span>
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            Protected attribute: <strong>{f.protected_attribute}</strong> | N={f.n} | Library: {f.library}
          </div>
        </div>
        <p style={{ fontSize: 12, color: '#475569', margin: '0 0 12px' }}>{f.interpretation}</p>
        {groupData.length > 0 && (
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={groupData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 0.5]} tick={{ fontSize: 11 }} />
              <Tooltip formatter={v => fmt(v)} />
              <Bar dataKey="selection_rate" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Selection Rate">
                {groupData.map((e, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </Card>

      {/* Robustness curve */}
      <Card title="Robustness Under Noise">
        {robCurve.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={robCurve}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="noise_level" tick={{ fontSize: 11 }} label={{ value: 'Noise Level', position: 'bottom', fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={2} dot />
              <Line type="monotone" dataKey="prediction_change_rate" stroke="#f59e0b" strokeWidth={2} dot />
            </LineChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No robustness data available</p>}
      </Card>

      {/* Consistency per disease */}
      <Card title="Consistency (Variance by Disease)">
        {consData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={consData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="variance" fill="#06b6d4" radius={[4, 4, 0, 0]}>
                {consData.map((e, i) => (
                  <Cell key={i} fill={e.status === 'HIGHLY_CONSISTENT' ? '#10b981' : '#3b82f6'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No consistency data available</p>}
      </Card>

      {/* Robustness levels table */}
      <Card title="Noise Tolerance Levels">
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead><tr style={{ borderBottom: '1px solid #e2e8f0' }}>
            <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Noise Level</th>
            <th style={{ textAlign: 'right', padding: 6, color: '#64748b' }}>Accuracy Drop</th>
            <th style={{ textAlign: 'center', padding: 6, color: '#64748b' }}>Status</th>
          </tr></thead>
          <tbody>
            {robLevels.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{r.level}</td>
                <td style={{ padding: 6, textAlign: 'right', fontFamily: 'monospace' }}>{r.accuracy_drop}</td>
                <td style={{ padding: 6, textAlign: 'center' }}><StatusBadge status={r.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Error patterns */}
      <Card title="Error Patterns">
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead><tr style={{ borderBottom: '1px solid #e2e8f0' }}>
            <th style={{ textAlign: 'left', padding: 6, color: '#64748b' }}>Pattern</th>
            <th style={{ textAlign: 'right', padding: 6, color: '#64748b' }}>Count</th>
          </tr></thead>
          <tbody>
            {Object.entries(errPatterns).map(([k, v]) => (
              <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6 }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</td>
                <td style={{ padding: 6, textAlign: 'right', fontWeight: 600 }}>{fmt(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return <p style={{ color: '#94a3b8' }}>No definitions available</p>
  return (
    <Card title="Metric Definitions">
      <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
        <thead><tr style={{ borderBottom: '2px solid #e2e8f0' }}>
          <th style={{ textAlign: 'left', padding: 8, color: '#334155', width: '30%' }}>Metric</th>
          <th style={{ textAlign: 'left', padding: 8, color: '#334155' }}>Definition</th>
        </tr></thead>
        <tbody>
          {Object.entries(defs).map(([k, v]) => (
            <tr key={k} style={{ borderBottom: '1px solid #f1f5f9' }}>
              <td style={{ padding: 8, fontWeight: 500, fontFamily: 'monospace', fontSize: 12 }}>
                {k.replace(/_/g, ' ')}
              </td>
              <td style={{ padding: 8, color: '#475569' }}>{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
