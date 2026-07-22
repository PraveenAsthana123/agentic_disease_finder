import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

export default function DeepchecksDashboard() {
  const [overview, setOverview] = useState(null)
  const [suites, setSuites] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, su, df] = await Promise.all([
          axios.get(`${API_URL}/api/deepchecks/overview`),
          axios.get(`${API_URL}/api/deepchecks/suites`),
          axios.get(`${API_URL}/api/deepchecks/definitions`)
        ])
        setOverview(ov.data)
        setSuites(su.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Deepchecks data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🔍</div>
      Running Deepchecks data & model validation on real EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Deepchecks validation not available. Ensure model and data are present.'}
    </div>
  )

  const checks = overview.checks || []
  const totalChecks = checks.length
  const passed = checks.filter(c => c.status === 'pass' || c.status === 'passed').length
  const failed = checks.filter(c => c.status === 'fail' || c.status === 'failed').length
  const warnings = checks.filter(c => c.status === 'warning').length

  const kpiItems = [
    { label: 'Total Checks', value: totalChecks, icon: '📋', color: '#1e88e5' },
    { label: 'Passed', value: passed, icon: '✅', color: '#4caf50' },
    { label: 'Failed', value: failed, icon: '❌', color: '#f44336' },
    { label: 'Warnings', value: warnings, icon: '⚠️', color: '#ff9800' },
  ]

  const pieData = [
    { name: 'Passed', value: passed },
    { name: 'Failed', value: failed },
    { name: 'Warning', value: warnings },
  ].filter(d => d.value > 0)
  const PIE_COLORS = ['#4caf50', '#f44336', '#ff9800']

  const statusColor = (status) => {
    if (status === 'pass' || status === 'passed') return '#4caf50'
    if (status === 'fail' || status === 'failed') return '#f44336'
    if (status === 'warning') return '#ff9800'
    return '#94a3b8'
  }

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
    border: '1px solid #e5e7eb',
  }

  const kpiStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    minWidth: 150,
    flex: 1,
  })

  const suiteData = suites?.suites || suites || {}

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          🔍 Deepchecks Data & Model Validation
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          Validation via <b>deepchecks</b>
          {overview.library_version ? ` v${overview.library_version}` : ''}
          {overview.file ? ` — ${overview.file}` : ''}
          {overview.n_checks != null ? ` · ${overview.n_checks} checks` : ''}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiStyle(kpi.color)}>
            <div style={{ fontSize: 18, marginBottom: 4 }}>{kpi.icon}</div>
            <div style={{ fontSize: 26, fontWeight: 700, color: kpi.color }}>
              {kpi.value}
            </div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{kpi.label}</div>
          </div>
        ))}
      </div>

      {/* Check Results Table + Pie Chart */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Check Results Table */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Check Results</h3>
          <div style={{ maxHeight: 360, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Check Name</th>
                  <th style={{ textAlign: 'center', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Status</th>
                  <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Value</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Details</th>
                </tr>
              </thead>
              <tbody>
                {checks.map((check, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {check.name || check.check_name || `Check ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <span style={{
                        display: 'inline-block',
                        padding: '2px 10px',
                        borderRadius: 12,
                        fontSize: 11,
                        fontWeight: 600,
                        color: '#fff',
                        background: statusColor(check.status),
                      }}>
                        {(check.status || 'unknown').toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', color: '#334155', fontFamily: 'monospace' }}>
                      {check.value != null ? (typeof check.value === 'number' ? check.value.toFixed(4) : String(check.value)) : '—'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#64748b', fontSize: 12, maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {check.details || check.detail || '—'}
                    </td>
                  </tr>
                ))}
                {checks.length === 0 && (
                  <tr>
                    <td colSpan={4} style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No check results</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Pass/Fail Pie Chart */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Pass / Fail Distribution</h3>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {pieData.map((entry, idx) => (
                    <Cell key={idx} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No results to chart</div>
          )}
        </div>
      </div>

      {/* Suite Breakdown */}
      {suiteData && Object.keys(suiteData).length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>Suite Breakdown</h3>
          <div style={{ display: 'grid', gap: 16 }}>
            {Object.entries(suiteData).map(([suiteName, suiteInfo]) => {
              const suiteChecks = suiteInfo?.checks || suiteInfo || []
              const checksList = Array.isArray(suiteChecks) ? suiteChecks : []
              return (
                <div key={suiteName} style={{ background: '#f8fafc', borderRadius: 8, padding: 16, border: '1px solid #e2e8f0' }}>
                  <div style={{ fontWeight: 600, color: '#1e293b', fontSize: 14, marginBottom: 10, textTransform: 'capitalize' }}>
                    {suiteName.replace(/_/g, ' ')}
                    {suiteInfo?.description && (
                      <span style={{ fontWeight: 400, color: '#64748b', fontSize: 12, marginLeft: 10 }}>
                        — {suiteInfo.description}
                      </span>
                    )}
                  </div>
                  {checksList.length > 0 ? (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                      {checksList.map((chk, j) => {
                        const name = typeof chk === 'string' ? chk : (chk.name || chk.check_name || `Check ${j + 1}`)
                        const st = typeof chk === 'object' ? chk.status : null
                        return (
                          <span key={j} style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 5,
                            background: '#fff',
                            border: '1px solid #e2e8f0',
                            borderRadius: 6,
                            padding: '4px 10px',
                            fontSize: 12,
                            color: '#334155',
                          }}>
                            {st && (
                              <span style={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                background: statusColor(st),
                                display: 'inline-block',
                              }} />
                            )}
                            {name}
                          </span>
                        )
                      })}
                    </div>
                  ) : (
                    <div style={{ color: '#94a3b8', fontSize: 12 }}>No checks in this suite</div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '▾ Hide' : '▸ Show'} Check Definitions & Clinical Relevance
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Check Name</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Clinical Relevance</th>
                </tr>
              </thead>
              <tbody>
                {(Array.isArray(defs) ? defs : Object.values(defs).filter(d => typeof d === 'object' && d.name)).map((def, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {def.name || def.check_name || `Check ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569', lineHeight: 1.4 }}>
                      {def.description || '—'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#1e88e5', fontSize: 12 }}>
                      {def.clinical_relevance || '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
