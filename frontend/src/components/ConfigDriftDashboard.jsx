import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, AreaChart, Area, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16'
]

const sevColor = (sev) => {
  if (sev === 'critical') return '#ef4444'
  if (sev === 'high') return '#f97316'
  if (sev === 'medium') return '#f59e0b'
  if (sev === 'low') return '#6366f1'
  return '#8b5cf6'
}

const healthColor = (score) => {
  if (score >= 80) return '#10b981'
  if (score >= 60) return '#f59e0b'
  return '#ef4444'
}

const statusColor = (status) => {
  if (status === 'drifted') return '#ef4444'
  if (status === 'warning') return '#f59e0b'
  return '#10b981'
}

const card = {
  background: '#ffffff',
  borderRadius: 12,
  padding: 20,
  boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
  marginBottom: 16,
}

const badge = (bg) => ({
  display: 'inline-block',
  padding: '2px 10px',
  borderRadius: 12,
  fontSize: 12,
  fontWeight: 600,
  color: '#fff',
  background: bg,
})

export default function ConfigDriftDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoints = {
      overview: '/api/config-drift/overview',
      drifts: '/api/config-drift/breakdown',
      files: '/api/config-drift/breakdown',
      env: '/api/config-drift/breakdown',
      definitions: '/api/config-drift/definitions',
    }
    const url = endpoints[tab] || endpoints.overview
    axios.get(API + url)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'definitions') setDefinitions(r.data)
        else setBreakdown(r.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'drifts', label: 'Drift Details' },
    { id: 'files', label: 'Config Files' },
    { id: 'env', label: 'Env Audit' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const renderOverview = () => {
    if (!overview) return null
    const d = overview
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Health Score</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: healthColor(d.health_score) }}>{d.health_score}</div>
            <span style={badge(healthColor(d.health_score))}>{d.health_status}</span>
          </div>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Config Files</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: '#6366f1' }}>{d.total_config_files}</div>
          </div>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Total Drifts</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: d.total_drifts > 0 ? '#ef4444' : '#10b981' }}>{d.total_drifts}</div>
          </div>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Critical</div>
            <div style={{ fontSize: 36, fontWeight: 700, color: '#ef4444' }}>{d.critical_drifts}</div>
          </div>
        </div>

        {/* Severity Distribution */}
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>Drift Severity Distribution</h4>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={d.severity_distribution.filter(s => s.count > 0)} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label={({ severity, count }) => `${severity}: ${count}`}>
                {d.severity_distribution.map((s, i) => (
                  <Cell key={s.severity} fill={sevColor(s.severity)} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          {d.severity_distribution.every(s => s.count === 0) && (
            <div style={{ textAlign: 'center', color: '#10b981', fontWeight: 600, padding: 20 }}>No drifts detected</div>
          )}
        </div>

        {/* Change Trend */}
        {d.change_trend && d.change_trend.length > 0 && (
          <div style={card}>
            <h4 style={{ marginTop: 0 }}>Config Change Trend (7 days)</h4>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={d.change_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="changes" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Category Summary */}
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>Category Summary</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Category</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Files</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Drifts</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Git Tracked</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.values(d.category_summary).map(cat => (
                <tr key={cat.name} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8 }}>{cat.name}</td>
                  <td style={{ textAlign: 'center', padding: 8 }}>{cat.file_count}</td>
                  <td style={{ textAlign: 'center', padding: 8, color: cat.drift_count > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{cat.drift_count}</td>
                  <td style={{ textAlign: 'center', padding: 8 }}>{cat.git_tracked}</td>
                  <td style={{ textAlign: 'center', padding: 8 }}><span style={badge(statusColor(cat.status))}>{cat.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Recent Changes */}
        {d.recent_changes && d.recent_changes.length > 0 && (
          <div style={card}>
            <h4 style={{ marginTop: 0 }}>Recent Config Changes</h4>
            {d.recent_changes.slice(0, 8).map((ch, i) => (
              <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f3f4f6' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <code style={{ fontSize: 12, color: '#6366f1' }}>{ch.commit}</code>
                  <span style={{ fontSize: 11, color: '#9ca3af' }}>{ch.date}</span>
                </div>
                <div style={{ fontSize: 13, marginTop: 2 }}>{ch.message}</div>
                {ch.files && ch.files.length > 0 && (
                  <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>{ch.files.join(', ')}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  const renderDrifts = () => {
    if (!breakdown) return null
    const drifts = breakdown.drifts || []
    if (drifts.length === 0) {
      return (
        <div style={card}>
          <div style={{ textAlign: 'center', padding: 40, color: '#10b981' }}>
            <div style={{ fontSize: 48 }}>&#10003;</div>
            <div style={{ fontSize: 18, fontWeight: 600, marginTop: 8 }}>No Config Drift Detected</div>
            <div style={{ fontSize: 13, color: '#6b7280', marginTop: 4 }}>All configuration files are in sync with version control</div>
          </div>
        </div>
      )
    }
    return (
      <div>
        <div style={{ ...card, background: '#fef2f2', borderLeft: '4px solid #ef4444' }}>
          <strong>{drifts.length}</strong> drift(s) detected across config files
        </div>
        {drifts.map((d, i) => (
          <div key={i} style={{ ...card, borderLeft: `4px solid ${sevColor(d.severity)}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <code style={{ fontSize: 13, fontWeight: 600 }}>{d.path}</code>
              <span style={badge(sevColor(d.severity))}>{d.severity}</span>
            </div>
            <div style={{ fontSize: 13, color: '#374151' }}>{d.description}</div>
            <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: 12, color: '#6b7280' }}>
              <span>Type: <strong>{d.type}</strong></span>
              <span>Category: {d.category}</span>
              <span>Hash: <code>{d.hash}</code></span>
            </div>
            <div style={{ marginTop: 8, fontSize: 12, color: '#6366f1', fontStyle: 'italic' }}>{d.recommendation}</div>
          </div>
        ))}
      </div>
    )
  }

  const renderFiles = () => {
    if (!breakdown) return null
    const cats = breakdown.files_by_category || {}
    return (
      <div>
        <div style={card}>
          <div style={{ fontSize: 13, color: '#6b7280' }}>Total tracked: <strong>{breakdown.total_files}</strong></div>
        </div>
        {Object.entries(cats).map(([catId, files]) => (
          <div key={catId} style={card}>
            <h4 style={{ marginTop: 0, color: '#6366f1' }}>{files[0]?.category_name || catId} ({files.length})</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: 6 }}>File</th>
                  <th style={{ textAlign: 'center', padding: 6 }}>Size</th>
                  <th style={{ textAlign: 'center', padding: 6 }}>Hash</th>
                  <th style={{ textAlign: 'center', padding: 6 }}>Git</th>
                </tr>
              </thead>
              <tbody>
                {files.map(f => (
                  <tr key={f.path} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: 6 }}><code>{f.path}</code></td>
                    <td style={{ textAlign: 'center', padding: 6 }}>{(f.size_bytes / 1024).toFixed(1)}K</td>
                    <td style={{ textAlign: 'center', padding: 6 }}><code style={{ fontSize: 10 }}>{f.hash}</code></td>
                    <td style={{ textAlign: 'center', padding: 6 }}>{f.tracked_in_git ? '\u2705' : '\u274c'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    )
  }

  const renderEnv = () => {
    if (!breakdown) return null
    const audit = breakdown.env_audit || []
    const summary = breakdown.env_summary || {}
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12, marginBottom: 16 }}>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Total Vars</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#6366f1' }}>{summary.total}</div>
          </div>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Set</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#10b981' }}>{summary.set}</div>
          </div>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Using Default</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: '#f59e0b' }}>{summary.default}</div>
          </div>
          <div style={card}>
            <div style={{ fontSize: 13, color: '#6b7280' }}>Missing Required</div>
            <div style={{ fontSize: 28, fontWeight: 700, color: summary.missing_required > 0 ? '#ef4444' : '#10b981' }}>{summary.missing_required}</div>
          </div>
        </div>
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>Environment Variable Audit</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Variable</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Category</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Expected</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Actual</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {audit.map(v => (
                <tr key={v.name} style={{ borderBottom: '1px solid #f3f4f6' }}>
                  <td style={{ padding: 8 }}><code>{v.name}</code></td>
                  <td style={{ textAlign: 'center', padding: 8 }}><span style={badge('#6366f1')}>{v.category}</span></td>
                  <td style={{ padding: 8, fontSize: 12, color: '#6b7280' }}>{v.expected_default}</td>
                  <td style={{ padding: 8, fontSize: 12 }}>{v.actual}</td>
                  <td style={{ textAlign: 'center', padding: 8 }}>
                    <span style={badge(v.status === 'set' ? '#10b981' : v.status === 'missing' ? '#ef4444' : '#f59e0b')}>{v.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  const renderDefinitions = () => {
    if (!definitions) return null
    return (
      <div>
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>{definitions.title}</h4>
          {(definitions.terms || []).map(t => (
            <div key={t.term} style={{ padding: '8px 0', borderBottom: '1px solid #f3f4f6' }}>
              <strong style={{ color: '#6366f1' }}>{t.term}</strong>
              <div style={{ fontSize: 13, color: '#374151', marginTop: 2 }}>{t.definition}</div>
            </div>
          ))}
        </div>
        <div style={card}>
          <h4 style={{ marginTop: 0 }}>Severity Levels</h4>
          {(definitions.severity_levels || []).map(s => (
            <div key={s.level} style={{ padding: '8px 0', borderBottom: '1px solid #f3f4f6' }}>
              <span style={badge(sevColor(s.level))}>{s.level}</span>
              <span style={{ marginLeft: 12, fontSize: 13 }}>{s.description}</span>
              {s.examples && <div style={{ fontSize: 11, color: '#6b7280', marginTop: 2 }}>e.g. {s.examples.join(', ')}</div>}
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1100, margin: '0 auto' }}>
      <h2 style={{ marginBottom: 4 }}>Config Drift Monitor</h2>
      <p style={{ color: '#6b7280', fontSize: 13, marginTop: 0, marginBottom: 16 }}>
        Environment / config drift detection, versioning, and remediation
      </p>
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400,
            background: tab === t.id ? '#6366f1' : '#f3f4f6',
            color: tab === t.id ? '#fff' : '#374151',
            fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>
      {loading && <div style={{ textAlign: 'center', padding: 40, color: '#6b7280' }}>Loading...</div>}
      {error && <div style={{ ...card, background: '#fef2f2', color: '#ef4444' }}>Error: {error}</div>}
      {!loading && !error && tab === 'overview' && renderOverview()}
      {!loading && !error && tab === 'drifts' && renderDrifts()}
      {!loading && !error && tab === 'files' && renderFiles()}
      {!loading && !error && tab === 'env' && renderEnv()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
