import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'

const STAGE_COLORS = {
  active: '#22c55e',
  flagged: '#eab308',
  approved: '#f97316',
  archived: '#8b5cf6',
  audit_closed: '#94a3b8'
}

const REASON_COLORS = {
  age: '#3b82f6',
  drift: '#ef4444',
  accuracy: '#f97316'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

function StageBadge({ stage }) {
  const color = STAGE_COLORS[stage] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{stage || 'unknown'}</span>
  )
}

export default function ModelRetirementDashboard() {
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
          axios.get(`${API_URL}/model-retirement/overview`),
          axios.get(`${API_URL}/model-retirement/breakdown`),
          axios.get(`${API_URL}/model-retirement/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Model Retirement data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No model retirement data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'models', label: 'Models' },
    { id: 'analytics', label: 'Analytics' },
    { id: 'history', label: 'History' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const stageSummaryData = overview?.stage_summary
    ? overview.stage_summary.map(s => ({
        name: s.stage, value: s.count, color: STAGE_COLORS[s.stage] || '#94a3b8'
      }))
    : []

  const accuracyDistData = overview?.accuracy_distribution
    ? overview.accuracy_distribution.map(b => ({ name: b.bucket, count: b.count }))
    : []

  /* Breakdown data prep */
  const timeline = breakdown?.retirement_timeline || []
  const accuracyDrift = breakdown?.accuracy_vs_drift || []
  const sizeComparison = breakdown?.model_size_comparison || []
  const ageDistribution = breakdown?.age_distribution || []
  const stageProgression = breakdown?.stage_progression || []
  const gitHistory = breakdown?.git_model_history || []

  /* Reason breakdown */
  const reasonCounts = {}
  timeline.forEach(m => {
    const reason = (m.retirement_reason || '').split(' ')[0]
    const key = reason.startsWith('age') ? 'age' : reason.startsWith('drift') ? 'drift' : reason.startsWith('accuracy') ? 'accuracy' : 'other'
    reasonCounts[key] = (reasonCounts[key] || 0) + 1
  })
  const reasonData = Object.entries(reasonCounts).map(([k, v]) => ({
    name: k.charAt(0).toUpperCase() + k.slice(1), value: v, color: REASON_COLORS[k] || '#94a3b8'
  }))

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Model Retirement Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          ML model lifecycle management — staleness, drift, accuracy degradation, and retirement workflow
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Models" value={fmt(overview.total_models)} />
              <KPI label="Active" value={fmt(overview.active_models)} color="#22c55e" />
              <KPI label="Flagged" value={fmt(overview.flagged_for_retirement)} color="#f97316" />
              <KPI label="Retirement Rate" value={`${fmt(overview.retirement_rate)}%`} color={overview.retirement_rate > 50 ? '#ef4444' : '#22c55e'} />
              <KPI label="Avg Age (days)" value={fmt(overview.avg_model_age_days)} />
            </div>
          </Card>

          <Card title="Oldest Model">
            <div style={{ textAlign: 'center', padding: 10 }}>
              <div style={{ fontSize: 18, fontWeight: 600, color: '#ef4444' }}>{overview.oldest_model?.name || '--'}</div>
              <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{fmt(overview.oldest_model?.age_days)} days old</div>
            </div>
          </Card>
          <Card title="Newest Model">
            <div style={{ textAlign: 'center', padding: 10 }}>
              <div style={{ fontSize: 18, fontWeight: 600, color: '#22c55e' }}>{overview.newest_model?.name || '--'}</div>
              <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>{fmt(overview.newest_model?.age_days)} days old</div>
            </div>
          </Card>
          <Card title="Retirement Reasons">
            {reasonData.length > 0 ? (
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={reasonData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={65} label={({ name, value }) => `${name}: ${value}`}>
                    {reasonData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>

          <Card title="Stage Distribution" span={2}>
            {stageSummaryData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={stageSummaryData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="value" name="Models">
                    {stageSummaryData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>
          <Card title="Accuracy Distribution">
            {accuracyDistData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={accuracyDistData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>
        </div>
      )}

      {/* Tab 2: Models */}
      {tab === 'models' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="All Models" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Model</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Disease</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Age (d)</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Accuracy</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Drift</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Size (KB)</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Stage</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview?.models || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{m.name}</td>
                      <td style={{ padding: '8px 12px' }}>{m.disease}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: m.age_days > 90 ? '#ef4444' : m.age_days > 30 ? '#f97316' : '#22c55e' }}>{fmt(m.age_days)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: m.accuracy != null && m.accuracy < 0.8 ? '#ef4444' : '#1e293b' }}>{m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{m.drift_status || '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(m.file_size_kb)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><StageBadge stage={m.retirement_stage} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{m.retirement_reason || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Stage Progression" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Model</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Disease</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Age (d)</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Accuracy</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Stage</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {stageProgression.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{m.name}</td>
                      <td style={{ padding: '8px 12px' }}>{m.disease}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(m.age_days)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><StageBadge stage={m.stage} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{m.reason || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Analytics */}
      {tab === 'analytics' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Model Size Comparison">
            {sizeComparison.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={sizeComparison} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={120} />
                  <Tooltip formatter={(v) => `${v} KB`} />
                  <Bar dataKey="size_kb" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>

          <Card title="Age Distribution">
            {ageDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={ageDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>

          <Card title="Accuracy vs Drift" span={2}>
            {accuracyDrift.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Model</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Disease</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Accuracy</th>
                      <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Drift Frac</th>
                      <th style={{ textAlign: 'center', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Stage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {accuracyDrift.map((m, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 500 }}>{m.name}</td>
                        <td style={{ padding: '8px 12px' }}>{m.disease}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right' }}>{m.accuracy != null ? (m.accuracy * 100).toFixed(1) + '%' : '--'}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: m.drift_frac > 0.5 ? '#ef4444' : '#22c55e' }}>{m.drift_frac != null ? (m.drift_frac * 100).toFixed(0) + '%' : '--'}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'center' }}><StageBadge stage={m.stage} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No data</div>}
          </Card>
        </div>
      )}

      {/* Tab 4: History */}
      {tab === 'history' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Git Model History">
            {gitHistory.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Hash</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Date</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Author</th>
                      <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gitHistory.map((h, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 12 }}>{h.hash}</td>
                        <td style={{ padding: '8px 12px' }}>{h.date}</td>
                        <td style={{ padding: '8px 12px' }}>{h.author}</td>
                        <td style={{ padding: '8px 12px', fontSize: 12, color: '#334155', maxWidth: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{h.message}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No git history available</div>}
          </Card>

          <Card title="Training History (Recent Events)">
            {(breakdown?.training_history || []).length > 0 ? (
              <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                {breakdown.training_history.slice(0, 20).map((ev, i) => (
                  <div key={i} style={{ padding: '8px 12px', borderBottom: '1px solid #f1f5f9', fontSize: 12 }}>
                    <span style={{ color: '#64748b', fontFamily: 'monospace' }}>{ev.ts}</span>
                    <span style={{ marginLeft: 8, padding: '1px 6px', borderRadius: 4, fontSize: 10, fontWeight: 600, background: '#e0f2fe', color: '#0369a1' }}>{ev.level}</span>
                    <span style={{ marginLeft: 8, color: '#334155' }}>{ev.event}</span>
                  </div>
                ))}
              </div>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 20 }}>No training history</div>}
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Retirement Stages">
            <div style={{ display: 'grid', gap: 12 }}>
              {(defs.stages || []).map((s, i) => (
                <div key={i} style={{ padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                  <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{s.stage}</div>
                  <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{s.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Retirement Criteria">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Criterion</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Threshold</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Source</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b', fontWeight: 600 }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.retirement_criteria || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{c.criterion}</td>
                      <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: '#ef4444' }}>{c.threshold}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{c.source}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{c.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Metrics Glossary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 10 }}>
              {(defs.metrics || []).map((m, i) => (
                <div key={i} style={{ padding: '10px 14px', background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{m.term}</div>
                  <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.4 }}>{m.definition}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Clinical Significance">
            <div style={{ display: 'grid', gap: 10 }}>
              {(defs.clinical_significance || []).map((c, i) => (
                <div key={i} style={{ padding: '10px 14px', background: '#fef2f2', borderRadius: 8, borderLeft: '3px solid #ef4444' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#991b1b', marginBottom: 2 }}>{c.aspect}</div>
                  <div style={{ fontSize: 12, color: '#7f1d1d', lineHeight: 1.4 }}>{c.description}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
