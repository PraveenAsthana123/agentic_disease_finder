import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#607d8b']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: decimals }) : String(v)
}

function freshnessColor(score) {
  if (score >= 70) return '#4caf50'
  if (score >= 40) return '#ff9800'
  return '#f44336'
}

function riskColor(level) {
  const l = (level || '').toLowerCase()
  if (l === 'fresh') return '#4caf50'
  if (l === 'aging') return '#ff9800'
  if (l === 'stale') return '#f44336'
  if (l === 'critical') return '#8b0000'
  return '#64748b'
}

function priorityColor(p) {
  const l = (p || '').toLowerCase()
  if (l === 'high') return '#f44336'
  if (l === 'medium') return '#ff9800'
  return '#1e88e5'
}

export default function ContentFreshnessDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedType, setExpandedType] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, bd, df] = await Promise.all([
          axios.get(`${API_URL}/api/content-freshness/overview`),
          axios.get(`${API_URL}/api/content-freshness/breakdown`),
          axios.get(`${API_URL}/api/content-freshness/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load Content Freshness data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128337;</div>
      Loading Content Freshness data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Content Freshness data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const freshnessByType = overview.freshness_by_type || []
  const stalenessDistribution = overview.staleness_distribution || []
  const decayRisks = overview.decay_risks || []
  const queueStats = overview.queue_stats || {}
  const ingestionTimeline = overview.ingestion_timeline || []
  const patientFreshness = breakdown?.patient_freshness || []
  const typeDetails = breakdown?.type_details || []
  const refreshRecommendations = breakdown?.refresh_recommendations || []
  const updateActivity = breakdown?.update_activity || []
  const definitions = defs?.definitions || []

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'documents', label: 'Documents' },
    { id: 'activity', label: 'Activity' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const kpis = [
    { label: 'Total Documents', value: fmt(s.total_documents), color: COLORS[0] },
    { label: 'Document Types', value: fmt(s.document_types), color: COLORS[1] },
    { label: 'Avg Age (hrs)', value: fmt(s.avg_age_hours, 1), color: COLORS[3] },
    { label: 'Freshness Score', value: fmt(s.freshness_score, 1), color: freshnessColor(s.freshness_score) }
  ]

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        <span style={{ fontSize: 22 }}>&#128337;</span>
        <h2 style={{ margin: 0, fontSize: 18 }}>Content Freshness Dashboard</h2>
        <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 'auto' }}>
          RAG document freshness &middot; {overview.generated_at}
        </span>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '2px solid #e5e7eb', paddingBottom: 2 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              padding: '6px 14px', fontSize: 12, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : 'transparent',
              color: tab === t.id ? '#fff' : '#64748b',
              border: 'none', borderRadius: '6px 6px 0 0', cursor: 'pointer'
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10, marginBottom: 18 }}>
            {kpis.map((k, i) => (
              <div key={i} style={{
                background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 12px',
                borderLeft: `3px solid ${k.color}`
              }}>
                <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>{k.label}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#1e293b', marginTop: 2 }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Freshness by Type + Staleness Distribution side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 14 }}>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Freshness by Type</h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={freshnessByType}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="type" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Bar dataKey="freshness_score" radius={[2, 2, 0, 0]}>
                    {freshnessByType.map((entry, i) => (
                      <Cell key={i} fill={freshnessColor(entry.freshness_score)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Staleness Distribution</h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={stalenessDistribution} dataKey="count" nameKey="bucket" cx="50%" cy="50%" outerRadius={70}
                    label={({ bucket, count }) => `${bucket}: ${count}`}>
                    {stalenessDistribution.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Decay Risk table */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Decay Risk</h3>
            {decayRisks.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>No decay risk data available</div>
            ) : (
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Risk Level</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Count</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Threshold</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Doc Types</th>
                  </tr>
                </thead>
                <tbody>
                  {decayRisks.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '4px 8px', fontWeight: 600, color: riskColor(r.risk_level) }}>
                        {r.risk_level}
                      </td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmt(r.count)}</td>
                      <td style={{ padding: '4px 8px', fontSize: 10, fontFamily: 'monospace' }}>{r.threshold || '--'}</td>
                      <td style={{ padding: '4px 8px', fontSize: 10 }}>
                        {Array.isArray(r.doc_types) ? r.doc_types.join(', ') : (r.doc_types || '--')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Queue Stats + Ingestion Timeline side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 14, marginBottom: 14 }}>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Queue Stats</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{
                  background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: 10
                }}>
                  <div style={{ fontSize: 10, color: '#1e40af' }}>Total Queued</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#1e293b' }}>{fmt(queueStats.total_queued)}</div>
                </div>
                <div style={{
                  background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: 10
                }}>
                  <div style={{ fontSize: 10, color: '#1e40af' }}>Pending Operations</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#1e293b' }}>{fmt(queueStats.pending_operations)}</div>
                </div>
              </div>
            </div>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Ingestion Timeline</h3>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={ingestionTimeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="date" tick={{ fontSize: 9 }} tickFormatter={d => d ? d.slice(5) : ''} />
                  <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="count" stroke="#1e88e5" strokeWidth={2} dot={{ r: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Documents Tab */}
      {tab === 'documents' && (
        <>
          {/* Per-Patient Freshness */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Per-Patient Freshness</h3>
            {patientFreshness.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>No patient freshness data available</div>
            ) : (
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Docs</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Avg Age (hrs)</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Freshness</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Doc Types</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Last Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {patientFreshness.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '4px 8px', fontWeight: 600, fontFamily: 'monospace', fontSize: 10 }}>{p.patient_id}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmt(p.doc_count)}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmt(p.avg_age_hours, 1)}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 700, color: freshnessColor(p.freshness_score) }}>
                        {fmt(p.freshness_score, 1)}
                      </td>
                      <td style={{ padding: '4px 8px', fontSize: 10 }}>
                        {Array.isArray(p.doc_types) ? p.doc_types.join(', ') : (p.doc_types || '--')}
                      </td>
                      <td style={{ padding: '4px 8px', fontSize: 10, color: '#94a3b8' }}>{p.last_updated || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Per-Type Detail (expandable) */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Per-Type Detail</h3>
            {typeDetails.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>No type detail data available</div>
            ) : (
              typeDetails.map((td, ti) => (
                <div key={ti} style={{ marginBottom: 8 }}>
                  <button
                    onClick={() => setExpandedType(expandedType === td.type ? null : td.type)}
                    style={{
                      width: '100%', textAlign: 'left', padding: '8px 12px', fontSize: 12,
                      fontWeight: 600, background: expandedType === td.type ? '#e2e8f0' : '#f1f5f9',
                      border: '1px solid #e2e8f0', borderRadius: 6, cursor: 'pointer',
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                    }}
                  >
                    <span>{td.type} ({(td.documents || []).length} docs)</span>
                    <span style={{ fontSize: 10, color: '#64748b' }}>{expandedType === td.type ? '▲' : '▼'}</span>
                  </button>
                  {expandedType === td.type && (
                    <div style={{ padding: '8px 0' }}>
                      <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                            <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Embedding ID</th>
                            <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Patient ID</th>
                            <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Age (hrs)</th>
                            <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Content Preview</th>
                            <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Created At</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(td.documents || []).map((doc, di) => (
                            <tr key={di} style={{ borderBottom: '1px solid #f1f5f9' }}>
                              <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 10 }}>{doc.embedding_id || '--'}</td>
                              <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 10 }}>{doc.patient_id || '--'}</td>
                              <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmt(doc.age, 1)}</td>
                              <td style={{ padding: '4px 8px', fontSize: 10, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {doc.content_preview || '--'}
                              </td>
                              <td style={{ padding: '4px 8px', fontSize: 10, color: '#94a3b8' }}>{doc.created_at || '--'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Refresh Recommendations */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Refresh Recommendations</h3>
            {refreshRecommendations.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>No refresh recommendations</div>
            ) : (
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Reason</th>
                    <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Priority</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Affected</th>
                  </tr>
                </thead>
                <tbody>
                  {refreshRecommendations.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '4px 8px', fontWeight: 600 }}>{r.type}</td>
                      <td style={{ padding: '4px 8px' }}>{r.reason || '--'}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 700,
                          color: '#fff', background: priorityColor(r.priority)
                        }}>
                          {r.priority}
                        </span>
                      </td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmt(r.affected_count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}

      {/* Activity Tab */}
      {tab === 'activity' && (
        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
          <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Update Activity Timeline</h3>
          {updateActivity.length === 0 ? (
            <div style={{ color: '#94a3b8', fontSize: 12 }}>No update activity data available</div>
          ) : (
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Date</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px', color: '#64748b' }}>Transactions</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Components</th>
                </tr>
              </thead>
              <tbody>
                {updateActivity.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 10 }}>{a.date || '--'}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{fmt(a.transaction_count)}</td>
                    <td style={{ padding: '4px 8px', fontSize: 10 }}>
                      {Array.isArray(a.components) ? a.components.join(', ') : (a.components || '--')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && (
        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
          <h3 style={{ fontSize: 13, margin: '0 0 12px', color: '#334155' }}>Metric Definitions</h3>
          {definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 12, paddingBottom: 10, borderBottom: i < definitions.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
              <div style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{d.term}</div>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>{d.definition}</div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2, fontFamily: 'monospace' }}>Source: {d.source}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
