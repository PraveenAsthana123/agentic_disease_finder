import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function scoreBadge(score) {
  const color = score >= 80 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444'
  return { color, bg: `${color}18`, border: `${color}44` }
}

export default function RetrievalDashboard() {
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
          axios.get(`${API_URL}/api/retrieval/overview`),
          axios.get(`${API_URL}/api/retrieval/breakdown`),
          axios.get(`${API_URL}/api/retrieval/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load retrieval data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128269;</div>
      Loading retrieval data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Retrieval data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const vectorStore = overview.vector_store || {}
  const queryVolumeDaily = overview.query_volume_daily || []
  const queueOps = overview.queue_operations || []
  const recentQueries = breakdown?.recent_queries || []
  const queryTextAnalysis = breakdown?.query_text_analysis || []
  const queriesByPatient = breakdown?.queries_by_patient || []
  const patientRetrieval = breakdown?.patient_retrieval || []
  const typeDist = breakdown?.type_distribution || []
  const embeddingTimeline = breakdown?.embedding_timeline || []

  const readinessBadge = scoreBadge(s.retrieval_readiness_pct || 0)

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#1e40af' : '#f1f5f9', color: active ? '#fff' : '#64748b',
    border: 'none', fontSize: 13, marginRight: 4
  })

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 110
  })

  const kpiItems = [
    { label: 'Total Queries', value: s.total_queries, color: '#3b82f6' },
    { label: 'Unique Patients Queried', value: s.unique_patients_queried, color: '#10b981' },
    { label: 'Query Rate/Day', value: s.query_rate_per_day, color: '#f59e0b' },
    { label: 'Vector Store Size', value: s.vector_store_size, color: '#8b5cf6' },
    { label: 'Retrieval Readiness', value: s.retrieval_readiness_pct != null ? `${s.retrieval_readiness_pct}%` : '--', color: readinessBadge.color },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'queries', label: 'Queries' },
    { id: 'coverage', label: 'Coverage' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const coveredCount = patientRetrieval.filter(p => p.has_vectors).length
  const totalCount = patientRetrieval.length
  const coverageRate = totalCount > 0 ? ((coveredCount / totalCount) * 100).toFixed(1) : 0

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Retrieval Dashboard</h2>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>Patient-chat retrieval analytics</span>
      </div>

      {/* Tab bar */}
      <div style={{ marginBottom: 18 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ OVERVIEW TAB ═══ */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {kpiItems.map((k, i) => (
              <div key={i} style={kpiStyle(k.color)}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: k.color }}>
                  {typeof k.value === 'number' ? fmt(k.value) : (k.value || '--')}
                </div>
              </div>
            ))}
          </div>

          {/* Daily query volume line chart */}
          {queryVolumeDaily.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Daily Query Volume</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={queryVolumeDaily}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={10} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Vector store summary cards */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Vector Store Summary</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#3b82f6' }}>{fmt(vectorStore.embeddings)}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Embeddings</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#10b981' }}>{fmt(vectorStore.metadata)}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Metadata</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#f59e0b' }}>{fmt(vectorStore.queue)}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Queue</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#8b5cf6' }}>{fmt(vectorStore.collections)}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Collections</div>
              </div>
            </div>
          </div>

          {/* Queue operations bar chart */}
          {queueOps.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Queue Operations</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={queueOps}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="operation" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* ═══ QUERIES TAB ═══ */}
      {tab === 'queries' && (
        <>
          {/* Recent queries table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>
              Recent Queries ({recentQueries.length})
            </h4>
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>ID</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Query Text</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Actor</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {recentQueries.map((q, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{q.id}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{q.patient_id || '--'}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {q.query_text || '--'}
                      </td>
                      <td style={{ padding: '6px 10px' }}>{q.actor || '--'}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{q.timestamp || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts row: Query word frequency + Queries by patient */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Query Word Frequency (Top 20)</h4>
              {queryTextAnalysis.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={queryTextAnalysis.slice(0, 20)} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" fontSize={11} />
                    <YAxis type="category" dataKey="word" fontSize={10} width={80} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No query text data</div>
              )}
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Queries by Patient</h4>
              {queriesByPatient.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={queriesByPatient}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="patient_id" fontSize={10} angle={-30} textAnchor="end" height={60} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No patient query data</div>
              )}
            </div>
          </div>
        </>
      )}

      {/* ═══ COVERAGE TAB ═══ */}
      {tab === 'coverage' && (
        <>
          {/* Coverage rate KPI */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            <div style={kpiStyle(coveredCount === totalCount ? '#10b981' : '#f59e0b')}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Coverage Rate</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: coveredCount === totalCount ? '#10b981' : '#f59e0b' }}>
                {coverageRate}%
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>{coveredCount} / {totalCount} patients</div>
            </div>
          </div>

          {/* Per-patient retrieval table */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>
              Per-Patient Retrieval ({coveredCount} covered, {totalCount - coveredCount} missing)
            </h4>
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient ID</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Name</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Query Count</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Vector Count</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Has Vectors</th>
                  </tr>
                </thead>
                <tbody>
                  {patientRetrieval.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{p.name || '--'}</td>
                      <td style={{ padding: '6px 10px' }}>{p.disease || '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{fmt(p.query_count)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{fmt(p.vector_count)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                          background: p.has_vectors ? '#10b98122' : '#ef444422',
                          color: p.has_vectors ? '#10b981' : '#ef4444'
                        }}>
                          {p.has_vectors ? 'Yes' : 'No'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts row: Type distribution + Embedding timeline */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Type Distribution</h4>
              {typeDist.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={typeDist} dataKey="count" nameKey="type" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                      label={({ type, count }) => `${type}: ${count}`}>
                      {typeDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No type data</div>
              )}
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Embedding Timeline</h4>
              {embeddingTimeline.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={embeddingTimeline}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" fontSize={10} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Line type="monotone" dataKey="count" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No timeline data</div>
              )}
            </div>
          </div>
        </>
      )}

      {/* ═══ DEFINITIONS TAB ═══ */}
      {tab === 'definitions' && defs?.metrics && (
        <div style={cardStyle}>
          <h4 style={{ margin: '0 0 16px', color: '#334155' }}>Metric Definitions</h4>
          {Object.entries(defs.metrics).map(([key, m]) => (
            <div key={key} style={{ marginBottom: 16, paddingBottom: 14, borderBottom: '1px solid #f1f5f9' }}>
              <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{m.label}</div>
              <div style={{ fontSize: 13, color: '#475569', marginBottom: 4 }}>{m.description}</div>
              <div style={{ fontSize: 11, color: '#94a3b8', fontFamily: 'monospace' }}>Source: {m.source}</div>
            </div>
          ))}
          {defs.data_sources && (
            <>
              <h4 style={{ margin: '16px 0 8px', color: '#334155' }}>Data Sources</h4>
              {Object.entries(defs.data_sources).map(([key, desc]) => (
                <div key={key} style={{ marginBottom: 8, fontSize: 12 }}>
                  <span style={{ fontWeight: 600, color: '#3b82f6' }}>{key}:</span>{' '}
                  <span style={{ color: '#64748b' }}>{desc}</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}
