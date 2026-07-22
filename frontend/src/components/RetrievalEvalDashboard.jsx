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

export default function RetrievalEvalDashboard() {
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
          axios.get(`${API_URL}/api/retrieval-eval/overview`),
          axios.get(`${API_URL}/api/retrieval-eval/breakdown`),
          axios.get(`${API_URL}/api/retrieval-eval/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load retrieval evaluation data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128269;</div>
      Loading retrieval evaluation data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Retrieval evaluation data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const cov = overview.coverage || {}
  const typeDist = overview.type_distribution || []
  const queueOps = overview.queue_operations || []
  const queueTimeline = overview.queue_timeline || []
  const patientCov = breakdown?.patient_coverage || []
  const documents = breakdown?.documents || []
  const analysisCov = breakdown?.analysis_coverage || {}
  const queueDetail = breakdown?.queue_detail || []

  const sb = scoreBadge(s.quality_score)

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
    { label: 'Quality Score', value: `${s.quality_score}%`, color: sb.color },
    { label: 'Coverage', value: `${s.coverage_pct}%`, color: s.coverage_pct >= 80 ? '#10b981' : '#f59e0b' },
    { label: 'Embeddings', value: s.total_embeddings, color: '#3b82f6' },
    { label: 'Queue Entries', value: s.total_queue_entries, color: '#8b5cf6' },
    { label: 'Queue Health', value: `${s.queue_health_pct}%`, color: s.queue_health_pct >= 50 ? '#10b981' : '#ef4444' },
    { label: 'Type Diversity', value: `${s.type_diversity_pct}%`, color: s.type_diversity_pct >= 70 ? '#10b981' : '#f59e0b' },
    { label: 'Collections', value: s.collections, color: '#06b6d4' },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'coverage', label: 'Coverage Detail' },
    { id: 'documents', label: 'Documents' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const coveredCount = patientCov.filter(p => p.has_vectors).length
  const missingCount = patientCov.filter(p => !p.has_vectors).length

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Retrieval Evaluation Dashboard</h2>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>ChromaDB + clinical.db</span>
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

          {/* Charts row: Type Distribution + Queue Operations */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Embedding Type Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={typeDist} dataKey="count" nameKey="type" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                    label={({ type, count }) => `${type}: ${count}`}>
                    {typeDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Queue Operations</h4>
              {queueOps.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={queueOps}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="operation" fontSize={11} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No queue data</div>
              )}
            </div>
          </div>

          {/* Coverage summary card */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Patient Coverage Summary</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#3b82f6' }}>{cov.clinical_patients}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Clinical Patients</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#10b981' }}>{cov.vectorized_patients}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Vectorized Patients</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#06b6d4' }}>{cov.clinical_analyses}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Analyses</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 700, color: '#8b5cf6' }}>{cov.clinical_uploads}</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>Uploads</div>
              </div>
            </div>
            {cov.missing_from_vector?.length > 0 && (
              <div style={{ marginTop: 12, padding: 10, background: '#fef2f2', borderRadius: 6, fontSize: 12, color: '#991b1b' }}>
                Missing from vector store: {cov.missing_from_vector.join(', ')}
              </div>
            )}
            {cov.extra_in_vector?.length > 0 && (
              <div style={{ marginTop: 8, padding: 10, background: '#eff6ff', borderRadius: 6, fontSize: 12, color: '#1e40af' }}>
                Extra in vector (no clinical record): {cov.extra_in_vector.join(', ')}
              </div>
            )}
          </div>

          {/* Queue timeline */}
          {queueTimeline.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Ingestion Queue Timeline</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={queueTimeline}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={10} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* ═══ COVERAGE DETAIL TAB ═══ */}
      {tab === 'coverage' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>
              Per-Patient Vector Coverage ({coveredCount} covered, {missingCount} missing)
            </h4>
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Vectors</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Types</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {patientCov.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>
                        {p.patient_id} {p.name ? `(${p.name})` : ''}
                      </td>
                      <td style={{ padding: '6px 10px' }}>{p.disease || '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{p.vector_count}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11 }}>
                        {p.types?.map(t => `${t.type}(${t.count})`).join(', ') || '--'}
                      </td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
                          background: p.has_vectors ? '#10b98122' : '#ef444422',
                          color: p.has_vectors ? '#10b981' : '#ef4444'
                        }}>
                          {p.has_vectors ? 'Covered' : 'Missing'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Analysis coverage */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>
              Analysis Coverage ({analysisCov.total_analyses || 0} analyses)
            </h4>
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>ID</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Label</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Confidence</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(analysisCov.vectorized_analyses || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{a.id}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{a.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{a.disease}</td>
                      <td style={{ padding: '6px 10px' }}>{a.label}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <span style={{
                          fontWeight: 600,
                          color: a.confidence >= 0.8 ? '#10b981' : a.confidence >= 0.5 ? '#f59e0b' : '#ef4444'
                        }}>
                          {a.confidence != null ? (a.confidence * 100).toFixed(1) + '%' : '--'}
                        </span>
                      </td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{a.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ═══ DOCUMENTS TAB ═══ */}
      {tab === 'documents' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>
              Embedded Documents ({documents.length})
            </h4>
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc', position: 'sticky', top: 0 }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>ID</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Type</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Document Preview</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.id}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{d.patient_id || '--'}</td>
                      <td style={{ padding: '6px 10px' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 11,
                          background: '#3b82f622', color: '#3b82f6', fontWeight: 600
                        }}>
                          {d.type}
                        </span>
                      </td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {d.document || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Queue detail */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Recent Queue Entries (last 50)</h4>
            <div style={{ maxHeight: 300, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Seq</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Operation</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Topic</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {queueDetail.map((q, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{q.seq_id}</td>
                      <td style={{ padding: '6px 10px' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 10, fontSize: 11,
                          background: q.operation === 'ADD' ? '#10b98122' : '#f59e0b22',
                          color: q.operation === 'ADD' ? '#10b981' : '#f59e0b', fontWeight: 600
                        }}>
                          {q.operation}
                        </span>
                      </td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{q.topic || '--'}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{q.created_at || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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
          <h4 style={{ margin: '16px 0 8px', color: '#334155' }}>Data Sources</h4>
          {defs.data_sources && Object.entries(defs.data_sources).map(([key, desc]) => (
            <div key={key} style={{ marginBottom: 8, fontSize: 12 }}>
              <span style={{ fontWeight: 600, color: '#3b82f6' }}>{key}:</span>{' '}
              <span style={{ color: '#64748b' }}>{desc}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
