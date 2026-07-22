import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const TYPE_COLORS = {
  analysis: '#3b82f6',
  patient: '#22c55e',
  mri_findings: '#8b5cf6',
  neuropsych: '#f97316',
  medications: '#ef4444',
  survey: '#06b6d4',
  hitl_reviews: '#eab308',
}
const TYPE_LABELS = {
  analysis: 'Analysis',
  patient: 'Patient',
  mri_findings: 'MRI Findings',
  neuropsych: 'Neuropsych',
  medications: 'Medications',
  survey: 'Survey',
  hitl_reviews: 'HITL Reviews',
}
const PIE_COLORS = ['#3b82f6', '#22c55e', '#8b5cf6', '#f97316', '#ef4444', '#06b6d4', '#eab308', '#94a3b8']

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

function TypeBadge({ type }) {
  const color = TYPE_COLORS[type] || '#94a3b8'
  const label = TYPE_LABELS[type] || type || 'unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function RAGMetadataFilterDashboard() {
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
          axios.get(`${API_URL}/api/rag-metadata-filter/overview`),
          axios.get(`${API_URL}/api/rag-metadata-filter/breakdown`),
          axios.get(`${API_URL}/api/rag-metadata-filter/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading RAG Metadata Filter data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No metadata filter data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'types', label: 'Type Analysis' },
    { id: 'patients', label: 'Patient Matrix' },
    { id: 'queries', label: 'Query Filter' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const summary = overview?.summary || {}
  const typeDist = (overview?.type_distribution || []).map(d => ({
    ...d, name: TYPE_LABELS[d.type] || d.type, color: TYPE_COLORS[d.type] || '#94a3b8'
  }))
  const patientEmbCounts = overview?.patient_embedding_counts || []
  const typePatientSummary = breakdown?.type_patient_summary || []
  const patientMatrix = breakdown?.patient_type_matrix || []
  const recentEmbs = breakdown?.recent_embeddings || []
  const completeness = breakdown?.metadata_completeness || {}
  const filterQueries = breakdown?.filterable_queries || []
  const filterApplicability = breakdown?.filter_applicability || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>RAG Metadata Filter</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Metadata-driven retrieval filtering — type/patient dimensions, coverage, filter readiness
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: 600, cursor: 'pointer', border: 'none',
            borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            color: tab === t.id ? '#3b82f6' : '#64748b', background: 'none', marginBottom: -2,
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title="Key Metrics" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Total Embeddings" value={fmt(summary.total_embeddings)} color="#3b82f6" />
              <KPI label="Document Types" value={fmt(summary.document_types)} color="#8b5cf6" />
              <KPI label="Metadata Keys" value={fmt(summary.metadata_keys_count)} color="#06b6d4" />
              <KPI label="Patients Covered" value={fmt(summary.patients_with_vectors)} sub={`of ${fmt(summary.total_patients)}`} color="#22c55e" />
              <KPI label="Coverage" value={`${fmt(summary.coverage_pct)}%`} color="#f97316" />
              <KPI label="Filter Readiness" value={`${fmt(summary.filter_readiness_pct)}%`} sub="both dims" color="#10b981" />
            </div>
          </Card>

          <Card title="Type Distribution" span={2}>
            {typeDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={typeDist} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                    {typeDist.map((d, i) => <Cell key={i} fill={d.color || PIE_COLORS[i % PIE_COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No type data</div>}
          </Card>

          <Card title="Metadata Completeness">
            <div style={{ display: 'grid', gap: 12 }}>
              {[
                { label: 'Has patient_id', val: completeness.pct_patient_id, color: '#22c55e' },
                { label: 'Has type', val: completeness.pct_type, color: '#3b82f6' },
                { label: 'Has both', val: completeness.pct_both, color: '#10b981' },
              ].map(({ label, val, color }) => (
                <div key={label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                    <span style={{ color: '#334155' }}>{label}</span>
                    <span style={{ fontWeight: 600, color }}>{fmt(val)}%</span>
                  </div>
                  <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8 }}>
                    <div style={{ background: color, borderRadius: 4, height: 8, width: `${val || 0}%`, transition: 'width 0.3s' }} />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Patient Embedding Distribution" span={3}>
            {patientEmbCounts.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={patientEmbCounts.slice(0, 25)} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="embedding_count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No patient data</div>}
          </Card>
        </div>
      )}

      {/* ── Type Analysis Tab ── */}
      {tab === 'types' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Embeddings by Type" span={1}>
            {typePatientSummary.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={typePatientSummary} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={80} />
                  <Tooltip />
                  <Bar dataKey="total_embeddings" fill="#3b82f6" radius={[0, 4, 4, 0]} name="Embeddings" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          <Card title="Patient Coverage by Type" span={1}>
            {typePatientSummary.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={typePatientSummary} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="type" tick={{ fontSize: 11 }} width={80} />
                  <Tooltip />
                  <Bar dataKey="patients" fill="#22c55e" radius={[0, 4, 4, 0]} name="Patients" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          <Card title="Type-Patient Summary" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Embeddings</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Patients</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#64748b' }}>Avg/Patient</th>
                  </tr>
                </thead>
                <tbody>
                  {typePatientSummary.map((t, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}><TypeBadge type={t.type} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>{fmt(t.total_embeddings)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(t.patients)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: '#64748b' }}>{(t.total_embeddings / Math.max(t.patients, 1)).toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Patient Matrix Tab ── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Patient Metadata Matrix (Top 20)" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Name</th>
                    {Object.keys(TYPE_LABELS).map(t => (
                      <th key={t} style={{ textAlign: 'center', padding: '8px 6px', color: TYPE_COLORS[t] || '#64748b', fontSize: 10 }}>
                        {TYPE_LABELS[t]}
                      </th>
                    ))}
                    <th style={{ textAlign: 'right', padding: '8px 10px', color: '#64748b' }}>Total</th>
                  </tr>
                </thead>
                <tbody>
                  {patientMatrix.map((pm, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{pm.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{pm.patient_name || ''}</td>
                      {Object.keys(TYPE_LABELS).map(t => {
                        const val = pm.types?.[t] || 0
                        return (
                          <td key={t} style={{ textAlign: 'center', padding: '6px', color: val > 0 ? TYPE_COLORS[t] : '#e2e8f0', fontWeight: val > 0 ? 600 : 400 }}>
                            {val > 0 ? val : '-'}
                          </td>
                        )
                      })}
                      <td style={{ textAlign: 'right', padding: '6px 10px', fontWeight: 700 }}>{pm.total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Recent Embeddings" span={1}>
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>ID</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {recentEmbs.map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 10 }}>{e.embedding_id?.slice(0, 12) || e.id}</td>
                      <td style={{ padding: '6px 10px' }}>{e.patient_id || '--'}</td>
                      <td style={{ padding: '6px 10px' }}><TypeBadge type={e.type} /></td>
                      <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{e.created_at || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Query Filter Tab ── */}
      {tab === 'queries' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title="Filter Applicability" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Queries Sampled" value={fmt(filterApplicability.total_queries_sampled)} color="#3b82f6" />
              <KPI label="Filter Applicable" value={fmt(filterApplicability.filter_applicable)} color="#22c55e" />
              <KPI label="Applicability Rate" value={`${fmt(filterApplicability.pct_applicable)}%`} color="#10b981" />
            </div>
          </Card>

          <Card title="Recent Queries with Filter Analysis" span={3}>
            <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Query</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Patient Filter</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Type Filters</th>
                    <th style={{ textAlign: 'center', padding: '8px 10px', color: '#64748b' }}>Filterable</th>
                    <th style={{ textAlign: 'left', padding: '8px 10px', color: '#64748b' }}>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {filterQueries.map((q, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 10 }}>{q.patient_id || '--'}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{q.query_text || '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: q.has_patient_filter ? '#22c55e' : '#e2e8f0' }} />
                      </td>
                      <td style={{ padding: '6px 10px' }}>
                        {q.detected_type_filters?.length > 0
                          ? q.detected_type_filters.map((t, j) => <TypeBadge key={j} type={t} />)
                          : <span style={{ color: '#94a3b8' }}>none</span>}
                      </td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 10, fontWeight: 600,
                          background: q.filter_applicable ? '#22c55e22' : '#ef444422',
                          color: q.filter_applicable ? '#22c55e' : '#ef4444'
                        }}>{q.filter_applicable ? 'Yes' : 'No'}</span>
                      </td>
                      <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 11 }}>{q.timestamp || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Metrics" span={1}>
            {(defs?.metrics || []).map((m, i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{m.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{m.description}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Source: {m.source}</div>
              </div>
            ))}
          </Card>

          <Card title="Filter Dimensions" span={1}>
            {(defs?.filter_dimensions || []).map((d, i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.key}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{d.description}</div>
                <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 2, fontStyle: 'italic' }}>{d.example}</div>
              </div>
            ))}
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {Object.entries(defs?.glossary || {}).map(([term, desc], i) => (
                <div key={i}>
                  <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{term}: </span>
                  <span style={{ fontSize: 12, color: '#64748b' }}>{desc}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
