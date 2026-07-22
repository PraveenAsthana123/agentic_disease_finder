import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function CitationDashboard() {
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
          axios.get(`${API_URL}/api/citation/overview`),
          axios.get(`${API_URL}/api/citation/breakdown`),
          axios.get(`${API_URL}/api/citation/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load citation data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8, animation: 'spin 1.5s linear infinite' }}>&#9881;</div>
      Loading citation data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Citation data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const docTypes = overview.citations_by_doc_type || []
  const dailyTrend = overview.daily_citation_trend || []
  const qualityDist = overview.quality_distribution || []
  const diseaseCov = breakdown?.disease_coverage || []
  const topPatients = breakdown?.top_cited_patients || []
  const componentRates = breakdown?.component_citation_rates || []
  const gapPatients = breakdown?.uncited_patients || []
  const documents = breakdown?.documents || []
  const docTypeDist = breakdown?.document_type_distribution || []
  const citationGaps = breakdown?.citation_gaps || []
  const definitions = defs?.definitions || defs?.metrics || []

  const cardStyle = { background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', marginBottom: 18 }
  const kpiStyle = { background: '#f8fafc', borderRadius: 10, padding: '14px 18px', minWidth: 140, textAlign: 'center' }
  const sectionTitle = { fontSize: 15, fontWeight: 700, color: '#1e293b', marginBottom: 12 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#3b82f6' : '#f1f5f9', color: active ? '#fff' : '#475569',
    border: 'none', fontSize: 13, marginRight: 4
  })
  const thStyle = { padding: '8px 12px', textAlign: 'left', fontSize: 12, color: '#64748b', borderBottom: '1px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '8px 12px', fontSize: 13, color: '#334155', borderBottom: '1px solid #f1f5f9' }

  const kpiItems = [
    { label: 'Citation Rate', value: `${fmt(s.citation_rate_pct)}%` },
    { label: 'Source Coverage', value: `${fmt(s.source_coverage_pct)}%` },
    { label: 'Citation Quality', value: `${fmt(s.citation_quality_score)}` },
    { label: 'Total Documents', value: fmt(s.total_documents) },
    { label: 'Total Responses', value: fmt(s.total_responses) },
    { label: 'Faithfulness', value: `${fmt(s.faithfulness_pct)}%` },
  ]

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'coverage', label: 'Coverage Detail' },
    { id: 'documents', label: 'Documents' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Citation Dashboard</h2>
        <span style={{ fontSize: 12, color: '#94a3b8' }}>real clinical.db + ChromaDB citation tracking</span>
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
              <div key={i} style={kpiStyle}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{k.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Charts row: Doc Type Bar + Daily Trend Line */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={sectionTitle}>Citations by Document Type</h4>
              {docTypes.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={docTypes}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" fontSize={11} angle={-20} textAnchor="end" height={50} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                      {docTypes.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No document type data</div>
              )}
            </div>

            <div style={cardStyle}>
              <h4 style={sectionTitle}>Daily Citation Volume</h4>
              {dailyTrend.length > 0 ? (
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={dailyTrend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" fontSize={11} />
                    <YAxis fontSize={11} />
                    <Tooltip />
                    <Line type="monotone" dataKey="citations" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No daily trend data</div>
              )}
            </div>
          </div>

          {/* Quality Distribution Pie */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Citation Quality Distribution</h4>
            {qualityDist.length > 0 ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <ResponsiveContainer width="50%" height={240}>
                  <PieChart>
                    <Pie data={qualityDist} dataKey="count" nameKey="category" cx="50%" cy="50%" innerRadius={50} outerRadius={90}
                      label={({ category, count }) => `${category}: ${count}`}>
                      {qualityDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ marginLeft: 24 }}>
                  {qualityDist.map((q, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <div style={{ width: 12, height: 12, borderRadius: 3, background: COLORS[i % COLORS.length] }} />
                      <span style={{ fontSize: 13, color: '#334155' }}>{q.category}: {fmt(q.count)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No quality distribution data</div>
            )}
          </div>
        </>
      )}

      {/* ═══ COVERAGE DETAIL TAB ═══ */}
      {tab === 'coverage' && (
        <>
          {/* Per-disease citation coverage */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Per-Disease Citation Coverage</h4>
            {diseaseCov.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={diseaseCov}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" fontSize={11} angle={-15} textAnchor="end" height={50} />
                  <YAxis fontSize={11} unit="%" />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="coverage_pct" fill="#10b981" radius={[4, 4, 0, 0]} name="Coverage %" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No disease coverage data</div>
            )}
          </div>

          {/* Top cited patients table */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Top Cited Patients</h4>
            {topPatients.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Patient ID</th>
                      <th style={thStyle}>Name</th>
                      <th style={thStyle}>Disease</th>
                      <th style={thStyle}>Citation Count</th>
                      <th style={thStyle}>Coverage %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topPatients.map((p, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{fmt(p.patient_id)}</td>
                        <td style={tdStyle}>{p.name || '--'}</td>
                        <td style={tdStyle}>{p.disease || '--'}</td>
                        <td style={tdStyle}>{fmt(p.citation_count)}</td>
                        <td style={tdStyle}>
                          <span style={{
                            padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                            background: (p.coverage_pct || 0) >= 80 ? '#dcfce7' : '#fef9c3',
                            color: (p.coverage_pct || 0) >= 80 ? '#166534' : '#854d0e'
                          }}>
                            {fmt(p.coverage_pct)}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 40 }}>No patient citation data</div>
            )}
          </div>

          {/* Per-component citation rates */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Per-Component Citation Rates</h4>
            {componentRates.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={componentRates} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" fontSize={11} unit="%" />
                  <YAxis dataKey="component" type="category" fontSize={11} width={120} />
                  <Tooltip formatter={(v) => `${v}%`} />
                  <Bar dataKey="rate_pct" fill="#8b5cf6" radius={[0, 4, 4, 0]} name="Citation Rate %" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No component rate data</div>
            )}
          </div>

          {/* Gap analysis: uncited patients */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Citation Gap Analysis -- Uncited Patients</h4>
            {gapPatients.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Patient ID</th>
                      <th style={thStyle}>Name</th>
                      <th style={thStyle}>Disease</th>
                      <th style={thStyle}>Responses</th>
                      <th style={thStyle}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gapPatients.map((p, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{fmt(p.patient_id)}</td>
                        <td style={tdStyle}>{p.name || '--'}</td>
                        <td style={tdStyle}>{p.disease || '--'}</td>
                        <td style={tdStyle}>{fmt(p.response_count)}</td>
                        <td style={tdStyle}>
                          <span style={{ padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 600, background: '#fef2f2', color: '#991b1b' }}>
                            No Citations
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#10b981', textAlign: 'center', paddingTop: 40, fontWeight: 600 }}>
                All patients have citations -- no gaps detected
              </div>
            )}
          </div>
        </>
      )}

      {/* ═══ DOCUMENTS TAB ═══ */}
      {tab === 'documents' && (
        <>
          {/* Document inventory table */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Document Inventory</h4>
            {documents.length > 0 ? (
              <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={{ ...thStyle, position: 'sticky', top: 0, background: '#fff' }}>Document ID</th>
                      <th style={{ ...thStyle, position: 'sticky', top: 0, background: '#fff' }}>Type</th>
                      <th style={{ ...thStyle, position: 'sticky', top: 0, background: '#fff' }}>Patient</th>
                      <th style={{ ...thStyle, position: 'sticky', top: 0, background: '#fff' }}>Citations</th>
                      <th style={{ ...thStyle, position: 'sticky', top: 0, background: '#fff' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((d, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{fmt(d.doc_id)}</td>
                        <td style={tdStyle}>{d.type || '--'}</td>
                        <td style={tdStyle}>{d.patient || '--'}</td>
                        <td style={tdStyle}>{fmt(d.citation_count)}</td>
                        <td style={tdStyle}>
                          <span style={{
                            padding: '2px 8px', borderRadius: 6, fontSize: 12, fontWeight: 600,
                            background: d.cited ? '#dcfce7' : '#fef2f2',
                            color: d.cited ? '#166534' : '#991b1b'
                          }}>
                            {d.cited ? 'Cited' : 'Uncited'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 40 }}>No document data available</div>
            )}
          </div>

          {/* Document type distribution bar chart */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Document Type Distribution</h4>
            {docTypeDist.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={docTypeDist}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="type" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Documents">
                    {docTypeDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No document type data</div>
            )}
          </div>

          {/* Citation gap visualization */}
          <div style={cardStyle}>
            <h4 style={sectionTitle}>Citation Gap Visualization</h4>
            {citationGaps.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={citationGaps}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="category" fontSize={11} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="cited" stackId="a" fill="#10b981" name="Cited" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="uncited" stackId="a" fill="#ef4444" name="Uncited" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 80 }}>No citation gap data</div>
            )}
          </div>
        </>
      )}

      {/* ═══ DEFINITIONS TAB ═══ */}
      {tab === 'definitions' && (
        <>
          <div style={{ marginBottom: 12 }}>
            <h4 style={sectionTitle}>Metric Definitions</h4>
            <p style={{ fontSize: 13, color: '#64748b', margin: '0 0 16px' }}>
              Reference definitions for all citation metrics, scoring methodology, and clinical relevance.
            </p>
          </div>
          {definitions.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              {definitions.map((d, i) => (
                <div key={i} style={cardStyle}>
                  <h5 style={{ margin: '0 0 8px', fontSize: 14, fontWeight: 700, color: '#1e293b' }}>
                    {d.title || d.name || `Metric ${i + 1}`}
                  </h5>
                  <p style={{ margin: '0 0 8px', fontSize: 13, color: '#475569', lineHeight: 1.5 }}>
                    {d.description || '--'}
                  </p>
                  {d.computation && (
                    <div style={{ marginBottom: 6 }}>
                      <span style={{ fontSize: 11, fontWeight: 600, color: '#64748b' }}>Computation: </span>
                      <span style={{ fontSize: 12, color: '#334155', fontFamily: 'monospace' }}>{d.computation}</span>
                    </div>
                  )}
                  {d.clinical_relevance && (
                    <div>
                      <span style={{ fontSize: 11, fontWeight: 600, color: '#64748b' }}>Clinical Relevance: </span>
                      <span style={{ fontSize: 12, color: '#334155' }}>{d.clinical_relevance}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div style={{ ...cardStyle, textAlign: 'center', color: '#94a3b8' }}>
              No metric definitions available
            </div>
          )}
        </>
      )}
    </div>
  )
}
