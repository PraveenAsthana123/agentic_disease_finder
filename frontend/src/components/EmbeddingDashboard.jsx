import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#f97316', '#64748b']

export default function EmbeddingDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API}/api/embedding/overview`),
      axios.get(`${API}/api/embedding/breakdown`),
      axios.get(`${API}/api/embedding/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'extractions', label: 'Feature Extractions' },
    { id: 'staleness', label: 'Staleness & Refresh' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Embedding & Feature Engineering dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No embedding data available.'}</div>

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#1e293b' }}>Embedding & Feature Engineering Dashboard</h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Feature extraction pipeline, embedding quality, dimensionality analysis, and refresh tracking
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI cards row 1 */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Features Extracted" value={overview.kpis.total_features_extracted} sub="total analyses" color="#3b82f6" /></Card>
            <Card><KPI label="Patients with Features" value={overview.kpis.total_patients_with_features} sub={`${overview.kpis.feature_coverage_pct}% coverage`} color="#10b981" /></Card>
            <Card><KPI label="Avg Confidence" value={overview.kpis.avg_feature_confidence} sub="feature quality" color="#f59e0b" /></Card>
            <Card><KPI label="Good Quality" value={`${overview.kpis.good_quality_pct}%`} sub="signal quality rate" color="#8b5cf6" /></Card>
          </div>

          {/* KPI cards row 2 */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Feature Dimensions" value={overview.kpis.total_feature_dimensions} sub="per extraction" color="#06b6d4" /></Card>
            <Card><KPI label="Refresh Events" value={overview.kpis.embedding_refresh_events} sub="pipeline runs" color="#ec4899" /></Card>
            <Card><KPI label="Unique Diseases" value={overview.kpis.unique_diseases} sub="disease coverage" color="#f97316" /></Card>
            <Card><KPI label="Unique Files" value={overview.kpis.unique_files || 0} sub="input sources" color="#64748b" /></Card>
          </div>

          {/* Charts row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Feature Type Distribution">
              {(overview.feature_type_distribution || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={overview.feature_type_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                      {overview.feature_type_distribution.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No feature type data</div>}
            </Card>

            <Card title="Confidence Distribution">
              {(overview.confidence_distribution || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={overview.confidence_distribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="bucket" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No confidence data</div>}
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Daily Extraction Trend">
              {(overview.daily_extraction_trend || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={overview.daily_extraction_trend}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} />
                    <Tooltip />
                    <Line type="monotone" dataKey="extractions" stroke="#3b82f6" strokeWidth={2} name="Extractions" />
                  </LineChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily trend data</div>}
            </Card>

            <Card title="Signal Quality Distribution">
              {(overview.signal_quality_distribution || []).length ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={overview.signal_quality_distribution} dataKey="count" nameKey="quality" cx="50%" cy="50%" outerRadius={80} label={({ quality, count }) => `${quality}: ${count}`}>
                      {overview.signal_quality_distribution.map((d, i) => (
                        <Cell key={i} fill={d.quality === 'Good' ? '#10b981' : '#ef4444'} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No signal quality data</div>}
            </Card>
          </div>

          {/* Disease feature coverage */}
          <Card title="Disease Feature Coverage">
            {(overview.disease_feature_coverage || []).length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.disease_feature_coverage}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="features_extracted" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Features" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No disease coverage data</div>}
          </Card>
        </>
      )}

      {/* ── Patient Profiles Tab ── */}
      {tab === 'patients' && breakdown && (
        <>
          <Card title="Per-Patient Feature Profiles" span={2}>
            {(breakdown.patient_profiles || []).length ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient ID</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Name</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Features</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Avg Confidence</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Signal Quality</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Latest Extraction</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.patient_profiles.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{p.name || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>{p.disease}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={p.n_features} color="#3b82f6" />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{p.avg_confidence}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={p.signal_quality || 'N/A'} color={p.signal_quality === 'Good' ? '#10b981' : '#ef4444'} />
                        </td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(p.latest_extraction || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No patient feature profiles yet</div>}
          </Card>
        </>
      )}

      {/* ── Feature Extractions Tab ── */}
      {tab === 'extractions' && breakdown && (
        <>
          <Card title="Recent Feature Extractions" span={2}>
            {(breakdown.recent_extractions || []).length ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>ID</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Disease</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Prediction</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Confidence</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Signal Quality</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.recent_extractions.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px' }}>{e.id}</td>
                        <td style={{ padding: '6px 10px' }}>{e.patient_id}</td>
                        <td style={{ padding: '6px 10px' }}>{e.disease}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={e.predicted_label || 'N/A'} color="#3b82f6" />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{e.confidence}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={e.signal_quality || 'N/A'} color={e.signal_quality === 'Good' ? '#10b981' : '#ef4444'} />
                        </td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(e.created_at || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No recent extractions</div>}
          </Card>

          {/* Feature dimension analysis */}
          <Card title="Feature Dimension Analysis" span={2}>
            {(breakdown.feature_dimension_analysis || []).length ? (
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={breakdown.feature_dimension_analysis}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature_type" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="extraction_count" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Extractions" />
                  <Bar dataKey="dimensions" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Dimensions" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No dimension data</div>}
          </Card>

          {/* Extraction event log */}
          <Card title="Extraction Event Log" span={2}>
            {(breakdown.extraction_event_log || []).length ? (
              <div style={{ overflowX: 'auto', marginTop: 8 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f8fafc' }}>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Component</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Action</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Actor</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Detail</th>
                      <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.extraction_event_log.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px' }}>{e.patient_id || '-'}</td>
                        <td style={{ padding: '6px 10px' }}>
                          <Badge text={e.component} color="#06b6d4" />
                        </td>
                        <td style={{ padding: '6px 10px' }}>{e.action}</td>
                        <td style={{ padding: '6px 10px' }}>{e.actor || '-'}</td>
                        <td style={{ padding: '6px 10px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || '-'}</td>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{(e.timestamp || '').slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No extraction events</div>}
          </Card>
        </>
      )}

      {/* ── Staleness & Refresh Tab ── */}
      {tab === 'staleness' && breakdown && (
        <>
          <Card title="Feature Staleness Analysis — Days Since Last Extraction" span={2}>
            {(breakdown.staleness_analysis || []).length ? (
              <>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={breakdown.staleness_analysis}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} />
                    <YAxis tick={{ fontSize: 10 }} label={{ value: 'Days', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="days_since_extraction" fill="#f97316" radius={[4, 4, 0, 0]} name="Days Since Extraction" />
                  </BarChart>
                </ResponsiveContainer>
                <div style={{ overflowX: 'auto', marginTop: 16 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: '#f8fafc' }}>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Patient ID</th>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Name</th>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Days Since Extraction</th>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Last Extraction</th>
                        <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {breakdown.staleness_analysis.map((s, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.patient_id}</td>
                          <td style={{ padding: '6px 10px' }}>{s.name || '-'}</td>
                          <td style={{ padding: '6px 10px' }}>{s.days_since_extraction}</td>
                          <td style={{ padding: '6px 10px', color: '#64748b' }}>{(s.last_extraction || '').slice(0, 16)}</td>
                          <td style={{ padding: '6px 10px' }}>
                            <Badge text={s.status || (s.days_since_extraction > 30 ? 'STALE' : s.days_since_extraction > 7 ? 'AGING' : 'FRESH')} color={s.status === 'Stale' ? '#ef4444' : s.status === 'Recent' ? '#f59e0b' : '#10b981'} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No staleness data available</div>}
          </Card>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && definitions && (
        <>
          {(definitions.sections || []).map((sec, si) => (
            <Card key={si} title={sec.title} span={2}>
              <div style={{ marginTop: 8 }}>
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{item.definition}</div>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}
    </div>
  )
}
