import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316', '#14b8a6']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function QualityBadge({ level }) {
  const colors = {
    'Good': { bg: '#ecfdf5', fg: '#065f46' },
    'Poor': { bg: '#fef2f2', fg: '#991b1b' },
  }
  const c = colors[level] || { bg: '#f1f5f9', fg: '#475569' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.fg
    }}>{level || '--'}</span>
  )
}

function CompletenessBar({ rate }) {
  const color = rate >= 95 ? '#10b981' : rate >= 80 ? '#f59e0b' : '#ef4444'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 8, background: '#f1f5f9', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ width: `${Math.min(rate, 100)}%`, height: '100%', background: color, borderRadius: 4 }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color, minWidth: 40, textAlign: 'right' }}>{rate}%</span>
    </div>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'uploads', label: 'Upload Quality' },
  { id: 'outliers', label: 'Outliers & Dupes' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DataQualityDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/data-quality/overview`),
      axios.get(`${API_URL}/api/data-quality/breakdown`),
      axios.get(`${API_URL}/api/data-quality/definitions`),
    ])
      .then(([oRes, bRes, dRes]) => {
        setOverview(oRes.data)
        setBreakdown(bRes.data)
        setDefs(dRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Data Quality...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40 }}>Data quality information not available</div>

  const k = overview.kpis || {}
  const fc = overview.field_completeness || {}
  const sqd = overview.signal_quality_distribution || {}
  const cs = overview.confidence_stats || {}
  const cov = overview.coverage || {}

  // Prepare chart data
  const completenessData = Object.entries(fc).map(([field, info]) => ({
    name: field.charAt(0).toUpperCase() + field.slice(1),
    rate: info.rate,
    filled: info.filled,
    missing: info.total - info.filled,
  }))

  const signalData = Object.entries(sqd).map(([quality, count], i) => ({
    name: quality, value: count, fill: quality === 'Good' ? '#10b981' : '#ef4444'
  }))

  const formatData = (overview.format_distribution || []).map((f, i) => ({
    ...f, name: f.format, value: f.count, fill: COLORS[i % COLORS.length]
  }))

  const confData = overview.confidence_distribution || []
  const dailyTrend = overview.daily_trend || []
  const nullData = (overview.null_field_distribution || []).map(f => ({
    ...f, name: f.field.charAt(0).toUpperCase() + f.field.slice(1)
  }))

  const diseaseData = (overview.disease_distribution || []).map((d, i) => ({
    ...d, name: d.disease, value: d.count, fill: COLORS[i % COLORS.length]
  }))

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Data Quality / Cleaning Dashboard</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 13 }}>
          Field completeness, deduplication, signal quality, confidence profiling, and coverage analytics
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#1e293b' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Total Patients" value={k.total_patients} /></Card>
            <Card><KPI label="Total Uploads" value={k.total_uploads} /></Card>
            <Card><KPI label="Total Analyses" value={k.total_analyses} /></Card>
            <Card><KPI label="Completeness" value={`${k.overall_completeness_pct}%`} color={k.overall_completeness_pct >= 80 ? '#10b981' : '#f59e0b'} /></Card>
            <Card><KPI label="Good Signal" value={`${k.good_signal_pct}%`} color={k.good_signal_pct >= 80 ? '#10b981' : '#f59e0b'} /></Card>
            <Card><KPI label="Avg Confidence" value={k.avg_confidence} color={k.avg_confidence >= 0.6 ? '#3b82f6' : '#ef4444'} /></Card>
            <Card><KPI label="Duplicate Files" value={k.duplicate_files} color={k.duplicate_files > 0 ? '#f59e0b' : '#10b981'} /></Card>
            <Card><KPI label="Upload Coverage" value={`${k.upload_coverage_pct}%`} /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16, marginBottom: 24 }}>
            {/* Field Completeness */}
            <Card title="Field Completeness">
              {completenessData.map(f => (
                <div key={f.name} style={{ marginBottom: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 3 }}>
                    <span style={{ color: '#334155', fontWeight: 500 }}>{f.name}</span>
                    <span style={{ color: '#64748b' }}>{f.filled}/{f.filled + f.missing}</span>
                  </div>
                  <CompletenessBar rate={f.rate} />
                </div>
              ))}
            </Card>

            {/* Signal Quality Distribution */}
            <Card title="Signal Quality Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={signalData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    innerRadius={50} outerRadius={80} paddingAngle={5}>
                    {signalData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16, marginBottom: 24 }}>
            {/* Confidence Distribution */}
            <Card title="Confidence Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={confData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: 8, fontSize: 11, color: '#64748b' }}>
                <span>Mean: {cs.mean}</span>
                <span>Min: {cs.min}</span>
                <span>Max: {cs.max}</span>
                <span>Std: {cs.std}</span>
              </div>
            </Card>

            {/* File Format Distribution */}
            <Card title="File Format Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={formatData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    innerRadius={45} outerRadius={75} paddingAngle={5}>
                    {formatData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16, marginBottom: 24 }}>
            {/* Daily Upload Quality Trend */}
            <Card title="Daily Upload Quality Trend" span={2}>
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={dailyTrend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Legend />
                  <Area type="monotone" dataKey="good" name="Good" stackId="1" fill="#10b981" stroke="#10b981" fillOpacity={0.6} />
                  <Area type="monotone" dataKey="poor" name="Poor" stackId="1" fill="#ef4444" stroke="#ef4444" fillOpacity={0.6} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16 }}>
            {/* Missing Fields */}
            <Card title="Missing Fields by Category">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={nullData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={80} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="present" name="Present" stackId="a" fill="#10b981" />
                  <Bar dataKey="missing" name="Missing" stackId="a" fill="#ef4444" />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            {/* Coverage Stats */}
            <Card title="Patient Coverage">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 12 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 32, fontWeight: 700, color: '#3b82f6' }}>{cov.patients_with_uploads}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>With Uploads</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{cov.upload_coverage_pct}% of {k.total_patients}</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 32, fontWeight: 700, color: '#8b5cf6' }}>{cov.patients_with_analyses}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>With Analyses</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{cov.analysis_coverage_pct}% of {k.total_patients}</div>
                </div>
              </div>
              <div style={{ marginTop: 20 }}>
                <div style={{ fontSize: 13, fontWeight: 500, color: '#334155', marginBottom: 6 }}>Disease Distribution</div>
                {diseaseData.map(d => (
                  <div key={d.name} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 12, borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ color: '#475569' }}>{d.name}</span>
                    <span style={{ fontWeight: 600 }}>{d.count}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}

      {/* ── Patient Profiles Tab ── */}
      {tab === 'patients' && (
        <Card title={`Patient Data Quality Profiles (${(breakdown?.patient_profiles || []).length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'Completeness', 'Missing Fields', 'Uploads', 'Analyses', 'Avg Conf', 'Signal'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown?.patient_profiles || []).map((p, i) => (
                  <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafbfc' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <CompletenessBar rate={p.completeness_pct} />
                    </td>
                    <td style={{ padding: '6px 10px', color: p.missing_fields.length > 0 ? '#ef4444' : '#10b981', fontSize: 11 }}>
                      {p.missing_fields.length > 0 ? p.missing_fields.join(', ') : 'None'}
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.uploads}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.analyses}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.avg_confidence != null ? p.avg_confidence : '--'}</td>
                    <td style={{ padding: '6px 10px' }}><QualityBadge level={p.latest_signal_quality} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Upload Quality Tab ── */}
      {tab === 'uploads' && (
        <Card title={`Upload Quality Detail (${(breakdown?.upload_quality || []).length})`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                  {['ID', 'Patient', 'File', 'Disease', 'Signal', 'Confidence', 'Prediction', 'Uploaded'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown?.upload_quality || []).map((u, i) => (
                  <tr key={u.upload_id} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafbfc' }}>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{u.upload_id}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{u.patient_id}</td>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{u.file_name}</td>
                    <td style={{ padding: '6px 10px' }}>{u.disease}</td>
                    <td style={{ padding: '6px 10px' }}><QualityBadge level={u.signal_quality} /></td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: u.confidence && u.confidence < 0.55 ? '#ef4444' : '#1e293b' }}>
                      {u.confidence != null ? u.confidence : '--'}
                    </td>
                    <td style={{ padding: '6px 10px' }}>{u.predicted_label || '--'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{u.uploaded_at?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Outliers & Dupes Tab ── */}
      {tab === 'outliers' && (
        <>
          <Card title={`Outlier Analyses — Low Confidence / Poor Signal (${(breakdown?.outliers || []).length})`}>
            {(breakdown?.outliers || []).length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#10b981', fontSize: 14 }}>No outliers detected</div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#fef2f2', borderBottom: '2px solid #fecaca' }}>
                      {['ID', 'Patient', 'File', 'Signal', 'Confidence', 'Prediction', 'Analyzed'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#991b1b', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown?.outliers || []).map((o, i) => (
                      <tr key={o.analysis_id} style={{ borderBottom: '1px solid #fef2f2', background: i % 2 === 0 ? '#fff' : '#fffbfb' }}>
                        <td style={{ padding: '6px 10px', color: '#64748b' }}>{o.analysis_id}</td>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{o.patient_id}</td>
                        <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{o.file_name}</td>
                        <td style={{ padding: '6px 10px' }}><QualityBadge level={o.signal_quality} /></td>
                        <td style={{ padding: '6px 10px', fontWeight: 700, color: '#ef4444' }}>{o.confidence}</td>
                        <td style={{ padding: '6px 10px' }}>{o.predicted_label}</td>
                        <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{o.analyzed_at?.slice(0, 16)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <div style={{ marginTop: 16 }}>
            <Card title={`Duplicate File Log (${(breakdown?.duplicate_log || []).length})`}>
              {(breakdown?.duplicate_log || []).length === 0 ? (
                <div style={{ padding: 20, textAlign: 'center', color: '#10b981', fontSize: 14 }}>No duplicates found</div>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: '#fff7ed', borderBottom: '2px solid #fed7aa' }}>
                        {['File Name', 'Patient', 'Uploaded At'].map(h => (
                          <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#9a3412', fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(breakdown?.duplicate_log || []).map((d, i) => (
                        <tr key={`${d.file_name}-${d.patient_id}-${i}`} style={{ borderBottom: '1px solid #fff7ed', background: i % 2 === 0 ? '#fff' : '#fffcf5' }}>
                          <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.file_name}</td>
                          <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.patient_id}</td>
                          <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{d.uploaded_at?.slice(0, 16)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>
          </div>

          {(breakdown?.quality_event_log || []).length > 0 && (
            <div style={{ marginTop: 16 }}>
              <Card title="Recent Quality Events">
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr style={{ background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                        {['Timestamp', 'Actor', 'Component', 'Action', 'Patient', 'Detail'].map(h => (
                          <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', fontWeight: 600 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(breakdown?.quality_event_log || []).map((e, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 === 0 ? '#fff' : '#fafbfc' }}>
                          <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{e.timestamp?.slice(0, 16)}</td>
                          <td style={{ padding: '6px 10px' }}>{e.actor}</td>
                          <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{e.component}</td>
                          <td style={{ padding: '6px 10px' }}>{e.action}</td>
                          <td style={{ padding: '6px 10px', fontWeight: 600 }}>{e.patient_id || '--'}</td>
                          <td style={{ padding: '6px 10px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          )}
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && (
        <>
          {(defs?.sections || []).map((sec, si) => (
            <div key={si} style={{ marginBottom: 16 }}>
              <Card title={sec.title}>
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: ii < sec.items.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 3 }}>{item.term}</div>
                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.definition}</div>
                  </div>
                ))}
              </Card>
            </div>
          ))}
        </>
      )}
    </div>
  )
}
