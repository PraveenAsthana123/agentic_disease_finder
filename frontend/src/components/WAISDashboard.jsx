import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = '/api'
const IQ_COLORS = {
  'Extremely High': '#16a34a', 'Superior': '#22c55e', 'High Average': '#3b82f6',
  'Average': '#64748b', 'Low Average': '#eab308', 'Borderline': '#f97316',
  'Extremely Low': '#ef4444'
}
const INDEX_COLORS = { VCI: '#3b82f6', PRI: '#22c55e', WMI: '#f97316', PSI: '#8b5cf6' }

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

function IQBadge({ classification }) {
  const color = IQ_COLORS[classification] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{classification}</span>
  )
}

function ScoreBar({ score, max, label, mean }) {
  const pct = Math.round(score / max * 100)
  const color = score >= 10 ? '#22c55e' : score >= 7 ? '#3b82f6' : score >= 4 ? '#eab308' : '#ef4444'
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ color: '#475569' }}>{label}</span>
        <span style={{ fontWeight: 600 }}>{score}/19</span>
      </div>
      <div style={{ background: '#f1f5f9', borderRadius: 4, height: 8, position: 'relative' }}>
        <div style={{ width: `${pct}%`, background: color, borderRadius: 4, height: '100%' }} />
        {mean && <div style={{
          position: 'absolute', left: `${mean / max * 100}%`, top: -2, width: 2, height: 12,
          background: '#1e293b', opacity: 0.3
        }} />}
      </div>
    </div>
  )
}

export default function WAISDashboard() {
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
          axios.get(`${API_URL}/wais-dashboard/overview`),
          axios.get(`${API_URL}/wais-dashboard/breakdown`),
          axios.get(`${API_URL}/wais-dashboard/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading WAIS data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'indices', label: 'Indices & Subtests' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const distData = overview?.iq_distribution
    ? Object.entries(overview.iq_distribution).map(([k, v]) => ({
        name: k, value: v, color: IQ_COLORS[k] || '#94a3b8'
      }))
    : []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>WAIS Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Wechsler Adult Intelligence Scale (WAIS-IV) — IQ assessment (FSIQ mean=100, SD=15)
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

      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPIs */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Assessments" value={fmt(overview.total_assessments)} />
              <KPI label="Unique Patients" value={fmt(overview.unique_patients)} />
              <KPI label="Mean FSIQ" value={fmt(overview.avg_fsiq)} sub="mean=100, SD=15" />
              <KPI label="FSIQ Range" value={`${fmt(overview.min_fsiq)}-${fmt(overview.max_fsiq)}`} />
              <KPI label="Below Average" value={
                fmt((overview.iq_distribution['Low Average'] || 0) +
                    (overview.iq_distribution['Borderline'] || 0) +
                    (overview.iq_distribution['Extremely Low'] || 0))
              } sub="patients" color="#f97316" />
            </div>
          </Card>

          {/* IQ Distribution Pie */}
          <Card title="IQ Classification Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={distData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={80} paddingAngle={2}>
                  {distData.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {distData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Index Comparison Bar */}
          <Card title="Index Score Comparison (All Patients)" span={2}>
            {breakdown?.patient_index_comparison && (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={breakdown.patient_index_comparison}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="patient_id" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis domain={[40, 160]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="VCI" fill={INDEX_COLORS.VCI} name="VCI" />
                  <Bar dataKey="PRI" fill={INDEX_COLORS.PRI} name="PRI" />
                  <Bar dataKey="WMI" fill={INDEX_COLORS.WMI} name="WMI" />
                  <Bar dataKey="PSI" fill={INDEX_COLORS.PSI} name="PSI" />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Per-Patient Summary Table */}
          <Card title="Per-Patient FSIQ Scores" span={3}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>FSIQ</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>VCI</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>PRI</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>WMI</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>PSI</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Classification</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.patient_summary || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 700 }}>{p.fsiq}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.vci}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.pri}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.wmi}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.psi}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><IQBadge classification={p.classification} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'indices' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Index Profile */}
          <Card title="Index Score Profile (Group Average)" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={breakdown.index_profile} layout="vertical" margin={{ left: 150 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[40, 160]} />
                <YAxis type="category" dataKey="label" width={140} tick={{ fontSize: 12 }} />
                <Tooltip formatter={v => `${v} (mean=100)`} />
                <Bar dataKey="avg" radius={[0, 4, 4, 0]}>
                  {(breakdown.index_profile || []).map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              Dashed line at 100 = population mean
            </div>
          </Card>

          {/* Subtest Scaled Scores (weakest first) */}
          <Card title="Subtest Scaled Scores (weakest first)" span={2}>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={breakdown.subtest_summary} layout="vertical" margin={{ left: 130 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 19]} />
                <YAxis type="category" dataKey="label" width={120} tick={{ fontSize: 11 }} />
                <Tooltip formatter={v => `${v}/19 (mean=10)`} />
                <Bar dataKey="avg_scaled" radius={[0, 4, 4, 0]}>
                  {(breakdown.subtest_summary || []).map((d, i) => {
                    const color = d.avg_scaled >= 10 ? '#22c55e' : d.avg_scaled >= 7 ? '#3b82f6' : d.avg_scaled >= 4 ? '#eab308' : '#ef4444'
                    return <Cell key={i} fill={color} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly Trend */}
          {breakdown.trend && breakdown.trend.length > 0 && (
            <Card title="Monthly FSIQ Trend" span={2}>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={breakdown.trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis domain={[60, 140]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="avg_fsiq" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Avg FSIQ" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Per-Patient Index Profiles & History">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              {Object.entries(breakdown.patient_history || {}).map(([pid, hist]) => {
                const latest = hist[hist.length - 1]
                return (
                  <div key={pid} style={{ marginBottom: 20, padding: '12px 16px', background: '#f8fafc', borderRadius: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                      <div>
                        <span style={{ fontWeight: 700, fontSize: 14 }}>{pid}</span>
                        <span style={{ marginLeft: 12, fontSize: 12, color: '#64748b' }}>
                          FSIQ: {latest.fsiq} | VCI: {latest.vci} | PRI: {latest.pri} | WMI: {latest.wmi} | PSI: {latest.psi}
                        </span>
                      </div>
                      <IQBadge classification={latest.classification} />
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                      {[
                        { label: 'VCI', val: latest.vci, color: INDEX_COLORS.VCI },
                        { label: 'PRI', val: latest.pri, color: INDEX_COLORS.PRI },
                        { label: 'WMI', val: latest.wmi, color: INDEX_COLORS.WMI },
                        { label: 'PSI', val: latest.psi, color: INDEX_COLORS.PSI },
                      ].map(idx => (
                        <div key={idx.label} style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 2 }}>{idx.label}</div>
                          <div style={{ fontSize: 20, fontWeight: 700, color: idx.color }}>{idx.val}</div>
                          <div style={{
                            height: 4, borderRadius: 2, marginTop: 4,
                            background: '#f1f5f9', position: 'relative'
                          }}>
                            <div style={{
                              width: `${Math.min(100, Math.max(0, (idx.val - 40) / 120 * 100))}%`,
                              background: idx.color, borderRadius: 2, height: '100%'
                            }} />
                          </div>
                        </div>
                      ))}
                    </div>
                    {hist.length > 1 && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
                        {hist.length} assessments | First: {hist[0].date} | Latest: {latest.date}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <Card title={defs.title}>
          <table style={{ width: '100%', fontSize: 13, borderCollapse: 'collapse' }}>
            <tbody>
              {(defs.definitions || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap', verticalAlign: 'top', color: '#334155', width: 260 }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  )
}
