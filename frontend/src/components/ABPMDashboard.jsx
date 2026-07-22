import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function SeverityBadge({ severity }) {
  const colors = { Normal: '#10b981', Mild: '#f59e0b', Moderate: '#f97316', Severe: '#ef4444' }
  const color = colors[severity] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{severity || '--'}</span>
  )
}

function FlagBadge({ flag }) {
  const colors = { normal: '#10b981', borderline: '#f59e0b', high: '#ef4444', low: '#ef4444', critical: '#991b1b' }
  const color = colors[(flag || '').toLowerCase()] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{flag || '--'}</span>
  )
}

function DippingBadge({ category }) {
  const colors = {
    extreme_dipper: '#8b5cf6', normal_dipper: '#10b981',
    non_dipper: '#f59e0b', reverse_dipper: '#ef4444'
  }
  const labels = {
    extreme_dipper: 'Extreme Dipper', normal_dipper: 'Normal Dipper',
    non_dipper: 'Non-Dipper', reverse_dipper: 'Reverse Dipper'
  }
  const color = colors[category] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{labels[category] || category || '--'}</span>
  )
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

export default function ABPMDashboard() {
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
          axios.get(`${API_URL}/api/abpm/overview`),
          axios.get(`${API_URL}/api/abpm/breakdown`),
          axios.get(`${API_URL}/api/abpm/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load ABPM/Holter data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading ABPM/Holter data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (overview?.available === false) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>ABPM/Holter data not available</div>

  const tabs = ['overview', 'parameters', 'patients', 'definitions']
  const kpis = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>ABPM / Holter Monitoring Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Ambulatory blood pressure monitoring + Holter ECG analysis — {fmt(kpis.total_studies)} studies, {fmt(kpis.abnormal_count)} abnormal ({fmt(kpis.abnormal_rate_pct)}%)
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#f1f5f9',
            color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {/* ── Overview tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Total Studies" value={kpis.total_studies} />
              <KPI label="Abnormal Count" value={kpis.abnormal_count} color="#ef4444" />
              <KPI label="Abnormal Rate" value={kpis.abnormal_rate_pct} sub="%" color="#ef4444" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
              <KPI label="Mean Systolic (24h)" value={kpis.mean_systolic_24h} sub="mmHg" color="#3b82f6" />
              <KPI label="Mean Diastolic (24h)" value={kpis.mean_diastolic_24h} sub="mmHg" color="#3b82f6" />
              <KPI label="Mean QTc" value={kpis.mean_qtc_ms} sub="ms" color="#8b5cf6" />
            </div>
          </Card>

          <Card title="Severity Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={Object.entries(overview?.severity_distribution || {}).map(([name, value]) => ({ name, value }))}
                  cx="50%" cy="50%" outerRadius={90} dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {Object.keys(overview?.severity_distribution || {}).map((_, i) => (
                    <Cell key={i} fill={[COLORS[0], COLORS[2], '#f97316', COLORS[3]][i] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Dipping Status Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={Object.entries(overview?.dipping_distribution || {}).map(([name, value]) => ({
                    name: name.replace(/_/g, ' '), value
                  }))}
                  cx="50%" cy="50%" outerRadius={90} dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {Object.keys(overview?.dipping_distribution || {}).map((_, i) => (
                    <Cell key={i} fill={[COLORS[4], COLORS[0], COLORS[2], COLORS[3]][i] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Diagnostic Pattern Distribution" span={4}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={Object.entries(overview?.pattern_distribution || {}).map(([name, count]) => ({
                name: name.replace(/_/g, ' '), count
              }))} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={180} />
                <Tooltip />
                <Bar dataKey="count" name="Studies" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                  {Object.keys(overview?.pattern_distribution || {}).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Patient Summary" span={4}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Name</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Age</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Disease</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Severity</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Pattern</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Cardiac Score</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>SBP 24h</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>DBP 24h</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Dipping %</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Dipping</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>QTc (ms)</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>PVCs</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview?.patient_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{row.name}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{row.age}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.disease}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><SeverityBadge severity={row.severity} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                        <span style={{
                          display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                          background: '#3b82f622', color: '#3b82f6', fontWeight: 600, fontSize: 11
                        }}>{row.pattern_label}</span>
                      </td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600 }}>{fmt(row.cardiac_score)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.systolic_24h)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.diastolic_24h)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.dipping_pct)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><DippingBadge category={row.dipping_category} /></td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.qtc_ms)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.pvc_count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Parameters tab ── */}
      {tab === 'parameters' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="ABPM Parameter Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Parameter</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Mean</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Min</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Max</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Unit</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Ref Range</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Abnormal N</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Abnormal %</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.abpm_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.parameter}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.mean)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.min)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.max)}</td>
                      <td style={{ padding: '8px 12px' }}>{row.unit}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.ref_low}–{row.ref_high}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{row.abnormal_n}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.abnormal_pct > 25 ? '#ef4444' : '#64748b', fontWeight: row.abnormal_pct > 25 ? 600 : 400 }}>
                        {fmt(row.abnormal_pct)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Holter Parameter Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Parameter</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Mean</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Min</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Max</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Unit</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Ref Range</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Abnormal N</th>
                    <th style={{ textAlign: 'right', padding: '8px 12px', color: '#475569' }}>Abnormal %</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.holter_summary || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.parameter}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.mean)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.min)}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{fmt(row.max)}</td>
                      <td style={{ padding: '8px 12px' }}>{row.unit}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.ref_low}–{row.ref_high}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right' }}>{row.abnormal_n}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'right', color: row.abnormal_pct > 25 ? '#ef4444' : '#64748b', fontWeight: row.abnormal_pct > 25 ? 600 : 400 }}>
                        {fmt(row.abnormal_pct)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Systolic BP Histogram (24h)" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={breakdown.systolic_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 11 }} tickFormatter={v => `${v}`} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={l => `${l} mmHg`} />
                <Bar dataKey="count" name="Patients" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="QTc Histogram" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={breakdown.qtc_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 11 }} tickFormatter={v => `${v}`} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={l => `${l} ms`} />
                <Bar dataKey="count" name="Patients" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Dipping % Histogram" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={breakdown.dipping_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 11 }} tickFormatter={v => `${v}`} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [v, 'Count']} labelFormatter={l => `${l}%`} />
                <Bar dataKey="count" name="Patients" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ── Patients tab ── */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* High-risk patients table */}
          {(() => {
            const highRisk = (breakdown.patient_detail_cards || []).filter(
              p => p.severity === 'Severe' || p.severity === 'Moderate'
            )
            if (highRisk.length === 0) return null
            return (
              <Card title="High-Risk Patients (Moderate / Severe)">
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#fef2f2' }}>
                        <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Patient</th>
                        <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Name</th>
                        <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Age</th>
                        <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Disease</th>
                        <th style={{ textAlign: 'center', padding: '8px 12px', color: '#991b1b' }}>Severity</th>
                        <th style={{ textAlign: 'left', padding: '8px 12px', color: '#991b1b' }}>Pattern</th>
                        <th style={{ textAlign: 'right', padding: '8px 12px', color: '#991b1b' }}>Cardiac Score</th>
                        <th style={{ textAlign: 'center', padding: '8px 12px', color: '#991b1b' }}>Dipping</th>
                      </tr>
                    </thead>
                    <tbody>
                      {highRisk.map((row, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                          <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.patient_id}</td>
                          <td style={{ padding: '8px 12px' }}>{row.name}</td>
                          <td style={{ padding: '8px 12px', textAlign: 'right' }}>{row.age}</td>
                          <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.disease}</td>
                          <td style={{ padding: '8px 12px', textAlign: 'center' }}><SeverityBadge severity={row.severity} /></td>
                          <td style={{ padding: '8px 12px' }}>
                            <span style={{
                              display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                              background: '#3b82f622', color: '#3b82f6', fontWeight: 600, fontSize: 11
                            }}>{row.pattern_label}</span>
                          </td>
                          <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 600, color: '#ef4444' }}>{fmt(row.cardiac_score)}</td>
                          <td style={{ padding: '8px 12px', textAlign: 'center' }}><DippingBadge category={row.dipping_category} /></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )
          })()}

          {/* Per-patient detail cards */}
          <Card title="Patient Detail Cards">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              {(breakdown.patient_detail_cards || []).map((card, i) => (
                <div key={i} style={{
                  padding: 16, background: '#f8fafc', borderRadius: 10,
                  border: card.is_abnormal ? '2px solid #fca5a5' : '1px solid #e2e8f0'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: 14, color: '#1e293b' }}>{card.name}</span>
                      <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>Age {card.age} | {card.disease}</span>
                    </div>
                    <SeverityBadge severity={card.severity} />
                  </div>

                  <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                    <span style={{
                      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
                      background: '#3b82f622', color: '#3b82f6', fontWeight: 600, fontSize: 11
                    }}>{card.pattern_label}</span>
                    <DippingBadge category={card.dipping_category} />
                  </div>

                  {/* Cardiac score progress bar */}
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>Cardiac Score: {fmt(card.cardiac_score)}/100</div>
                    <div style={{ height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{
                        width: `${Math.min(card.cardiac_score || 0, 100)}%`, height: '100%',
                        background: (card.cardiac_score || 0) > 70 ? '#ef4444' : (card.cardiac_score || 0) > 45 ? '#f97316' : (card.cardiac_score || 0) > 20 ? '#f59e0b' : '#10b981',
                        borderRadius: 4
                      }} />
                    </div>
                  </div>

                  {/* Key ABPM values */}
                  <div style={{ fontSize: 12, marginBottom: 6 }}>
                    <strong style={{ color: '#334155' }}>ABPM:</strong>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
                      {card.abpm?.systolic_24h_mmhg && (
                        <span>SBP {fmt(card.abpm.systolic_24h_mmhg.value)} mmHg <FlagBadge flag={card.abpm.systolic_24h_mmhg.flag} /></span>
                      )}
                      {card.abpm?.diastolic_24h_mmhg && (
                        <span>DBP {fmt(card.abpm.diastolic_24h_mmhg.value)} mmHg <FlagBadge flag={card.abpm.diastolic_24h_mmhg.flag} /></span>
                      )}
                      {card.abpm?.dipping_pct && (
                        <span>Dip {fmt(card.abpm.dipping_pct.value)}% <FlagBadge flag={card.abpm.dipping_pct.flag} /></span>
                      )}
                    </div>
                  </div>

                  {/* Key Holter values */}
                  <div style={{ fontSize: 12 }}>
                    <strong style={{ color: '#334155' }}>Holter:</strong>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 4 }}>
                      {card.holter?.qtc_ms && (
                        <span>QTc {fmt(card.holter.qtc_ms.value)} ms <FlagBadge flag={card.holter.qtc_ms.flag} /></span>
                      )}
                      {card.holter?.pvc_count && (
                        <span>PVCs {fmt(card.holter.pvc_count.value)} <FlagBadge flag={card.holter.pvc_count.flag} /></span>
                      )}
                      {card.holter?.vt_runs && (
                        <span>VT {fmt(card.holter.vt_runs.value)} <FlagBadge flag={card.holter.vt_runs.flag} /></span>
                      )}
                      {card.holter?.af_episodes && (
                        <span>AF {fmt(card.holter.af_episodes.value)} <FlagBadge flag={card.holter.af_episodes.flag} /></span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Protocol">
            <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{defs.protocol?.description}</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginTop: 16 }}>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>ABPM Recording</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{defs.protocol?.recording_methods?.abpm}</div>
              </div>
              <div style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>Holter Recording</div>
                <div style={{ fontSize: 12, color: '#64748b' }}>{defs.protocol?.recording_methods?.holter}</div>
              </div>
            </div>
          </Card>

          <Card title="Indications">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.protocol?.indications || []).map((item, i) => (
                <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6, lineHeight: 1.5 }}>{item}</li>
              ))}
            </ul>
          </Card>

          <Card title="ABPM Parameters">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Parameter</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Unit</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Ref Range</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.parameters?.abpm || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.label}</td>
                      <td style={{ padding: '8px 12px' }}>{row.unit}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.ref_low}–{row.ref_high}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 400 }}>{row.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Holter Parameters">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Parameter</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Unit</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Ref Range</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.parameters?.holter || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.label}</td>
                      <td style={{ padding: '8px 12px' }}>{row.unit}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.ref_low}–{row.ref_high}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 400 }}>{row.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Dipping Categories">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Category</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Range</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.dipping_categories || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}><DippingBadge category={row.category} /></td>
                      <td style={{ padding: '8px 12px' }}>{row.range}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.risk}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Diagnostic Patterns">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {(defs.diagnostic_patterns || []).map((row, i) => (
                <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#334155', marginBottom: 4 }}>{row.label}</div>
                  <div style={{ fontSize: 12, color: '#64748b' }}>{row.description}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Severity Levels">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', background: '#f8fafc' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Level</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Score Range</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.severity_levels || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px' }}><SeverityBadge severity={row.level} /></td>
                      <td style={{ padding: '8px 12px', fontWeight: 500 }}>{row.score_range}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{row.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Clinical Significance">
            <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{defs.clinical_significance}</p>
          </Card>

          <Card title="Standards & References">
            <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, margin: 0 }}>{defs.protocol?.standard}</p>
          </Card>
        </div>
      )}
    </div>
  )
}
