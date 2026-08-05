import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#4caf50', '#ff9800', '#f44336', '#9c27b0', '#1e88e5', '#00bcd4']
const SEVERITY_COLORS = { Mild: '#4caf50', Moderate: '#ff9800', Severe: '#f44336', Critical: '#9c27b0' }

function LSSSDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedPatient, setSelectedPatient] = useState('')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/lsss/overview`),
      axios.get(`${API_URL}/api/lsss/breakdown`),
      axios.get(`${API_URL}/api/lsss/definitions`),
    ])
      .then(([ov, bk, df]) => {
        setOverview(ov.data)
        setBreakdown(bk.data)
        setDefinitions(df.data)
        setError(null)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'patients', label: 'Per Patient' },
    { id: 'items', label: 'Item Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 32, textAlign: 'center' }}>Loading LSSS data…</div>
  if (error) return <div style={{ padding: 32, color: '#f44336' }}>Error: {error}</div>
  if (!overview) return null

  return (
    <div style={{ padding: 24, fontFamily: 'sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
          Liverpool Seizure Severity Scale (LSSS)
        </h2>
        <p style={{ margin: '4px 0 0', color: '#666', fontSize: 14 }}>
          Patient-reported seizure severity · 20-item scale · max score 80 (higher = worse)
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid #ddd' }}>
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            style={{
              padding: '8px 20px', border: 'none', cursor: 'pointer', fontWeight: 600,
              borderBottom: activeTab === t.id ? '3px solid #1e88e5' : '3px solid transparent',
              background: 'none', color: activeTab === t.id ? '#1e88e5' : '#555',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && overview && (
        <div>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 16, marginBottom: 32 }}>
            {[
              { label: 'Total Assessments', value: overview.total_assessments, color: '#1e88e5' },
              { label: 'Unique Patients', value: overview.unique_patients, color: '#7c4dff' },
              { label: 'Avg Score', value: `${overview.avg_score} / 80`, color: '#ff9800' },
              { label: 'High-Risk Patients', value: overview.high_risk_patient_count, color: '#f44336' },
              { label: 'Score Range', value: `${overview.min_score}–${overview.max_score}`, color: '#4caf50' },
            ].map(kpi => (
              <div key={kpi.label} style={{
                background: '#fff', border: `2px solid ${kpi.color}20`,
                borderRadius: 10, padding: 20, textAlign: 'center', boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
              }}>
                <div style={{ fontSize: 26, fontWeight: 800, color: kpi.color }}>{kpi.value}</div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{kpi.label}</div>
              </div>
            ))}
          </div>

          {/* Two charts side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 32 }}>
            {/* Severity Distribution */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
              <h4 style={{ margin: '0 0 16px', fontSize: 14, color: '#333' }}>Severity Distribution</h4>
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie
                    data={Object.entries(overview.severity_distribution || {}).map(([k, v]) => ({ name: k, value: v }))}
                    cx="50%" cy="50%" outerRadius={80}
                    dataKey="value" nameKey="name" label={({ name, value }) => `${name}: ${value}`}
                  >
                    {Object.keys(overview.severity_distribution || {}).map((k, i) => (
                      <Cell key={k} fill={SEVERITY_COLORS[k] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Score Histogram */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
              <h4 style={{ margin: '0 0 16px', fontSize: 14, color: '#333' }}>Score Distribution (bins of 10)</h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview.score_histogram || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#1e88e5" radius={4} name="Assessments" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Monthly Trend */}
          {(overview.monthly_trend || []).length > 0 && (
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
              <h4 style={{ margin: '0 0 16px', fontSize: 14, color: '#333' }}>Monthly Trend — Assessments & Avg Score</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={overview.monthly_trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" domain={[0, 80]} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="assessments" stroke="#1e88e5" strokeWidth={2} name="Assessments" dot />
                  <Line yAxisId="right" type="monotone" dataKey="avg_score" stroke="#ff9800" strokeWidth={2} name="Avg Score" dot />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Per Patient Tab */}
      {activeTab === 'patients' && breakdown && (
        <div>
          <h4 style={{ margin: '0 0 16px', fontSize: 14, color: '#333' }}>
            Per-Patient LSSS Summary ({(breakdown.patient_summary || []).length} patients)
          </h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  {['Patient', 'Assessments', 'Avg Score', 'Min', 'Max', 'Latest', 'Level', 'Trend', 'Last Date'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, borderBottom: '2px solid #ddd', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.patient_summary || []).map((p, i) => {
                  const trendColor = p.trend === 'worsening' ? '#f44336' : p.trend === 'improving' ? '#4caf50' : '#888'
                  const trendIcon = p.trend === 'worsening' ? '↑' : p.trend === 'improving' ? '↓' : '→'
                  return (
                    <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{p.assessments}</td>
                      <td style={{ padding: '8px 12px' }}>{p.avg_score}</td>
                      <td style={{ padding: '8px 12px' }}>{p.min_score}</td>
                      <td style={{ padding: '8px 12px' }}>{p.max_score}</td>
                      <td style={{ padding: '8px 12px', fontWeight: 700 }}>{p.latest_score}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <span style={{
                          background: SEVERITY_COLORS[p.latest_level] + '20',
                          color: SEVERITY_COLORS[p.latest_level],
                          padding: '2px 8px', borderRadius: 4, fontWeight: 700, fontSize: 12,
                        }}>
                          {p.latest_level}
                        </span>
                      </td>
                      <td style={{ padding: '8px 12px', color: trendColor, fontWeight: 700 }}>
                        {trendIcon} {p.trend}
                      </td>
                      <td style={{ padding: '8px 12px', color: '#666' }}>{p.latest_date}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Recent assessment log */}
          <h4 style={{ margin: '32px 0 16px', fontSize: 14, color: '#333' }}>Recent Assessment Log</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  {['Date', 'Patient', 'Score', 'Level', 'Interpretation', 'Examiner'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, borderBottom: '2px solid #ddd' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.assessment_log || []).slice(0, 30).map((a, i) => (
                  <tr key={a.id} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                    <td style={{ padding: '8px 12px', color: '#666' }}>{a.date}</td>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{a.patient_id}</td>
                    <td style={{ padding: '8px 12px', fontWeight: 700 }}>{a.score} / {a.max_score}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        background: (SEVERITY_COLORS[a.level?.charAt(0).toUpperCase() + a.level?.slice(1)] || '#aaa') + '20',
                        color: SEVERITY_COLORS[a.level?.charAt(0).toUpperCase() + a.level?.slice(1)] || '#aaa',
                        padding: '2px 8px', borderRadius: 4, fontWeight: 700, fontSize: 12,
                      }}>
                        {a.level}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px', color: '#555' }}>{a.interpretation}</td>
                    <td style={{ padding: '8px 12px', color: '#666' }}>{a.examiner}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Item Analysis Tab */}
      {activeTab === 'items' && breakdown && (
        <div>
          <h4 style={{ margin: '0 0 16px', fontSize: 14, color: '#333' }}>
            Average Score per LSSS Item (1–4 scale; higher = more severe)
          </h4>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={(breakdown.item_averages || []).map(item => ({
                ...item,
                label: definitions?.item_labels?.[item.item] || item.item,
              }))}
              layout="vertical"
              margin={{ left: 160, right: 30, top: 8, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 4]} tick={{ fontSize: 11 }} />
              <YAxis dataKey="label" type="category" tick={{ fontSize: 11 }} width={160} />
              <Tooltip formatter={(v) => v.toFixed(2)} />
              <Bar dataKey="avg_score" fill="#7c4dff" radius={3} name="Avg Score" />
            </BarChart>
          </ResponsiveContainer>

          <div style={{ marginTop: 24, display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px,1fr))', gap: 12 }}>
            {(breakdown.item_averages || []).map(item => (
              <div key={item.item} style={{
                background: '#fff', borderRadius: 8, padding: 14,
                border: '1px solid #eee', display: 'flex', alignItems: 'center', gap: 12,
              }}>
                <div style={{
                  width: 48, height: 48, borderRadius: '50%', flexShrink: 0,
                  background: `conic-gradient(#7c4dff ${(item.avg_score / 4) * 100}%, #eee 0%)`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, fontWeight: 700,
                }}>
                  {item.avg_score.toFixed(1)}
                </div>
                <div>
                  <div style={{ fontWeight: 700, fontSize: 13 }}>{definitions?.item_labels?.[item.item] || item.item}</div>
                  <div style={{ color: '#888', fontSize: 12 }}>{item.responses} responses</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {activeTab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
          <div style={{ background: '#fff', borderRadius: 10, padding: 24, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', gridColumn: '1/-1' }}>
            <h4 style={{ margin: '0 0 8px', color: '#1e88e5' }}>About LSSS</h4>
            <p style={{ margin: 0, color: '#555', lineHeight: 1.6 }}>{definitions.description}</p>
          </div>

          <div style={{ background: '#fff', borderRadius: 10, padding: 24, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
            <h4 style={{ margin: '0 0 16px', color: '#333' }}>Severity Thresholds</h4>
            {(definitions.severity_thresholds || []).map(t => (
              <div key={t.level} style={{ marginBottom: 12, padding: 12, borderRadius: 8, background: (SEVERITY_COLORS[t.level] || '#aaa') + '10', borderLeft: `4px solid ${SEVERITY_COLORS[t.level] || '#aaa'}` }}>
                <div style={{ fontWeight: 700, color: SEVERITY_COLORS[t.level] || '#aaa' }}>
                  {t.level} ({t.min}–{t.max})
                </div>
                <div style={{ fontSize: 13, color: '#555', marginTop: 4 }}>{t.description}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff', borderRadius: 10, padding: 24, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
            <h4 style={{ margin: '0 0 16px', color: '#333' }}>Subscales</h4>
            {(definitions.subscales || []).map(s => (
              <div key={s.name} style={{ marginBottom: 16 }}>
                <div style={{ fontWeight: 700, color: '#333' }}>{s.name} Subscale</div>
                <div style={{ fontSize: 13, color: '#555', margin: '4px 0' }}>{s.description}</div>
                <div style={{ fontSize: 12, color: '#888' }}>Items: {s.items.join(', ')}</div>
              </div>
            ))}

            <h4 style={{ margin: '24px 0 12px', color: '#333' }}>Clinical Uses</h4>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(definitions.clinical_use || []).map(u => (
                <li key={u} style={{ fontSize: 13, color: '#555', marginBottom: 4 }}>{u}</li>
              ))}
            </ul>
          </div>

          <div style={{ background: '#fff', borderRadius: 10, padding: 24, boxShadow: '0 1px 4px rgba(0,0,0,0.06)', gridColumn: '1/-1' }}>
            <h4 style={{ margin: '0 0 16px', color: '#333' }}>References</h4>
            {(definitions.references || []).map((r, i) => (
              <div key={i} style={{ fontSize: 13, color: '#555', marginBottom: 8, paddingLeft: 16, borderLeft: '3px solid #1e88e5' }}>{r}</div>
            ))}
            <div style={{ marginTop: 12, fontSize: 12, color: '#888' }}>
              Data source: {definitions.data_source}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default LSSSDashboard
