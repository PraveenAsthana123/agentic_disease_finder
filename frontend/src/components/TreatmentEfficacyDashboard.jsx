import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, ScatterChart, Scatter, LineChart, Line } from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const RESPONSE_COLORS = { Excellent: '#10b981', Good: '#3b82f6', Partial: '#f59e0b', Poor: '#ef4444' }

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const TABS = ['Overview', 'Per-Patient', 'Per-Drug', 'Correlation', 'Definitions']

export default function TreatmentEfficacyDashboard() {
  const [tab, setTab] = useState(0)
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch overview on mount
  useEffect(() => {
    setLoading(true)
    axios.get(`${API_URL}/api/treatment-efficacy/overview`)
      .then(r => { setOverview(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  // Lazy fetch breakdown when Per-Patient, Per-Drug, or Correlation tabs selected
  useEffect(() => {
    if ((tab === 1 || tab === 2 || tab === 3) && !breakdown) {
      setLoading(true)
      axios.get(`${API_URL}/api/treatment-efficacy/breakdown`)
        .then(r => { setBreakdown(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, breakdown])

  // Lazy fetch definitions
  useEffect(() => {
    if (tab === 4 && !definitions) {
      setLoading(true)
      axios.get(`${API_URL}/api/treatment-efficacy/definitions`)
        .then(r => { setDefinitions(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, definitions])

  if (error) return <div style={{ padding: 40, color: '#ef4444', textAlign: 'center' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 20px', fontSize: 22, color: '#0f172a' }}>Treatment Efficacy Dashboard</h2>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            borderRadius: '8px 8px 0 0',
            background: tab === i ? '#3b82f6' : 'transparent',
            color: tab === i ? '#fff' : '#64748b',
          }}>{t}</button>
        ))}
      </div>

      {loading && !overview && <div style={{ textAlign: 'center', padding: 60, color: '#94a3b8' }}>Loading...</div>}

      {/* Tab 0: Overview */}
      {tab === 0 && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
          {/* KPI row */}
          <Card span={5}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Patients" value={overview.total_patients} color="#3b82f6" />
              <KPI label="Adherence %" value={overview.overall_adherence_pct != null ? `${overview.overall_adherence_pct}%` : '--'} color="#10b981" />
              <KPI label="On-Time %" value={overview.on_time_pct != null ? `${overview.on_time_pct}%` : '--'} color="#06b6d4" />
              <KPI label="Total Seizures" value={overview.total_seizures} color="#ef4444" />
              <KPI label="Unique Drugs" value={overview.unique_drugs} color="#8b5cf6" />
            </div>
          </Card>

          {/* Treatment response pie chart */}
          <Card title="Treatment Response Categories" span={2}>
            {overview.treatment_response_categories && overview.treatment_response_categories.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={overview.treatment_response_categories} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={90} label={({ category, pct }) => `${category} ${pct}%`}>
                    {overview.treatment_response_categories.map((entry, i) => (
                      <Cell key={i} fill={RESPONSE_COLORS[entry.category] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          {/* Monthly trend line chart */}
          <Card title="Monthly Adherence & Seizure Trend" span={3}>
            {overview.monthly_adherence_trend && overview.monthly_adherence_trend.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={overview.monthly_adherence_trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11 }} label={{ value: 'Adherence %', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} label={{ value: 'Seizure Count', angle: 90, position: 'insideRight', style: { fontSize: 11 } }} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="adherence_pct" stroke="#3b82f6" name="Adherence %" strokeWidth={2} />
                  <Line yAxisId="right" type="monotone" dataKey="seizure_count" stroke="#ef4444" name="Seizure Count" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          {/* Top side effects bar chart */}
          <Card title="Top Side Effects" span={5}>
            {overview.side_effect_profile && overview.side_effect_profile.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={overview.side_effect_profile}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="side_effect" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>
        </div>
      )}

      {/* Tab 1: Per-Patient */}
      {tab === 1 && (
        <Card title="Per-Patient Treatment Data">
          {!breakdown ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient ID', 'Drugs', 'Adherence %', 'On-Time %', 'Total Doses', 'Missed', 'Seizures', 'Avg Severity', 'Top Side Effect', 'Avg Mood', 'Response'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{Array.isArray(p.drug_names) ? p.drug_names.join(', ') : p.drug_names}</td>
                      <td style={{ padding: '6px 10px' }}>{p.adherence_pct}%</td>
                      <td style={{ padding: '6px 10px' }}>{p.on_time_pct}%</td>
                      <td style={{ padding: '6px 10px' }}>{p.total_doses}</td>
                      <td style={{ padding: '6px 10px' }}>{p.missed_doses}</td>
                      <td style={{ padding: '6px 10px' }}>{p.seizure_count}</td>
                      <td style={{ padding: '6px 10px' }}>{p.avg_seizure_severity}</td>
                      <td style={{ padding: '6px 10px' }}>{p.most_common_side_effect || '--'}</td>
                      <td style={{ padding: '6px 10px' }}>{p.avg_mood}</td>
                      <td style={{ padding: '6px 10px' }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                          background: `${RESPONSE_COLORS[p.response_category] || '#94a3b8'}20`,
                          color: RESPONSE_COLORS[p.response_category] || '#94a3b8'
                        }}>{p.response_category}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}

      {/* Tab 2: Per-Drug */}
      {tab === 2 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Drug Adherence Rates">
            {!breakdown ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
              breakdown.per_drug && breakdown.per_drug.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={breakdown.per_drug}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="drug_name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="avg_adherence_pct" fill="#3b82f6" name="Avg Adherence %" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
            )}
          </Card>

          <Card title="Drug Details">
            {breakdown && breakdown.per_drug ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      {['Drug Name', 'Patient Count', 'Avg Adherence %', 'Avg Side Effect Severity', 'Most Common Side Effects'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.per_drug.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.drug_name}</td>
                        <td style={{ padding: '6px 10px' }}>{d.patient_count}</td>
                        <td style={{ padding: '6px 10px' }}>{d.avg_adherence_pct}%</td>
                        <td style={{ padding: '6px 10px' }}>{d.avg_side_effect_severity}</td>
                        <td style={{ padding: '6px 10px' }}>{Array.isArray(d.most_common_side_effects) ? d.most_common_side_effects.join(', ') : d.most_common_side_effects || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div>}
          </Card>
        </div>
      )}

      {/* Tab 3: Correlation */}
      {tab === 3 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {/* Scatter: adherence vs seizure count */}
          <Card title="Adherence % vs Seizure Count" span={2}>
            {!breakdown || !overview ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
              overview.adherence_vs_seizures && overview.adherence_vs_seizures.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="adherence_pct" name="Adherence %" tick={{ fontSize: 11 }} label={{ value: 'Adherence %', position: 'insideBottom', offset: -5, style: { fontSize: 11 } }} />
                    <YAxis dataKey="seizure_count" name="Seizure Count" tick={{ fontSize: 11 }} label={{ value: 'Seizure Count', angle: -90, position: 'insideLeft', style: { fontSize: 11 } }} />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter data={overview.adherence_vs_seizures} fill="#3b82f6" />
                  </ScatterChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
            )}
          </Card>

          {/* Adherence by time of day */}
          <Card title="Adherence by Time of Day">
            {!breakdown ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
              breakdown.adherence_by_time_of_day ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={[
                    { time: 'Morning', pct: breakdown.adherence_by_time_of_day.morning },
                    { time: 'Afternoon', pct: breakdown.adherence_by_time_of_day.afternoon },
                    { time: 'Evening', pct: breakdown.adherence_by_time_of_day.evening },
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                    <Tooltip />
                    <Bar dataKey="pct" name="Adherence %" radius={[4, 4, 0, 0]}>
                      {['#f59e0b', '#3b82f6', '#8b5cf6'].map((c, i) => <Cell key={i} fill={c} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
            )}
          </Card>

          {/* Side effects by drug grouped bar */}
          <Card title="Side Effects by Drug" span={3}>
            {!breakdown ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
              breakdown.side_effects_by_drug && breakdown.side_effects_by_drug.length > 0 ? (() => {
                // Build grouped data: one bar per drug, sub-bars per side effect
                const allSideEffects = [...new Set(breakdown.side_effects_by_drug.flatMap(d => (d.side_effects || []).map(s => s.name)))]
                const grouped = breakdown.side_effects_by_drug.map(d => {
                  const row = { drug_name: d.drug_name }
                  ;(d.side_effects || []).forEach(s => { row[s.name] = s.count })
                  return row
                })
                return (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={grouped}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="drug_name" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Legend />
                      {allSideEffects.map((se, i) => (
                        <Bar key={se} dataKey={se} fill={COLORS[i % COLORS.length]} radius={[2, 2, 0, 0]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                )
              })() : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 4: Definitions */}
      {tab === 4 && (
        <Card title="Term Definitions">
          {!definitions ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
            <dl style={{ margin: 0 }}>
              {Object.entries(definitions).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 16, padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                  <dt style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{term}</dt>
                  <dd style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{def}</dd>
                </div>
              ))}
            </dl>
          )}
        </Card>
      )}
    </div>
  )
}
