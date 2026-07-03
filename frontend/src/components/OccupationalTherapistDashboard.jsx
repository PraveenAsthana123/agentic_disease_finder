import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

const COLORS = ['#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6']
const SEV_COLORS = { low: '#10b981', moderate: '#f59e0b', high: '#ef4444' }

export default function OccupationalTherapistDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/occupational-therapist/overview`),
      axios.get(`${API_URL}/api/occupational-therapist/breakdown`),
      axios.get(`${API_URL}/api/occupational-therapist/definitions`),
    ]).then(([ov, bd, df]) => {
      setOverview(ov.data)
      setBreakdown(bd.data)
      setDefs(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading OT dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'adl', label: 'ADL (Barthel)' },
    { id: 'qol', label: 'Quality of Life' },
    { id: 'sleepiness', label: 'Sleepiness' },
    { id: 'profiles', label: 'Patient Profiles' },
    { id: 'rehab', label: 'Rehab Candidates' },
    { id: 'aed', label: 'AED Functional Risk' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 8 }}>
        Occupational Therapist Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Functional outcome assessment, ADL independence, quality of life, and rehabilitation planning
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '7px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#0f766e' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview */}
      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Total Patients" value={k.total_patients} color="#0f766e" /></Card>
            <Card><KPI label="Barthel Assessments" value={k.barthel_assessments} /></Card>
            <Card><KPI label="QOLIE Assessments" value={k.qolie_assessments} /></Card>
            <Card><KPI label="Epworth Assessments" value={k.epworth_assessments} /></Card>
            <Card><KPI label="LSSS Assessments" value={k.lsss_assessments} /></Card>
            <Card><KPI label="ADL Impaired" value={k.barthel_impaired} sub="Barthel < 91" color="#ef4444" /></Card>
            <Card><KPI label="Poor QoL" value={k.qolie_poor} sub="QOLIE < 50" color="#f59e0b" /></Card>
            <Card><KPI label="Excessive Sleepiness" value={k.ess_elevated} sub="ESS >= 11" color="#f59e0b" /></Card>
            <Card><KPI label="Avg Barthel" value={k.avg_barthel} color="#06b6d4" /></Card>
            <Card><KPI label="Avg QOLIE" value={k.avg_qolie} color="#8b5cf6" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Barthel Index Distribution (ADL Independence)">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview?.barthel_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#0f766e" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Quality of Life Distribution (QOLIE-31)">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={(overview?.qolie_distribution || []).filter(d => d.count > 0)}
                       dataKey="count" nameKey="level" cx="50%" cy="50%"
                       outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview?.qolie_distribution || []).filter(d => d.count > 0).map((_, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Seizure Severity Distribution (LSSS)">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview?.lsss_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ef4444" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Daytime Sleepiness Distribution (Epworth)">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={overview?.ess_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* ADL (Barthel) Tab */}
      {tab === 'adl' && (
        <>
          <Card title="ADL Item-Level Analysis (Average Scores Across All Patients)" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview?.barthel_items || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 10]} />
                <YAxis dataKey="item" type="category" width={120} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => `${v}/10`} />
                <Bar dataKey="avg_score" fill="#0f766e" radius={[0,6,6,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ marginTop: 16 }}>
            <Card title="Barthel Independence Levels">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 8 }}>Score Range</th>
                    <th style={{ textAlign: 'left', padding: 8 }}>Classification</th>
                    <th style={{ textAlign: 'right', padding: 8 }}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview?.barthel_distribution || []).map((d, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 8, color: '#475569' }}>
                        {['0-20', '21-60', '61-90', '91-99', '100'][i]}
                      </td>
                      <td style={{ padding: 8 }}>
                        <span style={{
                          padding: '2px 8px', borderRadius: 12, fontSize: 12,
                          background: i <= 1 ? '#fef2f2' : i <= 2 ? '#fffbeb' : '#f0fdf4',
                          color: i <= 1 ? '#dc2626' : i <= 2 ? '#d97706' : '#16a34a',
                        }}>{d.level}</span>
                      </td>
                      <td style={{ padding: 8, textAlign: 'right', fontWeight: 600 }}>{d.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        </>
      )}

      {/* Quality of Life Tab */}
      {tab === 'qol' && (
        <>
          <Card title="QOLIE-31 Domain Scores (Average Across All Patients)">
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={overview?.qolie_domains || []} cx="50%" cy="50%" outerRadius={120}>
                <PolarGrid />
                <PolarAngleAxis dataKey="domain" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 100]} />
                <Radar dataKey="avg_score" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="QOLIE-31 Domain Breakdown">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={overview?.qolie_domains || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="domain" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={80} />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={(v) => `${v}/100`} />
                  <Bar dataKey="avg_score" fill="#8b5cf6" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="QoL Classification Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={(overview?.qolie_distribution || []).filter(d => d.count > 0)}
                       dataKey="count" nameKey="level" cx="50%" cy="50%"
                       outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                    {(overview?.qolie_distribution || []).filter(d => d.count > 0).map((_, i) => (
                      <Cell key={i} fill={['#ef4444', '#f59e0b', '#06b6d4', '#10b981'][i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* Sleepiness Tab */}
      {tab === 'sleepiness' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Epworth Sleepiness Scale Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={overview?.ess_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#f59e0b" radius={[6,6,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="ESS Scoring Guide">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Score Range</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Classification</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Clinical Action</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['0-5', 'Lower Normal', 'No action needed'],
                  ['6-10', 'Higher Normal', 'Monitor'],
                  ['11-12', 'Mild Sleepiness', 'Review AED regimen, sleep hygiene'],
                  ['13-15', 'Moderate Sleepiness', 'Investigate causes, consider referral'],
                  ['16-24', 'Severe Sleepiness', 'Urgent review, driving restriction'],
                ].map(([range, level, action], i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: 8 }}>{range}</td>
                    <td style={{ padding: 8 }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 12, fontSize: 12,
                        background: i <= 1 ? '#f0fdf4' : i <= 2 ? '#fffbeb' : '#fef2f2',
                        color: i <= 1 ? '#16a34a' : i <= 2 ? '#d97706' : '#dc2626',
                      }}>{level}</span>
                    </td>
                    <td style={{ padding: 8, color: '#475569', fontSize: 12 }}>{action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* Patient Profiles Tab */}
      {tab === 'profiles' && (
        <Card title={`Patient Functional Profiles (${(breakdown?.profiles || []).length} patients)`}>
          <div style={{ maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                  <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>Barthel</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>QOLIE</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>ESS</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>LSSS</th>
                  <th style={{ textAlign: 'center', padding: 8 }}>Seizures</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Risk Factors</th>
                  <th style={{ textAlign: 'left', padding: 8 }}>Suggested Goals</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.profiles || []).sort((a, b) => b.risk_count - a.risk_count).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: p.risk_count >= 3 ? '#fef2f2' : undefined }}>
                    <td style={{ padding: 8 }}>
                      <div style={{ fontWeight: 600 }}>{p.patient_id}</div>
                      <div style={{ fontSize: 11, color: '#94a3b8' }}>{p.name}{p.age ? `, ${p.age}${p.gender ? p.gender[0] : ''}` : ''}</div>
                    </td>
                    <td style={{ textAlign: 'center', padding: 8 }}>
                      {p.barthel_score != null ? (
                        <span style={{
                          padding: '2px 6px', borderRadius: 8, fontSize: 11,
                          background: p.barthel_score >= 91 ? '#f0fdf4' : p.barthel_score >= 61 ? '#fffbeb' : '#fef2f2',
                          color: p.barthel_score >= 91 ? '#16a34a' : p.barthel_score >= 61 ? '#d97706' : '#dc2626',
                        }}>{p.barthel_score}</span>
                      ) : '--'}
                    </td>
                    <td style={{ textAlign: 'center', padding: 8 }}>
                      {p.qolie_score != null ? (
                        <span style={{
                          padding: '2px 6px', borderRadius: 8, fontSize: 11,
                          background: p.qolie_score >= 70 ? '#f0fdf4' : p.qolie_score >= 50 ? '#fffbeb' : '#fef2f2',
                          color: p.qolie_score >= 70 ? '#16a34a' : p.qolie_score >= 50 ? '#d97706' : '#dc2626',
                        }}>{p.qolie_score}</span>
                      ) : '--'}
                    </td>
                    <td style={{ textAlign: 'center', padding: 8 }}>
                      {p.epworth_score != null ? (
                        <span style={{
                          padding: '2px 6px', borderRadius: 8, fontSize: 11,
                          background: p.epworth_score < 11 ? '#f0fdf4' : '#fef2f2',
                          color: p.epworth_score < 11 ? '#16a34a' : '#dc2626',
                        }}>{p.epworth_score}</span>
                      ) : '--'}
                    </td>
                    <td style={{ textAlign: 'center', padding: 8 }}>
                      {p.lsss_score != null ? (
                        <span style={{
                          padding: '2px 6px', borderRadius: 8, fontSize: 11,
                          background: p.lsss_score < 41 ? '#fffbeb' : '#fef2f2',
                          color: p.lsss_score < 41 ? '#d97706' : '#dc2626',
                        }}>{p.lsss_score}</span>
                      ) : '--'}
                    </td>
                    <td style={{ textAlign: 'center', padding: 8, fontWeight: 600,
                      color: p.seizure_count >= 3 ? '#dc2626' : '#475569' }}>{p.seizure_count}</td>
                    <td style={{ padding: 8, fontSize: 11, color: '#475569' }}>
                      {p.risk_factors.length > 0 ? p.risk_factors.map((rf, j) => (
                        <div key={j}>{rf}</div>
                      )) : <span style={{ color: '#94a3b8' }}>None identified</span>}
                    </td>
                    <td style={{ padding: 8, fontSize: 11 }}>
                      {p.suggested_goals.map((g, j) => (
                        <span key={j} style={{
                          display: 'inline-block', padding: '1px 6px', borderRadius: 8, margin: '1px 2px',
                          background: '#f0fdf4', color: '#0f766e', fontSize: 10,
                        }}>{g}</span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Rehab Candidates Tab */}
      {tab === 'rehab' && (
        <>
          <Card title={`Rehabilitation Candidates (>= 2 risk factors): ${(breakdown?.rehab_candidates || []).length} patients`}>
            <div style={{ maxHeight: 500, overflowY: 'auto' }}>
              {(breakdown?.rehab_candidates || []).sort((a, b) => b.risk_count - a.risk_count).map((p, i) => (
                <div key={i} style={{
                  padding: 14, marginBottom: 10, borderRadius: 10,
                  border: '1px solid #fecaca', background: '#fef2f2',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <div>
                      <span style={{ fontWeight: 700, fontSize: 14 }}>{p.patient_id}</span>
                      <span style={{ color: '#64748b', fontSize: 12, marginLeft: 8 }}>
                        {p.name}{p.age ? `, ${p.age}${p.gender ? p.gender[0] : ''}` : ''} | {p.disease || 'Unknown'}
                      </span>
                    </div>
                    <span style={{
                      padding: '2px 10px', borderRadius: 12, fontSize: 12, fontWeight: 600,
                      background: p.risk_count >= 4 ? '#dc2626' : '#f59e0b',
                      color: '#fff',
                    }}>{p.risk_count} risk factors</span>
                  </div>

                  <div style={{ display: 'flex', gap: 16, fontSize: 12, marginBottom: 8 }}>
                    <span>Barthel: <strong>{p.barthel_score ?? '--'}</strong> ({p.barthel_level || '--'})</span>
                    <span>QOLIE: <strong>{p.qolie_score ?? '--'}</strong> ({p.qolie_level || '--'})</span>
                    <span>ESS: <strong>{p.epworth_score ?? '--'}</strong> ({p.epworth_level || '--'})</span>
                    <span>LSSS: <strong>{p.lsss_score ?? '--'}</strong> ({p.lsss_level || '--'})</span>
                  </div>

                  <div style={{ fontSize: 11, color: '#475569' }}>
                    <strong>Risk factors:</strong> {p.risk_factors.join(' | ')}
                  </div>
                  <div style={{ fontSize: 11, marginTop: 4 }}>
                    <strong>Suggested goals:</strong>{' '}
                    {p.suggested_goals.map((g, j) => (
                      <span key={j} style={{
                        display: 'inline-block', padding: '1px 6px', borderRadius: 8, margin: '1px 2px',
                        background: '#ecfdf5', color: '#0f766e', fontSize: 10,
                      }}>{g}</span>
                    ))}
                  </div>
                  {p.medications.length > 0 && (
                    <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                      <strong>Medications:</strong> {p.medications.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>

          <div style={{ marginTop: 16 }}>
            <Card title="OT Goal Domains (Epilepsy-Specific)">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
                {(breakdown?.goal_domains || []).map((g, i) => (
                  <div key={i} style={{
                    padding: 12, borderRadius: 8, background: '#f0fdfa', textAlign: 'center',
                    border: '1px solid #99f6e4', fontSize: 12, color: '#0f766e', fontWeight: 500,
                  }}>{g}</div>
                ))}
              </div>
            </Card>
          </div>
        </>
      )}

      {/* AED Functional Risk Tab */}
      {tab === 'aed' && (
        <Card title="AED Functional Side-Effect Profiling">
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: 8 }}>Drug</th>
                <th style={{ textAlign: 'center', padding: 8 }}>Severity</th>
                <th style={{ textAlign: 'left', padding: 8 }}>Functional Effects</th>
              </tr>
            </thead>
            <tbody>
              {(overview?.aed_functional_risk || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: 8, fontWeight: 600 }}>{d.drug}</td>
                  <td style={{ textAlign: 'center', padding: 8 }}>
                    <span style={{
                      padding: '2px 10px', borderRadius: 12, fontSize: 11,
                      background: d.severity === 'high' ? '#fef2f2' : d.severity === 'moderate' ? '#fffbeb' : '#f0fdf4',
                      color: SEV_COLORS[d.severity] || '#475569',
                      fontWeight: 600,
                    }}>{d.severity}</span>
                  </td>
                  <td style={{ padding: 8, color: '#475569', fontSize: 12 }}>{d.effects.join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <>
          <Card title="OT Concepts">
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 14 }}>
                <div style={{ fontWeight: 600, color: '#0f766e', fontSize: 14 }}>{c.term}</div>
                <div style={{ color: '#475569', fontSize: 13, marginTop: 2 }}>{c.definition}</div>
              </div>
            ))}
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <Card title="Quality Metrics">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 6 }}>Metric</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>Target</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.quality_metrics || []).map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6 }}>{m.metric}</td>
                      <td style={{ padding: 6, color: '#16a34a', fontWeight: 500 }}>{m.target}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card title="Compliance Standards">
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: 6 }}>Standard</th>
                    <th style={{ textAlign: 'left', padding: 6 }}>Scope</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.compliance || []).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: 6, fontWeight: 600 }}>{c.standard}</td>
                      <td style={{ padding: 6, color: '#475569' }}>{c.scope}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          </div>

          <div style={{ marginTop: 16 }}>
            <Card title="Remediation Strategies">
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
                {(defs.remediation || []).map((r, i) => (
                  <li key={i} style={{ marginBottom: 6, color: '#475569' }}>{r}</li>
                ))}
              </ul>
            </Card>
          </div>
        </>
      )}
    </div>
  )
}
