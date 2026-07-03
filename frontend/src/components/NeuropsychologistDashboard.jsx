import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const LEVEL_COLORS = {
  'Extremely Low': '#991b1b',
  'Very Low': '#dc2626',
  'Low': '#ef4444',
  'Low Average': '#f59e0b',
  'Average': '#10b981',
  'High Average': '#3b82f6',
  'Superior': '#7c3aed',
  'Very Superior': '#6d28d9',
}

const SEVERITY_COLORS = {
  high: '#ef4444',
  moderate: '#f59e0b',
  low: '#3b82f6',
  minimal: '#10b981',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'wais', label: 'WAIS Profiles' },
  { id: 'digit', label: 'Digit Span' },
  { id: 'screening', label: 'Cognitive Screening' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'aed', label: 'AED Cognitive Risk' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function NeuropsychologistDashboard() {
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
      axios.get(`${API_URL}/api/neuropsychologist/overview`),
      axios.get(`${API_URL}/api/neuropsychologist/breakdown`),
      axios.get(`${API_URL}/api/neuropsychologist/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Neuropsychologist Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, color: '#64748b' }}>{overview?.message || 'No data'}</div>

  const ov = overview
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Neuropsychologist Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Cognitive assessment battery, IQ profiles, memory/executive function analysis, AED cognitive impact
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card title="Patients Assessed"><KPI value={ov.total_patients_assessed} label="Patients" color="#3b82f6" /></Card>
          <Card title="Total Assessments"><KPI value={ov.total_assessments} label="Assessments" sub={`WAIS ${ov.wais_count} / MoCA ${ov.moca_count} / MMSE ${ov.mmse_count} / DS ${ov.digit_span_count}`} color="#8b5cf6" /></Card>
          <Card title="Mean FSIQ"><KPI value={ov.avg_fsiq} label="Full-Scale IQ" sub={`${ov.iq_below_average} below average`} color={ov.avg_fsiq < 90 ? '#f59e0b' : '#10b981'} /></Card>
          <Card title="Mean MoCA"><KPI value={ov.avg_moca} label="out of 30" sub={`${ov.moca_impaired} impaired (<26)`} color={ov.avg_moca < 26 ? '#ef4444' : '#10b981'} /></Card>

          <Card title="Mean MMSE"><KPI value={ov.avg_mmse} label="out of 30" color="#06b6d4" /></Card>
          <Card title="Mean Digit Span"><KPI value={ov.avg_digit_span} label="Scaled Score" sub="Mean=10, SD=3" color="#8b5cf6" /></Card>
          <Card title="Depression (PHQ-9)"><KPI value={ov.mood_comorbidity?.phq9_elevated} label={`of ${ov.mood_comorbidity?.phq9_assessed} elevated`} color="#ef4444" /></Card>
          <Card title="Anxiety (GAD-7)"><KPI value={ov.mood_comorbidity?.gad7_elevated} label={`of ${ov.mood_comorbidity?.gad7_assessed} elevated`} color="#f59e0b" /></Card>

          {/* IQ Distribution */}
          <Card title="IQ Classification Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.iq_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients">
                  {(ov.iq_distribution || []).map((e, i) => (
                    <Cell key={i} fill={LEVEL_COLORS[e.level] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* WAIS Index Profile Radar */}
          <Card title="WAIS Index Profile (Mean Subtest Scores)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={ov.index_profile || []}>
                <PolarGrid />
                <PolarAngleAxis dataKey="label" fontSize={11} />
                <PolarRadiusAxis domain={[0, 15]} fontSize={10} />
                <Radar dataKey="mean_score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} name="Mean Score" />
                <Tooltip formatter={v => v?.toFixed(1)} />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* MoCA Distribution */}
          <Card title="MoCA Score Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.moca_distribution || []} dataKey="count" nameKey="category" cx="50%" cy="50%" outerRadius={80} label={e => `${e.category}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.moca_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Digit Span Profile */}
          <Card title="Digit Span Profile (Mean Spans)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.digit_span_profile || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" fontSize={12} />
                <YAxis fontSize={11} domain={[0, 10]} />
                <Tooltip />
                <Bar dataKey="mean_span" name="Mean Span" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Daily Activity */}
          {ov.daily_activity?.length > 0 && (
            <Card title="Daily Pipeline Activity" span={4}>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={ov.daily_activity}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={10} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* WAIS PROFILES TAB */}
      {tab === 'wais' && (
        <Card title="WAIS-IV Subtest Profiles" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient', 'FSIQ', 'Level', 'VCI', 'PRI', 'WMI', 'PSI', 'Interpretation', 'Date'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.wais_profiles || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px' }}>{p.patient_id}</td>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{p.fsiq}</td>
                    <td style={{ padding: '6px' }}><Badge text={p.level || '--'} color={LEVEL_COLORS[p.level] || '#64748b'} /></td>
                    <td style={{ padding: '6px' }}>{p.VCI ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.PRI ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.WMI ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.PSI ?? '--'}</td>
                    <td style={{ padding: '6px', maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.interpretation}</td>
                    <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>{p.date?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DIGIT SPAN TAB */}
      {tab === 'digit' && (
        <Card title="Digit Span Detailed Profiles" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient', 'Scaled Score', 'Level', 'Forward Span', 'Backward Span', 'Sequencing Span', 'Forward Raw', 'Backward Raw', 'Total Raw', 'Date'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.digit_span_profiles || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px' }}>{p.patient_id}</td>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{p.scaled_score}</td>
                    <td style={{ padding: '6px' }}><Badge text={p.level || '--'} color={LEVEL_COLORS[p.level] || '#64748b'} /></td>
                    <td style={{ padding: '6px' }}>{p.forward_span ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.backward_span ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.sequencing_span ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.forward_raw ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.backward_raw ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.total_raw ?? '--'}</td>
                    <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>{p.date?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* COGNITIVE SCREENING TAB */}
      {tab === 'screening' && (
        <Card title="Cognitive Screening (MoCA + MMSE)" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient', 'Instrument', 'Score', 'Max', 'Level', 'Interpretation', 'Date'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.cognitive_screening || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px' }}>{s.patient_id}</td>
                    <td style={{ padding: '6px' }}><Badge text={s.instrument} color={s.instrument === 'MOCA' ? '#3b82f6' : '#8b5cf6'} /></td>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{s.score}</td>
                    <td style={{ padding: '6px' }}>{s.max_score}</td>
                    <td style={{ padding: '6px' }}>{s.level}</td>
                    <td style={{ padding: '6px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.interpretation}</td>
                    <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>{s.date?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* PATIENT PROFILES TAB */}
      {tab === 'patients' && (
        <Card title="Per-Patient Cognitive Profiles" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['Patient', 'Age', 'Gender', 'Disease', 'FSIQ', 'IQ Level', 'MoCA', 'PHQ-9', 'GAD-7', 'Seizures', 'Tests', 'Risk Factors'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patient_profiles || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0', background: p.risk_factors?.length > 2 ? '#fef2f2' : undefined }}>
                    <td style={{ padding: '6px' }}>{p.patient_id}</td>
                    <td style={{ padding: '6px' }}>{p.age ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.gender ?? '--'}</td>
                    <td style={{ padding: '6px', fontSize: 11 }}>{p.disease ?? '--'}</td>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{p.fsiq ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.iq_level ? <Badge text={p.iq_level} color={LEVEL_COLORS[p.iq_level] || '#64748b'} /> : '--'}</td>
                    <td style={{ padding: '6px' }}>{p.moca_score ?? '--'}</td>
                    <td style={{ padding: '6px', color: p.phq9_score >= 10 ? '#ef4444' : undefined }}>{p.phq9_score ?? '--'}</td>
                    <td style={{ padding: '6px', color: p.gad7_score >= 10 ? '#f59e0b' : undefined }}>{p.gad7_score ?? '--'}</td>
                    <td style={{ padding: '6px' }}>{p.seizure_count}</td>
                    <td style={{ padding: '6px', fontSize: 11 }}>{p.instruments_completed?.join(', ')}</td>
                    <td style={{ padding: '6px', fontSize: 11 }}>
                      {p.risk_factors?.length > 0 ? (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                          {p.risk_factors.map((rf, j) => (
                            <Badge key={j} text={rf} color="#ef4444" />
                          ))}
                        </div>
                      ) : <span style={{ color: '#10b981' }}>None</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* AED COGNITIVE RISK TAB */}
      {tab === 'aed' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="AED Cognitive Side-Effect Profiles">
            {(ov.aed_cognitive_risk || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>
                No assessed patients currently on AEDs with documented cognitive effects.
                AED cognitive risk data is derived from medication records cross-referenced
                with the neuropsychology knowledge base.
              </p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      {['Drug', 'Cognitive Severity', 'Patients on Drug', 'Known Cognitive Effects'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.aed_cognitive_risk || []).map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{d.drug}</td>
                        <td style={{ padding: '6px' }}><Badge text={d.severity} color={SEVERITY_COLORS[d.severity] || '#64748b'} /></td>
                        <td style={{ padding: '6px' }}>{d.patients_on_drug}</td>
                        <td style={{ padding: '6px', fontSize: 11 }}>{d.effects?.join('; ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
          <Card title="AED Cognitive Risk Reference">
            <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
              Evidence-based cognitive side-effect severity ratings for common anti-seizure medications.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
              {[
                { level: 'High', color: '#ef4444', drugs: 'Topiramate, Phenobarbital' },
                { level: 'Moderate', color: '#f59e0b', drugs: 'Carbamazepine, Phenytoin, Zonisamide, Perampanel, Clobazam' },
                { level: 'Low', color: '#3b82f6', drugs: 'Valproate, Levetiracetam, Lacosamide, Oxcarbazepine, Gabapentin, Pregabalin' },
                { level: 'Minimal', color: '#10b981', drugs: 'Lamotrigine, Brivaracetam' },
              ].map((g, i) => (
                <div key={i} style={{ padding: 10, borderRadius: 8, border: `1px solid ${g.color}20`, background: g.color + '08' }}>
                  <div style={{ fontWeight: 600, color: g.color, fontSize: 13, marginBottom: 4 }}>{g.level} Risk</div>
                  <div style={{ fontSize: 12, color: '#475569' }}>{g.drugs}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* PIPELINE LOG TAB */}
      {tab === 'pipeline' && (
        <Card title="Recent Pipeline Activity" span={4}>
          {ov.daily_activity?.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={ov.daily_activity}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#64748b' }}>No pipeline activity recorded.</p>}
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {['concepts', 'quality_metrics', 'compliance', 'remediation_strategies'].map(section => (
            defs[section]?.length > 0 && (
              <Card key={section} title={section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
                {defs[section].map((item, i) => (
                  <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs[section].length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                    <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                  </div>
                ))}
              </Card>
            )
          ))}
        </div>
      )}
    </div>
  )
}
