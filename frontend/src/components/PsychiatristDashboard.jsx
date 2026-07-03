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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const SEVERITY_COLORS = {
  'Minimal': '#10b981',
  'Mild': '#f59e0b',
  'Moderate': '#f97316',
  'Moderately Severe': '#ef4444',
  'Severe': '#991b1b',
  'No Risk': '#10b981',
  'Low Risk': '#f59e0b',
  'Moderate Risk': '#f97316',
  'High Risk': '#ef4444',
}

const AED_SEV_COLORS = {
  'beneficial': '#10b981',
  'low': '#3b82f6',
  'moderate': '#f59e0b',
  'high': '#ef4444',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'depression', label: 'Depression (PHQ-9)' },
  { id: 'anxiety', label: 'Anxiety (GAD-7)' },
  { id: 'suicidality', label: 'Suicidality (C-SSRS)' },
  { id: 'profiles', label: 'Patient Profiles' },
  { id: 'pnes', label: 'PNES Screening' },
  { id: 'aed-psych', label: 'AED Psychiatric Risk' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PsychiatristDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/psychiatrist/overview`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/psychiatrist/breakdown`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/psychiatrist/definitions`).then(r => r.data).catch(() => null),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Psychiatrist Dashboard...</div>

  const kpis = ov?.kpis || {}

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Psychiatrist Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Psychiatric comorbidity assessment, mood/anxiety screening, PNES differential, suicidality risk
      </p>

      {/* Tab Navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          {/* KPI Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Patients Screened" value={kpis.total_patients} color="#3b82f6" /></Card>
            <Card><KPI label="PHQ-9 Assessments" value={kpis.phq9_assessments} sub={`${kpis.phq9_elevated || 0} elevated`} color="#8b5cf6" /></Card>
            <Card><KPI label="GAD-7 Assessments" value={kpis.gad7_assessments} sub={`${kpis.gad7_elevated || 0} elevated`} color="#f59e0b" /></Card>
            <Card><KPI label="C-SSRS Screenings" value={kpis.cssrs_assessments} sub={`${kpis.cssrs_positive || 0} positive`} color="#ef4444" /></Card>
            <Card><KPI label="NDDI-E Assessments" value={kpis.nddie_assessments} color="#10b981" /></Card>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Avg PHQ-9 Score" value={kpis.avg_phq9} sub="/27" color="#8b5cf6" /></Card>
            <Card><KPI label="Avg GAD-7 Score" value={kpis.avg_gad7} sub="/21" color="#f59e0b" /></Card>
            <Card><KPI label="PNES Candidates" value={bd?.pnes_candidates?.length ?? '--'} color="#ec4899" /></Card>
          </div>

          {/* Charts Row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="PHQ-9 Depression Severity">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ov?.phq9_severity || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4,4,0,0]}>
                    {(ov?.phq9_severity || []).map((e, i) => (
                      <Cell key={i} fill={SEVERITY_COLORS[e.level] || COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="GAD-7 Anxiety Severity">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={ov?.gad7_severity || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="level" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4,4,0,0]}>
                    {(ov?.gad7_severity || []).map((e, i) => (
                      <Cell key={i} fill={SEVERITY_COLORS[e.level] || COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="C-SSRS Suicidality Risk">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={ov?.cssrs_risk || []} dataKey="count" nameKey="level" cx="50%" cy="50%"
                    outerRadius={80} label={({ level, count }) => count > 0 ? `${level}: ${count}` : ''}>
                    {(ov?.cssrs_risk || []).map((e, i) => (
                      <Cell key={i} fill={SEVERITY_COLORS[e.level] || COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {/* Depression Tab */}
      {tab === 'depression' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="PHQ-9 Score Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov?.phq9_severity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {(ov?.phq9_severity || []).map((e, i) => (
                    <Cell key={i} fill={SEVERITY_COLORS[e.level] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Patients with Depression (PHQ-9 >= 10)" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Patient</th>
                    <th style={{ padding: '8px 6px' }}>PHQ-9</th>
                    <th style={{ padding: '8px 6px' }}>Level</th>
                    <th style={{ padding: '8px 6px' }}>Top PHQ-9 Items</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.patient_profiles.filter(p => p.latest_phq9 >= 10).map(p => (
                    <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px' }}>{p.patient_id}</td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.latest_phq9}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge text={p.phq9_level} color={SEVERITY_COLORS[p.phq9_level] || '#64748b'} />
                      </td>
                      <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>
                        {(p.phq9_items || []).filter(i => i.score >= 2).map(i => i.item).join(', ') || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Anxiety Tab */}
      {tab === 'anxiety' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="GAD-7 Score Distribution" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov?.gad7_severity || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {(ov?.gad7_severity || []).map((e, i) => (
                    <Cell key={i} fill={SEVERITY_COLORS[e.level] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Patients with Anxiety (GAD-7 >= 10)" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Patient</th>
                    <th style={{ padding: '8px 6px' }}>GAD-7</th>
                    <th style={{ padding: '8px 6px' }}>Level</th>
                    <th style={{ padding: '8px 6px' }}>Top GAD-7 Items</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.patient_profiles.filter(p => p.latest_gad7 >= 10).map(p => (
                    <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px' }}>{p.patient_id}</td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.latest_gad7}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge text={p.gad7_level} color={SEVERITY_COLORS[p.gad7_level] || '#64748b'} />
                      </td>
                      <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>
                        {(p.gad7_items || []).filter(i => i.score >= 2).map(i => i.item).join(', ') || '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Suicidality Tab */}
      {tab === 'suicidality' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="C-SSRS Risk Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={ov?.cssrs_risk || []} dataKey="count" nameKey="level" cx="50%" cy="50%"
                  outerRadius={90} label={({ level, count }) => `${level}: ${count}`}>
                  {(ov?.cssrs_risk || []).map((e, i) => (
                    <Cell key={i} fill={SEVERITY_COLORS[e.level] || COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
          <Card title="C-SSRS Positive Patients">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              {bd?.patient_profiles?.filter(p => p.latest_cssrs > 0).length === 0 ? (
                <p style={{ color: '#64748b', fontSize: 13 }}>No patients with positive C-SSRS screens.</p>
              ) : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 6px' }}>Patient</th>
                      <th style={{ padding: '8px 6px' }}>C-SSRS Score</th>
                      <th style={{ padding: '8px 6px' }}>Risk Flags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd?.patient_profiles?.filter(p => p.latest_cssrs > 0).map(p => (
                      <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px' }}>{p.patient_id}</td>
                        <td style={{ padding: '6px', fontWeight: 600, color: '#ef4444' }}>{p.latest_cssrs}</td>
                        <td style={{ padding: '6px', fontSize: 11 }}>
                          {p.risk_flags.map((f, i) => <div key={i}>{f}</div>)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </Card>
        </div>
      )}

      {/* Patient Profiles Tab */}
      {tab === 'profiles' && bd && (
        <Card title={`Psychiatric Profiles (${bd.patient_profiles.length} patients)`}>
          <div style={{ maxHeight: 600, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px' }}>Patient</th>
                  <th style={{ padding: '8px 6px' }}>Age/Gender</th>
                  <th style={{ padding: '8px 6px' }}>PHQ-9</th>
                  <th style={{ padding: '8px 6px' }}>GAD-7</th>
                  <th style={{ padding: '8px 6px' }}>C-SSRS</th>
                  <th style={{ padding: '8px 6px' }}>NDDI-E</th>
                  <th style={{ padding: '8px 6px' }}>Seizures</th>
                  <th style={{ padding: '8px 6px' }}>Risk Flags</th>
                </tr>
              </thead>
              <tbody>
                {bd.patient_profiles.map(p => (
                  <tr key={p.patient_id} style={{
                    borderBottom: '1px solid #f1f5f9',
                    background: p.risk_flags.length >= 3 ? '#fef2f218' : undefined
                  }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px', fontSize: 11 }}>
                      {p.age ?? '--'} / {p.gender || '--'}
                    </td>
                    <td style={{ padding: '6px' }}>
                      {p.latest_phq9 != null ? (
                        <Badge text={`${p.latest_phq9} ${p.phq9_level}`}
                          color={SEVERITY_COLORS[p.phq9_level] || '#64748b'} />
                      ) : '--'}
                    </td>
                    <td style={{ padding: '6px' }}>
                      {p.latest_gad7 != null ? (
                        <Badge text={`${p.latest_gad7} ${p.gad7_level}`}
                          color={SEVERITY_COLORS[p.gad7_level] || '#64748b'} />
                      ) : '--'}
                    </td>
                    <td style={{ padding: '6px' }}>
                      {p.latest_cssrs != null ? (
                        <span style={{ color: p.latest_cssrs > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>
                          {p.latest_cssrs}
                        </span>
                      ) : '--'}
                    </td>
                    <td style={{ padding: '6px' }}>
                      {p.latest_nddie != null ? p.latest_nddie : '--'}
                    </td>
                    <td style={{ padding: '6px' }}>{p.seizure_count || '--'}</td>
                    <td style={{ padding: '6px', fontSize: 11, maxWidth: 250 }}>
                      {p.risk_flags.length > 0 ? p.risk_flags.map((f, i) => (
                        <div key={i} style={{ color: '#dc2626' }}>{f}</div>
                      )) : <span style={{ color: '#10b981' }}>No flags</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* PNES Screening Tab */}
      {tab === 'pnes' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title={`PNES Screening Candidates (${bd.pnes_candidates.length})`}>
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>
              Patients with >= 2 risk factors: psychiatric comorbidity + seizure burden
            </p>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Patient</th>
                    <th style={{ padding: '8px 6px' }}>Risk Factors</th>
                    <th style={{ padding: '8px 6px' }}>Flags</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.pnes_candidates.map(p => (
                    <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px' }}>
                        <Badge text={`${p.pnes_risk_factors} factors`} color="#ec4899" />
                      </td>
                      <td style={{ padding: '6px', fontSize: 11 }}>
                        {p.flags.map((f, i) => <div key={i}>{f}</div>)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <Card title="PNES Differentiating Features (Clinical Reference)">
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>
              Features suggesting psychogenic vs. epileptic seizures
            </p>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {bd.pnes_features.map((f, i) => (
                <li key={i} style={{ fontSize: 13, marginBottom: 4, color: '#334155' }}>{f}</li>
              ))}
            </ul>
          </Card>
          <Card title="Mood-Seizure Correlation" span={2}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bd.mood_seizure_correlation}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="phq9" name="PHQ-9" fill="#8b5cf6" radius={[4,4,0,0]} />
                <Bar dataKey="seizures" name="Seizures" fill="#ef4444" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* AED Psychiatric Risk Tab */}
      {tab === 'aed-psych' && (
        <Card title="AED Psychiatric Side-Effect Profile">
          <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>
            Evidence-based psychiatric effects of antiepileptic drugs observed in this cohort
          </p>
          <div style={{ maxHeight: 500, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px' }}>Drug</th>
                  <th style={{ padding: '8px 6px' }}>Risk Level</th>
                  <th style={{ padding: '8px 6px' }}>Psychiatric Effects</th>
                </tr>
              </thead>
              <tbody>
                {(ov?.aed_psychiatric_risk || []).map(a => (
                  <tr key={a.drug} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{a.drug}</td>
                    <td style={{ padding: '6px' }}>
                      <Badge text={a.severity} color={AED_SEV_COLORS[a.severity] || '#64748b'} />
                    </td>
                    <td style={{ padding: '6px', fontSize: 12, color: '#475569' }}>
                      {a.effects.join(', ')}
                    </td>
                  </tr>
                ))}
                {(ov?.aed_psychiatric_risk || []).length === 0 && (
                  <tr><td colSpan={3} style={{ padding: 12, color: '#94a3b8', textAlign: 'center' }}>No AED data in current cohort</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Psychiatry Concepts (${defs.concepts?.length || 0})`}>
            {(defs.concepts || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid #f1f5f9' }}>
                <strong style={{ fontSize: 14, color: '#1e293b' }}>{c.name}</strong>
                <p style={{ margin: '4px 0 0', fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{c.description}</p>
              </div>
            ))}
          </Card>
          <Card title="Quality Metrics">
            {(defs.quality_metrics || []).map((m, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <strong style={{ fontSize: 13, color: '#334155' }}>{m.name}</strong>
                <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{m.description}</p>
              </div>
            ))}
          </Card>
          <Card title="Compliance & Guidelines">
            {(defs.compliance || []).map((c, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <strong style={{ fontSize: 13, color: '#334155' }}>{c.name}</strong>
                <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{c.description}</p>
              </div>
            ))}
          </Card>
          <Card title="Remediation Strategies">
            {(defs.remediation_strategies || []).map((s, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
                <strong style={{ fontSize: 13, color: '#334155' }}>{s.name}</strong>
                <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{s.description}</p>
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
