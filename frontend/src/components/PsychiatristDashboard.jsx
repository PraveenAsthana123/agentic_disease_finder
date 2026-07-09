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

const RISK_COLORS = {
  'minimal': '#10b981',
  'mild': '#f59e0b',
  'moderate': '#f97316',
  'severe': '#ef4444',
}

const TREATMENT_COLORS = {
  'untreated': '#ef4444',
  'under_treatment': '#f59e0b',
  'stable': '#10b981',
  'treatment_resistant': '#991b1b',
  'none': '#94a3b8',
}

const PRIORITY_COLORS = {
  'critical': '#991b1b',
  'high': '#ef4444',
  'moderate': '#f59e0b',
  'standard': '#3b82f6',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'threshold-alerts', label: 'Threshold Alerts' },
  { id: 'comorbidity', label: 'Comorbidity & Risk' },
  { id: 'depression', label: 'Depression (PHQ-9)' },
  { id: 'anxiety', label: 'Anxiety (GAD-7)' },
  { id: 'suicidality', label: 'Suicidality (C-SSRS)' },
  { id: 'profiles', label: 'Patient Profiles' },
  { id: 'assessment', label: 'Assessment Report' },
  { id: 'pnes', label: 'PNES Screening' },
  { id: 'aed-psych', label: 'AED Psychiatric Risk' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PsychiatristDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [flags, setFlags] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/psychiatrist/overview`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/psychiatrist/breakdown`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/psychiatrist/definitions`).then(r => r.data).catch(() => null),
      axios.get(`${API_URL}/api/psychiatrist/threshold-flags`).then(r => r.data).catch(() => null),
    ]).then(([o, b, d, f]) => { setOv(o); setBd(b); setDefs(d); setFlags(f); setLoading(false) })
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
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Comorbidity Screen Rate" value={kpis.comorbidity_screen_rate != null ? `${kpis.comorbidity_screen_rate}%` : '--'} sub={`${kpis.screened_patients || 0} screened`} color="#06b6d4" /></Card>
            <Card><KPI label="Avg Behavioral Risk" value={kpis.avg_behavioral_risk} sub="/100" color="#f97316" /></Card>
            <Card><KPI label="Referrals to Neurology" value={kpis.referrals_to_neurology} sub="AED review" color="#3b82f6" /></Card>
            <Card><KPI label="High-Risk Patients" value={kpis.high_risk_patients} sub="risk ≥ 60" color="#ef4444" /></Card>
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

      {/* Threshold Alerts Tab */}
      {tab === 'threshold-alerts' && flags && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <Card title="Total Flagged Patients">
              <KPI label="Above threshold" value={flags.total_flagged} color="#ef4444" />
            </Card>
            <Card title="Total Alerts">
              <KPI label="Across all instruments" value={flags.total_alerts} color="#f59e0b" />
            </Card>
            <Card title="Critical Priority">
              <KPI label="C-SSRS positive" value={(flags.severity_distribution || []).find(s => s.priority === 'critical')?.count || 0} color="#991b1b" />
            </Card>
            <Card title="High Priority">
              <KPI label="3+ threshold violations" value={(flags.severity_distribution || []).find(s => s.priority === 'high')?.count || 0} color="#ef4444" />
            </Card>
          </div>
          <Card title="Instrument Threshold Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 6px' }}>Instrument</th>
                  <th style={{ padding: '8px 6px' }}>Threshold</th>
                  <th style={{ padding: '8px 6px' }}>Flagged</th>
                  <th style={{ padding: '8px 6px' }}>Recommended Action</th>
                </tr>
              </thead>
              <tbody>
                {(flags.instrument_summary || []).map(inst => (
                  <tr key={inst.instrument} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px', fontWeight: 600 }}>{inst.instrument}</td>
                    <td style={{ padding: '6px' }}>{'>='} {inst.cutoff}</td>
                    <td style={{ padding: '6px' }}>
                      <Badge text={`${inst.flagged_count} patients`} color={inst.flagged_count > 0 ? '#ef4444' : '#10b981'} />
                    </td>
                    <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>{inst.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card title="Priority Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={flags.severity_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="priority" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {(flags.severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={PRIORITY_COLORS[e.priority] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Flagged Patients (ordered by priority)" span={2}>
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ padding: '8px 6px' }}>Priority</th>
                    <th style={{ padding: '8px 6px' }}>Patient</th>
                    <th style={{ padding: '8px 6px' }}>Age/Gender</th>
                    <th style={{ padding: '8px 6px' }}>Alerts</th>
                    <th style={{ padding: '8px 6px' }}>Details</th>
                    <th style={{ padding: '8px 6px' }}>AED Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {(flags.flagged_patients || []).map(p => (
                    <tr key={p.patient_id} style={{
                      borderBottom: '1px solid #f1f5f9',
                      background: p.priority === 'critical' ? '#fef2f218' : undefined
                    }}>
                      <td style={{ padding: '6px' }}>
                        <Badge text={p.priority.toUpperCase()} color={PRIORITY_COLORS[p.priority] || '#64748b'} />
                      </td>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.name || p.patient_id}</td>
                      <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>{p.age || '--'} / {p.gender || '--'}</td>
                      <td style={{ padding: '6px' }}>
                        {p.alerts.map((a, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>
                            <Badge text={`${a.instrument}: ${a.score}`} color={
                              a.instrument === 'C-SSRS' ? '#991b1b' :
                              a.score >= 15 ? '#ef4444' :
                              a.score >= 10 ? '#f59e0b' : '#3b82f6'
                            } />
                            <span style={{ fontSize: 10, color: '#94a3b8', marginLeft: 4 }}>{a.severity}</span>
                          </div>
                        ))}
                      </td>
                      <td style={{ padding: '6px', fontSize: 11, color: '#64748b' }}>
                        {p.alerts.map((a, i) => (
                          <div key={i} style={{ marginBottom: 2 }}>{a.action}</div>
                        ))}
                      </td>
                      <td style={{ padding: '6px', fontSize: 11 }}>
                        {p.risky_aeds.length > 0
                          ? p.risky_aeds.map((a, i) => (
                              <div key={i} style={{ color: '#ef4444' }}>{a.drug} ({a.severity})</div>
                            ))
                          : <span style={{ color: '#10b981' }}>None</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* Comorbidity & Risk Tab */}
      {tab === 'comorbidity' && ov && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Psychiatric Comorbidity Distribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={(ov.comorbidity_distribution || []).slice(0, 10)} layout="vertical"
                margin={{ left: 120, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="condition" tick={{ fontSize: 11 }} width={120} />
                <Tooltip />
                <Bar dataKey="count" fill="#06b6d4" radius={[0,4,4,0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Behavioral Risk Severity">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={ov.risk_severity_distribution || []} dataKey="count" nameKey="level"
                  cx="50%" cy="50%" outerRadius={100}
                  label={({ level, count }) => count > 0 ? `${level}: ${count}` : ''}>
                  {(ov.risk_severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={RISK_COLORS[e.level] || COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
          <Card title="Treatment Status" span={1}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.treatment_status || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="status" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[4,4,0,0]}>
                  {(ov.treatment_status || []).map((e, i) => (
                    <Cell key={i} fill={TREATMENT_COLORS[e.status] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
          <Card title={`Referral Summary (${ov.referral_summary?.total || 0} total)`}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
              <div style={{ textAlign: 'center', padding: 16, background: '#eff6ff', borderRadius: 8 }}>
                <div style={{ fontSize: 32, fontWeight: 700, color: '#3b82f6' }}>{ov.referral_summary?.to_neurology || 0}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>To Neurology</div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>AED psychiatric review</div>
              </div>
              <div style={{ textAlign: 'center', padding: 16, background: '#f0fdf4', borderRadius: 8 }}>
                <div style={{ fontSize: 32, fontWeight: 700, color: '#10b981' }}>{ov.referral_summary?.from_neurology || 0}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>From Neurology</div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>Comorbidity assessment</div>
              </div>
            </div>
          </Card>
          <Card title="Patient Comorbidity Detail" span={2}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 6px' }}>Patient</th>
                    <th style={{ padding: '8px 6px' }}>Comorbidities</th>
                    <th style={{ padding: '8px 6px' }}>Risk Score</th>
                    <th style={{ padding: '8px 6px' }}>Severity</th>
                    <th style={{ padding: '8px 6px' }}>Treatment</th>
                    <th style={{ padding: '8px 6px' }}>Screened</th>
                    <th style={{ padding: '8px 6px' }}>Referrals</th>
                  </tr>
                </thead>
                <tbody>
                  {bd.patient_profiles.filter(p => p.comorbidity_count > 0 || p.behavioral_risk_score != null).map(p => (
                    <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9',
                      background: p.behavioral_risk_score >= 60 ? '#fef2f218' : undefined }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px', fontSize: 11, maxWidth: 200 }}>
                        {p.comorbidities.length > 0
                          ? p.comorbidities.map((c, i) => <div key={i}>{c}</div>)
                          : <span style={{ color: '#94a3b8' }}>None identified</span>}
                      </td>
                      <td style={{ padding: '6px', fontWeight: 600,
                        color: p.behavioral_risk_score >= 60 ? '#ef4444' : p.behavioral_risk_score >= 35 ? '#f97316' : '#10b981' }}>
                        {p.behavioral_risk_score != null ? p.behavioral_risk_score : '--'}
                      </td>
                      <td style={{ padding: '6px' }}>
                        {p.risk_severity && <Badge text={p.risk_severity} color={RISK_COLORS[p.risk_severity] || '#64748b'} />}
                      </td>
                      <td style={{ padding: '6px' }}>
                        {p.treatment_status && p.treatment_status !== 'none' && (
                          <Badge text={p.treatment_status.replace('_', ' ')} color={TREATMENT_COLORS[p.treatment_status] || '#64748b'} />
                        )}
                      </td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>
                        {p.screened ? <span style={{ color: '#10b981', fontWeight: 600 }}>Yes</span>
                          : <span style={{ color: '#ef4444' }}>No</span>}
                      </td>
                      <td style={{ padding: '6px', fontSize: 11 }}>
                        {p.referrals.length > 0
                          ? p.referrals.map((r, i) => (
                            <div key={i}><Badge text={r.action.replace('refer_to_', '→ ')} color="#3b82f6" /></div>
                          ))
                          : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
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

      {/* Assessment Report Tab */}
      {tab === 'assessment' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Psychiatric Assessment Report — Per-Patient Summary">
            <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 12px' }}>
              Comprehensive psychiatric assessment integrating PHQ-9, GAD-7, C-SSRS, NDDI-E, comorbidity screening, behavioral risk, and referral history.
            </p>
            <div style={{ maxHeight: 700, overflow: 'auto' }}>
              {bd.patient_profiles.map(p => (
                <div key={p.patient_id} style={{
                  border: '1px solid #e2e8f0', borderRadius: 10, padding: 16, marginBottom: 12,
                  background: p.behavioral_risk_score >= 60 ? '#fef2f208' : '#fff'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                    <div>
                      <strong style={{ fontSize: 15, color: '#1e293b' }}>{p.patient_id}</strong>
                      <span style={{ fontSize: 12, color: '#64748b', marginLeft: 8 }}>
                        {p.age ? `${p.age}y` : ''} {p.gender || ''} — {p.disease || 'epilepsy'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: 6 }}>
                      {p.risk_severity && <Badge text={`Risk: ${p.risk_severity}`} color={RISK_COLORS[p.risk_severity] || '#64748b'} />}
                      {p.behavioral_risk_score != null && (
                        <Badge text={`Score: ${p.behavioral_risk_score}`}
                          color={p.behavioral_risk_score >= 60 ? '#ef4444' : p.behavioral_risk_score >= 35 ? '#f97316' : '#10b981'} />
                      )}
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginBottom: 10 }}>
                    <div style={{ padding: 8, background: '#f8fafc', borderRadius: 6, textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#64748b' }}>PHQ-9</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: SEVERITY_COLORS[p.phq9_level] || '#334155' }}>
                        {p.latest_phq9 ?? '--'}
                      </div>
                      {p.phq9_level && <div style={{ fontSize: 10, color: '#94a3b8' }}>{p.phq9_level}</div>}
                    </div>
                    <div style={{ padding: 8, background: '#f8fafc', borderRadius: 6, textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#64748b' }}>GAD-7</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: SEVERITY_COLORS[p.gad7_level] || '#334155' }}>
                        {p.latest_gad7 ?? '--'}
                      </div>
                      {p.gad7_level && <div style={{ fontSize: 10, color: '#94a3b8' }}>{p.gad7_level}</div>}
                    </div>
                    <div style={{ padding: 8, background: '#f8fafc', borderRadius: 6, textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#64748b' }}>C-SSRS</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: p.latest_cssrs > 0 ? '#ef4444' : '#10b981' }}>
                        {p.latest_cssrs ?? '--'}
                      </div>
                    </div>
                    <div style={{ padding: 8, background: '#f8fafc', borderRadius: 6, textAlign: 'center' }}>
                      <div style={{ fontSize: 11, color: '#64748b' }}>NDDI-E</div>
                      <div style={{ fontSize: 18, fontWeight: 700, color: (p.latest_nddie || 0) >= 15 ? '#ef4444' : '#334155' }}>
                        {p.latest_nddie ?? '--'}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, fontSize: 12 }}>
                    <div>
                      <div style={{ fontWeight: 600, color: '#334155', marginBottom: 4 }}>Comorbidities</div>
                      {p.comorbidities.length > 0
                        ? p.comorbidities.map((c, i) => <div key={i} style={{ color: '#475569' }}>• {c}</div>)
                        : <div style={{ color: '#94a3b8' }}>None identified</div>}
                    </div>
                    <div>
                      <div style={{ fontWeight: 600, color: '#334155', marginBottom: 4 }}>Risk Flags</div>
                      {p.risk_flags.length > 0
                        ? p.risk_flags.map((f, i) => <div key={i} style={{ color: '#dc2626' }}>• {f}</div>)
                        : <div style={{ color: '#10b981' }}>No flags</div>}
                    </div>
                    <div>
                      <div style={{ fontWeight: 600, color: '#334155', marginBottom: 4 }}>Referrals & Status</div>
                      {p.treatment_status && p.treatment_status !== 'none' && (
                        <div style={{ marginBottom: 4 }}>
                          Treatment: <Badge text={p.treatment_status.replace('_', ' ')} color={TREATMENT_COLORS[p.treatment_status] || '#64748b'} />
                        </div>
                      )}
                      {p.referrals.length > 0
                        ? p.referrals.map((r, i) => (
                          <div key={i} style={{ color: '#475569' }}>• {r.action.replace('refer_to_', '→ ')} ({r.actor})</div>
                        ))
                        : <div style={{ color: '#94a3b8' }}>No referrals</div>}
                      {p.screening_date && (
                        <div style={{ color: '#94a3b8', marginTop: 4 }}>Screened: {p.screening_date}</div>
                      )}
                    </div>
                  </div>
                  {p.medications.length > 0 && (
                    <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
                      <strong>Medications:</strong> {p.medications.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>
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
