import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Legend
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
  const colors = {
    red: { bg: '#fee2e2', text: '#991b1b' },
    amber: { bg: '#fef3c7', text: '#92400e' },
    green: { bg: '#dcfce7', text: '#166534' },
    blue: { bg: '#dbeafe', text: '#1e40af' },
    purple: { bg: '#f3e8ff', text: '#6b21a8' },
  }
  const c = colors[color] || colors.blue
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 9999,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.text, marginRight: 4, marginBottom: 4
    }}>{text}</span>
  )
}

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')
const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316']
const STAGE_COLORS = { N1: '#94a3b8', N2: '#3b82f6', N3: '#1e40af', REM: '#8b5cf6', Wake: '#f59e0b' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'studies', label: 'Study Records' },
  { id: 'seizure_sleep', label: 'Seizure–Sleep' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SleepStagingDashboard() {
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
      axios.get(`${API_URL}/api/sleep-staging/overview`),
      axios.get(`${API_URL}/api/sleep-staging/breakdown`),
      axios.get(`${API_URL}/api/sleep-staging/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading sleep staging data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const a = overview?.averages || {}
  const ssl = overview?.seizure_sleep_link || {}

  /* ── Overview Tab ───────────────────────────────── */
  const renderOverview = () => {
    const stageData = [
      { stage: 'N1', pct: a.n1_pct, normal: '2-5%', fill: STAGE_COLORS.N1 },
      { stage: 'N2', pct: a.n2_pct, normal: '45-55%', fill: STAGE_COLORS.N2 },
      { stage: 'N3', pct: a.n3_pct, normal: '15-25%', fill: STAGE_COLORS.N3 },
      { stage: 'REM', pct: a.rem_pct, normal: '20-25%', fill: STAGE_COLORS.REM },
    ]

    const pieData = stageData.map(s => ({ name: s.stage, value: s.pct }))

    return (
      <>
        {/* KPI row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Total Studies" value={overview?.total_studies} color="#3b82f6" /></Card>
          <Card><KPI label="Patients Studied" value={overview?.total_patients} color="#8b5cf6" /></Card>
          <Card><KPI label="Avg Sleep Efficiency" value={pct(a.sleep_efficiency)} sub="Normal ≥85%" color={a.sleep_efficiency >= 85 ? '#10b981' : '#f59e0b'} /></Card>
          <Card><KPI label="Avg TST" value={`${fmt(a.tst_minutes)} min`} sub={`~${(a.tst_minutes / 60).toFixed(1)}h`} /></Card>
          <Card><KPI label="Avg Arousal Index" value={fmt(a.arousal_index)} sub="Normal <15/h" color={a.arousal_index > 15 ? '#ef4444' : '#10b981'} /></Card>
          <Card><KPI label="Avg Spindle Density" value={`${fmt(a.spindle_density)}/min`} sub="Normal 3-8" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
          {/* Sleep Architecture Pie */}
          <Card title="Sleep Architecture (Average Stage %)">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={100} label={({ name, value }) => `${name} ${value}%`}>
                  {pieData.map((e, i) => <Cell key={i} fill={stageData[i].fill} />)}
                </Pie>
                <Tooltip formatter={v => `${v}%`} />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Efficiency Distribution */}
          <Card title="Sleep Efficiency Distribution">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={overview?.efficiency_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {/* Study Type Distribution */}
          <Card title="Study Type Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview?.study_type_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="type" type="category" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Stage comparison vs normal */}
          <Card title="Stage % vs Normal Range">
            {stageData.map(s => (
              <div key={s.stage} style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                <div style={{ width: 40, fontWeight: 600, color: s.fill }}>{s.stage}</div>
                <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 6, height: 20, position: 'relative', overflow: 'hidden' }}>
                  <div style={{ width: `${Math.min(s.pct, 100)}%`, height: '100%', background: s.fill, borderRadius: 6 }} />
                </div>
                <div style={{ width: 60, textAlign: 'right', fontSize: 13, fontWeight: 600 }}>{s.pct}%</div>
                <div style={{ width: 70, textAlign: 'right', fontSize: 11, color: '#94a3b8' }}>{s.normal}</div>
              </div>
            ))}
          </Card>
        </div>
      </>
    )
  }

  /* ── Patient Profiles Tab ───────────────────────── */
  const renderPatients = () => {
    const patients = breakdown?.patients || []
    return (
      <Card title={`Patient Sleep Profiles (${patients.length} patients)`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'Studies', 'Avg SE%', 'Avg TST', 'Avg REM%', 'Avg N3%', 'Arousal Idx', 'Sz in Sleep', 'Flags'].map(h =>
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px' }}>{p.patient_id}</td>
                  <td style={{ padding: '6px' }}>{p.studies}</td>
                  <td style={{ padding: '6px', color: p.avg_sleep_efficiency < 75 ? '#ef4444' : p.avg_sleep_efficiency < 85 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>
                    {p.avg_sleep_efficiency}%
                  </td>
                  <td style={{ padding: '6px' }}>{p.avg_tst_minutes} min</td>
                  <td style={{ padding: '6px', color: p.avg_rem_pct < 12 ? '#ef4444' : '#1e293b' }}>{p.avg_rem_pct}%</td>
                  <td style={{ padding: '6px' }}>{p.avg_n3_pct}%</td>
                  <td style={{ padding: '6px', color: p.avg_arousal_index > 25 ? '#ef4444' : '#1e293b' }}>{p.avg_arousal_index}</td>
                  <td style={{ padding: '6px' }}>{p.total_seizures_sleep}</td>
                  <td style={{ padding: '6px' }}>
                    {p.flags?.map(f => (
                      <Badge key={f} text={f.replace(/_/g, ' ')}
                        color={f.includes('seizure') ? 'red' : f.includes('low') || f.includes('suppressed') ? 'amber' : f.includes('high') ? 'red' : 'blue'} />
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  /* ── Study Records Tab ──────────────────────────── */
  const renderStudies = () => {
    const studies = breakdown?.recent_studies || []
    return (
      <Card title={`Recent Sleep Studies (${studies.length})`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                {['Patient', 'Date', 'Type', 'SE%', 'N1', 'N2', 'N3', 'REM', 'Wake', 'AI', 'Sz', 'Notes'].map(h =>
                  <th key={h} style={{ padding: '6px 4px', textAlign: 'left', fontSize: 11, color: '#64748b' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {studies.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px' }}>{s.patient_id}</td>
                  <td style={{ padding: '4px' }}>{s.study_date}</td>
                  <td style={{ padding: '4px' }}><Badge text={s.study_type} color="blue" /></td>
                  <td style={{ padding: '4px', fontWeight: 600, color: s.sleep_efficiency < 75 ? '#ef4444' : s.sleep_efficiency < 85 ? '#f59e0b' : '#10b981' }}>
                    {s.sleep_efficiency}%
                  </td>
                  <td style={{ padding: '4px' }}>{s.n1_pct}%</td>
                  <td style={{ padding: '4px' }}>{s.n2_pct}%</td>
                  <td style={{ padding: '4px' }}>{s.n3_pct}%</td>
                  <td style={{ padding: '4px' }}>{s.rem_pct}%</td>
                  <td style={{ padding: '4px' }}>{s.wake_pct}%</td>
                  <td style={{ padding: '4px', color: s.arousal_index > 25 ? '#ef4444' : '#1e293b' }}>{s.arousal_index}</td>
                  <td style={{ padding: '4px' }}>{s.seizures_during_sleep > 0 ? <Badge text={`${s.seizures_during_sleep} sz`} color="red" /> : '0'}</td>
                  <td style={{ padding: '4px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                    title={s.clinical_notes}>{s.clinical_notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    )
  }

  /* ── Seizure–Sleep Tab ──────────────────────────── */
  const renderSeizureSleep = () => {
    const radarData = [
      { metric: 'Seizures in Sleep', value: ssl.studies_with_seizures || 0 },
      { metric: 'IED NREM Activation', value: ssl.ied_nrem_activation_count || 0 },
      { metric: 'OSA (AHI>5)', value: ssl.osa_prevalence || 0 },
      { metric: 'PLM (index>15)', value: ssl.plm_prevalence || 0 },
    ]

    // Patients with seizures during sleep — detailed
    const szPatients = (breakdown?.patients || []).filter(p => p.total_seizures_sleep > 0)

    return (
      <>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 20 }}>
          <Card><KPI label="Seizures During Sleep" value={ssl.total_seizures_during_sleep} color="#ef4444" sub="Total captured" /></Card>
          <Card><KPI label="Studies with Seizures" value={ssl.studies_with_seizures} color="#f59e0b" /></Card>
          <Card><KPI label="IED NREM Activation" value={ssl.ied_nrem_activation_count} color="#8b5cf6" sub="Studies confirming" /></Card>
          <Card><KPI label="OSA Prevalence" value={ssl.osa_prevalence} sub="AHI > 5" color="#06b6d4" /></Card>
          <Card><KPI label="PLM Prevalence" value={ssl.plm_prevalence} sub="Index > 15" color="#ec4899" /></Card>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <Card title="Comorbid Sleep Disorder Prevalence">
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis />
                <Radar name="Count" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          <Card title={`Patients with Nocturnal Seizures (${szPatients.length})`}>
            {szPatients.length === 0 ? (
              <div style={{ color: '#64748b', padding: 20, textAlign: 'center' }}>No seizures captured during sleep</div>
            ) : (
              <div style={{ maxHeight: 260, overflowY: 'auto' }}>
                {szPatients.map((p, i) => (
                  <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <div style={{ fontWeight: 600 }}>{p.patient_id}</div>
                      <div style={{ fontSize: 11, color: '#64748b' }}>SE: {p.avg_sleep_efficiency}% | REM: {p.avg_rem_pct}%</div>
                    </div>
                    <Badge text={`${p.total_seizures_sleep} seizure(s)`} color="red" />
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </>
    )
  }

  /* ── Definitions Tab ────────────────────────────── */
  const renderDefinitions = () => {
    const defs = definitions || {}
    return (
      <>
        <Card title="AASM Sleep Stage Criteria" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Stage', 'EEG Pattern', 'Key Features', 'Normal Duration'].map(h =>
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontSize: 12, color: '#64748b' }}>{h}</th>
                  )}
                </tr>
              </thead>
              <tbody>
                {(defs.sleep_stages || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{s.stage}</td>
                    <td style={{ padding: '8px 6px' }}>{s.eeg_pattern}</td>
                    <td style={{ padding: '8px 6px' }}>{s.key_features}</td>
                    <td style={{ padding: '8px 6px' }}>{s.duration_normal}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Quantitative Sleep Metrics" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
            {(defs.key_metrics || []).map((m, i) => (
              <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{m.metric}</div>
                <div style={{ fontSize: 12, color: '#475569', marginBottom: 4 }}>{m.definition}</div>
                <div style={{ fontSize: 11, color: '#10b981', fontWeight: 600 }}>Normal: {m.normal_adult}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Epilepsy–Sleep Interaction" span={2}>
          {(defs.epilepsy_sleep_interaction || []).map((item, i) => (
            <div key={i} style={{ padding: 12, marginBottom: 8, background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#3b82f6', marginBottom: 4 }}>{item.topic}</div>
              <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
            </div>
          ))}
        </Card>
      </>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Sleep Staging Dashboard</h2>
        <p style={{ margin: '4px 0 0', color: '#64748b', fontSize: 14 }}>
          EEG-based sleep architecture analysis — AASM staging, efficiency metrics, seizure–sleep interaction
        </p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer',
            fontWeight: tab === t.id ? 700 : 400, fontSize: 13,
            background: tab === t.id ? '#3b82f6' : 'transparent',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'patients' && renderPatients()}
      {tab === 'studies' && renderStudies()}
      {tab === 'seizure_sleep' && renderSeizureSleep()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
