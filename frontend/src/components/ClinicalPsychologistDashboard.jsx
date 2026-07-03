import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
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

const fmt = v => (v != null ? v : '--')

const IMPAIRMENT_COLORS = {
  none: '#2ecc71', mild: '#f39c12', moderate: '#e67e22', severe: '#e74c3c'
}

const BATTERY_COLORS = {
  Full: '#3498db', Screening: '#9b59b6', 'Follow-up': '#2ecc71'
}

const MOOD_COLORS = {
  none: '#2ecc71', minimal: '#2ecc71', mild: '#f39c12', moderate: '#e67e22',
  'moderately severe': '#e74c3c', severe: '#e74c3c',
  None: '#2ecc71', Minimal: '#2ecc71', Mild: '#f39c12', Moderate: '#e67e22',
  'Moderately Severe': '#e74c3c', Severe: '#e74c3c'
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

function ImpairmentBadge({ level }) {
  const key = level?.toLowerCase()
  return <Badge text={level || '--'} color={IMPAIRMENT_COLORS[key] || '#64748b'} />
}

function BatteryBadge({ type }) {
  return <Badge text={type || '--'} color={BATTERY_COLORS[type] || '#64748b'} />
}

function MoodBadge({ level }) {
  return <Badge text={level || '--'} color={MOOD_COLORS[level] || '#64748b'} />
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'cognitive', label: 'Cognitive Profile' },
  { id: 'mood', label: 'Mood & Comorbidity' },
  { id: 'patients', label: 'Patient Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ClinicalPsychologistDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState({})

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/api/clinical-psychologist/overview`),
      axios.get(`${API_URL}/api/clinical-psychologist/breakdown`),
      axios.get(`${API_URL}/api/clinical-psychologist/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const toggleExpand = id => setExpanded(prev => ({ ...prev, [id]: !prev[id] }))

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Clinical Psychologist Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Clinical Psychologist Dashboard
      </h2>
      <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>
        Neuropsychological assessment, cognitive profiling, mood screening, and comorbidity analysis for epilepsy patients
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
          {/* KPI Row 1 */}
          <Card title="Total Assessments"><KPI value={ov.total_assessments} label="Assessments" color="#3b82f6" /></Card>
          <Card title="Patients Assessed"><KPI value={ov.patients_assessed} label="Patients" color="#8b5cf6" /></Card>
          <Card title="Avg MoCA Score"><KPI value={ov.avg_moca != null ? ov.avg_moca.toFixed(1) : '--'} label="MoCA (0-30)" color="#10b981" /></Card>
          <Card title="Avg MMSE Score"><KPI value={ov.avg_mmse != null ? ov.avg_mmse.toFixed(1) : '--'} label="MMSE (0-30)" color="#06b6d4" /></Card>

          {/* KPI Row 2 */}
          <Card title="MoCA Impairment Rate"><KPI value={ov.moca_impairment_rate != null ? `${ov.moca_impairment_rate}%` : '--'} label="Below cutoff" color="#e74c3c" /></Card>
          <Card title="Avg PHQ-9"><KPI value={ov.avg_phq9 != null ? ov.avg_phq9.toFixed(1) : '--'} label="Depression (0-27)" color="#ef4444" /></Card>
          <Card title="Avg GAD-7"><KPI value={ov.avg_gad7 != null ? ov.avg_gad7.toFixed(1) : '--'} label="Anxiety (0-21)" color="#f59e0b" /></Card>
          <Card title="Impairment Rate"><KPI value={ov.impairment_rate != null ? `${ov.impairment_rate}%` : '--'} label="Any impairment" color="#e67e22" /></Card>

          {/* Depression Severity Pie Chart */}
          <Card title="Depression Severity Distribution (PHQ-9)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.depression_severity_distribution || []} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label={e => `${e.severity}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.depression_severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={MOOD_COLORS[e.severity] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Anxiety Severity Pie Chart */}
          <Card title="Anxiety Severity Distribution (GAD-7)" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.anxiety_severity_distribution || []} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label={e => `${e.severity}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.anxiety_severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={MOOD_COLORS[e.severity] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Impairment Level Pie Chart */}
          <Card title="Impairment Level Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.impairment_distribution || []} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={80} label={e => `${e.level}: ${e.count}`} labelLine fontSize={11}>
                  {(ov.impairment_distribution || []).map((e, i) => (
                    <Cell key={i} fill={IMPAIRMENT_COLORS[e.level?.toLowerCase()] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Cognitive Index Means Bar Chart */}
          <Card title="Cognitive Index Means" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.cognitive_index_means || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="mean" name="Mean Score" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Referral Reason Distribution Bar Chart */}
          <Card title="Referral Reason Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.referral_reason_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" fontSize={11} />
                <YAxis dataKey="reason" type="category" fontSize={11} width={120} />
                <Tooltip />
                <Bar dataKey="count" name="Count">
                  {(ov.referral_reason_distribution || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Battery Type Distribution */}
          <Card title="Battery Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.battery_type_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="battery_type" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Count">
                  {(ov.battery_type_distribution || []).map((e, i) => (
                    <Cell key={i} fill={BATTERY_COLORS[e.battery_type] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* COGNITIVE PROFILE TAB */}
      {tab === 'cognitive' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Cognitive Index Means Bar Chart */}
          <Card title="Cognitive Domain Index Means (Memory, Attention, Executive, Language, Processing Speed)">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ov.cognitive_index_means || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" fontSize={12} />
                <YAxis fontSize={11} domain={[0, 'auto']} />
                <Tooltip />
                <Bar dataKey="mean" name="Mean Score" fill="#8b5cf6" radius={[4, 4, 0, 0]}>
                  {(ov.cognitive_index_means || []).map((e, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Trail Making Test Stats */}
          <Card title="Trail Making Test Statistics">
            {ov.trail_making_stats ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#3b82f618' }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: '#3b82f6' }}>{fmt(ov.trail_making_stats.avg_trail_a)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Avg Trail A (sec)</div>
                </div>
                <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#ef444418' }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{fmt(ov.trail_making_stats.avg_trail_b)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Avg Trail B (sec)</div>
                </div>
                <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#8b5cf618' }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: '#8b5cf6' }}>{fmt(ov.trail_making_stats.b_a_ratio)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>B:A Ratio</div>
                </div>
              </div>
            ) : (
              <p style={{ color: '#64748b', fontSize: 13 }}>No trail making test data available.</p>
            )}
          </Card>

          {/* Memory Lateralization Cross-Tab */}
          <Card title="Memory Lateralization Cross-Tab">
            {(bd.memory_lateralization || []).length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      {['Lateralization', 'Count', 'Avg Memory Index', 'Avg Verbal Memory', 'Avg Visual Memory'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.memory_lateralization || []).map((m, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(m.lateralization)}</td>
                        <td style={{ padding: '6px' }}>{fmt(m.count)}</td>
                        <td style={{ padding: '6px' }}>{m.avg_memory_index != null ? m.avg_memory_index.toFixed(1) : '--'}</td>
                        <td style={{ padding: '6px' }}>{m.avg_verbal_memory != null ? m.avg_verbal_memory.toFixed(1) : '--'}</td>
                        <td style={{ padding: '6px' }}>{m.avg_visual_memory != null ? m.avg_visual_memory.toFixed(1) : '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: '#64748b', fontSize: 13 }}>No memory lateralization data available.</p>
            )}
          </Card>
        </div>
      )}

      {/* MOOD & COMORBIDITY TAB */}
      {tab === 'mood' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Depression Distribution Bar Chart */}
          <Card title="Depression Distribution (PHQ-9 Severity Levels)">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.depression_severity_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients">
                  {(ov.depression_severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={MOOD_COLORS[e.severity] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Anxiety Distribution Bar Chart */}
          <Card title="Anxiety Distribution (GAD-7 Severity Levels)">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ov.anxiety_severity_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="severity" fontSize={11} angle={-20} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="count" name="Patients">
                  {(ov.anxiety_severity_distribution || []).map((e, i) => (
                    <Cell key={i} fill={MOOD_COLORS[e.severity] || COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Combined Mood Comorbidity Stats */}
          <Card title="Combined Mood Comorbidity Statistics">
            {ov.mood_comorbidity ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
                <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#ef444418' }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: '#ef4444' }}>{fmt(ov.mood_comorbidity.phq9_elevated)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>PHQ-9 Elevated</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Score &ge; 10</div>
                </div>
                <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#f59e0b18' }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: '#f59e0b' }}>{fmt(ov.mood_comorbidity.gad7_elevated)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>GAD-7 Elevated</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Score &ge; 10</div>
                </div>
                <div style={{ textAlign: 'center', padding: 20, borderRadius: 8, background: '#e74c3c18' }}>
                  <div style={{ fontSize: 28, fontWeight: 700, color: '#e74c3c' }}>{fmt(ov.mood_comorbidity.both_elevated)}</div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>Both Elevated</div>
                  <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Comorbid depression + anxiety</div>
                </div>
              </div>
            ) : (
              <p style={{ color: '#64748b', fontSize: 13 }}>No mood comorbidity data available.</p>
            )}
          </Card>
        </div>
      )}

      {/* PATIENT DETAIL TAB */}
      {tab === 'patients' && (
        <Card title="Per-Patient Neuropsychological Assessment Detail" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f1f5f9' }}>
                  {['', 'Patient', 'Age', 'Gender', 'Disease', 'Battery', 'Impairment', 'MoCA'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(bd.patients || []).map((p, i) => {
                  const isExp = expanded[p.patient_id]
                  const assessments = p.assessments || [p]
                  return (
                    <React.Fragment key={i}>
                      <tr
                        style={{ borderBottom: '1px solid #e2e8f0', cursor: 'pointer' }}
                        onClick={() => toggleExpand(p.patient_id)}
                      >
                        <td style={{ padding: '6px', fontSize: 14, transition: 'transform 0.2s', transform: isExp ? 'rotate(90deg)' : 'rotate(0deg)', display: 'inline-block', width: 20 }}>&#9654;</td>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                        <td style={{ padding: '6px' }}>{fmt(p.age)}</td>
                        <td style={{ padding: '6px' }}>{fmt(p.gender)}</td>
                        <td style={{ padding: '6px', fontSize: 11 }}>{fmt(p.disease)}</td>
                        <td style={{ padding: '6px' }}><BatteryBadge type={p.battery_type || (assessments[0] && assessments[0].battery_type)} /></td>
                        <td style={{ padding: '6px' }}><ImpairmentBadge level={p.impairment_level || (assessments[0] && assessments[0].impairment_level)} /></td>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{fmt(p.moca_score || (assessments[0] && assessments[0].moca_score))}</td>
                      </tr>
                      {isExp && assessments.map((a, ai) => (
                        <tr key={`${i}-${ai}`}>
                          <td colSpan={8} style={{ padding: '12px 20px', background: '#f8fafc' }}>
                            <div>
                              {assessments.length > 1 && (
                                <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>
                                  Assessment {ai + 1} {a.assessment_date ? `(${a.assessment_date})` : ''}
                                </h4>
                              )}
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                                {/* Neuropsych Scores */}
                                <div>
                                  <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Neuropsych Scores</h4>
                                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
                                    <div><span style={{ color: '#64748b' }}>MoCA:</span> {fmt(a.moca_score)}</div>
                                    <div><span style={{ color: '#64748b' }}>MMSE:</span> {fmt(a.mmse_score)}</div>
                                    <div><span style={{ color: '#64748b' }}>PHQ-9:</span> {fmt(a.phq9_score)}</div>
                                    <div><span style={{ color: '#64748b' }}>GAD-7:</span> {fmt(a.gad7_score)}</div>
                                    <div><span style={{ color: '#64748b' }}>Memory Index:</span> {fmt(a.memory_index)}</div>
                                    <div><span style={{ color: '#64748b' }}>Attention Index:</span> {fmt(a.attention_index)}</div>
                                    <div><span style={{ color: '#64748b' }}>Executive Index:</span> {fmt(a.executive_index)}</div>
                                    <div><span style={{ color: '#64748b' }}>Language Index:</span> {fmt(a.language_index)}</div>
                                    <div><span style={{ color: '#64748b' }}>Processing Speed:</span> {fmt(a.processing_speed_index)}</div>
                                    <div><span style={{ color: '#64748b' }}>Trail A (sec):</span> {fmt(a.trail_a)}</div>
                                    <div><span style={{ color: '#64748b' }}>Trail B (sec):</span> {fmt(a.trail_b)}</div>
                                  </div>
                                </div>
                                {/* Assessment Details */}
                                <div>
                                  <h4 style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Assessment Details</h4>
                                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 12 }}>
                                    <div><span style={{ color: '#64748b' }}>Battery Type:</span> <BatteryBadge type={a.battery_type} /></div>
                                    <div><span style={{ color: '#64748b' }}>Impairment:</span> <ImpairmentBadge level={a.impairment_level} /></div>
                                    <div><span style={{ color: '#64748b' }}>Depression Severity:</span> <MoodBadge level={a.depression_severity} /></div>
                                    <div><span style={{ color: '#64748b' }}>Anxiety Severity:</span> <MoodBadge level={a.anxiety_severity} /></div>
                                    <div><span style={{ color: '#64748b' }}>Lateralization:</span> {fmt(a.lateralization_hypothesis)}</div>
                                    <div><span style={{ color: '#64748b' }}>Assessor:</span> {fmt(a.assessor)}</div>
                                    <div><span style={{ color: '#64748b' }}>Date:</span> {fmt(a.assessment_date)}</div>
                                  </div>
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Clinical Neuropsychology Concepts">
            {(defs.concepts || []).length === 0 ? (
              <p style={{ color: '#64748b', fontSize: 13 }}>No definitions available.</p>
            ) : (
              (defs.concepts || []).map((item, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < defs.concepts.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{item.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{item.description}</div>
                </div>
              ))
            )}
          </Card>
        </div>
      )}
    </div>
  )
}
