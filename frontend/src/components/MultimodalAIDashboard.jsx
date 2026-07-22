import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const MOD_COLORS = {
  'EEG Analyses': '#3b82f6',
  'MRI Findings': '#8b5cf6',
  'Neuropsych': '#f59e0b',
  'Seizure Diary': '#ef4444',
  'Clinical Assessments': '#10b981'
}

const CONC_COLORS = {
  concordant: '#10b981',
  discordant: '#ef4444',
  indeterminate: '#f59e0b',
  insufficient_data: '#94a3b8'
}

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981', '#06b6d4', '#ec4899']

export default function MultimodalAIDashboard() {
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
      axios.get(`${API}/api/multimodal-ai/overview`),
      axios.get(`${API}/api/multimodal-ai/breakdown`),
      axios.get(`${API}/api/multimodal-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'modality', label: 'Modality Analysis' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'correlation', label: 'Correlation Matrix' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Multimodal AI dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Multimodal AI</h2>
        <Badge
          text={`${overview?.kpis?.total_patients || 0} patients`}
          color="#3b82f6"
        />
        <Badge
          text={`${overview?.kpis?.total_records || 0} records`}
          color="#8b5cf6"
        />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>
          Avg modalities/patient: {overview?.kpis?.avg_modalities_per_patient ?? 'N/A'}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#1e293b' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569', fontSize: 13, fontWeight: 500
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'modality' && renderModality()}
      {tab === 'patients' && renderPatients()}
      {tab === 'correlation' && renderCorrelation()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview?.kpis || {}
    const kpis = [
      { label: 'Total Patients', value: k.total_patients, color: '#3b82f6' },
      { label: 'Total Records', value: k.total_records, color: '#8b5cf6' },
      { label: 'Avg Modalities / Patient', value: k.avg_modalities_per_patient, color: '#06b6d4' },
      { label: 'Patients w/ Full Coverage', value: k.patients_with_full_coverage, color: '#10b981' },
      { label: 'EEG-MRI Concordance Rate', value: k.eeg_mri_concordance_rate != null ? (k.eeg_mri_concordance_rate * 100).toFixed(1) + '%' : 'N/A', color: '#f59e0b' },
    ]

    const modalities = overview?.modalities || []
    const coverageDist = overview?.modality_coverage_distribution || []
    const concordance = overview?.concordance_summary || {}

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Modality Coverage (%)" span={3}>
          {modalities.length > 0 && (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={modalities} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="modality" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(2) + '%' : v} />
                <Bar dataKey="coverage_pct" name="Coverage %" radius={[4, 4, 0, 0]}>
                  {modalities.map((m, i) => (
                    <Cell key={i} fill={MOD_COLORS[m.modality] || PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Coverage Distribution" span={2}>
          {coverageDist.length > 0 && (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={coverageDist}
                  dataKey="patient_count"
                  nameKey="modality_count"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={({ modality_count, patient_count }) => `${modality_count} mod: ${patient_count} pts`}
                >
                  {coverageDist.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v, n) => [v + ' patients', `${n} modalities`]} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Concordance Summary" span={2}>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', padding: '8px 0' }}>
            {[
              { label: 'Concordant', value: concordance.concordant ?? 0, color: CONC_COLORS.concordant },
              { label: 'Discordant', value: concordance.discordant ?? 0, color: CONC_COLORS.discordant },
              { label: 'Indeterminate', value: concordance.indeterminate ?? 0, color: CONC_COLORS.indeterminate },
            ].map((c, i) => (
              <div key={i} style={{ textAlign: 'center', flex: 1 }}>
                <div style={{ fontSize: 32, fontWeight: 700, color: c.color }}>{c.value}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{c.label}</div>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Modality Record Counts" span={3}>
          {modalities.length > 0 && (
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              {modalities.map((m, i) => (
                <div key={i} style={{
                  flex: '1 1 160px', padding: '12px 16px', borderRadius: 8,
                  background: (MOD_COLORS[m.modality] || PIE_COLORS[i]) + '12',
                  borderLeft: `4px solid ${MOD_COLORS[m.modality] || PIE_COLORS[i]}`
                }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>{m.modality}</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: MOD_COLORS[m.modality] || PIE_COLORS[i] }}>{m.count}</div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>{m.patient_count} patients</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{m.coverage_pct?.toFixed(1)}% coverage</div>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    )
  }

  function renderModality() {
    const mriDist = breakdown?.mri_lesion_distribution || []
    const eegDist = breakdown?.eeg_disease_distribution || []
    const confByMod = breakdown?.confidence_by_modality_count || []

    // Build timeline data: group by month, one line per modality
    const rawTimeline = overview?.modality_timeline || []
    const months = [...new Set(rawTimeline.map(r => r.month))].sort()
    const modNames = [...new Set(rawTimeline.map(r => r.modality))]
    const timelineData = months.map(month => {
      const row = { month }
      modNames.forEach(mod => {
        const found = rawTimeline.find(r => r.month === month && r.modality === mod)
        row[mod] = found ? found.count : 0
      })
      return row
    })

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="MRI Lesion Distribution">
          {mriDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={mriDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="lesion_type" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No MRI lesion data available.</p>
          )}
        </Card>

        <Card title="EEG Disease Distribution">
          {eegDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={eegDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" tick={{ fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No EEG disease data available.</p>
          )}
        </Card>

        <Card title="Confidence by Modality Count">
          {confByMod.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={confByMod} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="modality_count" tick={{ fontSize: 11 }} label={{ value: 'Modality Count', position: 'insideBottom', offset: -4, fontSize: 11 }} height={50} />
                <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
                <Tooltip formatter={(v) => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v} />
                <Bar dataKey="avg_confidence" name="Avg Confidence" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No confidence data available.</p>
          )}
        </Card>

        <Card title="Modality Timeline (by Month)">
          {timelineData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={timelineData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {modNames.map((mod, i) => (
                  <Line
                    key={mod}
                    type="monotone"
                    dataKey={mod}
                    stroke={MOD_COLORS[mod] || PIE_COLORS[i % PIE_COLORS.length]}
                    strokeWidth={2}
                    dot={{ fill: MOD_COLORS[mod] || PIE_COLORS[i % PIE_COLORS.length] }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No timeline data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const patients = breakdown?.patient_profiles || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Profiles (${patients.length} patients)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient ID</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Modalities</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>EEG</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>MRI</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Seizures</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Assessments</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Concordance</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => {
                  const concColor = CONC_COLORS[p.concordance_status] || '#94a3b8'
                  const mri = p.mri_summary
                  const eeg = p.eeg_summary
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
                          <span style={{ fontSize: 12, color: '#64748b', marginRight: 4 }}>
                            {p.modality_count}x
                          </span>
                          {(p.modalities_available || []).map((mod, mi) => (
                            <Badge key={mi} text={mod} color={MOD_COLORS[mod] || '#64748b'} />
                          ))}
                        </div>
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 12, color: '#334155' }}>
                        {eeg ? (
                          <span>
                            <span style={{ fontWeight: 600 }}>{eeg.disease || 'N/A'}</span>
                            {eeg.confidence != null && (
                              <span style={{ color: '#94a3b8', marginLeft: 4 }}>
                                {(eeg.confidence * 100).toFixed(0)}%
                              </span>
                            )}
                          </span>
                        ) : (
                          <span style={{ color: '#cbd5e1' }}>—</span>
                        )}
                      </td>
                      <td style={{ padding: '10px 12px', fontSize: 12, color: '#334155' }}>
                        {mri ? (
                          <span>
                            <span style={{ fontWeight: 600 }}>{mri.lesion_type || 'N/A'}</span>
                            {mri.lesion_location && (
                              <span style={{ color: '#94a3b8', marginLeft: 4 }}>
                                {mri.laterality ? `${mri.laterality} ` : ''}{mri.lesion_location}
                              </span>
                            )}
                          </span>
                        ) : (
                          <span style={{ color: '#cbd5e1' }}>—</span>
                        )}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        {p.seizure_count > 0
                          ? <Badge text={p.seizure_count} color="#ef4444" />
                          : <span style={{ color: '#94a3b8' }}>0</span>
                        }
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        {p.assessment_count > 0
                          ? <Badge text={p.assessment_count} color="#10b981" />
                          : <span style={{ color: '#94a3b8' }}>0</span>
                        }
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge
                          text={p.concordance_status?.replace(/_/g, ' ') || 'N/A'}
                          color={concColor}
                        />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderCorrelation() {
    const matrix = breakdown?.modality_correlation_matrix || []

    // Collect unique modalities from the matrix
    const modSet = new Set()
    matrix.forEach(r => {
      modSet.add(r.modality_a)
      modSet.add(r.modality_b)
    })
    const mods = [...modSet].sort()

    // Build a lookup for quick access
    const lookup = {}
    matrix.forEach(r => {
      lookup[`${r.modality_a}||${r.modality_b}`] = r.patient_count
      lookup[`${r.modality_b}||${r.modality_a}`] = r.patient_count
    })

    const maxCount = Math.max(...matrix.map(r => r.patient_count), 1)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Modality Co-occurrence (Patient Count)">
          {matrix.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569', minWidth: 160 }}>Modality</th>
                    {mods.map((mod, i) => (
                      <th key={i} style={{
                        padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569',
                        minWidth: 130, fontSize: 11, whiteSpace: 'nowrap'
                      }}>{mod}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {mods.map((rowMod, ri) => (
                    <tr key={ri} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, color: '#334155', fontSize: 12, whiteSpace: 'nowrap' }}>
                        {rowMod}
                      </td>
                      {mods.map((colMod, ci) => {
                        if (rowMod === colMod) {
                          return (
                            <td key={ci} style={{
                              padding: '10px 12px', textAlign: 'center',
                              background: '#f1f5f9', color: '#94a3b8', fontSize: 12
                            }}>—</td>
                          )
                        }
                        const count = lookup[`${rowMod}||${colMod}`] ?? null
                        const intensity = count != null ? count / maxCount : 0
                        const bg = count != null
                          ? `rgba(59,130,246,${(intensity * 0.6 + 0.1).toFixed(2)})`
                          : '#fafafa'
                        return (
                          <td key={ci} style={{
                            padding: '10px 12px', textAlign: 'center',
                            background: bg,
                            color: intensity > 0.5 ? '#fff' : '#1e293b',
                            fontWeight: count != null ? 600 : 400,
                            fontSize: 13
                          }}>
                            {count != null ? count : <span style={{ color: '#cbd5e1' }}>0</span>}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No correlation data available.</p>
          )}
        </Card>

        <Card title="Modality Pair Patient Overlap (Bar)">
          {matrix.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(200, matrix.length * 36)}>
              <BarChart
                data={matrix.map(r => ({ pair: `${r.modality_a} + ${r.modality_b}`, patient_count: r.patient_count }))}
                layout="vertical"
                margin={{ left: 220 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="pair" tick={{ fontSize: 11 }} width={210} />
                <Tooltip />
                <Bar dataKey="patient_count" name="Patients" fill="#3b82f6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No correlation data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions?.sections) return null
    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.sections.map((sec, si) => (
          <Card key={si} title={sec.title}>
            {(sec.items || []).map((item, ii) => (
              <div key={ii} style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>{item.term}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.description}</div>
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }
}
