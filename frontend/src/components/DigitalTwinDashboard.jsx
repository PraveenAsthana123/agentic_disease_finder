import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line
} from 'recharts'

const API = '/api'

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

const COMPLETENESS_COLORS = {
  high: '#10b981',
  medium: '#f59e0b',
  low: '#ef4444'
}

const DOMAIN_COLORS = {
  eeg: '#3b82f6',
  seizures: '#ef4444',
  medications: '#8b5cf6',
  mri: '#06b6d4',
  assessments: '#f59e0b',
  hitl: '#10b981'
}

const DOMAIN_LABELS = {
  eeg: 'EEG',
  seizures: 'Seizures',
  medications: 'Medications',
  mri: 'MRI',
  assessments: 'Assessments',
  hitl: 'HITL'
}

const READINESS_COLORS = {
  'Full Twin': '#10b981',
  'Partial Twin': '#3b82f6',
  'Minimal Twin': '#f59e0b',
  'No Data': '#94a3b8'
}

const GENDER_COLORS = {
  Female: '#ec4899',
  Male: '#3b82f6',
  Unknown: '#94a3b8'
}

const AGE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#06b6d4']

function completenessColor(score) {
  if (score >= 70) return COMPLETENESS_COLORS.high
  if (score >= 40) return COMPLETENESS_COLORS.medium
  return COMPLETENESS_COLORS.low
}

export default function DigitalTwinDashboard() {
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
      axios.get(`${API}/digital-twin/overview`),
      axios.get(`${API}/digital-twin/breakdown`),
      axios.get(`${API}/digital-twin/definitions`),
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
    { id: 'patients', label: 'Patient Twins' },
    { id: 'domains', label: 'Data Domains' },
    { id: 'cross', label: 'Cross-Analysis' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Digital Twin dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No digital twin data available.</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Patient Digital Twin</h2>
        <Badge
          text={`Avg Completeness: ${overview.kpi_cards?.avg_completeness_score != null ? (typeof overview.kpi_cards.avg_completeness_score === 'number' ? overview.kpi_cards.avg_completeness_score.toFixed(1) + '%' : overview.kpi_cards.avg_completeness_score) : 'N/A'}`}
          color={completenessColor(overview.kpi_cards?.avg_completeness_score || 0)}
        />
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
      {tab === 'patients' && renderPatients()}
      {tab === 'domains' && renderDomains()}
      {tab === 'cross' && renderCross()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpi_cards || {}
    const kpis = [
      { label: 'Total Patients', value: k.total_patients, color: '#3b82f6' },
      { label: 'With Analyses', value: k.patients_with_analyses, color: '#8b5cf6' },
      { label: 'With Seizures', value: k.patients_with_seizures, color: '#ef4444' },
      { label: 'With Medications', value: k.patients_with_medications, color: '#06b6d4' },
      { label: 'With MRI', value: k.patients_with_mri, color: '#f59e0b' },
      { label: 'With Assessments', value: k.patients_with_assessments, color: '#10b981' },
      { label: 'Total Data Points', value: k.total_data_points, color: '#f97316' },
      { label: 'Avg Completeness', value: k.avg_completeness_score != null ? (typeof k.avg_completeness_score === 'number' ? k.avg_completeness_score.toFixed(1) + '%' : k.avg_completeness_score) : 'N/A', color: completenessColor(k.avg_completeness_score || 0) },
    ]

    const completenessDist = overview.completeness_distribution || []
    const twinReadiness = overview.twin_readiness || []
    const domainCoverage = overview.domain_coverage || []
    const genderDist = overview.gender_distribution || []
    const ageDist = overview.age_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Completeness Distribution" span={2}>
          {completenessDist.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={completenessDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {completenessDist.map((d, i) => <Cell key={i} fill={i >= completenessDist.length * 0.6 ? COMPLETENESS_COLORS.high : i >= completenessDist.length * 0.3 ? COMPLETENESS_COLORS.medium : COMPLETENESS_COLORS.low} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Twin Readiness" span={2}>
          {twinReadiness.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={twinReadiness} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={90} label={({ level, count }) => `${level}: ${count}`}>
                  {twinReadiness.map((t, i) => <Cell key={i} fill={READINESS_COLORS[t.level] || '#94a3b8'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Domain Coverage" span={2}>
          {domainCoverage.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={domainCoverage} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, 100]} unit="%" />
                <YAxis type="category" dataKey="domain" tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(1) + '%' : v} />
                <Bar dataKey="coverage_pct" name="Coverage %" radius={[0, 4, 4, 0]}>
                  {domainCoverage.map((d, i) => {
                    const domainKey = (d.domain || '').toLowerCase()
                    return <Cell key={i} fill={DOMAIN_COLORS[domainKey] || AGE_COLORS[i % AGE_COLORS.length]} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Gender Distribution" span={1}>
          {genderDist.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={genderDist} dataKey="count" nameKey="gender" cx="50%" cy="50%" outerRadius={80} label={({ gender, count }) => `${gender}: ${count}`}>
                  {genderDist.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Age Distribution" span={1}>
          {ageDist.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ageDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bracket" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Patients" radius={[4, 4, 0, 0]}>
                  {ageDist.map((a, i) => <Cell key={i} fill={AGE_COLORS[i % AGE_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const patients = breakdown?.patient_twins || []
    const domainKeys = ['eeg', 'seizures', 'medications', 'mri', 'assessments', 'hitl']

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Digital Twins (${patients.length} patients)`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Age</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Gender</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Completeness</th>
                  {domainKeys.map(dk => (
                    <th key={dk} style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: DOMAIN_COLORS[dk] || '#475569' }}>{DOMAIN_LABELS[dk] || dk}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => {
                  const score = p.completeness_score != null ? (typeof p.completeness_score === 'number' ? p.completeness_score : parseFloat(p.completeness_score)) : 0
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.age != null ? p.age : 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={p.gender || 'Unknown'} color={GENDER_COLORS[p.gender] || GENDER_COLORS.Unknown} />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                          <div style={{ width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(score, 100)}%`, height: '100%', background: completenessColor(score), borderRadius: 4 }} />
                          </div>
                          <span style={{ fontSize: 11, fontFamily: 'monospace', color: completenessColor(score), fontWeight: 600 }}>
                            {typeof p.completeness_score === 'number' ? p.completeness_score.toFixed(0) + '%' : p.completeness_score || 'N/A'}
                          </span>
                        </div>
                      </td>
                      {domainKeys.map(dk => {
                        const has = p.domains && p.domains[dk]
                        return (
                          <td key={dk} style={{ padding: '10px 12px', textAlign: 'center', fontSize: 16 }}>
                            {has ? <span style={{ color: '#10b981' }}>{'\u2713'}</span> : <span style={{ color: '#cbd5e1' }}>{'\u2717'}</span>}
                          </td>
                        )
                      })}
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

  function renderDomains() {
    const domainCoverage = overview.domain_coverage || []
    const domainCorrelation = breakdown?.domain_correlation || []
    const topComplete = breakdown?.top_complete_patients || []

    const allDomains = [...new Set(domainCorrelation.flatMap(d => [d.domain_a, d.domain_b]))]

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Domain Coverage" span={2}>
          {domainCoverage.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={domainCoverage} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="domain" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v : v} />
                <Bar dataKey="patients_covered" name="Patients Covered" radius={[4, 4, 0, 0]}>
                  {domainCoverage.map((d, i) => {
                    const domainKey = (d.domain || '').toLowerCase()
                    return <Cell key={i} fill={DOMAIN_COLORS[domainKey] || AGE_COLORS[i % AGE_COLORS.length]} />
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Domain Correlation Matrix" span={2}>
          {domainCorrelation.length > 0 && allDomains.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Domain A</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Domain B</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Overlap Count</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Overlap %</th>
                  </tr>
                </thead>
                <tbody>
                  {domainCorrelation.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{row.domain_a}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{row.domain_b}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center', fontFamily: 'monospace' }}>{row.overlap_count}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'center' }}>
                          <div style={{ width: 60, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                            <div style={{ width: `${Math.min(row.overlap_pct || 0, 100)}%`, height: '100%', background: '#3b82f6', borderRadius: 4 }} />
                          </div>
                          <span style={{ fontSize: 11, fontFamily: 'monospace', color: '#475569' }}>
                            {row.overlap_pct != null ? (typeof row.overlap_pct === 'number' ? row.overlap_pct.toFixed(1) + '%' : row.overlap_pct) : 'N/A'}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No domain correlation data available.</p>
          )}
        </Card>

        <Card title="Top 10 Most Complete Patients" span={2}>
          {topComplete.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Rank</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Completeness</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Summary</th>
                  </tr>
                </thead>
                <tbody>
                  {topComplete.map((p, i) => {
                    const score = p.completeness_score != null ? (typeof p.completeness_score === 'number' ? p.completeness_score : parseFloat(p.completeness_score)) : 0
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 700, color: '#3b82f6' }}>{i + 1}</td>
                        <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id}</td>
                        <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                        <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                          <Badge
                            text={typeof p.completeness_score === 'number' ? p.completeness_score.toFixed(1) + '%' : (p.completeness_score || 'N/A')}
                            color={completenessColor(score)}
                          />
                        </td>
                        <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>
                          {[p.eeg_summary, p.seizure_summary, p.medication_summary].filter(Boolean).join(' | ') || 'N/A'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No top patient data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderCross() {
    const medEegCross = breakdown?.medication_eeg_cross || []
    const domainCorrelation = breakdown?.domain_correlation || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Medication-EEG Cross Analysis" span={2}>
          {medEegCross.length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Drug</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {medEegCross.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{row.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <Badge text={row.drug || 'N/A'} color="#8b5cf6" />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                        {row.confidence != null ? (typeof row.confidence === 'number' ? row.confidence.toFixed(3) : row.confidence) : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No medication-EEG cross data available.</p>
          )}
        </Card>

        <Card title="Domain Pair Overlap Analysis" span={2}>
          {domainCorrelation.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={domainCorrelation} margin={{ left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={(d) => `${d.domain_a} - ${d.domain_b}`} tick={{ fontSize: 10, angle: -30, textAnchor: 'end' }} interval={0} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(1) + '%' : v} />
                <Bar dataKey="overlap_pct" name="Overlap %" radius={[4, 4, 0, 0]} fill="#3b82f6">
                  {domainCorrelation.map((d, i) => <Cell key={i} fill={AGE_COLORS[i % AGE_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No domain pair overlap data available.</p>
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
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{item.definition}</div>
              </div>
            ))}
          </Card>
        ))}
      </div>
    )
  }
}
