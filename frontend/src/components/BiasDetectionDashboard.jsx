import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line
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

const GENDER_COLORS = {
  Female: '#ec4899',
  Male: '#3b82f6',
  Unknown: '#94a3b8'
}

const AGE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']

export default function BiasDetectionDashboard() {
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
      axios.get(`${API}/api/bias-detection/overview`),
      axios.get(`${API}/api/bias-detection/breakdown`),
      axios.get(`${API}/api/bias-detection/definitions`),
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
    { id: 'parity', label: 'Demographic Parity' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'intersectional', label: 'Intersectional' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Bias Detection dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No bias detection data available.</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Bias Detection AI</h2>
        <Badge
          text={`Fairness Index: ${overview.kpi_cards?.fairness_index != null ? (typeof overview.kpi_cards.fairness_index === 'number' ? overview.kpi_cards.fairness_index.toFixed(2) : overview.kpi_cards.fairness_index) : 'N/A'}`}
          color={overview.kpi_cards?.fairness_index >= 0.8 ? '#10b981' : overview.kpi_cards?.fairness_index >= 0.5 ? '#f59e0b' : '#ef4444'}
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
      {tab === 'parity' && renderParity()}
      {tab === 'patients' && renderPatients()}
      {tab === 'intersectional' && renderIntersectional()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpi_cards || {}
    const kpis = [
      { label: 'Total Patients', value: k.total_patients, color: '#3b82f6' },
      { label: 'Gender Groups', value: k.gender_groups, color: '#8b5cf6' },
      { label: 'Age Groups', value: k.age_groups, color: '#06b6d4' },
      { label: 'Total Analyses', value: k.total_analyses, color: '#f59e0b' },
      { label: 'Total Assessments', value: k.total_assessments, color: '#10b981' },
      { label: 'Representation Gap', value: k.representation_gap != null ? (typeof k.representation_gap === 'number' ? k.representation_gap.toFixed(2) : k.representation_gap) : 'N/A', color: '#f97316' },
      { label: 'Avg Confidence Gap', value: k.avg_confidence_gap != null ? (typeof k.avg_confidence_gap === 'number' ? k.avg_confidence_gap.toFixed(3) : k.avg_confidence_gap) : 'N/A', color: '#ef4444' },
      { label: 'Fairness Index', value: k.fairness_index != null ? (typeof k.fairness_index === 'number' ? k.fairness_index.toFixed(2) : k.fairness_index) : 'N/A', color: k.fairness_index >= 0.8 ? '#10b981' : '#f59e0b' },
    ]

    const genderDist = overview.gender_distribution || []
    const ageDist = overview.age_distribution || []
    const confByGender = overview.confidence_by_gender || []
    const confByAge = overview.confidence_by_age_group || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Gender Distribution" span={2}>
          {genderDist.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={genderDist} dataKey="count" nameKey="gender" cx="50%" cy="50%" outerRadius={90} label={({ gender, count, percentage }) => `${gender}: ${count} (${percentage != null ? percentage + '%' : ''})`}>
                  {genderDist.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Age Distribution" span={2}>
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

        <Card title="Confidence by Gender" span={2}>
          {confByGender.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confByGender} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(3) : v} />
                <Bar dataKey="avg_confidence" name="Avg Confidence" radius={[4, 4, 0, 0]}>
                  {confByGender.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Confidence by Age Group" span={2}>
          {confByAge.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confByAge} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="age_group" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(3) : v} />
                <Bar dataKey="avg_confidence" name="Avg Confidence" radius={[4, 4, 0, 0]}>
                  {confByAge.map((a, i) => <Cell key={i} fill={AGE_COLORS[i % AGE_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderParity() {
    const assessCoverage = overview.assessment_coverage_by_gender || []
    const medAccess = overview.medication_access_by_gender || []
    const seizureBurden = overview.seizure_burden_by_gender || []
    const confHistogram = breakdown?.confidence_histogram || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Assessment Coverage by Gender" span={1}>
          {assessCoverage.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={assessCoverage} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(1) : v} />
                <Bar dataKey="avg_assessments" name="Avg Assessments" radius={[4, 4, 0, 0]}>
                  {assessCoverage.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Medication Access by Gender" span={1}>
          {medAccess.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={medAccess} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v} />
                <Bar dataKey="fraction_with_meds" name="Fraction with Meds" radius={[4, 4, 0, 0]}>
                  {medAccess.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Seizure Burden by Gender" span={1}>
          {seizureBurden.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={seizureBurden} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(2) : v} />
                <Bar dataKey="avg_seizures" name="Avg Seizures" radius={[4, 4, 0, 0]}>
                  {seizureBurden.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Confidence Histogram by Gender" span={1}>
          {confHistogram.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confHistogram} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="Female" name="Female" stackId="a" fill={GENDER_COLORS.Female} />
                <Bar dataKey="Male" name="Male" stackId="a" fill={GENDER_COLORS.Male} />
                <Bar dataKey="Unknown" name="Unknown" stackId="a" fill={GENDER_COLORS.Unknown} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const patients = breakdown?.per_patient_bias_profile || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Bias Profiles (${patients.length} patients)`}>
          <div style={{ overflowX: 'auto', maxHeight: 600, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Age</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Gender</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Analyses</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Avg Confidence</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Assessments</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Medications</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Seizures</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Instruments</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, fontFamily: 'monospace' }}>{p.patient_id}</td>
                    <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.age != null ? p.age : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={p.gender || 'Unknown'} color={GENDER_COLORS[p.gender] || GENDER_COLORS.Unknown} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.num_analyses != null ? p.num_analyses : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {p.avg_confidence != null ? (typeof p.avg_confidence === 'number' ? p.avg_confidence.toFixed(3) : p.avg_confidence) : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.num_assessments != null ? p.num_assessments : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.num_medications != null ? p.num_medications : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>{p.num_seizures != null ? p.num_seizures : 'N/A'}</td>
                    <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>
                      {(p.instruments_administered || []).slice(0, 4).join(', ')}
                      {(p.instruments_administered || []).length > 4 && ` +${p.instruments_administered.length - 4} more`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  function renderIntersectional() {
    const intersectional = breakdown?.intersectional_analysis || []
    const instrumentByGender = breakdown?.instrument_by_gender || []
    const mriCoverage = breakdown?.mri_coverage_by_gender || []
    const disparity = breakdown?.disparity_metrics || {}

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Intersectional Analysis (Age x Gender)" span={2}>
          {intersectional.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Age Group</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Gender</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Avg Confidence</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {intersectional.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{row.age_group}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={row.gender || 'Unknown'} color={GENDER_COLORS[row.gender] || GENDER_COLORS.Unknown} />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                        {row.avg_confidence != null ? (typeof row.avg_confidence === 'number' ? row.avg_confidence.toFixed(3) : row.avg_confidence) : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{row.count != null ? row.count : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No intersectional data available.</p>
          )}
        </Card>

        <Card title="Instrument Scores by Gender" span={2}>
          {instrumentByGender.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Instrument</th>
                    {Object.keys(GENDER_COLORS).map(g => (
                      <th key={g} style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: GENDER_COLORS[g] }}>{g} (Avg / N)</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {instrumentByGender.map((inst, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{inst.instrument}</td>
                      {Object.keys(GENDER_COLORS).map(g => {
                        const gs = inst.gender_scores?.[g]
                        return (
                          <td key={g} style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                            {gs ? `${typeof gs.avg_score === 'number' ? gs.avg_score.toFixed(1) : gs.avg_score} / ${gs.count}` : <span style={{ color: '#94a3b8' }}>--</span>}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No instrument-by-gender data available.</p>
          )}
        </Card>

        <Card title="MRI Coverage by Gender" span={1}>
          {mriCoverage.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={mriCoverage} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="gender" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(1) + '%' : v} />
                <Bar dataKey="coverage_pct" name="MRI Coverage %" radius={[4, 4, 0, 0]}>
                  {mriCoverage.map((g, i) => <Cell key={i} fill={GENDER_COLORS[g.gender] || '#64748b'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No MRI coverage data available.</p>
          )}
        </Card>

        <Card title="Disparity Metrics" span={1}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '8px 0' }}>
            {[
              { label: 'Statistical Parity Difference', key: 'statistical_parity_diff', desc: 'Difference in positive outcome rates between groups' },
              { label: 'Disparate Impact Ratio', key: 'disparate_impact_ratio', desc: 'Ratio of positive rates (ideal = 1.0, legal threshold >= 0.8)' },
              { label: 'Equalized Odds Difference', key: 'equalized_odds_diff', desc: 'Max difference in TPR/FPR between groups' },
            ].map((metric, i) => (
              <div key={i} style={{ borderBottom: i < 2 ? '1px solid #f1f5f9' : 'none', paddingBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{metric.label}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{metric.desc}</div>
                  </div>
                  <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'monospace', color: '#3b82f6' }}>
                    {disparity[metric.key] != null ? (typeof disparity[metric.key] === 'number' ? disparity[metric.key].toFixed(3) : disparity[metric.key]) : 'N/A'}
                  </div>
                </div>
              </div>
            ))}
          </div>
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
