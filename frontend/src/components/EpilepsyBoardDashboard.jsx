import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (window._env_ && window._env_.API_URL) || '/api'
const COLORS = ['#3b82f6','#22c55e','#f97316','#ef4444','#8b5cf6','#ec4899','#14b8a6','#eab308','#6366f1','#f43f5e','#06b6d4','#84cc16','#d946ef','#0ea5e9','#a855f7','#10b981']
const REC_COLORS = {
  'Surgical evaluation': '#ef4444',
  'Medication adjustment': '#f97316',
  'Continued monitoring': '#3b82f6',
  'VNS consideration': '#8b5cf6',
  'Dietary therapy': '#22c55e'
}

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

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
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: (color || '#94a3b8') + '22', color: color || '#94a3b8'
    }}>{text}</span>
  )
}

export default function EpilepsyBoardDashboard() {
  const [data, setData] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, bd, df] = await Promise.all([
          axios.get(`${API_URL}/epilepsy-board/overview`),
          axios.get(`${API_URL}/epilepsy-board/breakdown`),
          axios.get(`${API_URL}/epilepsy-board/definitions`)
        ])
        setData({ overview: ov.data, breakdown: bd.data, definitions: df.data })
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading epilepsy board data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!data.overview) return <div style={{ padding: 40, color: '#64748b' }}>No board review data available.</div>

  const tabs = ['overview', 'cases', 'imaging', 'concordance', 'definitions']
  const ov = data.overview
  const charts = ov.charts || {}
  const summary = ov.summary || {}
  const bd = data.breakdown || {}
  const defs = data.definitions || {}

  // Convert neuropsych_profile dict to array for bar chart
  const npProfile = charts.neuropsych_profile || {}
  const npChartData = [
    { metric: 'MoCA', avg: npProfile.avg_moca },
    { metric: 'PHQ-9', avg: npProfile.avg_phq9 },
    { metric: 'GAD-7', avg: npProfile.avg_gad7 },
    { metric: 'Memory', avg: npProfile.avg_memory_index },
    { metric: 'Attention', avg: npProfile.avg_attention_index },
  ].filter(d => d.avg != null)

  // Convert seizure severity breakdown from dict to array
  const sevBreakdown = summary.seizure_severity_breakdown || {}
  const sevChartData = Object.entries(sevBreakdown).map(([k, v]) => ({ severity: k, count: v })).filter(d => d.count > 0)

  // Convert concordance to by_role with agree/disagree/total
  const concordanceByRole = (bd.review_concordance || []).map(c => ({
    role: c.role,
    agree: (c.responses || {}).agree || 0,
    disagree: (c.responses || {}).disagree || 0,
    total: ((c.responses || {}).agree || 0) + ((c.responses || {}).disagree || 0) + ((c.responses || {}).unknown || 0)
  }))

  // Recommendation distribution from case summaries
  const recCounts = {}
  for (const c of (bd.patient_case_summaries || [])) {
    const r = c.board_recommendation || 'Pending'
    recCounts[r] = (recCounts[r] || 0) + 1
  }
  const recDistribution = Object.entries(recCounts).map(([k, v]) => ({ recommendation: k, count: v }))

  const renderOverview = () => {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Board Review KPIs">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
            <KPI label="Patients Reviewed" value={fmt(summary.total_patients_reviewed)} color="#3b82f6" />
            <KPI label="Expert Reviews" value={fmt(summary.total_expert_reviews)} color="#22c55e" />
            <KPI label="Clinical Decisions" value={fmt(summary.total_clinical_decisions)} color="#8b5cf6" />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 16 }}>
            <KPI label="AI Agreement Rate" value={summary.ai_agreement_rate_pct != null ? summary.ai_agreement_rate_pct + '%' : '--'} color={summary.ai_agreement_rate_pct >= 80 ? '#22c55e' : '#f97316'} />
            <KPI label="MRI Lesion-Positive" value={summary.mri_lesion_positive_rate_pct != null ? summary.mri_lesion_positive_rate_pct + '%' : '--'} color="#ef4444" />
            <KPI label="Surgical Candidates" value={fmt(summary.surgical_candidate_count)} color="#ef4444" />
          </div>
        </Card>

        <Card title="Reviews by Expert Role">
          {(charts.review_by_role || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={charts.review_by_role}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[6,6,0,0]}>
                  {(charts.review_by_role || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No expert review data</div>}
        </Card>

        <Card title="MRI Lesion Type Distribution">
          {(charts.lesion_type_distribution || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={charts.lesion_type_distribution} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {(charts.lesion_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No MRI data</div>}
        </Card>

        <Card title="Epilepsy Type Distribution">
          {(charts.epilepsy_type_distribution || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={charts.epilepsy_type_distribution} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {(charts.epilepsy_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No epilepsy type data</div>}
        </Card>

        <Card title="Seizure Events by Month" span={2}>
          {(charts.seizure_frequency_trend || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={charts.seizure_frequency_trend}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#ef4444" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No seizure trend data</div>}
        </Card>

        <Card title="Medication Distribution">
          {(charts.medication_distribution || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={charts.medication_distribution} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis dataKey="drug" type="category" width={120} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#22c55e" radius={[0,6,6,0]}>
                  {(charts.medication_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No medication data</div>}
        </Card>

        <Card title="Neuropsych Profile (Avg Scores)">
          {npChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={npChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="avg" fill="#8b5cf6" radius={[6,6,0,0]}>
                  {npChartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No neuropsych data</div>}
        </Card>

        {sevChartData.length > 0 && (
          <Card title="Seizure Severity Breakdown">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={sevChartData} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {sevChartData.map((e, i) => (
                    <Cell key={i} fill={e.severity === 'Severe' ? '#ef4444' : e.severity === 'Moderate' ? '#f97316' : '#22c55e'} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>
    )
  }

  const renderCases = () => {
    const cases = bd.patient_case_summaries || []
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title={`Patient Case Summaries (${cases.length} patients)`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'Age/Sex', 'Epilepsy Type', 'Years', 'Neurologist', 'MRI Lesion', 'Seizures', 'Meds', 'MoCA', 'PHQ-9', 'Recommendation'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cases.map((c, i) => {
                  const np = c.latest_neuropsych || {}
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 6px', fontWeight: 600 }}>{c.patient_id}</td>
                      <td style={{ padding: '8px 6px' }}>{c.age ? `${c.age}/${c.sex || '?'}` : '--'}</td>
                      <td style={{ padding: '8px 6px' }}>{c.epilepsy_type || '--'}</td>
                      <td style={{ padding: '8px 6px' }}>{c.years_with_epilepsy != null ? c.years_with_epilepsy : '--'}</td>
                      <td style={{ padding: '8px 6px', fontSize: 11 }}>{c.primary_neurologist || '--'}</td>
                      <td style={{ padding: '8px 6px' }}>
                        {c.mri_lesion ? <Badge text={c.mri_lesion} color={c.mri_lesion === 'Normal' || c.mri_lesion === 'None' || c.mri_lesion === 'NL' || c.mri_lesion === 'NRM' ? '#22c55e' : '#ef4444'} /> : '--'}
                      </td>
                      <td style={{ padding: '8px 6px' }}>{c.seizure_count}</td>
                      <td style={{ padding: '8px 6px' }}>{c.medication_count}</td>
                      <td style={{ padding: '8px 6px' }}>{np.moca != null ? np.moca : '--'}</td>
                      <td style={{ padding: '8px 6px' }}>{np.phq9 != null ? np.phq9 : '--'}</td>
                      <td style={{ padding: '8px 6px' }}>
                        <Badge text={c.board_recommendation || 'Pending'} color={REC_COLORS[c.board_recommendation] || '#94a3b8'} />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>

        {cases.filter(c => (c.expert_reviews || []).length > 0).length > 0 && (
          <Card title="Expert Reviews per Patient" span={2}>
            {cases.filter(c => (c.expert_reviews || []).length > 0).map((c, i) => (
              <div key={i} style={{ marginBottom: 16, padding: 12, background: '#f8fafc', borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 8, color: '#1e293b' }}>{c.patient_id}</div>
                {c.expert_reviews.map((r, j) => (
                  <div key={j} style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4, fontSize: 12 }}>
                    <Badge text={r.role} color="#3b82f6" />
                    <span style={{ color: '#475569' }}>{r.expert}: {r.finding}</span>
                    <Badge text={r.agree_with_ai === 'agree' ? 'Agrees w/ AI' : 'Disagrees'} color={r.agree_with_ai === 'agree' ? '#22c55e' : '#ef4444'} />
                  </div>
                ))}
                {c.clinical_decision && (
                  <div style={{ marginTop: 6, fontSize: 12, color: '#334155' }}>
                    <strong>Decision:</strong> {c.clinical_decision.final_decision} by {c.clinical_decision.reviewer}
                    {c.clinical_decision.note && <span> — {c.clinical_decision.note}</span>}
                  </div>
                )}
              </div>
            ))}
          </Card>
        )}
      </div>
    )
  }

  const renderImaging = () => {
    // Build MRI details from case summaries
    const cases = bd.patient_case_summaries || []
    const mriCases = cases.filter(c => c.mri_lesion)
    const medSummary = (bd.medication_summary || {}).per_drug_counts || []
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title={`MRI Findings (${mriCases.length} patients with MRI)`} span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Patient', 'Lesion Type', 'Location', 'Epilepsy Type', 'Seizures'].map(h => (
                    <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {mriCases.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600 }}>{m.patient_id}</td>
                    <td style={{ padding: '8px 6px' }}>
                      <Badge text={m.mri_lesion} color={m.mri_lesion === 'NL' || m.mri_lesion === 'NRM' || m.mri_lesion === 'Normal' || m.mri_lesion === 'None' ? '#22c55e' : '#ef4444'} />
                    </td>
                    <td style={{ padding: '8px 6px' }}>{m.mri_location || '--'}</td>
                    <td style={{ padding: '8px 6px' }}>{m.epilepsy_type || '--'}</td>
                    <td style={{ padding: '8px 6px' }}>{m.seizure_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Lesion Type Distribution">
          {(charts.lesion_type_distribution || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={charts.lesion_type_distribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="type" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" radius={[6,6,0,0]}>
                  {(charts.lesion_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No lesion data</div>}
        </Card>

        <Card title="Medication Summary">
          {medSummary.length > 0 ? (
            <div>
              {medSummary.map((m, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 13 }}>
                  <span style={{ color: '#334155' }}>{m.drug}</span>
                  <Badge text={`${m.count} patients`} color="#3b82f6" />
                </div>
              ))}
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No medication data</div>}
        </Card>
      </div>
    )
  }

  const renderConcordance = () => {
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Expert–AI Concordance" span={2}>
          {concordanceByRole.length > 0 ? (
            <div>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={concordanceByRole}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="role" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="agree" fill="#22c55e" name="Agree" radius={[6,6,0,0]} />
                  <Bar dataKey="disagree" fill="#ef4444" name="Disagree" radius={[6,6,0,0]} />
                </BarChart>
              </ResponsiveContainer>
              <div style={{ marginTop: 16 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                      <th style={{ padding: '8px 6px', textAlign: 'left' }}>Role</th>
                      <th style={{ padding: '8px 6px', textAlign: 'center' }}>Agree</th>
                      <th style={{ padding: '8px 6px', textAlign: 'center' }}>Disagree</th>
                      <th style={{ padding: '8px 6px', textAlign: 'center' }}>Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {concordanceByRole.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 6px' }}>{r.role}</td>
                        <td style={{ padding: '8px 6px', textAlign: 'center' }}><Badge text={r.agree} color="#22c55e" /></td>
                        <td style={{ padding: '8px 6px', textAlign: 'center' }}><Badge text={r.disagree} color="#ef4444" /></td>
                        <td style={{ padding: '8px 6px', textAlign: 'center' }}>{r.total > 0 ? ((r.agree / r.total) * 100).toFixed(0) + '%' : '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No concordance data</div>}
        </Card>

        <Card title="Board Recommendation Distribution">
          {recDistribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={recDistribution} dataKey="count" nameKey="recommendation" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {recDistribution.map((e, i) => (
                    <Cell key={i} fill={REC_COLORS[e.recommendation] || COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No recommendation data</div>}
        </Card>
      </div>
    )
  }

  const renderDefinitions = () => {
    const entries = defs.definitions || []
    return (
      <div style={{ display: 'grid', gap: 16 }}>
        <Card title="Metric Definitions" span={2}>
          {entries.length > 0 ? entries.map((d, i) => (
            <div key={i} style={{ padding: '10px 0', borderBottom: '1px solid #f1f5f9' }}>
              <div style={{ fontWeight: 600, color: '#1e293b', fontSize: 13, marginBottom: 2 }}>{d.metric}</div>
              <div style={{ color: '#64748b', fontSize: 12 }}>{d.definition}</div>
            </div>
          )) : <div style={{ color: '#94a3b8' }}>No definitions available</div>}
        </Card>
      </div>
    )
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Multidisciplinary Epilepsy Board Review</h2>
      <p style={{ margin: '0 0 20px', color: '#64748b', fontSize: 13 }}>
        Case review aggregation — expert concordance, MRI findings, seizure burden, medication profiles, board recommendations
      </p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            background: tab === t ? '#3b82f6' : '#f1f5f9', color: tab === t ? '#fff' : '#64748b'
          }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'overview' && renderOverview()}
      {tab === 'cases' && renderCases()}
      {tab === 'imaging' && renderImaging()}
      {tab === 'concordance' && renderConcordance()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
