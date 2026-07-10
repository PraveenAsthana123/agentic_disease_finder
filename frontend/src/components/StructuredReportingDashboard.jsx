import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, LineChart, Line } from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const GRADE_COLORS = { A: '#10b981', B: '#3b82f6', C: '#f59e0b', D: '#ef4444', F: '#94a3b8' }

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

const TABS = ['Overview', 'Per-Patient', 'Per-Template', 'Quality', 'Definitions']

export default function StructuredReportingDashboard() {
  const [tab, setTab] = useState(0)
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch overview on mount
  useEffect(() => {
    setLoading(true)
    axios.get(`${API_URL}/api/structured-reporting/overview`)
      .then(r => { setOverview(r.data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  // Lazy fetch breakdown when Per-Patient, Per-Template, or Quality tabs selected
  useEffect(() => {
    if ((tab === 1 || tab === 2 || tab === 3) && !breakdown) {
      setLoading(true)
      axios.get(`${API_URL}/api/structured-reporting/breakdown`)
        .then(r => { setBreakdown(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, breakdown])

  // Lazy fetch definitions
  useEffect(() => {
    if (tab === 4 && !definitions) {
      setLoading(true)
      axios.get(`${API_URL}/api/structured-reporting/definitions`)
        .then(r => { setDefinitions(r.data); setLoading(false) })
        .catch(e => { setError(e.message); setLoading(false) })
    }
  }, [tab, definitions])

  if (error) return <div style={{ padding: 40, color: '#ef4444', textAlign: 'center' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 20px', fontSize: 22, color: '#0f172a' }}>Structured Reporting Dashboard</h2>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            borderRadius: '8px 8px 0 0',
            background: tab === i ? '#3b82f6' : 'transparent',
            color: tab === i ? '#fff' : '#64748b',
          }}>{t}</button>
        ))}
      </div>

      {loading && !overview && <div style={{ textAlign: 'center', padding: 60, color: '#94a3b8' }}>Loading...</div>}

      {/* Tab 0: Overview */}
      {tab === 0 && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 16 }}>
          {/* KPI row */}
          <Card span={5}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Reports" value={overview.report_generation_stats?.total_reports_generated} color="#3b82f6" />
              <KPI label="Template Types" value={overview.report_template_registry ? Object.keys(overview.report_template_registry).length : '--'} color="#8b5cf6" />
              <KPI label="Coverage %" value={overview.patient_coverage?.coverage_pct != null ? `${overview.patient_coverage.coverage_pct}%` : '--'} color="#10b981" />
              <KPI label="Patients Covered" value={overview.patient_coverage?.patients_with_any_report} sub={overview.patient_coverage?.total_patients != null ? `of ${overview.patient_coverage.total_patients}` : undefined} color="#06b6d4" />
            </div>
          </Card>

          {/* Reports by modality pie chart */}
          <Card title="Reports by Modality" span={2}>
            {overview.report_generation_stats ? (() => {
              const gs = overview.report_generation_stats
              const data = [
                { name: 'EEG', value: gs.eeg_reports || 0 },
                { name: 'MRI', value: gs.mri_reports || 0 },
                { name: 'Neuropsych', value: gs.neuropsych_reports || 0 }
              ].filter(d => d.value > 0)
              return data.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <PieChart>
                    <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`}>
                      {data.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
            })() : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          {/* Template field counts bar chart */}
          <Card title="Template Field Counts" span={3}>
            {overview.report_template_registry ? (() => {
              const data = Object.entries(overview.report_template_registry).map(([name, t]) => ({
                name: name.replace(' Report', ''),
                total_fields: (t.expected_fields || []).length,
                mandatory_fields: (t.mandatory_fields || []).length,
                sections: (t.ilae_sections || []).length
              }))
              return data.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="total_fields" fill="#3b82f6" name="Total Fields" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="mandatory_fields" fill="#ef4444" name="Mandatory Fields" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>
            })() : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          {/* Monthly trends line chart */}
          <Card title="Monthly Report Trends" span={5}>
            {overview.monthly_trends && overview.monthly_trends.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={overview.monthly_trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="eeg" stroke="#3b82f6" name="EEG" strokeWidth={2} />
                  <Line type="monotone" dataKey="mri" stroke="#ef4444" name="MRI" strokeWidth={2} />
                  <Line type="monotone" dataKey="neuropsych" stroke="#10b981" name="Neuropsych" strokeWidth={2} />
                  <Line type="monotone" dataKey="total" stroke="#8b5cf6" name="Total" strokeWidth={2} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
          </Card>

          {/* AI-assisted stats */}
          {overview.ai_assisted_finding_capture && (
            <Card title="AI-Assisted Review" span={2}>
              <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16, marginTop: 8 }}>
                <KPI label="Total Reviewed" value={overview.ai_assisted_finding_capture.total_findings} color="#3b82f6" />
                <KPI label="Agreed with AI" value={overview.ai_assisted_finding_capture.agree_count} color="#10b981" />
                <KPI label="Disagreed" value={overview.ai_assisted_finding_capture.disagree_count} color="#ef4444" />
                <KPI label="Agreement %" value={overview.ai_assisted_finding_capture.agreement_rate_pct != null ? `${overview.ai_assisted_finding_capture.agreement_rate_pct}%` : '--'} color="#8b5cf6" />
              </div>
            </Card>
          )}

          {/* Completeness by modality */}
          {overview.template_completeness && (
            <Card title="Completeness by Modality" span={3}>
              <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16, marginTop: 8 }}>
                {Object.entries(overview.template_completeness).map(([name, tc]) => (
                  <KPI key={name} label={name.replace(' Report', '')} value={tc.avg_completeness_pct != null ? `${tc.avg_completeness_pct}%` : '--'} sub={`Mandatory: ${tc.mandatory_capture_rate_pct || 0}%`} color={name.includes('EEG') ? '#3b82f6' : name.includes('MRI') ? '#ef4444' : '#10b981'} />
                ))}
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Tab 1: Per-Patient */}
      {tab === 1 && (
        <Card title="Per-Patient Report Data">
          {!breakdown ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient ID', 'EEG', 'MRI', 'Neuropsych', 'Total Reports', 'Completeness'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient_inventory || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{p.eeg_count}</td>
                      <td style={{ padding: '6px 10px' }}>{p.mri_count}</td>
                      <td style={{ padding: '6px 10px' }}>{p.neuropsych_count}</td>
                      <td style={{ padding: '6px 10px' }}>{p.total_reports}</td>
                      <td style={{ padding: '6px 10px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div style={{ flex: 1, height: 8, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden', maxWidth: 120 }}>
                            <div style={{ width: `${Math.round((p.modality_count || 0) / 3 * 100)}%`, height: '100%', background: (p.modality_count || 0) >= 3 ? '#10b981' : (p.modality_count || 0) >= 2 ? '#f59e0b' : '#ef4444', borderRadius: 4 }} />
                          </div>
                          <span style={{ fontSize: 12, color: '#475569', minWidth: 36 }}>{p.modality_count}/3</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      )}

      {/* Tab 2: Per-Template */}
      {tab === 2 && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {!breakdown ? <Card><div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div></Card> : (
            <>
              {['EEG Report', 'MRI Report', 'Neuropsych Report'].map((tmpl, idx) => {
                const color = ['#3b82f6', '#ef4444', '#10b981'][idx]
                const fields = breakdown.field_coverage_heatmap?.[tmpl]
                const data = fields ? Object.entries(fields).map(([field, info]) => ({
                  field: field.replace(/_/g, ' '),
                  fill_pct: info.fill_rate_pct || 0
                })) : []
                return (
                  <Card key={tmpl} title={`${tmpl} Field Fill Rates`}>
                    {data.length > 0 ? (
                      <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={data}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="field" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={70} />
                          <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                          <Tooltip formatter={(v) => `${v}%`} />
                          <Bar dataKey="fill_pct" fill={color} name="Fill %" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
                  </Card>
                )
              })}
            </>
          )}
        </div>
      )}

      {/* Tab 3: Quality */}
      {tab === 3 && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {!breakdown ? <Card><div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div></Card> : (
            <>
              {/* Quality scores bar chart */}
              <Card title="Quality Scores by Patient" span={2}>
                {breakdown.report_quality_scores && breakdown.report_quality_scores.length > 0 ? (() => {
                  const data = breakdown.report_quality_scores.map(q => ({
                    ...q,
                    score: q.avg_quality_pct,
                    grade: q.avg_quality_pct >= 90 ? 'A' : q.avg_quality_pct >= 75 ? 'B' : q.avg_quality_pct >= 60 ? 'C' : q.avg_quality_pct >= 40 ? 'D' : 'F'
                  }))
                  return (
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="patient_id" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={50} />
                        <YAxis tick={{ fontSize: 11 }} domain={[0, 100]} />
                        <Tooltip />
                        <Bar dataKey="score" name="Quality %" radius={[4, 4, 0, 0]}>
                          {data.map((entry, i) => (
                            <Cell key={i} fill={GRADE_COLORS[entry.grade] || COLORS[i % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  )
                })() : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
              </Card>

              {/* Grade distribution pie chart */}
              <Card title="Grade Distribution">
                {breakdown.report_quality_scores && breakdown.report_quality_scores.length > 0 ? (() => {
                  const gradeCounts = {}
                  breakdown.report_quality_scores.forEach(q => {
                    const grade = q.avg_quality_pct >= 90 ? 'A' : q.avg_quality_pct >= 75 ? 'B' : q.avg_quality_pct >= 60 ? 'C' : q.avg_quality_pct >= 40 ? 'D' : 'F'
                    gradeCounts[grade] = (gradeCounts[grade] || 0) + 1
                  })
                  const data = Object.entries(gradeCounts).map(([name, value]) => ({ name, value }))
                  return (
                    <ResponsiveContainer width="100%" height={280}>
                      <PieChart>
                        <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name} (${value})`}>
                          {data.map((entry, i) => (
                            <Cell key={i} fill={GRADE_COLORS[entry.name] || COLORS[i % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  )
                })() : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
              </Card>

              {/* Cross-modality stats */}
              <Card title="Cross-Modality Coverage" span={3}>
                {breakdown.cross_modality_concordance ? (() => {
                  const cm = breakdown.cross_modality_concordance
                  return (
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16, marginBottom: 20 }}>
                        <KPI label="All 3 Modalities" value={cm.all_three_modalities?.count} color="#10b981" />
                        <KPI label="EEG + MRI" value={cm.eeg_and_mri?.count} color="#3b82f6" />
                        <KPI label="EEG + Neuropsych" value={cm.eeg_and_neuropsych?.count} color="#8b5cf6" />
                        <KPI label="MRI + Neuropsych" value={cm.mri_and_neuropsych?.count} color="#f59e0b" />
                      </div>
                    </div>
                  )
                })() : <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>No data</div>}
              </Card>
            </>
          )}
        </div>
      )}

      {/* Tab 4: Definitions */}
      {tab === 4 && (
        <Card title="Term Definitions">
          {!definitions ? <div style={{ color: '#94a3b8', textAlign: 'center', padding: 40 }}>Loading...</div> : (
            Array.isArray(definitions.terms) ? (
              <dl style={{ margin: 0 }}>
                {definitions.terms.map((item, i) => (
                  <div key={i} style={{ marginBottom: 16, padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                    <dt style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{item.term}</dt>
                    <dd style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{item.definition}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <dl style={{ margin: 0 }}>
                {Object.entries(definitions).map(([term, def]) => (
                  <div key={term} style={{ marginBottom: 16, padding: '12px 16px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #3b82f6' }}>
                    <dt style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{term}</dt>
                    <dd style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{def}</dd>
                  </div>
                ))}
              </dl>
            )
          )}
        </Card>
      )}
    </div>
  )
}
