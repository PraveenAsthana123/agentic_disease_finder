import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const API = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#eab308']
const STAGE_COLORS = { Wake: '#ef4444', N1: '#f97316', N2: '#3b82f6', N3: '#8b5cf6', REM: '#22c55e' }
const EFF_COLORS = { Poor: '#ef4444', Fair: '#f97316', Borderline: '#eab308', Normal: '#22c55e' }

const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 13, fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }
const tdStyle = { padding: '8px 12px', borderBottom: '1px solid #f1f5f9', fontSize: 13, color: '#334155' }

function Card({ title, children, span }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)', gridColumn: span ? `span ${span}` : undefined }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ label, color }) {
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{label}</span>
}

export default function SleepStageAnalysisDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/sleep-stage-analysis/overview`),
      axios.get(`${API}/sleep-stage-analysis/breakdown`),
      axios.get(`${API}/sleep-stage-analysis/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Sleep Stage Analysis...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#f97316' }}>{overview.note || 'Not available'}</div>

  const tabs = ['overview', 'stages', 'hypnogram', 'seizures', 'medications', 'definitions']
  const k = overview.kpis || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700, color: '#0f172a' }}>Sleep Stage Analysis</h2>
      <p style={{ margin: '0 0 18px', fontSize: 13, color: '#64748b' }}>Sleep architecture profiling for {overview.total_patients} epilepsy EEG patients — stage distribution, seizure correlation, ASM impact</p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 500, background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569', transition: 'all .15s' }}>{t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Sleep Architecture KPIs" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Sleep Time" value={`${k.total_sleep_time_min} min`} sub={`${(k.total_sleep_time_min / 60).toFixed(1)} hrs`} />
              <KPI label="Sleep Efficiency" value={`${k.sleep_efficiency_pct}%`} sub={k.sleep_efficiency_pct >= 85 ? 'Normal' : k.sleep_efficiency_pct >= 70 ? 'Fair' : 'Poor'} />
              <KPI label="Sleep Onset Latency" value={`${k.sleep_onset_latency_min} min`} sub={k.sleep_onset_latency_min <= 20 ? 'Normal' : 'Prolonged'} />
              <KPI label="WASO" value={`${k.waso_min} min`} sub="Wake After Sleep Onset" />
              <KPI label="Arousal Index" value={`${k.arousal_index_per_hour}/hr`} sub={k.arousal_index_per_hour > 25 ? 'Elevated' : 'Normal'} />
              <KPI label="REM Latency" value={`${k.rem_latency_min} min`} sub={k.rem_latency_min > 120 ? 'Delayed' : 'Normal'} />
              <KPI label="Recording Time" value={`${k.total_recording_time_min} min`} sub={`${(k.total_recording_time_min / 60).toFixed(1)} hrs`} />
              <KPI label="Fragmentation Index" value={k.sleep_fragmentation_index} sub={k.sleep_fragmentation_index > 30 ? 'High' : 'Normal'} />
            </div>
          </Card>

          <Card title="Sleep Stage Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={(overview.stage_distribution || []).map(s => ({ name: s.stage, value: s.pct }))} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={85} label={({ name, value }) => `${name}: ${value}%`}>
                  {(overview.stage_distribution || []).map((s, i) => <Cell key={i} fill={STAGE_COLORS[s.stage] || COLORS[i]} />)}
                </Pie>
                <Tooltip formatter={v => `${v}%`} />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Sleep Efficiency Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={overview.sleep_efficiency_histogram || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_label" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {(overview.sleep_efficiency_histogram || []).map((e, i) => <Cell key={i} fill={EFF_COLORS[e.category] || '#94a3b8'} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Stage vs Normal Range" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Stage</th>
                    <th style={thStyle}>Observed %</th>
                    <th style={thStyle}>Normal Range</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.stage_distribution || []).map((s, i) => {
                    const inRange = s.pct >= s.normal_min && s.pct <= s.normal_max
                    return (
                      <tr key={i}>
                        <td style={tdStyle}><span style={{ fontWeight: 600, color: STAGE_COLORS[s.stage] || '#334155' }}>{s.stage}</span></td>
                        <td style={tdStyle}>{s.pct}%</td>
                        <td style={tdStyle}>{s.normal_min}%–{s.normal_max}%</td>
                        <td style={tdStyle}><Badge label={inRange ? 'Normal' : 'Abnormal'} color={inRange ? '#22c55e' : '#ef4444'} /></td>
                        <td style={{ ...tdStyle, fontSize: 12 }}>{s.interpretation}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Stages Tab ── */}
      {tab === 'stages' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {(breakdown.stage_details || []).map((s, i) => (
            <Card key={i} title={s.stage}>
              <div style={{ fontSize: 13, color: '#334155', lineHeight: 1.6 }}>
                <div style={{ marginBottom: 8 }}><strong>EEG Pattern:</strong> {s.eeg_pattern}</div>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 8 }}>
                  <span><strong>Duration:</strong> {s.duration_pct}%</span>
                  <span><strong>Normal:</strong> {s.normal_pct_range}</span>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <Badge label={`Epileptogenic: ${s.epileptogenic_potential}`} color={s.epileptogenic_potential === 'Highest' ? '#ef4444' : s.epileptogenic_potential === 'Low' || s.epileptogenic_potential === 'Lowest' ? '#22c55e' : '#f97316'} />
                  <span style={{ marginLeft: 8 }}><Badge label={`Seizure: ${s.seizure_pct}%`} color={s.seizure_pct > 30 ? '#ef4444' : '#3b82f6'} /></span>
                  <span style={{ marginLeft: 8 }}><Badge label={`IED: ${s.ied_activation}`} color='#8b5cf6' /></span>
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}><strong>Scoring:</strong> {s.scoring_rule}</div>
                <div style={{ fontSize: 12, color: '#475569' }}>{s.clinical_notes}</div>
                {s.key_features && (
                  <div style={{ marginTop: 8, display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                    {s.key_features.map((f, j) => <Badge key={j} label={f} color='#3b82f6' />)}
                  </div>
                )}
              </div>
            </Card>
          ))}

          <Card title="Sleep Scoring Reliability" span={2}>
            {breakdown.sleep_scoring_reliability && (
              <div>
                <div style={{ marginBottom: 12 }}>
                  <KPI label="Overall Agreement (Cohen's Kappa)" value={breakdown.sleep_scoring_reliability.overall_agreement_kappa} />
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>Stage</th>
                      <th style={thStyle}>Kappa</th>
                      <th style={thStyle}>Agreement Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.sleep_scoring_reliability.by_stage || []).map((s, i) => (
                      <tr key={i}>
                        <td style={tdStyle}>{s.stage}</td>
                        <td style={tdStyle}>{s.kappa}</td>
                        <td style={tdStyle}><Badge label={s.agreement} color={s.kappa >= 0.8 ? '#22c55e' : s.kappa >= 0.6 ? '#3b82f6' : '#f97316'} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ marginTop: 8, fontSize: 12, color: '#64748b' }}>{breakdown.sleep_scoring_reliability.note}</div>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ── Hypnogram Tab ── */}
      {tab === 'hypnogram' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Hypnogram — Sleep Stage Progression Over Night">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={breakdown.hypnogram_data || []} margin={{ left: 10, right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time_label" tick={{ fontSize: 10 }} interval={8} />
                <YAxis type="number" domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} tickFormatter={v => ['REM', 'N3', 'N2', 'N1', 'Wake'][v] || ''} tick={{ fontSize: 11 }} />
                <Tooltip labelFormatter={l => `Time: ${l}`} formatter={(v) => ['REM', 'N3', 'N2', 'N1', 'Wake'][v] || v} />
                <Line type="stepAfter" dataKey="level" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 8 }}>Y-axis: Wake (top) to REM (bottom). Step pattern shows sleep stage transitions throughout the recording.</div>
          </Card>

          <Card title="Arousal Analysis">
            {breakdown.arousal_analysis && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16, marginBottom: 16 }}>
                  <KPI label="Overall Arousal Index" value={`${breakdown.arousal_analysis.overall_arousal_index}/hr`} sub={`Normal < ${breakdown.arousal_analysis.normal_threshold}/hr`} />
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
                  <div>
                    <h4 style={{ fontSize: 13, fontWeight: 600, color: '#475569', marginBottom: 8 }}>Arousal Index by Stage</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={breakdown.arousal_analysis.by_stage || []} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="stage" width={50} tick={{ fontSize: 12 }} />
                        <Tooltip />
                        <Bar dataKey="arousal_index" radius={[0, 6, 6, 0]}>
                          {(breakdown.arousal_analysis.by_stage || []).map((s, i) => <Cell key={i} fill={STAGE_COLORS[s.stage] || COLORS[i]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <h4 style={{ fontSize: 13, fontWeight: 600, color: '#475569', marginBottom: 8 }}>Arousal Causes</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie data={(breakdown.arousal_analysis.arousal_causes || []).map(c => ({ name: c.cause, value: c.pct }))} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, value }) => `${name}: ${value}%`}>
                          {(breakdown.arousal_analysis.arousal_causes || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                        </Pie>
                        <Tooltip formatter={v => `${v}%`} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* ── Seizures Tab ── */}
      {tab === 'seizures' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Seizure-Sleep Correlation" span={2}>
            {overview.seizure_correlation_summary && (() => {
              const sc = overview.seizure_correlation_summary
              return (
                <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
                  <KPI label="Seizures During Sleep" value={`${sc.pct_seizures_during_sleep}%`} />
                  <KPI label="Seizures During Wake" value={`${sc.pct_seizures_during_wake}%`} />
                  <KPI label="Most Epileptogenic" value={sc.most_epileptogenic_stage} sub="Sleep stage" />
                  <KPI label="Least Epileptogenic" value={sc.least_epileptogenic_stage} sub="Sleep stage" />
                  <KPI label="Sleep Deprivation Trigger" value={`${sc.sleep_deprivation_trigger_pct}%`} />
                  <KPI label="Nocturnal Prevalence" value={`${sc.nocturnal_seizure_prevalence_pct}%`} />
                  <KPI label="IED Activation in Sleep" value={`${sc.ied_activation_during_sleep_pct}%`} />
                </div>
              )
            })()}
          </Card>

          {breakdown && (
            <Card title="Seizure Probability by Stage">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={breakdown.seizure_by_stage || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="stage" tick={{ fontSize: 12 }} />
                  <YAxis label={{ value: 'Probability %', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                  <Tooltip />
                  <Bar dataKey="probability_pct" radius={[6, 6, 0, 0]}>
                    {(breakdown.seizure_by_stage || []).map((s, i) => <Cell key={i} fill={s.color || COLORS[i]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {breakdown && (
            <Card title="IED Activation Ratio by Stage">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={breakdown.seizure_by_stage || []} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" label={{ value: 'IED Ratio (x baseline)', position: 'insideBottom', fontSize: 12, offset: -5 }} />
                  <YAxis type="category" dataKey="stage" width={50} tick={{ fontSize: 12 }} />
                  <Tooltip formatter={v => `${v}x baseline`} />
                  <Bar dataKey="ied_ratio" radius={[0, 6, 6, 0]}>
                    {(breakdown.seizure_by_stage || []).map((s, i) => <Cell key={i} fill={s.color || COLORS[i]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {breakdown && (
            <Card title="Seizure Types by Stage" span={2}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Stage</th>
                    <th style={thStyle}>Seizure Probability</th>
                    <th style={thStyle}>IED Ratio</th>
                    <th style={thStyle}>Predominant Seizure Type</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.seizure_by_stage || []).map((s, i) => (
                    <tr key={i}>
                      <td style={tdStyle}><span style={{ fontWeight: 600, color: s.color || '#334155' }}>{s.stage}</span></td>
                      <td style={tdStyle}>{s.probability_pct}%</td>
                      <td style={tdStyle}>{s.ied_ratio}x</td>
                      <td style={tdStyle}>{s.seizure_type}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </div>
      )}

      {/* ── Medications Tab ── */}
      {tab === 'medications' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="ASM Sleep Impact Summary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Medication</th>
                  <th style={thStyle}>Sleep Impact</th>
                  <th style={thStyle}>N2 Effect</th>
                  <th style={thStyle}>REM Effect</th>
                  <th style={thStyle}>Overall</th>
                </tr>
              </thead>
              <tbody>
                {(overview.asm_sleep_impact_summary || []).map((a, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{a.asm}</td>
                    <td style={tdStyle}><Badge label={a.sleep_impact} color={a.sleep_impact === 'Minimal' || a.sleep_impact === 'Beneficial' ? '#22c55e' : a.sleep_impact === 'Significant' ? '#ef4444' : '#f97316'} /></td>
                    <td style={tdStyle}>{a.n2_effect}</td>
                    <td style={tdStyle}>{a.rem_effect}</td>
                    <td style={tdStyle}>{a.overall}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {breakdown && (
            <Card title="Detailed ASM Impact on Sleep Architecture" span={2}>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      <th style={thStyle}>ASM</th>
                      <th style={thStyle}>Mechanism</th>
                      <th style={thStyle}>N1</th>
                      <th style={thStyle}>N2</th>
                      <th style={thStyle}>N3</th>
                      <th style={thStyle}>REM</th>
                      <th style={thStyle}>Latency</th>
                      <th style={thStyle}>Efficiency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.asm_detailed_impact || []).map((a, i) => (
                      <tr key={i}>
                        <td style={{ ...tdStyle, fontWeight: 600 }}>{a.asm}</td>
                        <td style={{ ...tdStyle, fontSize: 12, maxWidth: 200 }}>{a.mechanism}</td>
                        <td style={tdStyle}>{a.n1_effect}</td>
                        <td style={tdStyle}>{a.n2_effect}</td>
                        <td style={tdStyle}>{a.n3_effect}</td>
                        <td style={tdStyle}>{a.rem_effect}</td>
                        <td style={tdStyle}>{a.latency_effect}</td>
                        <td style={tdStyle}>{a.efficiency_effect}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          <Card title="Sleep Stages" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Stage</th><th style={thStyle}>Definition</th></tr></thead>
              <tbody>
                {(defs.sleep_stages || []).map((s, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{s.term}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{s.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Sleep Parameters">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Parameter</th><th style={thStyle}>Description</th></tr></thead>
              <tbody>
                {(defs.sleep_parameters || []).map((p, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{p.term}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{p.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Scoring Criteria (AASM)">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Stage</th><th style={thStyle}>Rule</th></tr></thead>
              <tbody>
                {(defs.scoring_criteria || []).map((c, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{c.stage}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{c.rule}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Sleep-Epilepsy Interactions" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thStyle}>Interaction</th><th style={thStyle}>Description</th></tr></thead>
              <tbody>
                {(defs.sleep_epilepsy_interactions || []).map((s, i) => (
                  <tr key={i}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.interaction}</td>
                    <td style={{ ...tdStyle, fontSize: 12 }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 8 }}>
              {(defs.terms || []).map((t, i) => (
                <div key={i} style={{ padding: 8, background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>{t.term}:</span>{' '}
                  <span style={{ color: '#475569' }}>{t.definition}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
