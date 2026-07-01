import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

export default function ExerciseDashboard() {
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
      axios.get(`${API}/exercise/overview`),
      axios.get(`${API}/exercise/breakdown`),
      axios.get(`${API}/exercise/definitions`),
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
    { id: 'categories', label: 'Category Analysis' },
    { id: 'adl', label: 'ADL & Rehab' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Exercise/Rehab dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No exercise data available.</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Exercise / Rehab Recommendations</h2>
        <Badge text={`${overview.kpis?.total_patients || 0} Patients`} color="#3b82f6" />
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
      {tab === 'categories' && renderCategories()}
      {tab === 'adl' && renderADL()}
      {tab === 'patients' && renderPatients()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpis || {}
    const riskDist = (overview.risk_distribution || []).filter(d => d.count > 0)
    const fitnessDist = (overview.fitness_distribution || []).filter(d => d.count > 0)
    const complianceDist = overview.compliance_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        <Card><KPI label="Total Patients" value={k.total_patients ?? '--'} color="#3b82f6" /></Card>
        <Card><KPI label="Mean Compliance %" value={k.mean_compliance_pct != null ? k.mean_compliance_pct.toFixed(1) + '%' : '--'} color={k.mean_compliance_pct >= 70 ? '#10b981' : '#f59e0b'} /></Card>
        <Card><KPI label="Mean Rehab Score" value={k.mean_rehab_score != null ? k.mean_rehab_score.toFixed(1) : '--'} color="#8b5cf6" /></Card>
        <Card><KPI label="Mean ADL Score" value={k.mean_adl_score != null ? k.mean_adl_score.toFixed(1) : '--'} color="#06b6d4" /></Card>
        <Card><KPI label="Weekly Target (min)" value={k.mean_weekly_target_min != null ? k.mean_weekly_target_min.toFixed(0) : '--'} color="#64748b" /></Card>
        <Card><KPI label="High Risk Patients" value={k.high_risk_count ?? 0} color={k.high_risk_count > 0 ? '#ef4444' : '#10b981'} /></Card>

        <Card title="Risk Distribution" span={1}>
          {riskDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={riskDist} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={80} label={({ level, count }) => `${level}: ${count}`}>
                  {riskDist.map((d, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No risk data available.</p>
          )}
        </Card>

        <Card title="Fitness Distribution" span={1}>
          {fitnessDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={fitnessDist} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={80} label={({ level, count }) => `${level}: ${count}`}>
                  {fitnessDist.map((d, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No fitness data available.</p>
          )}
        </Card>

        <Card title="Compliance Distribution" span={1}>
          {complianceDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={complianceDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="level" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {complianceDist.map((d, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No compliance data available.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderCategories() {
    const catCompliance = overview.category_compliance || []
    const catDetails = breakdown?.category_details || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Category Compliance (Mean %)" span={1}>
          {catCompliance.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(250, catCompliance.length * 40)}>
              <BarChart data={catCompliance} layout="vertical" margin={{ left: 160 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis type="category" dataKey="category" width={150} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => v.toFixed(1) + '%'} />
                <Bar dataKey="mean_compliance_pct" radius={[0, 4, 4, 0]}>
                  {catCompliance.map((c, i) => (
                    <Cell key={i} fill={c.mean_compliance_pct >= 70 ? '#10b981' : c.mean_compliance_pct >= 50 ? '#f59e0b' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No category compliance data.</p>
          )}
        </Card>

        {catDetails.length > 0 && (
          <Card title={`Exercise Categories (${catDetails.length})`} span={1}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              {catDetails.map((cat, i) => (
                <div key={i} style={{ border: '1px solid #e2e8f0', borderRadius: 10, padding: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{cat.name}</div>
                    <Badge text={`${cat.recommended_count} recommended`} color="#10b981" />
                  </div>
                  {cat.examples && cat.examples.length > 0 && (
                    <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
                      Examples: {cat.examples.join(', ')}
                    </div>
                  )}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginTop: 8 }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: '#3b82f6' }}>{cat.mean_compliance_pct?.toFixed(1)}%</div>
                      <div style={{ fontSize: 10, color: '#94a3b8' }}>Compliance</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: '#8b5cf6' }}>{cat.mean_target_weekly_min?.toFixed(0)}</div>
                      <div style={{ fontSize: 10, color: '#94a3b8' }}>Target min/wk</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 16, fontWeight: 700, color: '#06b6d4' }}>{cat.mean_session_duration_min?.toFixed(0)}</div>
                      <div style={{ fontSize: 10, color: '#94a3b8' }}>Session (min)</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    )
  }

  function renderADL() {
    const adlSummary = breakdown?.adl_summary || []
    const complianceHist = breakdown?.compliance_histogram || []
    const rehabHist = breakdown?.rehab_score_histogram || []
    const adlHist = breakdown?.adl_histogram || []

    const radarData = adlSummary.map(d => ({ domain: d.domain, score: d.mean_score }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="ADL Domain Scores (Radar)" span={2}>
          {radarData.length > 0 ? (
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius={120}>
                <PolarGrid />
                <PolarAngleAxis dataKey="domain" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Radar name="Mean Score" dataKey="score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                <Tooltip formatter={(v) => v.toFixed(1)} />
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No ADL data available.</p>
          )}
        </Card>

        <Card title="Compliance Histogram" span={1}>
          {complianceHist.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={complianceHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {complianceHist.map((d, i) => (
                    <Cell key={i} fill={d.good ? '#10b981' : '#f59e0b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No compliance histogram data.</p>
          )}
        </Card>

        <Card title="Rehab Score Histogram" span={1}>
          {rehabHist.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={rehabHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {rehabHist.map((d, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No rehab score histogram data.</p>
          )}
        </Card>

        <Card title="ADL Score Histogram" span={2}>
          {adlHist.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={adlHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {adlHist.map((d, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No ADL histogram data.</p>
          )}
        </Card>
      </div>
    )
  }

  function renderPatients() {
    const patients = overview.patient_summary || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Summary (${patients.length})`} span={1}>
          {patients.length === 0 ? (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No patient data available.</p>
          ) : (
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Patient ID</th>
                    <th style={{ textAlign: 'left', padding: '8px 6px' }}>Name</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Age</th>
                    <th style={{ textAlign: 'center', padding: '8px 6px' }}>Risk</th>
                    <th style={{ textAlign: 'center', padding: '8px 6px' }}>Fitness</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Compliance %</th>
                    <th style={{ textAlign: 'right', padding: '8px 6px' }}>Rehab Score</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px' }}>{p.name}</td>
                      <td style={{ padding: '6px', textAlign: 'right' }}>{p.age}</td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>
                        <Badge text={p.risk_level} color={p.risk_level === 'High' ? '#ef4444' : p.risk_level === 'Moderate' ? '#f59e0b' : '#10b981'} />
                      </td>
                      <td style={{ padding: '6px', textAlign: 'center' }}>
                        <Badge text={p.fitness_level} color={p.fitness_level === 'Very Low' ? '#ef4444' : p.fitness_level === 'Low' ? '#f59e0b' : p.fitness_level === 'Good' ? '#10b981' : '#3b82f6'} />
                      </td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {p.mean_compliance_pct != null ? p.mean_compliance_pct.toFixed(1) : '--'}
                      </td>
                      <td style={{ padding: '6px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {p.rehab_score != null ? p.rehab_score.toFixed(1) : '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    )
  }

  function renderDefinitions() {
    if (!definitions) return null

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        {definitions.protocol && (
          <Card title="Protocol">
            <p style={{ margin: '0 0 8px', fontSize: 14, color: '#475569', lineHeight: 1.6 }}>{definitions.protocol.description}</p>
            {definitions.protocol.framework && (
              <div style={{ fontSize: 13, color: '#64748b', marginBottom: 6 }}>
                <strong>Framework:</strong> {typeof definitions.protocol.framework === 'string' ? definitions.protocol.framework : JSON.stringify(definitions.protocol.framework)}
              </div>
            )}
            {definitions.protocol.standard && (
              <div style={{ fontSize: 13, color: '#64748b', marginBottom: 6 }}>
                <strong>Standard:</strong> {definitions.protocol.standard}
              </div>
            )}
            {definitions.protocol.indications && definitions.protocol.indications.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4 }}>Indications:</div>
                <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
                  {definitions.protocol.indications.map((ind, i) => <li key={i}>{ind}</li>)}
                </ul>
              </div>
            )}
          </Card>
        )}

        {definitions.exercise_categories && definitions.exercise_categories.length > 0 && (
          <Card title="Exercise Categories">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Category</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Target</th>
                  <th style={{ textAlign: 'left', padding: '8px 6px' }}>Epilepsy Considerations</th>
                </tr>
              </thead>
              <tbody>
                {definitions.exercise_categories.map((cat, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600, color: '#334155', verticalAlign: 'top' }}>{cat.name}</td>
                    <td style={{ padding: '8px 6px', color: '#475569', lineHeight: 1.5 }}>
                      {cat.description}
                      {cat.examples && cat.examples.length > 0 && (
                        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>Examples: {cat.examples.join(', ')}</div>
                      )}
                    </td>
                    <td style={{ padding: '8px 6px', color: '#475569', verticalAlign: 'top' }}>{cat.target}</td>
                    <td style={{ padding: '8px 6px', color: '#475569', verticalAlign: 'top' }}>{cat.epilepsy_considerations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}

        {definitions.risk_levels && definitions.risk_levels.length > 0 && (
          <Card title="Risk Levels">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {definitions.risk_levels.map((rl, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600, width: '25%', verticalAlign: 'top', color: '#334155' }}>
                      {typeof rl === 'string' ? rl : rl.level || rl.name || JSON.stringify(rl)}
                    </td>
                    <td style={{ padding: '8px 6px', color: '#475569', lineHeight: 1.5 }}>
                      {typeof rl === 'string' ? '' : rl.description || rl.criteria || ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}

        {definitions.adl_domains && definitions.adl_domains.length > 0 && (
          <Card title="ADL Domains">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {definitions.adl_domains.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 6px', fontWeight: 600, width: '25%', verticalAlign: 'top', color: '#334155' }}>
                      {typeof d === 'string' ? d : d.domain || d.name || JSON.stringify(d)}
                    </td>
                    <td style={{ padding: '8px 6px', color: '#475569', lineHeight: 1.5 }}>
                      {typeof d === 'string' ? '' : d.description || d.items?.join(', ') || ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        )}

        {definitions.clinical_significance && definitions.clinical_significance.length > 0 && (
          <Card title="Clinical Significance">
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#475569', lineHeight: 1.8 }}>
              {definitions.clinical_significance.map((sig, i) => <li key={i}>{sig}</li>)}
            </ul>
          </Card>
        )}
      </div>
    )
  }
}
