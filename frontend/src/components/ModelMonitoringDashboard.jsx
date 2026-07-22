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

const DRIFT_COLORS = {
  none: '#10b981', low: '#84cc16', medium: '#f59e0b', high: '#ef4444', severe: '#dc2626'
}

const STATUS_COLORS = {
  PASS: '#10b981', OK: '#10b981', ok: '#10b981', '200': '#10b981',
  FAIL: '#ef4444', SEVERE: '#ef4444', WARNING: '#f59e0b'
}

export default function ModelMonitoringDashboard() {
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
      axios.get(`${API}/model-monitoring/overview`),
      axios.get(`${API}/model-monitoring/breakdown`),
      axios.get(`${API}/model-monitoring/definitions`),
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
    { id: 'drift', label: 'Drift Analysis' },
    { id: 'training', label: 'Training Metrics' },
    { id: 'quality', label: 'Data Quality' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Model Monitoring dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No monitoring data available.</div>

  const driftVerdict = overview.drift_verdict || 'Unknown'
  const verdictColor = driftVerdict.includes('SEVERE') ? '#ef4444' : driftVerdict.includes('HIGH') ? '#f97316' : driftVerdict.includes('LOW') ? '#84cc16' : driftVerdict === 'PASS' ? '#10b981' : '#f59e0b'

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Model Monitoring</h2>
        <Badge text={`Drift: ${driftVerdict}`} color={verdictColor} />
        <Badge
          text={`Consistency: ${overview.consistency_verdict || 'N/A'}`}
          color={STATUS_COLORS[overview.consistency_verdict] || '#64748b'}
        />
        <span style={{ fontSize: 12, color: '#94a3b8', marginLeft: 'auto' }}>Generated: {overview.generated_at}</span>
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
      {tab === 'drift' && renderDrift()}
      {tab === 'training' && renderTraining()}
      {tab === 'quality' && renderQuality()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const health = overview.system_health || {}
    const predictions = overview.model_predictions || {}
    const kpis = [
      { label: 'Drift Verdict', value: overview.drift_verdict || 'N/A', color: verdictColor },
      { label: 'Features Drifted', value: `${overview.n_features_drifted || 0}/${overview.n_features_total || 0}`, color: '#f59e0b' },
      { label: 'Consistency', value: overview.consistency_verdict || 'N/A', color: STATUS_COLORS[overview.consistency_verdict] || '#64748b' },
      { label: 'Data Quality Score', value: overview.data_quality_score != null ? overview.data_quality_score.toFixed(1) : 'N/A', sub: overview.data_quality_grade || '', color: '#3b82f6' },
      { label: 'Total Predictions', value: predictions.total_analyses || 0, color: '#8b5cf6' },
      { label: 'Avg Confidence', value: predictions.avg_confidence != null ? (predictions.avg_confidence * 100).toFixed(1) + '%' : 'N/A', color: '#06b6d4' },
      { label: 'Backend Status', value: health.backend_http || 'N/A', color: STATUS_COLORS[health.backend_http] || '#64748b' },
      { label: 'API Errors', value: health.api_errors != null ? health.api_errors : 'N/A', color: health.api_errors > 0 ? '#ef4444' : '#10b981' },
    ]

    const sevDist = overview.severity_distribution || []
    const confDist = predictions.confidence_distribution || []
    const timeline = overview.monitoring_timeline || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} sub={kp.sub} color={kp.color} /></Card>
        ))}

        <Card title="Drift Severity Distribution" span={2}>
          {sevDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={sevDist} dataKey="count" nameKey="severity" cx="50%" cy="50%" outerRadius={90}
                  label={({ severity, count }) => `${severity}: ${count}`}>
                  {sevDist.map((s, i) => <Cell key={i} fill={DRIFT_COLORS[s.severity] || '#64748b'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No drift data.</p>}
        </Card>

        <Card title="Prediction Confidence Distribution" span={2}>
          {confDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Predictions" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No predictions yet.</p>}
        </Card>

        <Card title="Predictions by Disease" span={2}>
          {(predictions.by_disease || []).length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={predictions.by_disease} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Predictions" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="avg_confidence" name="Avg Confidence" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No disease predictions.</p>}
        </Card>

        <Card title="Monitoring Timeline" span={2}>
          {timeline.length > 0 ? (
            <div>
              {timeline.map((ev, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '8px 0', borderBottom: i < timeline.length - 1 ? '1px solid #f1f5f9' : 'none'
                }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <Badge text={ev.source} color="#3b82f6" />
                    <span style={{ fontSize: 13, color: '#334155' }}>{ev.verdict || ev.status}</span>
                  </div>
                  <span style={{ fontSize: 12, color: '#94a3b8', fontFamily: 'monospace' }}>{ev.timestamp}</span>
                </div>
              ))}
            </div>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No monitoring events.</p>}
        </Card>
      </div>
    )
  }

  function renderDrift() {
    const topFeatures = overview.top_drifted_features || []
    const allFeatures = breakdown?.drift_features_all || []
    const consistencyChecks = breakdown?.consistency_checks || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Top Drifted Features (PSI)">
          {topFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(250, topFeatures.length * 28)}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Bar dataKey="psi" name="PSI" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No drift data.</p>}
        </Card>

        <Card title={`All Drifted Features (${allFeatures.length})`}>
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>PSI</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>KS Stat</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>KS p-value</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Severity</th>
                </tr>
              </thead>
              <tbody>
                {allFeatures.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.feature}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.psi === 'number' ? f.psi.toFixed(4) : f.psi}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.ks_stat === 'number' ? f.ks_stat.toFixed(4) : f.ks_stat}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.ks_p === 'number' ? f.ks_p.toFixed(6) : f.ks_p}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={f.severity} color={DRIFT_COLORS[f.severity] || '#64748b'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title={`Consistency Checks (${consistencyChecks.length} analyses)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Analysis ID</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Trust Conf</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>SHAP Conf</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Trust Label</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>SHAP Label</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Consistent</th>
                </tr>
              </thead>
              <tbody>
                {consistencyChecks.map((c, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace' }}>{c.analysis_id}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof c.trust_conf === 'number' ? c.trust_conf.toFixed(4) : c.trust_conf}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof c.shap_conf === 'number' ? c.shap_conf.toFixed(4) : c.shap_conf}</td>
                    <td style={{ padding: '10px 12px' }}>{c.trust_label}</td>
                    <td style={{ padding: '10px 12px' }}>{c.shap_label}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={c.consistent ? 'Yes' : 'No'} color={c.consistent ? '#10b981' : '#ef4444'} />
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

  function renderTraining() {
    const runs = breakdown?.training_runs || []
    const patientAcc = breakdown?.accuracy_breakdown || []
    const allOptions = breakdown?.accuracy_all_options || {}

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title="Training Run History">
          {runs.length > 0 ? (
            <div>
              {runs.map((r, i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: '12px 0', borderBottom: i < runs.length - 1 ? '1px solid #f1f5f9' : 'none'
                }}>
                  <div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#1e293b' }}>{r.script}</div>
                    <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{r.seconds ? `${r.seconds.toFixed(1)}s` : ''}</div>
                  </div>
                  <Badge text={r.ok ? 'Success' : `Exit ${r.exit_code}`} color={r.ok ? '#10b981' : '#ef4444'} />
                </div>
              ))}
            </div>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No training runs recorded.</p>}
        </Card>

        <Card title="Patient-Specific Accuracy">
          {patientAcc.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={patientAcc} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="patient" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
                <Tooltip formatter={(v) => typeof v === 'number' ? (v * 100).toFixed(1) + '%' : v} />
                <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="sensitivity" name="Sensitivity" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No patient accuracy data.</p>}
        </Card>

        {allOptions.summary && (
          <Card title="Cross-Patient Accuracy Summary">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              {Object.entries(allOptions.summary).map(([key, val], i) => (
                <div key={i} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 700, color: '#3b82f6' }}>
                    {typeof val === 'number' ? (val * 100).toFixed(1) + '%' : val}
                  </div>
                  <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{key.replace(/_/g, ' ')}</div>
                </div>
              ))}
            </div>
          </Card>
        )}

        <Card title="Per-Patient Predictions">
          {(breakdown?.per_patient_predictions || []).length > 0 ? (
            <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                    <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Analyses</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Avg Confidence</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Diseases</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.per_patient_predictions || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'center' }}>{p.count}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {p.avg_confidence != null ? (p.avg_confidence * 100).toFixed(1) + '%' : 'N/A'}
                      </td>
                      <td style={{ padding: '8px 10px', fontSize: 11, color: '#64748b' }}>{p.diseases}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No per-patient predictions.</p>}
        </Card>
      </div>
    )
  }

  function renderQuality() {
    const dims = overview.quality_dimensions || {}
    const coverage = overview.modality_coverage || {}
    const missing = breakdown?.missing_matrix || []

    const dimData = Object.entries(dims).filter(([, v]) => v && v.score != null).map(([k, v]) => ({
      dimension: k, score: v.score, basis: v.basis
    }))
    const covData = Object.entries(coverage).map(([k, v]) => ({ modality: k, coverage: v }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <Card title="Quality Dimensions" span={1}>
          {dimData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={dimData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                <Radar dataKey="score" name="Score %" fill="#3b82f6" fillOpacity={0.3} stroke="#3b82f6" />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No quality dimensions available.</p>}
          <div style={{ marginTop: 12 }}>
            {Object.entries(dims).map(([k, v], i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid #f1f5f9', fontSize: 13 }}>
                <span style={{ color: '#334155', fontWeight: 500 }}>{k}</span>
                <span style={{ fontFamily: 'monospace', color: v && v.score != null ? '#3b82f6' : '#94a3b8' }}>
                  {v && v.score != null ? v.score.toFixed(1) + '%' : 'N/A'}
                </span>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Modality Coverage" span={1}>
          {covData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={covData} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="modality" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={60} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} tickFormatter={v => v + '%'} />
                <Tooltip formatter={(v) => v.toFixed(1) + '%'} />
                <Bar dataKey="coverage" name="Coverage %" fill="#06b6d4" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No coverage data.</p>}
        </Card>

        <Card title="Missing Data Matrix" span={2}>
          {missing.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Modality</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Present</th>
                    <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Missing</th>
                    <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>% Missing</th>
                  </tr>
                </thead>
                <tbody>
                  {missing.map((m, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{m.modality}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{m.present}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        {m.missing > 0 ? <Badge text={m.missing} color="#f59e0b" /> : <span style={{ color: '#94a3b8' }}>0</span>}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {typeof m.pct_missing === 'number' ? m.pct_missing.toFixed(1) + '%' : m.pct_missing}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p style={{ color: '#94a3b8', fontSize: 14 }}>No missing data information.</p>}
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
