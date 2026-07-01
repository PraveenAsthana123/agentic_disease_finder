import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ScatterChart, Scatter, ZAxis, LineChart, Line
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

const CAT_COLORS = {
  'Time-Domain': '#3b82f6', 'Spectral': '#8b5cf6', 'Complexity': '#f59e0b',
  'Hjorth': '#ec4899', 'Connectivity': '#06b6d4', 'Other': '#64748b'
}

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#10b981', '#f97316', '#64748b']

export default function ExplainableAIDashboard() {
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
      axios.get(`${API}/explainable-ai/overview`),
      axios.get(`${API}/explainable-ai/breakdown`),
      axios.get(`${API}/explainable-ai/definitions`),
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
    { id: 'features', label: 'Feature Analysis' },
    { id: 'patients', label: 'Patient Profiles' },
    { id: 'counterfactuals', label: 'Counterfactuals' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Explainable AI dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No explainable AI data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Explainable AI</h2>
        <Badge
          text={`${overview.kpis?.total_features || 0} features analyzed`}
          color="#3b82f6"
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
      {tab === 'features' && renderFeatures()}
      {tab === 'patients' && renderPatients()}
      {tab === 'counterfactuals' && renderCounterfactuals()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpis || {}
    const kpis = [
      { label: 'Total Analyses', value: k.total_analyses, color: '#3b82f6' },
      { label: 'Patients', value: k.total_patients, color: '#8b5cf6' },
      { label: 'Features', value: k.total_features, color: '#06b6d4' },
      { label: 'Avg Confidence', value: typeof k.avg_confidence === 'number' ? (k.avg_confidence * 100).toFixed(1) + '%' : k.avg_confidence, color: '#10b981' },
      { label: 'Above Threshold', value: k.features_above_threshold, color: '#f59e0b' },
      { label: 'Top Feature', value: k.top_feature, color: '#1e293b' },
      { label: 'Model Diseases', value: k.model_diseases, color: '#f97316' },
      { label: 'Avg Signal Quality', value: typeof k.avg_signal_quality === 'number' ? k.avg_signal_quality.toFixed(2) : k.avg_signal_quality, color: '#64748b' },
    ]

    const globalFeatures = overview.global_feature_importance || []
    const catImportance = overview.category_importance || []
    const bandPower = overview.band_power_contribution || []
    const confDist = overview.confidence_distribution || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Top 20 Feature Importance" span={4}>
          {globalFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(300, globalFeatures.length * 25)}>
              <BarChart data={globalFeatures.slice(0, 20)} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={130} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Bar dataKey="importance" name="Importance" radius={[0, 4, 4, 0]}>
                  {globalFeatures.slice(0, 20).map((f, i) => (
                    <Cell key={i} fill={CAT_COLORS[f.category] || '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No feature importance data available.</p>
          )}
        </Card>

        <Card title="Category Importance (Radar)" span={2}>
          {catImportance.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={catImportance}>
                <PolarGrid />
                <PolarAngleAxis dataKey="category" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis tick={{ fontSize: 10 }} />
                <Radar dataKey="avg_importance" name="Avg Importance" fill="#8b5cf6" fillOpacity={0.3} stroke="#8b5cf6" />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
              </RadarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Band Power Contribution" span={2}>
          {bandPower.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={bandPower} dataKey="mean_contribution" nameKey="band" cx="50%" cy="50%" outerRadius={100} label={({ band, mean_contribution }) => `${band}: ${typeof mean_contribution === 'number' ? mean_contribution.toFixed(3) : mean_contribution}`}>
                  {bandPower.map((b, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Confidence Distribution" span={4}>
          {confDist.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={confDist} margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" name="Count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>
    )
  }

  function renderFeatures() {
    const shapDir = breakdown?.shap_direction_analysis || []
    const featureStats = breakdown?.per_feature_stats || []
    const diseaseImportance = overview?.disease_wise_importance || {}
    const diseaseNames = Object.keys(diseaseImportance)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="SHAP Direction Analysis" span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Category</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Importance</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Direction</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Class Means</th>
                </tr>
              </thead>
              <tbody>
                {shapDir.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{s.feature}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={s.category} color={CAT_COLORS[s.category] || '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof s.importance === 'number' ? s.importance.toFixed(4) : s.importance}
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={s.direction} color={s.direction === 'positive' ? '#10b981' : s.direction === 'negative' ? '#ef4444' : '#f59e0b'} />
                    </td>
                    <td style={{ padding: '10px 12px', fontSize: 11, color: '#64748b' }}>
                      {s.class_means ? Object.entries(s.class_means).map(([cls, val], ci) => (
                        <span key={ci} style={{ marginRight: 8 }}>
                          {cls}: <span style={{ fontFamily: 'monospace' }}>{typeof val === 'number' ? val.toFixed(3) : val}</span>
                        </span>
                      )) : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title={`Per-Feature Statistics (${featureStats.length} features)`} span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Rank</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Mean</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Std</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Min</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Max</th>
                </tr>
              </thead>
              <tbody>
                {featureStats.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.feature}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={f.category} color={CAT_COLORS[f.category] || '#64748b'} />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={`#${f.importance_rank}`} color="#3b82f6" />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.mean === 'number' ? f.mean.toFixed(4) : f.mean}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.std === 'number' ? f.std.toFixed(4) : f.std}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.min === 'number' ? f.min.toFixed(4) : f.min}</td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>{typeof f.max === 'number' ? f.max.toFixed(4) : f.max}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        {diseaseNames.map((disease, di) => {
          const features = (diseaseImportance[disease] || []).slice(0, 5)
          return (
            <Card key={di} title={`Top Features: ${disease}`}>
              {features.length > 0 ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={features} margin={{ left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="feature" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                    <Bar dataKey="importance" name="Importance" radius={[4, 4, 0, 0]}>
                      {features.map((f, fi) => (
                        <Cell key={fi} fill={CAT_COLORS[f.category] || '#64748b'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p style={{ color: '#94a3b8', fontSize: 14 }}>No features for this disease.</p>
              )}
            </Card>
          )
        })}
      </div>
    )
  }

  function renderPatients() {
    const patients = breakdown?.per_patient_profiles || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Explainability Profiles (${patients.length} patients)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Disease</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Predicted Label</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Confidence</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Signal Quality</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Top 3 Features</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={p.disease || 'Unknown'} color="#8b5cf6" />
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={p.predicted_label || 'N/A'} color="#3b82f6" />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {p.confidence != null ? (p.confidence * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>
                      {p.signal_quality != null ? (typeof p.signal_quality === 'number' ? p.signal_quality.toFixed(2) : p.signal_quality) : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {(p.top_3_features || []).map((tf, fi) => (
                          <Badge
                            key={fi}
                            text={`${tf.feature} (z=${typeof tf.zscore === 'number' ? tf.zscore.toFixed(2) : tf.zscore})`}
                            color={Math.abs(tf.zscore) > 2 ? '#ef4444' : Math.abs(tf.zscore) > 1 ? '#f59e0b' : '#10b981'}
                          />
                        ))}
                      </div>
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

  function renderCounterfactuals() {
    const counterfactuals = breakdown?.counterfactual_analysis || []
    const correlations = breakdown?.feature_correlation_matrix || []
    const topCorr = correlations.slice(0, 15).map(c => ({
      ...c,
      pair: `${c.feature_1} vs ${c.feature_2}`,
      abs_corr: Math.abs(typeof c.correlation === 'number' ? c.correlation : 0)
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Counterfactual Analysis (${counterfactuals.length} counterfactuals)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Predicted</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Counterfactual</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Flip Feature</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Current Value</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Delta Needed</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Class Separation</th>
                </tr>
              </thead>
              <tbody>
                {counterfactuals.map((cf, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{cf.patient_id}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={cf.predicted_label || 'N/A'} color="#3b82f6" />
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={cf.counterfactual_class || 'N/A'} color="#f59e0b" />
                    </td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={cf.flip_feature || 'N/A'} color={CAT_COLORS[cf.flip_feature_category] || '#64748b'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof cf.current_value === 'number' ? cf.current_value.toFixed(4) : cf.current_value}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof cf.delta_needed === 'number' ? cf.delta_needed.toFixed(4) : cf.delta_needed}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof cf.class_separation === 'number' ? cf.class_separation.toFixed(4) : cf.class_separation}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Feature Correlation Matrix (Top Pairs)">
            <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead style={{ position: 'sticky', top: 0 }}>
                  <tr style={{ background: '#f8fafc' }}>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature 1</th>
                    <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature 2</th>
                    <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Correlation</th>
                  </tr>
                </thead>
                <tbody>
                  {correlations.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 10px', fontWeight: 600, fontSize: 11 }}>{c.feature_1}</td>
                      <td style={{ padding: '8px 10px', fontWeight: 600, fontSize: 11 }}>{c.feature_2}</td>
                      <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                        <span style={{ color: (typeof c.correlation === 'number' ? c.correlation : 0) >= 0 ? '#10b981' : '#ef4444' }}>
                          {typeof c.correlation === 'number' ? c.correlation.toFixed(4) : c.correlation}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Top Correlations">
            {topCorr.length > 0 ? (
              <ResponsiveContainer width="100%" height={Math.max(300, topCorr.length * 25)}>
                <BarChart data={topCorr} layout="vertical" margin={{ left: 140 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, 1]} />
                  <YAxis type="category" dataKey="pair" tick={{ fontSize: 10 }} width={130} />
                  <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                  <Bar dataKey="abs_corr" name="|Correlation|" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p style={{ color: '#94a3b8', fontSize: 14 }}>No correlation data available.</p>
            )}
          </Card>
        </div>
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
