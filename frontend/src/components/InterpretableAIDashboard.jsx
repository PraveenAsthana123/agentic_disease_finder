import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ScatterChart, Scatter, ZAxis, LineChart, Line
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

const CAT_COLORS = {
  'Time-Domain': '#3b82f6', 'Spectral': '#8b5cf6', 'Complexity': '#f59e0b',
  'Hjorth': '#ec4899', 'Connectivity': '#06b6d4', 'Other': '#64748b'
}

const PIE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4', '#10b981', '#f97316', '#64748b']

export default function InterpretableAIDashboard() {
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
      axios.get(`${API}/api/interpretable-ai/overview`),
      axios.get(`${API}/api/interpretable-ai/breakdown`),
      axios.get(`${API}/api/interpretable-ai/definitions`),
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
    { id: 'models', label: 'Interpretable Models' },
    { id: 'rules', label: 'Decision Rules' },
    { id: 'paths', label: 'Patient Paths' },
    { id: 'definitions', label: 'Definitions' },
  ]

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Interpretable AI dashboard...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No interpretable AI data available.'}</div>

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Interpretable AI</h2>
        <Badge
          text={`${overview.kpis?.total_interpretable_models || 0} interpretable models`}
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
      {tab === 'models' && renderModels()}
      {tab === 'rules' && renderRules()}
      {tab === 'paths' && renderPaths()}
      {tab === 'definitions' && renderDefinitions()}
    </div>
  )

  function renderOverview() {
    const k = overview.kpis || {}
    const kpis = [
      { label: 'Interpretable Models', value: k.total_interpretable_models, color: '#3b82f6' },
      { label: 'Decision Tree Depth', value: k.decision_tree_depth, color: '#8b5cf6' },
      { label: 'DT Accuracy', value: typeof k.decision_tree_accuracy === 'number' ? (k.decision_tree_accuracy * 100).toFixed(1) + '%' : k.decision_tree_accuracy, color: '#10b981' },
      { label: 'LR Accuracy', value: typeof k.logistic_regression_accuracy === 'number' ? (k.logistic_regression_accuracy * 100).toFixed(1) + '%' : k.logistic_regression_accuracy, color: '#06b6d4' },
      { label: 'Total Rules', value: k.total_rules, color: '#f59e0b' },
      { label: 'Features Used', value: k.total_features_used, color: '#ec4899' },
      { label: 'Total Analyses', value: k.total_analyses, color: '#f97316' },
      { label: 'Total Patients', value: k.total_patients, color: '#64748b' },
    ]

    const dtFeatures = overview.decision_tree?.feature_importance || []
    const lrPositive = overview.logistic_regression?.top_positive_coefficients || []
    const lrNegative = overview.logistic_regression?.top_negative_coefficients || []
    const accuracyComparison = overview.accuracy_comparison || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {kpis.map((kp, i) => (
          <Card key={i}><KPI label={kp.label} value={kp.value} color={kp.color} /></Card>
        ))}

        <Card title="Decision Tree Feature Importance" span={4}>
          {dtFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(250, dtFeatures.length * 30)}>
              <BarChart data={dtFeatures} layout="vertical" margin={{ left: 140 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={130} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Bar dataKey="importance" name="Importance" radius={[0, 4, 4, 0]}>
                  {dtFeatures.map((f, i) => (
                    <Cell key={i} fill={CAT_COLORS[f.category] || '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No feature importance data available.</p>
          )}
        </Card>

        <Card title="LR Positive Coefficients (push toward high confidence)" span={2}>
          {lrPositive.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(200, lrPositive.length * 28)}>
              <BarChart data={lrPositive.slice(0, 10)} layout="vertical" margin={{ left: 130 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Bar dataKey="coefficient" name="Coefficient" fill="#10b981" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No positive coefficients.</p>
          )}
        </Card>

        <Card title="LR Negative Coefficients (push toward low confidence)" span={2}>
          {lrNegative.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(200, lrNegative.length * 28)}>
              <BarChart data={lrNegative.slice(0, 10).map(c => ({ ...c, abs_coef: Math.abs(c.coefficient) }))} layout="vertical" margin={{ left: 130 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={120} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Bar dataKey="abs_coef" name="|Coefficient|" fill="#ef4444" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No negative coefficients.</p>
          )}
        </Card>

        <Card title="Accuracy Comparison: Interpretable vs Black-Box" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Model</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Type</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Train Accuracy</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>CV Accuracy</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Parameters</th>
                </tr>
              </thead>
              <tbody>
                {accuracyComparison.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600 }}>{m.model}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={m.type} color={m.type === 'interpretable' ? '#10b981' : '#f59e0b'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {m.train_accuracy != null ? (m.train_accuracy * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {m.cv_accuracy != null ? (m.cv_accuracy * 100).toFixed(1) + '%' : 'N/A'}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {m.n_parameters != null ? m.n_parameters : (m.file_size_kb ? `${m.file_size_kb} KB` : 'N/A')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Decision Tree Structure (Text)" span={4}>
          <pre style={{
            background: '#f8fafc', padding: 16, borderRadius: 8, fontSize: 12,
            fontFamily: 'monospace', overflow: 'auto', maxHeight: 400, whiteSpace: 'pre-wrap',
            color: '#334155', lineHeight: 1.6
          }}>
            {overview.decision_tree?.tree_text || 'No tree data available.'}
          </pre>
        </Card>
      </div>
    )
  }

  function renderModels() {
    const impComparison = breakdown?.importance_comparison || []
    const dtFeatures = overview?.decision_tree?.feature_importance || []
    const lrCoefs = breakdown?.full_coefficients || []
    const perDisease = breakdown?.per_disease_models || {}
    const diseaseNames = Object.keys(perDisease)

    // Category breakdown from DT importance
    const catMap = {}
    dtFeatures.forEach(f => {
      const cat = f.category || 'Other'
      catMap[cat] = (catMap[cat] || 0) + f.importance
    })
    const catData = Object.entries(catMap).map(([cat, imp]) => ({ category: cat, importance: Math.round(imp * 10000) / 10000 }))
      .sort((a, b) => b.importance - a.importance)

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        <Card title="Feature Importance: Decision Tree vs Logistic Regression" span={2}>
          <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead style={{ position: 'sticky', top: 0 }}>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Feature</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Category</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>DT Importance</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>LR Importance</th>
                  <th style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>LR Coefficient</th>
                  <th style={{ padding: '8px 10px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Agreement</th>
                </tr>
              </thead>
              <tbody>
                {impComparison.map((f, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{f.feature}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={f.category} color={CAT_COLORS[f.category] || '#64748b'} />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof f.dt_importance === 'number' ? f.dt_importance.toFixed(4) : f.dt_importance}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof f.lr_importance === 'number' ? f.lr_importance.toFixed(4) : f.lr_importance}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace' }}>
                      <span style={{ color: (f.lr_coefficient || 0) >= 0 ? '#10b981' : '#ef4444' }}>
                        {typeof f.lr_coefficient === 'number' ? f.lr_coefficient.toFixed(4) : f.lr_coefficient}
                      </span>
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={f.agreement} color={
                        f.agreement === 'both_important' ? '#10b981' :
                        f.agreement === 'dt_only' ? '#3b82f6' :
                        f.agreement === 'lr_only' ? '#8b5cf6' : '#94a3b8'
                      } />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="DT Category Importance">
          {catData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={catData} dataKey="importance" nameKey="category" cx="50%" cy="50%" outerRadius={90}
                  label={({ category, importance }) => `${category}: ${importance.toFixed(3)}`}>
                  {catData.map((c, i) => <Cell key={i} fill={CAT_COLORS[c.category] || PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No category data.</p>
          )}
        </Card>

        <Card title="Full LR Coefficients (Top 15)">
          {lrCoefs.length > 0 ? (
            <ResponsiveContainer width="100%" height={Math.max(250, Math.min(lrCoefs.length, 15) * 25)}>
              <BarChart data={lrCoefs.slice(0, 15)} layout="vertical" margin={{ left: 130 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={120} />
                <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(4) : v} />
                <Bar dataKey="coefficient" name="Coefficient" radius={[0, 4, 4, 0]}>
                  {lrCoefs.slice(0, 15).map((c, i) => (
                    <Cell key={i} fill={c.direction === 'positive' ? '#10b981' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: '#94a3b8', fontSize: 14 }}>No coefficient data.</p>
          )}
        </Card>

        {diseaseNames.map((disease, di) => {
          const dm = perDisease[disease]
          return (
            <Card key={di} title={`Per-Disease: ${disease}`} span={2}>
              {dm.message ? (
                <p style={{ color: '#94a3b8', fontSize: 14 }}>{dm.message} (n={dm.n_samples})</p>
              ) : (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: '#3b82f6' }}>{dm.n_samples}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>Samples</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: '#10b981' }}>{typeof dm.dt_accuracy === 'number' ? (dm.dt_accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>DT Accuracy</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: '#8b5cf6' }}>{typeof dm.lr_accuracy === 'number' ? (dm.lr_accuracy * 100).toFixed(1) + '%' : 'N/A'}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>LR Accuracy</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: 20, fontWeight: 700, color: '#f59e0b' }}>{dm.dt_n_rules}</div>
                    <div style={{ fontSize: 11, color: '#64748b' }}>Rules</div>
                  </div>
                </div>
              )}
            </Card>
          )
        })}
      </div>
    )
  }

  function renderRules() {
    const allRules = breakdown?.all_decision_rules || []
    const topRules = overview?.top_decision_rules || []
    const modelFiles = overview?.model_files || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Decision Rules (${allRules.length} rules extracted)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>#</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Rule (Conditions)</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Prediction</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Samples</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Confidence</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Conditions</th>
                </tr>
              </thead>
              <tbody>
                {allRules.map((r, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 600, color: '#64748b' }}>{i + 1}</td>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace', fontSize: 11, maxWidth: 500, wordBreak: 'break-all' }}>
                      {r.rule}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={r.prediction} color={r.prediction === 'high_confidence' ? '#10b981' : '#f59e0b'} />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{r.samples}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                      {typeof r.confidence === 'number' ? (r.confidence * 100).toFixed(1) + '%' : r.confidence}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                      <Badge text={r.conditions_count} color="#3b82f6" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title={`Production Model Files (${modelFiles.length} models)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Filename</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Disease</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>Size (KB)</th>
                </tr>
              </thead>
              <tbody>
                {modelFiles.map((m, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '10px 12px', fontFamily: 'monospace', fontWeight: 600 }}>{m.filename}</td>
                    <td style={{ padding: '10px 12px' }}>
                      <Badge text={m.disease} color="#8b5cf6" />
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>{m.size_kb}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title="Label Strategy" span={1}>
          <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
            <p><strong>Strategy:</strong> {overview?.label_strategy || 'N/A'}</p>
            <p><strong>Median Confidence Threshold:</strong> {overview?.median_confidence != null ? (overview.median_confidence * 100).toFixed(2) + '%' : 'N/A'}</p>
            <p>Analyses with confidence at or above the median are labelled <Badge text="high_confidence" color="#10b981" />, those below are labelled <Badge text="low_confidence" color="#f59e0b" />.</p>
          </div>
        </Card>
      </div>
    )
  }

  function renderPaths() {
    const paths = breakdown?.per_patient_paths || []

    return (
      <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
        <Card title={`Patient Decision Paths (${paths.length} analyses)`}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Patient</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Name</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Actual</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>DT Prediction</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>DT Conf</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>LR Prediction</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: '#475569' }}>LR Conf</th>
                  <th style={{ padding: '10px 12px', textAlign: 'center', fontWeight: 600, color: '#475569' }}>Path Length</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: '#475569' }}>Top LR Contributions</th>
                </tr>
              </thead>
              <tbody>
                {paths.map((p, i) => {
                  const dtCorrect = p.dt_prediction === p.actual_label
                  const lrCorrect = p.lr_prediction === p.actual_label
                  return (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '10px 12px' }}>{p.name || 'N/A'}</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={p.actual_label} color="#64748b" />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={p.dt_prediction} color={dtCorrect ? '#10b981' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {typeof p.dt_confidence === 'number' ? (p.dt_confidence * 100).toFixed(1) + '%' : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={p.lr_prediction} color={lrCorrect ? '#10b981' : '#ef4444'} />
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {typeof p.lr_confidence === 'number' ? (p.lr_confidence * 100).toFixed(1) + '%' : 'N/A'}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>
                        <Badge text={p.dt_path_length} color="#3b82f6" />
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                          {(p.lr_top_contributions || []).map((c, ci) => (
                            <Badge
                              key={ci}
                              text={`${c.feature}: ${typeof c.contribution === 'number' ? c.contribution.toFixed(3) : c.contribution}`}
                              color={c.contribution >= 0 ? '#10b981' : '#ef4444'}
                            />
                          ))}
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>

        {paths.length > 0 && (
          <Card title="Decision Path Detail (First Patient)" span={1}>
            <div style={{ fontSize: 13, color: '#475569' }}>
              <p style={{ fontWeight: 600, marginBottom: 8 }}>
                Patient: {paths[0].patient_id} | Disease: {paths[0].disease} | Confidence: {paths[0].actual_confidence}
              </p>
              {(paths[0].dt_path || []).map((step, si) => (
                <div key={si} style={{
                  padding: '8px 12px', marginBottom: 4, borderRadius: 6,
                  background: step.type === 'leaf' ? '#f0fdf4' : '#f8fafc',
                  borderLeft: `3px solid ${step.type === 'leaf' ? '#10b981' : '#3b82f6'}`
                }}>
                  {step.type === 'decision' ? (
                    <span>
                      <strong>{step.feature}</strong> = {step.value} {step.direction.includes('left') ? '<=' : '>'} {step.threshold}
                      <span style={{ color: '#94a3b8', marginLeft: 8 }}>({step.direction})</span>
                    </span>
                  ) : (
                    <span>
                      <strong>LEAF:</strong> Prediction = <Badge text={step.prediction} color="#10b981" />
                      <span style={{ marginLeft: 8 }}>({step.samples} samples)</span>
                    </span>
                  )}
                </div>
              ))}
            </div>
          </Card>
        )}
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
