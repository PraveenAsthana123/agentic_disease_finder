import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const MODEL_LABELS = {
  rf: 'Random Forest',
  ensemble: 'Ensemble',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'models', label: 'Models' },
  { id: 'fold-analysis', label: 'Fold Analysis' },
  { id: 'comparison', label: 'Comparison' },
  { id: 'definitions', label: 'Definitions' },
]

export default function BenchmarkValidationDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedTerm, setExpandedTerm] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/benchmark-validation/overview`),
      axios.get(`${API_URL}/api/benchmark-validation/breakdown`),
      axios.get(`${API_URL}/api/benchmark-validation/definitions`),
    ])
      .then(([ovRes, bdRes, dfRes]) => {
        setOverview(ovRes.data)
        setBreakdown(bdRes.data)
        setDefinitions(dfRes.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Benchmark Validation data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.dataset) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No benchmark validation data available.</div>

  /* ── Find best model by accuracy ── */
  const models = overview?.models || {}
  const bestModel = Object.entries(models).sort((a, b) => b[1].accuracy_mean - a[1].accuracy_mean)[0]
  const bestKey = bestModel ? bestModel[0] : null
  const bestMetrics = bestModel ? bestModel[1] : {}

  /* ── Overview Tab ── */
  const renderOverview = () => (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Dataset Information" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Dataset" value={overview.dataset?.split('(')[0]?.trim() || 'N/A'} color="#3b82f6" />
          <KPI label="Total Samples" value={overview.n_samples} color="#10b981" />
          <KPI label="Features" value={overview.n_features} color="#8b5cf6" />
          <KPI label="Balance" value={overview.balance} sub="Seizure / Normal" color="#f59e0b" />
          <KPI label="CV Method" value={overview.cv} color="#06b6d4" />
          <KPI label="Generated" value={overview.generated_at?.split('T')[0] || 'N/A'} color="#64748b" />
        </div>
      </Card>

      <Card title="Best Model Performance">
        <div style={{ textAlign: 'center', marginBottom: 12 }}>
          <Badge text={MODEL_LABELS[bestKey] || bestKey} color="#3b82f6" />
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <KPI label="Accuracy" value={`${(bestMetrics.accuracy_mean * 100).toFixed(1)}%`} color="#10b981" />
          <KPI label="F1 Score" value={bestMetrics.f1_mean?.toFixed(3)} color="#3b82f6" />
          <KPI label="AUC" value={bestMetrics.auc_mean?.toFixed(3)} color="#8b5cf6" />
        </div>
      </Card>

      <Card title="Purpose">
        <p style={{ fontSize: 14, color: '#475569', lineHeight: 1.6, margin: 0 }}>
          {overview.purpose}
        </p>
      </Card>
    </div>
  )

  /* ── Models Tab ── */
  const renderModels = () => {
    const modelEntries = Object.entries(breakdown?.models || {})
    const chartData = modelEntries.map(([key, m]) => ({
      name: MODEL_LABELS[key] || key,
      Accuracy: +(m.accuracy_mean * 100).toFixed(1),
      F1: +(m.f1_mean * 100).toFixed(1),
      AUC: +(m.auc_mean * 100).toFixed(1),
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        {modelEntries.map(([key, m], idx) => (
          <Card key={key} title={MODEL_LABELS[key] || key}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 16 }}>
              <KPI label="Accuracy" value={`${(m.accuracy_mean * 100).toFixed(1)}%`} color={COLORS[idx * 2]} />
              <KPI label="F1 Score" value={m.f1_mean?.toFixed(3)} color={COLORS[idx * 2 + 1]} />
              <KPI label="AUC" value={m.auc_mean?.toFixed(3)} color={COLORS[idx * 2 + 2]} />
            </div>
            <div style={{ fontSize: 12, color: '#64748b' }}>
              Accuracy Std: {m.accuracy_std?.toFixed(4)} | Folds: {m.fold_acc?.length || 0}
            </div>
          </Card>
        ))}

        <Card title="Model Metrics Comparison" span={2}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => `${v}%`} />
              <Legend />
              <Bar dataKey="Accuracy" fill={COLORS[0]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill={COLORS[3]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="AUC" fill={COLORS[4]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    )
  }

  /* ── Fold Analysis Tab ── */
  const renderFoldAnalysis = () => {
    const modelEntries = Object.entries(breakdown?.models || {})
    const maxFolds = Math.max(...modelEntries.map(([, m]) => m.fold_acc?.length || 0), 0)

    const foldChartData = Array.from({ length: maxFolds }, (_, i) => {
      const point = { fold: `Fold ${i + 1}` }
      modelEntries.forEach(([key, m]) => {
        point[MODEL_LABELS[key] || key] = m.fold_acc?.[i] != null ? +(m.fold_acc[i] * 100).toFixed(2) : null
      })
      return point
    })

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Per-Fold Accuracy" span={2}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={foldChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fold" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => `${v}%`} />
              <Legend />
              {modelEntries.map(([key], idx) => (
                <Line key={key} type="monotone" dataKey={MODEL_LABELS[key] || key}
                      stroke={COLORS[idx]} strokeWidth={2} dot={{ r: 5 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Fold-by-Fold Breakdown" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', color: '#475569', fontWeight: 600 }}>Fold</th>
                  {modelEntries.map(([key]) => (
                    <th key={key} style={{ padding: '8px 10px', textAlign: 'center', color: '#475569', fontWeight: 600 }}>
                      {MODEL_LABELS[key] || key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {foldChartData.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{row.fold}</td>
                    {modelEntries.map(([key]) => (
                      <td key={key} style={{ padding: '8px 10px', textAlign: 'center' }}>
                        {row[MODEL_LABELS[key] || key] != null
                          ? <Badge text={`${row[MODEL_LABELS[key] || key]}%`} color="#10b981" />
                          : '-'}
                      </td>
                    ))}
                  </tr>
                ))}
                <tr style={{ borderTop: '2px solid #e2e8f0', fontWeight: 700 }}>
                  <td style={{ padding: '8px 10px' }}>Mean</td>
                  {modelEntries.map(([key, m]) => (
                    <td key={key} style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={`${(m.accuracy_mean * 100).toFixed(2)}%`} color="#3b82f6" />
                    </td>
                  ))}
                </tr>
                <tr>
                  <td style={{ padding: '8px 10px', fontWeight: 700 }}>Std Dev</td>
                  {modelEntries.map(([key, m]) => (
                    <td key={key} style={{ padding: '8px 10px', textAlign: 'center', color: '#64748b' }}>
                      {m.accuracy_std?.toFixed(4)}
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    )
  }

  /* ── Comparison Tab ── */
  const renderComparison = () => {
    const comparison = breakdown?.comparison || []
    const chartData = comparison.map(c => ({
      model: MODEL_LABELS[c.model] || c.model,
      Accuracy: +(c.accuracy * 100).toFixed(1),
      F1: +(c.f1 * 100).toFixed(1),
      AUC: +(c.auc * 100).toFixed(1),
    }))

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        <Card title="Model Comparison" span={2}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip formatter={v => `${v}%`} />
              <Legend />
              <Bar dataKey="Accuracy" fill={COLORS[0]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill={COLORS[3]} radius={[4, 4, 0, 0]} />
              <Bar dataKey="AUC" fill={COLORS[4]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Summary Table" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  {['Model', 'Accuracy', 'F1 Score', 'AUC'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: h === 'Model' ? 'left' : 'center', color: '#475569', fontWeight: 600 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.map((c, i) => (
                  <tr key={c.model} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>
                      <Badge text={MODEL_LABELS[c.model] || c.model} color={COLORS[i]} />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={`${(c.accuracy * 100).toFixed(1)}%`} color="#10b981" />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={c.f1.toFixed(3)} color="#3b82f6" />
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      <Badge text={c.auc.toFixed(3)} color="#8b5cf6" />
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

  /* ── Definitions Tab ── */
  const renderDefinitions = () => {
    const terms = definitions?.terms || []
    return (
      <Card title="Benchmark Validation Glossary">
        <div style={{ display: 'grid', gap: 8 }}>
          {terms.map((t, i) => (
            <div key={i}
                 onClick={() => setExpandedTerm(expandedTerm === i ? null : i)}
                 style={{
                   padding: '10px 14px', background: expandedTerm === i ? '#f0f9ff' : '#f8fafc',
                   borderRadius: 8, cursor: 'pointer', border: '1px solid',
                   borderColor: expandedTerm === i ? '#bfdbfe' : '#f1f5f9',
                   transition: 'all 0.15s'
                 }}>
              <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>
                {expandedTerm === i ? '\u25BC' : '\u25B6'} {t.term}
              </div>
              {expandedTerm === i && (
                <div style={{ marginTop: 6, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>
                  {t.definition}
                </div>
              )}
            </div>
          ))}
        </div>
      </Card>
    )
  }

  const renderTab = () => {
    switch (tab) {
      case 'overview': return renderOverview()
      case 'models': return renderModels()
      case 'fold-analysis': return renderFoldAnalysis()
      case 'comparison': return renderComparison()
      case 'definitions': return renderDefinitions()
      default: return renderOverview()
    }
  }

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#0f172a' }}>Benchmark Validation</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          External dataset validation results — Bonn University epilepsy EEG dataset
        </p>
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
              background: tab === t.id ? '#3b82f6' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {renderTab()}
    </div>
  )
}
