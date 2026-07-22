import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, LineChart, Line, Legend, AreaChart, Area
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

export default function TorchMetricsDashboard() {
  const [overview, setOverview] = useState(null)
  const [curves, setCurves] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, cv, df] = await Promise.all([
          axios.get(`${API_URL}/api/torchmetrics/overview`),
          axios.get(`${API_URL}/api/torchmetrics/curves`),
          axios.get(`${API_URL}/api/torchmetrics/definitions`)
        ])
        setOverview(ov.data)
        setCurves(cv.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load TorchMetrics data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>📊</div>
      Computing TorchMetrics evaluation on real EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'TorchMetrics evaluation not available. Ensure model and data are present.'}
    </div>
  )

  const { metrics, confusion_matrix, per_class } = overview

  // KPI card data
  const kpiItems = [
    { label: 'Accuracy', value: metrics.accuracy, color: '#1e88e5' },
    { label: 'F1 Score', value: metrics.f1, color: '#7c4dff' },
    { label: 'AUROC', value: metrics.auroc, color: '#4caf50' },
    { label: 'Cohen\'s Kappa', value: metrics.kappa, color: '#ff9800' },
    { label: 'MCC', value: metrics.mcc, color: '#f44336' },
    { label: 'Specificity', value: metrics.specificity, color: '#00bcd4' },
  ]

  const kpiColor = (v) => {
    if (v >= 0.8) return '#16a34a'
    if (v >= 0.6) return '#ca8a04'
    return '#dc2626'
  }

  // Per-class bar chart data
  const classBarData = per_class.map(c => ({
    name: c.class_name || `Class ${c.class_id}`,
    precision: c.precision,
    recall: c.recall,
    f1: c.f1,
  }))

  // ROC curve data
  const rocData = curves?.roc ? curves.roc.fpr.map((fpr, i) => ({
    fpr,
    tpr: curves.roc.tpr[i],
  })) : []

  // Diagonal reference for ROC
  const rocDiagonal = [{ fpr: 0, ref: 0 }, { fpr: 1, ref: 1 }]

  // Precision-Recall curve data
  const prData = curves?.pr ? curves.pr.recall.map((recall, i) => ({
    recall,
    precision: curves.pr.precision[i],
  })) : []

  // Radar data
  const radarData = [
    { metric: 'Accuracy', value: metrics.accuracy },
    { metric: 'Precision', value: metrics.precision },
    { metric: 'Recall', value: metrics.recall },
    { metric: 'F1', value: metrics.f1 },
    { metric: 'Specificity', value: metrics.specificity },
    { metric: 'AUROC', value: metrics.auroc },
    { metric: 'Kappa', value: metrics.kappa },
    { metric: 'MCC', value: metrics.mcc },
  ].filter(d => d.value != null)

  // Confusion matrix cells
  const cm = confusion_matrix || { tp: 0, tn: 0, fp: 0, fn: 0 }
  const cmMax = Math.max(cm.tp, cm.tn, cm.fp, cm.fn, 1)

  const cmCellStyle = (value, isCorrect) => ({
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
    borderRadius: 8,
    background: isCorrect
      ? `rgba(76, 175, 80, ${0.15 + 0.55 * (value / cmMax)})`
      : `rgba(244, 67, 54, ${0.10 + 0.40 * (value / cmMax)})`,
    border: `1px solid ${isCorrect ? '#a5d6a7' : '#ef9a9a'}`,
    minWidth: 100,
    minHeight: 80,
  })

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
    border: '1px solid #e5e7eb',
  }

  const kpiStyle = (color) => ({
    ...cardStyle,
    borderLeft: `4px solid ${color}`,
    minWidth: 150,
    flex: 1,
  })

  return (
    <div style={{ padding: 20, background: '#f8fafc', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>
          📊 TorchMetrics Model Evaluation
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          Evaluation via <b>torchmetrics</b>
          {overview.library_version ? ` v${overview.library_version}` : ''}
          {overview.model_type ? ` — ${overview.model_type}` : ''}
          {overview.n_features != null ? ` · ${overview.n_features} features` : ''}
          {overview.n_samples != null ? ` · ${overview.n_samples} samples` : ''}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => (
          <div key={kpi.label} style={kpiStyle(kpi.color)}>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: kpiColor(kpi.value) }}>
              {kpi.value != null ? kpi.value.toFixed(3) : '—'}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>
              {kpi.value != null && kpi.value >= 0.8 ? 'Good' : kpi.value != null && kpi.value >= 0.6 ? 'Fair' : kpi.value != null ? 'Low' : ''}
            </div>
          </div>
        ))}
      </div>

      {/* Confusion Matrix + Per-Class Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16, marginBottom: 16 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Confusion Matrix</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div style={cmCellStyle(cm.tp, true)}>
              <div style={{ fontSize: 11, color: '#475569', fontWeight: 600 }}>TP</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{cm.tp}</div>
            </div>
            <div style={cmCellStyle(cm.fp, false)}>
              <div style={{ fontSize: 11, color: '#475569', fontWeight: 600 }}>FP</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{cm.fp}</div>
            </div>
            <div style={cmCellStyle(cm.fn, false)}>
              <div style={{ fontSize: 11, color: '#475569', fontWeight: 600 }}>FN</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{cm.fn}</div>
            </div>
            <div style={cmCellStyle(cm.tn, true)}>
              <div style={{ fontSize: 11, color: '#475569', fontWeight: 600 }}>TN</div>
              <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{cm.tn}</div>
            </div>
          </div>
          <div style={{ marginTop: 10, fontSize: 11, color: '#94a3b8', textAlign: 'center' }}>
            Predicted &rarr; | Actual &darr;
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Per-Class Metrics</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={classBarData} margin={{ left: 10, right: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
              <Legend />
              <Bar dataKey="precision" name="Precision" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              <Bar dataKey="recall" name="Recall" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" name="F1" fill="#4caf50" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ROC Curve + Precision-Recall Curve */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>
            ROC Curve {curves?.roc?.auroc != null ? `(AUROC = ${curves.roc.auroc.toFixed(3)})` : ''}
          </h3>
          {rocData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart margin={{ left: 10, right: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={{ fontSize: 11 }}
                  label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fontSize: 12 }} />
                <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }}
                  label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                <Tooltip formatter={(v) => v.toFixed(3)} />
                <Legend />
                <Line data={rocData} type="monotone" dataKey="tpr" name={`ROC (AUC=${curves?.roc?.auroc?.toFixed(3) || '—'})`}
                  stroke="#1e88e5" dot={false} strokeWidth={2} />
                <Line data={rocDiagonal} type="monotone" dataKey="ref" name="Random"
                  stroke="#94a3b8" dot={false} strokeWidth={1} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No ROC data available</div>
          )}
        </div>

        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>
            Precision-Recall Curve {curves?.pr?.auprc != null ? `(AUPRC = ${curves.pr.auprc.toFixed(3)})` : ''}
          </h3>
          {prData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={prData} margin={{ left: 10, right: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={{ fontSize: 11 }}
                  label={{ value: 'Recall', position: 'insideBottom', offset: -2, fontSize: 12 }} />
                <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }}
                  label={{ value: 'Precision', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                <Tooltip formatter={(v) => v.toFixed(3)} />
                <Line type="monotone" dataKey="precision" name="PR Curve"
                  stroke="#7c4dff" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No PR data available</div>
          )}
        </div>
      </div>

      {/* Radar Chart */}
      <div style={{ ...cardStyle, marginBottom: 16 }}>
        <h3 style={{ margin: '0 0 12px', fontSize: 15, color: '#334155' }}>Metrics Radar Overview</h3>
        {radarData.length > 0 ? (
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: '#475569' }} />
              <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fontSize: 10 }} />
              <Radar name="Model" dataKey="value" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.25} strokeWidth={2} />
              <Tooltip formatter={(v) => v.toFixed(3)} />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No radar data</div>
        )}
      </div>

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '▾ Hide' : '▸ Show'} Metric Definitions & Clinical Relevance
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, display: 'grid', gap: 14 }}>
            {Object.values(defs).filter(d => typeof d === 'object' && d.name).map((def, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: 14, border: '1px solid #e2e8f0' }}>
                <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 6 }}>{def.name}</div>
                <div style={{ color: '#475569', fontSize: 13, lineHeight: 1.5 }}>{def.description}</div>
                {def.citation && (
                  <div style={{ marginTop: 4, fontSize: 11, color: '#64748b', fontStyle: 'italic' }}>{def.citation}</div>
                )}
                {def.clinical_relevance && (
                  <div style={{ marginTop: 4, fontSize: 12, color: '#1e88e5' }}>Clinical: {def.clinical_relevance}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
