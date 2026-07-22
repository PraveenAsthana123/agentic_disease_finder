import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function fairnessColor(metric, value) {
  if (value == null) return '#94a3b8'
  if (metric === 'disparate_impact') {
    if (value >= 0.8 && value <= 1.25) return '#4caf50'
    if (value >= 0.6 && value <= 1.5) return '#ff9800'
    return '#f44336'
  }
  const abs = Math.abs(value)
  if (abs <= 0.1) return '#4caf50'
  if (abs <= 0.2) return '#ff9800'
  return '#f44336'
}

function fmt(v, decimals = 4) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toFixed(decimals) : String(v)
}

export default function AIF360Dashboard() {
  const [overview, setOverview] = useState(null)
  const [groups, setGroups] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showDefs, setShowDefs] = useState(false)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, gr, df] = await Promise.all([
          axios.get(`${API_URL}/api/aif360/overview`),
          axios.get(`${API_URL}/api/aif360/groups`),
          axios.get(`${API_URL}/api/aif360/definitions`)
        ])
        setOverview(ov.data)
        setGroups(gr.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load AIF360 bias detection data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Running AIF360 bias detection on real EEG data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'AIF360 bias detection not available. Ensure model predictions and protected attributes are present.'}
    </div>
  )

  const datasetMetrics = overview.dataset_metrics || {}
  const classMetrics = overview.classification_metrics || {}
  const mitigation = overview.mitigation_results || {}
  const metadata = overview.metadata || {}

  const di = datasetMetrics.disparate_impact
  const spd = datasetMetrics.statistical_parity_difference
  const eod = classMetrics.equal_opportunity_difference
  const aod = classMetrics.average_odds_difference

  const kpiItems = [
    { label: 'Disparate Impact', value: di, metric: 'disparate_impact', ideal: '0.8 - 1.25', icon: '&#9878;' },
    { label: 'Stat. Parity Diff', value: spd, metric: 'statistical_parity_difference', ideal: 'close to 0', icon: '&#9878;' },
    { label: 'Equal Opp. Diff', value: eod, metric: 'equal_opportunity_difference', ideal: 'close to 0', icon: '&#9878;' },
    { label: 'Avg Odds Diff', value: aod, metric: 'average_odds_difference', ideal: 'close to 0', icon: '&#9878;' },
  ]

  // Build before/after mitigation chart data
  const beforeMit = mitigation.before_mitigation || {}
  const afterMit = mitigation.after_mitigation || {}
  const mitigationMetricKeys = Array.from(new Set([
    ...Object.keys(beforeMit),
    ...Object.keys(afterMit),
  ]))
  const mitigationChartData = mitigationMetricKeys.map(key => ({
    metric: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    Before: beforeMit[key] != null ? beforeMit[key] : null,
    After: afterMit[key] != null ? afterMit[key] : null,
  }))

  // Build group comparison data
  const groupList = Array.isArray(groups) ? groups : (groups?.groups || [])
  const privileged = groupList.find(g => g.group === 'privileged' || g.privileged === true) || groupList[0] || {}
  const unprivileged = groupList.find(g => g.group === 'unprivileged' || g.privileged === false) || groupList[1] || {}
  const groupMetricKeys = ['base_rate', 'positive_prediction_rate', 'true_positive_rate', 'false_positive_rate']
  const groupChartData = groupMetricKeys.map(key => ({
    metric: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
    Privileged: privileged[key] != null ? privileged[key] : null,
    Unprivileged: unprivileged[key] != null ? unprivileged[key] : null,
  }))

  // Definitions
  const defsList = Array.isArray(defs) ? defs : (defs ? Object.values(defs).filter(d => typeof d === 'object' && d.name) : [])

  const cardStyle = {
    background: '#ffffff',
    borderRadius: 12,
    padding: 20,
    boxShadow: '0 1px 4px rgba(0,0,0,0.07)',
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
          AIF360 Bias Detection Dashboard
        </h2>
        <p style={{ margin: '6px 0 0', color: '#64748b', fontSize: 14 }}>
          IBM AI Fairness 360 bias detection on real EEG data
          {metadata.library_version ? ` — aif360 v${metadata.library_version}` : ''}
          {metadata.protected_attribute ? ` | Protected attribute: ${metadata.protected_attribute}` : ''}
          {metadata.mitigation_method ? ` | Mitigation: ${metadata.mitigation_method}` : ''}
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 20, flexWrap: 'wrap' }}>
        {kpiItems.map(kpi => {
          const color = fairnessColor(kpi.metric, kpi.value)
          return (
            <div key={kpi.label} style={kpiStyle(color)}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{kpi.label}</div>
              <div style={{ fontSize: 26, fontWeight: 700, color }}>
                {fmt(kpi.value)}
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                Ideal: {kpi.ideal}
              </div>
            </div>
          )
        })}
      </div>

      {/* Before/After Mitigation Bar Chart */}
      {mitigationChartData.length > 0 && (
        <div style={{ ...cardStyle, marginBottom: 16 }}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Before / After Reweighing Mitigation
          </h3>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={mitigationChartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="metric" tick={{ fontSize: 12, fill: '#475569' }} angle={-20} textAnchor="end" height={70} />
              <YAxis tick={{ fontSize: 12, fill: '#475569' }} />
              <Tooltip
                contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                formatter={(val) => val != null ? val.toFixed(4) : '--'}
              />
              <Legend />
              <Bar dataKey="Before" fill="#f44336" radius={[4, 4, 0, 0]} />
              <Bar dataKey="After" fill="#4caf50" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Group Comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        {/* Group Bar Chart */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Group Comparison: Privileged vs Unprivileged
          </h3>
          {groupChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={groupChartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="metric" tick={{ fontSize: 11, fill: '#475569' }} angle={-15} textAnchor="end" height={70} />
                <YAxis tick={{ fontSize: 12, fill: '#475569' }} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}
                  formatter={(val) => val != null ? val.toFixed(4) : '--'}
                />
                <Legend />
                <Bar dataKey="Privileged" fill="#1e88e5" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Unprivileged" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No group data available</div>
          )}
        </div>

        {/* Group Table */}
        <div style={cardStyle}>
          <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155' }}>
            Group Statistics
          </h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Metric</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#1e88e5', fontWeight: 600 }}>Privileged</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#7c4dff', fontWeight: 600 }}>Unprivileged</th>
                <th style={{ textAlign: 'right', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Difference</th>
              </tr>
            </thead>
            <tbody>
              {groupMetricKeys.map(key => {
                const pVal = privileged[key]
                const uVal = unprivileged[key]
                const diff = (pVal != null && uVal != null) ? (pVal - uVal) : null
                const diffColor = diff != null ? (Math.abs(diff) <= 0.1 ? '#4caf50' : Math.abs(diff) <= 0.2 ? '#ff9800' : '#f44336') : '#94a3b8'
                return (
                  <tr key={key} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#334155' }}>
                      {fmt(pVal)}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: '#334155' }}>
                      {fmt(uVal)}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'right', fontFamily: 'monospace', color: diffColor, fontWeight: 600 }}>
                      {diff != null ? (diff >= 0 ? '+' : '') + fmt(diff) : '--'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Definitions toggle */}
      <div style={cardStyle}>
        <button onClick={() => setShowDefs(!showDefs)} style={{
          background: 'none', border: '1px solid #cbd5e1', borderRadius: 8,
          padding: '8px 16px', cursor: 'pointer', fontSize: 13, color: '#475569',
        }}>
          {showDefs ? '\u25BE Hide' : '\u25B8 Show'} Metric Definitions & Clinical Relevance
        </button>
        {showDefs && defs && (
          <div style={{ marginTop: 16, maxHeight: 400, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Metric Name</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', color: '#475569', fontWeight: 600 }}>Clinical Relevance</th>
                </tr>
              </thead>
              <tbody>
                {defsList.map((def, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 10px', color: '#1e293b', fontWeight: 500 }}>
                      {def.name || def.metric_name || `Metric ${i + 1}`}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#475569', lineHeight: 1.4 }}>
                      {def.description || '--'}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#1e88e5', fontSize: 12 }}>
                      {def.clinical_relevance || '--'}
                    </td>
                  </tr>
                ))}
                {defsList.length === 0 && (
                  <tr>
                    <td colSpan={3} style={{ padding: 20, textAlign: 'center', color: '#94a3b8' }}>No definitions available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
