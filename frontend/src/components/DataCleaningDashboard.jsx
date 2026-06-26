import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#6366f1', '#64748b', '#06b6d4']

const STEP_ICONS = { complete: '\u2705', pending_cron: '\u23F3', no_data: '\u26A0\uFE0F' }

export default function DataCleaningDashboard() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await axios.get(`${API_URL}/data-manager/cleaning`)
        setData(res.data)
      } catch (err) {
        setError(err.message || 'Failed to load data cleaning report')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>🧹</div>
      Running EEG data-cleaning analysis on real recordings...
    </div>
  )
  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca',
      borderRadius: 8, color: '#991b1b' }}>Error: {error}</div>
  )
  if (!data?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a',
      borderRadius: 8, color: '#92400e' }}>No EEG data available for cleaning analysis.</div>
  )

  const cq = data.channel_quality || {}
  const ns = data.nan_inf_stats || {}
  const ica = data.ica_artifact_removal || {}
  const steps = data.cleaning_steps || []

  // Pie chart data: channel verdicts
  const pieData = [
    { name: 'Good', value: cq.good || 0 },
    { name: 'Flat', value: cq.flat || 0 },
    { name: 'Noisy', value: cq.noisy || 0 },
    { name: 'Line Noise', value: cq.line_noise || 0 },
    { name: 'Disconnected', value: cq.disconnected || 0 }
  ].filter(d => d.value > 0)

  // Bar chart: per-subject breakdown
  const barData = (cq.per_subject || []).map(s => ({
    name: s.file?.replace('.edf', '') || '?',
    Good: s.good || 0,
    Bad: (s.flat || 0) + (s.noisy || 0) + (s.disconnected || 0) + (s.line_noise || 0)
  }))

  // Radar: cleaning dimensions
  const radarData = [
    { axis: 'Channel QC', val: cq.scanned > 0 ? Math.round((cq.good / Math.max(cq.scanned, 1)) * 100) : 0 },
    { axis: 'NaN-Free', val: Math.round((ns.clean_ratio || 0) * 100) },
    { axis: 'ICA Done', val: ica.available ? 100 : 0 },
    { axis: 'Quality', val: data.post_clean_quality_pct || 0 },
    { axis: 'Coverage', val: data.subjects_found > 0 ? Math.min(100, Math.round(data.edfs_sampled / data.subjects_found * 100)) : 0 }
  ]

  const card = { background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 }
  const kpiCard = { ...card, textAlign: 'center', flex: 1, minWidth: 140 }

  return (
    <div style={{ padding: 16 }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 20, color: '#1e293b' }}>
        🧹 Clinical Data Manager — Data Cleaning
      </h2>
      <p style={{ margin: '0 0 16px', color: '#64748b', fontSize: 13 }}>
        Real EEG signal-quality analysis on CHB-MIT recordings ({data.subjects_found} subjects, {data.edfs_sampled} files sampled)
      </p>

      {/* KPI tiles */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#16a34a' }}>
            {data.post_clean_quality_pct}%
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Post-Clean Quality</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#2563eb' }}>
            {cq.scanned}
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Channels Scanned</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#22c55e' }}>
            {cq.good}
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Good Channels</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#ef4444' }}>
            {(cq.flat || 0) + (cq.noisy || 0) + (cq.disconnected || 0)}
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>Bad Channels</div>
        </div>
        <div style={kpiCard}>
          <div style={{ fontSize: 24, fontWeight: 700, color: ns.nan_count > 0 ? '#ef4444' : '#22c55e' }}>
            {ns.nan_count + ns.inf_count}
          </div>
          <div style={{ fontSize: 12, color: '#64748b' }}>NaN/Inf Samples</div>
        </div>
      </div>

      {/* Charts row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
        {/* Channel verdict pie */}
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Channel Verdict Distribution</h3>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No channel data</p>}
        </div>

        {/* Per-subject bar */}
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Per-File: Good vs Bad Channels</h3>
          {barData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="Good" fill="#22c55e" />
                <Bar dataKey="Bad" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          ) : <p style={{ color: '#94a3b8', fontSize: 13 }}>No per-file data</p>}
        </div>

        {/* Cleaning radar */}
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>Cleaning Dimensions</h3>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
              <Radar dataKey="val" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Pipeline steps */}
      <div style={card}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#334155' }}>Cleaning Pipeline Steps</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Step</th>
              <th style={{ textAlign: 'left', padding: '6px 8px' }}>Name</th>
              <th style={{ textAlign: 'center', padding: '6px 8px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px 8px', fontWeight: 600 }}>{s.step}</td>
                <td style={{ padding: '6px 8px' }}>{s.name}</td>
                <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                  {STEP_ICONS[s.status] || s.status} {s.status.replace('_', ' ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ICA Artifact Removal summary */}
      {ica.available && (
        <div style={card}>
          <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#334155' }}>ICA Artifact Removal Summary</h3>
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', fontSize: 13 }}>
            <div><strong>Subjects cleaned:</strong> {ica.subjects_cleaned}</div>
            <div><strong>Mean variance removed:</strong> {ica.mean_variance_removed_pct}%</div>
            <div><strong>Mean components removed:</strong> {ica.mean_components_removed}</div>
          </div>
          {(ica.subjects || []).length > 0 && (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 10 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px' }}>Subject</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px' }}>Components Removed</th>
                  <th style={{ textAlign: 'right', padding: '4px 8px' }}>Variance Removed %</th>
                </tr>
              </thead>
              <tbody>
                {ica.subjects.map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '4px 8px' }}>{s.subject || s.file || `Subject ${i + 1}`}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{s.components_removed || 0}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right' }}>{s.variance_removed_pct || 0}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Challenges */}
      <div style={{ ...card, background: '#fffbeb', borderColor: '#fde68a' }}>
        <h3 style={{ margin: '0 0 8px', fontSize: 14, color: '#92400e' }}>Known Challenges</h3>
        <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13, color: '#92400e' }}>
          {(data.challenges || []).map((c, i) => <li key={i} style={{ marginBottom: 4 }}>{c}</li>)}
        </ul>
      </div>
    </div>
  )
}
