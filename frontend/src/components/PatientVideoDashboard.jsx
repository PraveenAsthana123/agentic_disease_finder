import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, LineChart, Line
} from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8010'
const COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#84cc16',
  '#e11d48', '#64748b'
]

export default function PatientVideoDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    const endpoints = {
      overview: '/api/patient-video/overview',
      breakdown: '/api/patient-video/breakdown',
      definitions: '/api/patient-video/definitions',
    }
    const url = endpoints[tab]
    if (!url) return
    axios.get(`${API}${url}`)
      .then(r => {
        if (tab === 'overview') setOverview(r.data)
        else if (tab === 'breakdown') setBreakdown(r.data)
        else setDefinitions(r.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [tab])

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'breakdown', label: 'Patient Breakdown' },
    { id: 'models', label: 'Model Comparison' },
    { id: 'pose', label: 'Pose & Motion' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const sty = {
    card: { background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 },
    kpi: { background: '#334155', borderRadius: 8, padding: 16, textAlign: 'center', flex: 1 },
    kpiVal: { fontSize: 28, fontWeight: 700, color: '#818cf8' },
    kpiLabel: { fontSize: 12, color: '#94a3b8', marginTop: 4 },
    grid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 },
    tab: (active) => ({
      padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
      background: active ? '#6366f1' : '#334155', color: active ? '#fff' : '#94a3b8', fontSize: 13
    }),
    badge: (level) => ({
      display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 700,
      background: level === 'high' ? '#ef444433' : level === 'moderate' ? '#f59e0b33' : '#10b98133',
      color: level === 'high' ? '#ef4444' : level === 'moderate' ? '#f59e0b' : '#10b981'
    }),
    table: { width: '100%', borderCollapse: 'collapse', fontSize: 13 },
    th: { textAlign: 'left', padding: '8px 10px', borderBottom: '1px solid #334155', color: '#94a3b8', fontWeight: 600 },
    td: { padding: '8px 10px', borderBottom: '1px solid #1e293b' },
  }

  // ── Tab: Overview ────────────────────────────────────────────────
  const renderOverview = () => {
    if (!overview) return null
    const d = overview
    return (<>
      <div style={sty.grid}>
        {[
          [d.total_patients, 'Patients Monitored'],
          [d.total_video_events, 'Video Events'],
          [d.seizure_events_detected, 'Seizure Events'],
          [d.automatism_events, 'Automatisms'],
          [d.fall_alerts, 'Fall Alerts'],
          [`${d.fall_alert_pct}%`, 'Fall Alert Rate'],
          [`${(d.average_confidence * 100).toFixed(1)}%`, 'Avg Confidence'],
        ].map(([v, l], i) => (
          <div key={i} style={sty.kpi}>
            <div style={{...sty.kpiVal, color: i === 4 ? '#ef4444' : '#818cf8'}}>{v}</div>
            <div style={sty.kpiLabel}>{l}</div>
          </div>
        ))}
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 12}}>
        <div style={sty.card}>
          <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Motor Pattern Distribution</h4>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={d.pattern_distribution.filter(p => p.count > 0)} layout="vertical"
              margin={{left: 120}}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis type="number" stroke="#64748b" />
              <YAxis type="category" dataKey="pattern" stroke="#94a3b8" fontSize={11} width={110} />
              <Tooltip contentStyle={{background: '#1e293b', border: '1px solid #334155'}} />
              <Bar dataKey="count" fill="#6366f1" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={sty.card}>
          <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Pattern Type Breakdown</h4>
          <ResponsiveContainer width="100%" height={320}>
            <PieChart>
              <Pie data={d.pattern_distribution.filter(p => p.count > 0)}
                dataKey="count" nameKey="pattern" cx="50%" cy="50%"
                outerRadius={110} label={({pattern, percent}) =>
                  `${pattern.split(' ')[0]} ${(percent*100).toFixed(0)}%`}
                labelLine={false} fontSize={10}>
                {d.pattern_distribution.filter(p => p.count > 0).map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{background: '#1e293b', border: '1px solid #334155'}} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={sty.card}>
        <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Confidence Score Distribution</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={d.confidence_histogram}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="bin" stroke="#64748b" fontSize={11} />
            <YAxis stroke="#64748b" />
            <Tooltip contentStyle={{background: '#1e293b', border: '1px solid #334155'}} />
            <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div style={sty.card}>
        <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Per-Class Metrics (Best Model)</h4>
        <div style={{overflowX: 'auto'}}>
          <table style={sty.table}>
            <thead>
              <tr>
                {['Pattern', 'Precision', 'Recall', 'F1', 'Support'].map(h => (
                  <th key={h} style={sty.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {d.per_class_metrics.map((m, i) => (
                <tr key={i} style={{color: '#e2e8f0'}}>
                  <td style={sty.td}>{m.pattern}</td>
                  <td style={sty.td}>{m.precision}</td>
                  <td style={sty.td}>{m.recall}</td>
                  <td style={{...sty.td, color: m.f1 >= 0.8 ? '#10b981' : m.f1 >= 0.65 ? '#f59e0b' : '#ef4444'}}>
                    {m.f1}
                  </td>
                  <td style={sty.td}>{m.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>)
  }

  // ── Tab: Patient Breakdown ───────────────────────────────────────
  const renderBreakdown = () => {
    if (!breakdown) return null
    return (<>
      <div style={{...sty.card, marginBottom: 8}}>
        <span style={{color: '#94a3b8', fontSize: 13}}>
          {breakdown.total_patients} patients with video seizure events
        </span>
      </div>
      {breakdown.patients.map((pt, i) => (
        <details key={i} style={{...sty.card, cursor: 'pointer'}}>
          <summary style={{color: '#e2e8f0', fontWeight: 600, fontSize: 14, listStyle: 'none'}}>
            <span>{pt.name}</span>
            <span style={{color: '#64748b', fontWeight: 400, marginLeft: 10}}>
              ({pt.patient_id}) · {pt.age}y {pt.sex}
            </span>
            <span style={{float: 'right', display: 'flex', gap: 8, alignItems: 'center'}}>
              <span style={sty.badge(pt.fall_risk_level)}>{pt.fall_risk_level} fall risk</span>
              <span style={{color: '#94a3b8', fontSize: 12}}>
                {pt.seizure_events} seizure · {pt.fall_alerts} fall · {pt.automatism_events} auto
              </span>
            </span>
          </summary>
          <div style={{marginTop: 10}}>
            <table style={sty.table}>
              <thead>
                <tr>
                  {['#', 'Motor Pattern', 'Conf', 'Duration', 'Fall', 'Landmarks', 'Body Segments'].map(h => (
                    <th key={h} style={sty.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pt.events.map((ev, j) => (
                  <tr key={j} style={{color: '#e2e8f0'}}>
                    <td style={sty.td}>{ev.event_id}</td>
                    <td style={sty.td}>{ev.motor_pattern}</td>
                    <td style={{...sty.td, color: ev.confidence >= 0.8 ? '#10b981' : '#f59e0b'}}>
                      {(ev.confidence * 100).toFixed(0)}%
                    </td>
                    <td style={sty.td}>{ev.duration_s}s</td>
                    <td style={sty.td}>
                      {ev.fall_alert
                        ? <span style={{color: '#ef4444', fontWeight: 700}}>ALERT</span>
                        : <span style={{color: '#64748b'}}>—</span>}
                    </td>
                    <td style={sty.td}>{ev.landmarks_detected}/33</td>
                    <td style={sty.td}>{ev.body_segments_involved.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      ))}
    </>)
  }

  // ── Tab: Model Comparison ────────────────────────────────────────
  const renderModels = () => {
    if (!overview) return null
    const mp = overview.model_performance
    return (<>
      <div style={sty.card}>
        <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Model Performance Comparison</h4>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={mp}>
            <PolarGrid stroke="#334155" />
            <PolarAngleAxis dataKey="abbrev" stroke="#94a3b8" fontSize={12} />
            <PolarRadiusAxis domain={[0.5, 1]} stroke="#64748b" fontSize={10} />
            <Radar name="Accuracy" dataKey="accuracy" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} />
            <Radar name="F1" dataKey="macro_f1" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
            <Radar name="AUC" dataKey="auc_roc" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} />
            <Tooltip contentStyle={{background: '#1e293b', border: '1px solid #334155'}} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12}}>
        <div style={sty.card}>
          <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Accuracy vs Latency</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={mp}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="abbrev" stroke="#64748b" fontSize={11} />
              <YAxis yAxisId="acc" domain={[0.6, 1]} stroke="#6366f1" fontSize={10} />
              <YAxis yAxisId="lat" orientation="right" stroke="#ef4444" fontSize={10} />
              <Tooltip contentStyle={{background: '#1e293b', border: '1px solid #334155'}} />
              <Bar yAxisId="acc" dataKey="accuracy" fill="#6366f1" radius={[4, 4, 0, 0]} name="Accuracy" />
              <Line yAxisId="lat" type="monotone" dataKey="latency_ms" stroke="#ef4444" name="Latency (ms)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div style={sty.card}>
          <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Model Details</h4>
          <table style={sty.table}>
            <thead>
              <tr>
                {['Model', 'Type', 'Acc', 'F1', 'AUC', 'Latency', 'FPS', 'Params'].map(h => (
                  <th key={h} style={sty.th}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {mp.map((m, i) => (
                <tr key={i} style={{color: '#e2e8f0'}}>
                  <td style={sty.td}>{m.abbrev}</td>
                  <td style={sty.td}>{m.type}</td>
                  <td style={sty.td}>{m.accuracy}</td>
                  <td style={sty.td}>{m.macro_f1}</td>
                  <td style={sty.td}>{m.auc_roc}</td>
                  <td style={sty.td}>{m.latency_ms}ms</td>
                  <td style={sty.td}>{m.fps}</td>
                  <td style={sty.td}>{m.params_M}M</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>)
  }

  // ── Tab: Pose & Motion ───────────────────────────────────────────
  const renderPose = () => {
    if (!overview) return null
    const pq = overview.pose_quality
    const sm = overview.segment_motion
    return (<>
      <div style={sty.grid}>
        {[
          [pq.avg_landmarks_detected + '/33', 'Avg Landmarks Detected'],
          [(pq.avg_landmark_confidence * 100).toFixed(1) + '%', 'Landmark Confidence'],
          [pq.occlusion_rate_pct + '%', 'Occlusion Rate'],
          [pq.tracking_loss_pct + '%', 'Tracking Loss'],
        ].map(([v, l], i) => (
          <div key={i} style={sty.kpi}>
            <div style={sty.kpiVal}>{v}</div>
            <div style={sty.kpiLabel}>{l}</div>
          </div>
        ))}
      </div>

      <div style={{...sty.card, marginTop: 12}}>
        <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Body Segment Motion (Avg Velocity)</h4>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={sm} layout="vertical" margin={{left: 100}}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis type="number" stroke="#64748b" label={{value: 'deg/s', position: 'insideBottomRight', fill: '#64748b'}} />
            <YAxis type="category" dataKey="segment" stroke="#94a3b8" fontSize={11} width={90} />
            <Tooltip contentStyle={{background: '#1e293b', border: '1px solid #334155'}} />
            <Bar dataKey="avg_velocity_deg_s" fill="#14b8a6" radius={[0, 4, 4, 0]} name="Avg Velocity" />
            <Bar dataKey="max_velocity_deg_s" fill="#f97316" radius={[0, 4, 4, 0]} name="Max Velocity" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>)
  }

  // ── Tab: Definitions ─────────────────────────────────────────────
  const renderDefinitions = () => {
    if (!definitions) return null
    const d = definitions
    return (<>
      <div style={sty.card}>
        <h3 style={{color: '#e2e8f0', margin: '0 0 6px'}}>{d.title}</h3>
        <p style={{color: '#94a3b8', fontSize: 13, lineHeight: 1.6}}>{d.description}</p>
      </div>

      <div style={sty.card}>
        <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Motor Patterns</h4>
        {d.motor_patterns.map((mp, i) => (
          <div key={i} style={{marginBottom: 10, paddingBottom: 8, borderBottom: '1px solid #334155'}}>
            <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
              <span style={{color: '#818cf8', fontWeight: 600, fontSize: 13}}>{mp.label}</span>
              <span style={{...sty.badge(mp.fall_risk >= 0.6 ? 'high' : mp.fall_risk >= 0.3 ? 'moderate' : 'low'), fontSize: 10}}>
                Fall risk: {(mp.fall_risk * 100).toFixed(0)}%
              </span>
            </div>
            <p style={{color: '#94a3b8', fontSize: 12, margin: '4px 0 0', lineHeight: 1.5}}>{mp.description}</p>
          </div>
        ))}
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12}}>
        <div style={sty.card}>
          <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>AI Models</h4>
          {d.models.map((m, i) => (
            <div key={i} style={{marginBottom: 10}}>
              <span style={{color: '#10b981', fontWeight: 600, fontSize: 13}}>{m.name}</span>
              <span style={{color: '#64748b', fontSize: 11, marginLeft: 8}}>({m.type})</span>
              <p style={{color: '#94a3b8', fontSize: 12, margin: '4px 0 0'}}>{m.description}</p>
            </div>
          ))}
        </div>
        <div style={sty.card}>
          <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>Fall Detection Criteria</h4>
          <p style={{color: '#94a3b8', fontSize: 12, lineHeight: 1.6}}>{d.fall_detection_criteria.method}</p>
          <h5 style={{color: '#e2e8f0', margin: '12px 0 6px'}}>Risk Levels</h5>
          {Object.entries(d.fall_detection_criteria.risk_levels).map(([k, v]) => (
            <div key={k} style={{display: 'flex', gap: 8, marginBottom: 4}}>
              <span style={sty.badge(k)}>{k}</span>
              <span style={{color: '#94a3b8', fontSize: 12}}>{v}</span>
            </div>
          ))}
          <h5 style={{color: '#e2e8f0', margin: '12px 0 6px'}}>Alert Actions</h5>
          <ul style={{color: '#94a3b8', fontSize: 12, paddingLeft: 18}}>
            {d.fall_detection_criteria.alert_actions.map((a, i) => (
              <li key={i} style={{marginBottom: 3}}>{a}</li>
            ))}
          </ul>
        </div>
      </div>

      <div style={sty.card}>
        <h4 style={{color: '#e2e8f0', margin: '0 0 10px'}}>References</h4>
        <ol style={{color: '#94a3b8', fontSize: 12, paddingLeft: 18, lineHeight: 1.8}}>
          {d.references.map((r, i) => <li key={i}>{r}</li>)}
        </ol>
      </div>
    </>)
  }

  return (
    <div style={{padding: 16}}>
      <h2 style={{color: '#e2e8f0', marginBottom: 4}}>Patient Video Seizure Analysis</h2>
      <p style={{color: '#64748b', fontSize: 13, marginBottom: 12}}>
        Video-based motor pattern detection, action recognition, fall detection, and automatism analysis
      </p>
      <div style={{display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap'}}>
        {tabs.map(t => (
          <button key={t.id} style={sty.tab(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>
      {loading && <div style={{color: '#94a3b8', padding: 20}}>Loading...</div>}
      {error && <div style={{color: '#ef4444', padding: 20}}>Error: {error}</div>}
      {!loading && !error && tab === 'overview' && renderOverview()}
      {!loading && !error && tab === 'breakdown' && renderBreakdown()}
      {!loading && !error && tab === 'models' && renderModels()}
      {!loading && !error && tab === 'pose' && renderPose()}
      {!loading && !error && tab === 'definitions' && renderDefinitions()}
    </div>
  )
}
