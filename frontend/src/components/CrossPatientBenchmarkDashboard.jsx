import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

const DIFFICULTY_COLORS = { easy: '#10b981', medium: '#f59e0b', hard: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

function DifficultyBadge({ level }) {
  const color = DIFFICULTY_COLORS[level] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{level || 'unknown'}</span>
  )
}

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

export default function CrossPatientBenchmarkDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/cross-patient-benchmark/overview`),
          axios.get(`${API_URL}/cross-patient-benchmark/breakdown`),
          axios.get(`${API_URL}/cross-patient-benchmark/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load cross-patient benchmark data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#128202;</div>
      Loading cross-patient benchmark data...
    </div>
  )
  if (error) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  )
  if (!overview) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      No cross-patient benchmark data available.
    </div>
  )

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'folds', label: 'Fold Detail' },
    { id: 'features', label: 'Feature Analysis' },
    { id: 'spatial', label: 'Spatial Patterns' }
  ]

  const kpi = overview.kpis || {}
  const foldPerf = overview.fold_performance || []
  const accDist = overview.accuracy_distribution || []
  const inSample = overview.in_sample_comparison || {}
  const featureSet = overview.feature_set || []
  const folds = (breakdown && breakdown.folds_detail) || []
  const genGap = (breakdown && breakdown.generalization_gap) || []
  const subjDiff = (breakdown && breakdown.subject_difficulty) || []
  const bandPower = (breakdown && breakdown.band_power_contribution) || []
  const spatial = (breakdown && breakdown.spatial_patterns) || []

  const gapChartData = [
    { name: 'In-Sample', accuracy: inSample.in_sample_accuracy, fill: '#10b981' },
    { name: 'Cross-Patient', accuracy: inSample.cross_patient_accuracy, fill: '#3b82f6' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>Cross-Patient Benchmark</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        LOSO cross-validation | {kpi.n_subjects} subjects | {kpi.n_folds} folds | {kpi.window_seconds}s windows | {kpi.feature_count} features
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
            <Card><KPI label="Mean Accuracy" value={fmtPct(kpi.mean_accuracy)} color="#3b82f6" /></Card>
            <Card><KPI label="Mean F1" value={fmtPct(kpi.mean_f1)} color="#8b5cf6" /></Card>
            <Card><KPI label="Generalization Gap" value={fmtPct(inSample.gap)} sub="in-sample vs cross-patient" color="#ef4444" /></Card>
            <Card><KPI label="Subjects" value={kpi.n_subjects} sub={`${kpi.n_folds} LOSO folds`} color="#10b981" /></Card>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <Card title="Per-Subject Accuracy (LOSO)">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={foldPerf} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="subject" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} tickFormatter={v => (v * 100) + '%'} />
                  <Tooltip formatter={v => fmtPct(v)} />
                  <Bar dataKey="accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Accuracy" />
                  <Bar dataKey="f1" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="F1" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="In-Sample vs Cross-Patient">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={gapChartData} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} tickFormatter={v => (v * 100) + '%'} />
                  <Tooltip formatter={v => fmtPct(v)} />
                  <Bar dataKey="accuracy" radius={[4, 4, 0, 0]} name="Accuracy">
                    {gapChartData.map((d, i) => <Cell key={i} fill={d.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ textAlign: 'center', fontSize: 12, color: '#ef4444', marginTop: 4 }}>
                Gap: {fmtPct(inSample.gap)}
              </div>
            </Card>
          </div>

          <Card title="Accuracy Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={accDist} margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin_start" tick={{ fontSize: 12 }} tickFormatter={v => (v * 100) + '%'} />
                <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                <Tooltip labelFormatter={v => `${(v * 100).toFixed(0)}%-${((v + 0.1) * 100).toFixed(0)}%`} />
                <Bar dataKey="count" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Subjects" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {tab === 'folds' && (
        <>
          <Card title="LOSO Fold Detail">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '8px 12px' }}>Held-Out Subject</th>
                    <th style={{ padding: '8px 12px' }}>Train Subjects</th>
                    <th style={{ padding: '8px 12px' }}>Accuracy</th>
                    <th style={{ padding: '8px 12px' }}>F1</th>
                    <th style={{ padding: '8px 12px' }}>Test Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {folds.map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{f.held_out_subject}</td>
                      <td style={{ padding: '8px 12px', color: '#64748b' }}>{(f.train_subjects || []).join(', ')}</td>
                      <td style={{ padding: '8px 12px' }}>{fmtPct(f.accuracy)}</td>
                      <td style={{ padding: '8px 12px' }}>{fmtPct(f.f1)}</td>
                      <td style={{ padding: '8px 12px' }}>{f.n_test}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
            <Card title="Generalization Gap by Subject">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 12px' }}>Subject</th>
                      <th style={{ padding: '8px 12px' }}>In-Sample</th>
                      <th style={{ padding: '8px 12px' }}>Cross-Patient</th>
                      <th style={{ padding: '8px 12px' }}>Gap</th>
                    </tr>
                  </thead>
                  <tbody>
                    {genGap.map((g, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{g.held_out_subject}</td>
                        <td style={{ padding: '8px 12px' }}>{fmtPct(g.in_sample_accuracy)}</td>
                        <td style={{ padding: '8px 12px' }}>{fmtPct(g.cross_patient_accuracy)}</td>
                        <td style={{ padding: '8px 12px', color: '#ef4444' }}>{fmtPct(g.gap)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card title="Subject Difficulty Ranking">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 12px' }}>Rank</th>
                      <th style={{ padding: '8px 12px' }}>Subject</th>
                      <th style={{ padding: '8px 12px' }}>Accuracy</th>
                      <th style={{ padding: '8px 12px' }}>Difficulty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {subjDiff.map((s, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '8px 12px' }}>#{s.rank}</td>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.subject}</td>
                        <td style={{ padding: '8px 12px' }}>{fmtPct(s.accuracy)}</td>
                        <td style={{ padding: '8px 12px' }}><DifficultyBadge level={s.difficulty} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </div>
        </>
      )}

      {tab === 'features' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Band Power Importance">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={bandPower} margin={{ left: 10, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="band" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={v => (v * 100).toFixed(0) + '%'} />
                  <Tooltip formatter={v => fmtPct(v)} />
                  <Bar dataKey="importance_normalized" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Normalized Importance" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Feature Categories">
              {featureSet.map((fs, i) => (
                <div key={i} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 4, textTransform: 'capitalize' }}>{fs.category}</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {(fs.features || []).map((f, j) => (
                      <span key={j} style={{
                        padding: '3px 10px', borderRadius: 12, background: COLORS[i % COLORS.length] + '18',
                        color: COLORS[i % COLORS.length], fontSize: 12, fontWeight: 500
                      }}>{f}</span>
                    ))}
                  </div>
                </div>
              ))}
            </Card>
          </div>
        </>
      )}

      {tab === 'spatial' && (
        <Card title="Dominant Electrodes by Held-Out Subject">
          {spatial.map((sp, i) => (
            <div key={i} style={{ marginBottom: 20 }}>
              <h4 style={{ fontSize: 14, color: '#334155', marginBottom: 8 }}>
                Held out: <strong>{sp.held_out_subject}</strong>
              </h4>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {(sp.dominant_electrodes || []).map((el, j) => (
                  <div key={j} style={{
                    padding: '6px 14px', borderRadius: 8, background: '#f1f5f9',
                    fontSize: 13, display: 'flex', alignItems: 'center', gap: 6
                  }}>
                    <span style={{ fontWeight: 700, color: '#1e293b' }}>{el.electrode}</span>
                    <span style={{ color: '#64748b' }}>{(el.contribution * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </Card>
      )}

      {defs && (
        <div style={{ marginTop: 20, padding: 16, background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
          <strong>{defs.dashboard_name}</strong> — {defs.description && defs.description.slice(0, 200)}...
          {defs.caveat && <div style={{ marginTop: 4, fontStyle: 'italic' }}>{defs.caveat}</div>}
        </div>
      )}
    </div>
  )
}
