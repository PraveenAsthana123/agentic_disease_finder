import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#16a34a', '#eab308', '#ef4444', '#8b5cf6', '#ec4899', '#f59e0b', '#06b6d4']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(3)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : String(v)
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

function StatusBadge({ status }) {
  const styles = {
    easy: { bg: '#dcfce7', color: '#166534' },
    moderate: { bg: '#fef9c3', color: '#854d0e' },
    hard: { bg: '#fee2e2', color: '#991b1b' },
  }
  const s = styles[status] || { bg: '#f1f5f9', color: '#475569' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11,
      fontWeight: 600, background: s.bg, color: s.color
    }}>{status}</span>
  )
}

export default function CrossPatientDashboard() {
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
          axios.get(`${API_URL}/api/cross-patient-benchmark/overview`),
          axios.get(`${API_URL}/api/cross-patient-benchmark/breakdown`),
          axios.get(`${API_URL}/api/cross-patient-benchmark/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Cross-Patient Benchmark data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'folds', label: 'Fold Analysis' },
    { id: 'features', label: 'Feature Importance' },
    { id: 'spatial', label: 'Spatial Patterns' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const foldPerf = overview?.fold_performance || []
  const comparison = overview?.in_sample_comparison || {}
  const featureSet = overview?.feature_set || []

  const foldsDetail = breakdown?.folds_detail || []
  const genGap = breakdown?.generalization_gap || []
  const subjectDiff = breakdown?.subject_difficulty || []
  const bandPower = breakdown?.band_power_contribution || []
  const spatialPatterns = breakdown?.spatial_patterns || []

  const terms = defs?.terms || []
  const references = defs?.references || []
  const interpretation = defs?.interpretation || {}

  const comparisonChart = [
    { label: 'In-Sample', accuracy: comparison.in_sample_accuracy || 0 },
    { label: 'Cross-Patient', accuracy: comparison.cross_patient_accuracy || 0 },
  ]

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        Cross-Patient Benchmark Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Leave-one-subject-out (LOSO) cross-patient generalization benchmark on CHB-MIT scalp EEG database
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
            color: tab === t.id ? '#2563eb' : '#64748b', background: 'none', border: 'none',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            cursor: 'pointer', marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {/* ─── Overview Tab ─── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
              <KPI label="Mean Accuracy" value={fmtPct(kpis.mean_accuracy)} color="#2563eb" />
              <KPI label="Mean F1" value={fmtPct(kpis.mean_f1)} color="#16a34a" />
              <KPI label="Subjects" value={fmt(kpis.n_subjects)} />
              <KPI label="Folds" value={fmt(kpis.n_folds)} />
              <KPI label="Window" value={`${kpis.window_seconds || '--'}s`} sub="Epoch length" />
              <KPI label="Features" value={fmt(kpis.feature_count)} sub="12 fast features" />
            </div>
          </Card>

          <Card title="In-Sample vs Cross-Patient Accuracy">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={comparisonChart}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => [fmtPct(v), 'Accuracy']} />
                <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
                  {comparisonChart.map((d, i) => (
                    <Cell key={i} fill={i === 0 ? '#16a34a' : '#3b82f6'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', fontSize: 12, color: '#ef4444', fontWeight: 600, marginTop: 8 }}>
              Generalization Gap: {fmtPct(comparison.gap)}
            </div>
          </Card>

          <Card title="Per-Fold Accuracy" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={foldPerf}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => [fmtPct(v)]} />
                <Legend />
                <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="f1" name="F1 Score" fill="#16a34a" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Feature Set" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
              {featureSet.map((f, i) => (
                <div key={i} style={{
                  padding: '8px 12px', borderRadius: 8, background: '#f0f9ff', border: '1px solid #bae6fd',
                  fontSize: 12, color: '#0c4a6e'
                }}>
                  <span style={{ fontWeight: 600 }}>{f.name}</span>
                  <span style={{ color: '#64748b', marginLeft: 6 }}>({f.category})</span>
                </div>
              ))}
            </div>
          </Card>

          {comparison.gap > 0 && (
            <Card span={3} title="Key Insight">
              <div style={{ padding: 12, borderRadius: 8, background: '#fef3c7', border: '1px solid #fde68a', fontSize: 13, color: '#92400e', lineHeight: 1.6 }}>
                In-sample accuracy is {fmtPct(comparison.in_sample_accuracy)} but cross-patient (LOSO) accuracy drops
                to {fmtPct(comparison.cross_patient_accuracy)} — a <strong>{fmtPct(comparison.gap)} gap</strong>.
                This is typical for EEG classification and highlights the need for domain adaptation,
                transfer learning, and larger multi-site datasets to improve cross-patient generalization.
              </div>
            </Card>
          )}
        </div>
      )}

      {/* ─── Fold Analysis Tab ─── */}
      {tab === 'folds' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Fold-by-Fold Detail">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>Held-Out Subject</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>Train Subjects</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>N Test</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>Accuracy</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>F1 Score</th>
                  <th style={{ textAlign: 'left', padding: '8px 10px', borderBottom: '2px solid #e2e8f0', color: '#475569', fontWeight: 600, fontSize: 12 }}>Difficulty</th>
                </tr>
              </thead>
              <tbody>
                {foldsDetail.map((f, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ padding: '7px 10px', borderBottom: '1px solid #f1f5f9', fontWeight: 600 }}>{f.held_out_subject}</td>
                    <td style={{ padding: '7px 10px', borderBottom: '1px solid #f1f5f9', fontSize: 11 }}>{(f.train_subjects || []).join(', ')}</td>
                    <td style={{ padding: '7px 10px', borderBottom: '1px solid #f1f5f9' }}>{f.n_test}</td>
                    <td style={{ padding: '7px 10px', borderBottom: '1px solid #f1f5f9', color: f.accuracy >= 0.7 ? '#16a34a' : '#ef4444', fontWeight: 600 }}>{fmtPct(f.accuracy)}</td>
                    <td style={{ padding: '7px 10px', borderBottom: '1px solid #f1f5f9', color: f.f1 >= 0.7 ? '#16a34a' : '#ef4444', fontWeight: 600 }}>{fmtPct(f.f1)}</td>
                    <td style={{ padding: '7px 10px', borderBottom: '1px solid #f1f5f9' }}>
                      <StatusBadge status={f.difficulty || 'moderate'} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Generalization Gap per Fold">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={genGap}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="subject" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => [fmtPct(v)]} />
                <Legend />
                <Bar dataKey="in_sample" name="In-Sample" fill="#16a34a" radius={[4, 4, 0, 0]} />
                <Bar dataKey="cross_patient" name="Cross-Patient" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="gap" name="Gap" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Subject Difficulty Ranking">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={subjectDiff} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <YAxis type="category" dataKey="subject" tick={{ fontSize: 12 }} width={60} />
                <Tooltip formatter={(v) => [fmtPct(v), 'Accuracy']} />
                <Bar dataKey="accuracy" radius={[0, 4, 4, 0]}>
                  {subjectDiff.map((d, i) => (
                    <Cell key={i} fill={d.accuracy >= 0.7 ? '#16a34a' : d.accuracy >= 0.5 ? '#eab308' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── Feature Importance Tab ─── */}
      {tab === 'features' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Band Power Contribution">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={bandPower}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="band" tick={{ fontSize: 11 }} />
                <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => [fmtPct(v), 'Importance']} />
                <Bar dataKey="importance" radius={[4, 4, 0, 0]}>
                  {bandPower.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Feature Importance Radar">
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={bandPower}>
                <PolarGrid />
                <PolarAngleAxis dataKey="band" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis tick={{ fontSize: 10 }} />
                <Radar name="Importance" dataKey="importance" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          <Card span={2} title="Feature Categories">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {['Statistical', 'Band Power', 'Hjorth'].map((cat, ci) => {
                const catFeatures = featureSet.filter(f => f.category === cat.toLowerCase().replace(' ', '_') || f.category === cat.toLowerCase())
                return (
                  <div key={ci} style={{
                    padding: 14, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0'
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 14, color: COLORS[ci], marginBottom: 8 }}>{cat}</div>
                    <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.8 }}>
                      {catFeatures.length > 0
                        ? catFeatures.map(f => f.name).join(', ')
                        : `${cat} features`}
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>
        </div>
      )}

      {/* ─── Spatial Patterns Tab ─── */}
      {tab === 'spatial' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {spatialPatterns.map((fold, fi) => (
            <Card key={fi} title={`Fold ${fi + 1}: Held-out ${fold.held_out_subject} — Top Electrode Contributions`}>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={fold.electrodes || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="electrode" tick={{ fontSize: 10 }} />
                  <YAxis tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip formatter={(v) => [fmtPct(v), 'Contribution']} />
                  <Bar dataKey="contribution" radius={[4, 4, 0, 0]}>
                    {(fold.electrodes || []).map((d, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          ))}
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {interpretation && interpretation.summary && (
            <Card title="Clinical Interpretation">
              <div style={{ padding: 14, borderRadius: 8, background: '#f0fdf4', border: '1px solid #bbf7d0', fontSize: 13, color: '#166534', lineHeight: 1.6 }}>
                {interpretation.summary}
              </div>
              {interpretation.clinical_meaning && (
                <div style={{ marginTop: 12, fontSize: 12, color: '#475569', lineHeight: 1.6 }}>
                  {interpretation.clinical_meaning}
                </div>
              )}
            </Card>
          )}

          <Card title="Key Terms">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
              {terms.map((t, i) => (
                <div key={i} style={{
                  padding: 14, borderRadius: 8, background: '#f8fafc', border: '1px solid #e2e8f0'
                }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 6 }}>
                    {t.term}
                  </div>
                  <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>
                    {t.definition}
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="References">
            <div style={{ display: 'grid', gap: 8 }}>
              {references.map((r, i) => (
                <div key={i} style={{
                  padding: 10, borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0',
                  fontSize: 12, color: '#475569', lineHeight: 1.5
                }}>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>{r.title || r.citation}</span>
                  {r.year && <span style={{ color: '#94a3b8', marginLeft: 8 }}>({r.year})</span>}
                  {r.relevance && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4, fontStyle: 'italic' }}>{r.relevance}</div>}
                </div>
              ))}
            </div>
          </Card>

          <Card title="Caveat">
            <div style={{ padding: 12, borderRadius: 8, background: '#fef3c7', border: '1px solid #fde68a', fontSize: 12, color: '#92400e', lineHeight: 1.6 }}>
              Bounded subset: 3 subjects (chb02, chb03, chb04), 1 seizure EDF each, capped windows.
              This is an honest cross-patient signal from real CHB-MIT data, not the full-dataset benchmark.
              Generalization to the full 23-patient CHB-MIT dataset is pending.
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
