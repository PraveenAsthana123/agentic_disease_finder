import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value ?? '--'}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

const fmt = v => (v != null ? v : '--')
const pct = v => (v != null ? `${v}%` : '--')

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']
const CLS_COLORS = { pnes_likely: '#ef4444', epilepsy_likely: '#3b82f6', mixed: '#f59e0b', indeterminate: '#94a3b8' }
const STATUS_COLORS = { pending: '#f59e0b', reviewed: '#3b82f6', confirmed_pnes: '#ef4444', confirmed_epilepsy: '#10b981' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'by_patient', label: 'By Patient' },
  { id: 'recent', label: 'Recent Screenings' },
  { id: 'features', label: 'Feature Analysis' },
  { id: 'definitions', label: 'Definitions' },
]

export default function PnesScreeningDashboard() {
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
      axios.get(`${API_URL}/api/pnes-screening/overview`),
      axios.get(`${API_URL}/api/pnes-screening/breakdown`),
      axios.get(`${API_URL}/api/pnes-screening/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading PNES Screening Dashboard...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const ov = overview || {}
  const bd = breakdown || {}
  const defs = definitions || {}

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#0f172a' }}>PNES Screening Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Psychogenic Non-Epileptic Seizures — semiological scoring, EEG findings, differential classification
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#e2e8f0',
            color: tab === t.id ? '#fff' : '#334155', fontWeight: tab === t.id ? 600 : 400,
            fontSize: 13
          }}>{t.label}</button>
        ))}
      </div>

      {!ov.available && tab !== 'definitions' && (
        <Card><p style={{ color: '#94a3b8' }}>{ov.reason || 'No data available'}</p></Card>
      )}

      {/* ─── OVERVIEW TAB ─── */}
      {tab === 'overview' && ov.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          <Card>
            <KPI label="Total Screenings" value={fmt(ov.total_screenings)} sub={`${ov.total_patients} patients`} color="#3b82f6" />
          </Card>
          <Card>
            <KPI label="Avg PNES Probability" value={fmt(ov.avg_pnes_probability)} sub="0 = epilepsy, 1 = PNES" color="#ef4444" />
          </Card>
          <Card>
            <KPI label="Avg Confidence" value={fmt(ov.avg_confidence)} sub="Model certainty" color="#8b5cf6" />
          </Card>
          <Card>
            <KPI label="Video-EEG Recommended" value={pct(ov.video_eeg_recommendation_rate)} sub="Of all screenings" color="#f59e0b" />
          </Card>

          <Card>
            <KPI label="Psychiatry Referral Rate" value={pct(ov.psychiatry_referral_rate)} sub="Referred for psych eval" color="#ec4899" />
          </Card>
          <Card>
            <KPI label="Duration > 2 min" value={pct(ov.duration_gt2min_rate)} sub="Suggestive of PNES" color="#06b6d4" />
          </Card>
          <Card>
            <KPI label="Ictal EEG Normal" value={fmt(ov.eeg_ictal_normal_count)} sub="Events with normal ictal EEG" color="#10b981" />
          </Card>
          <Card>
            <KPI label="Screenings/Patient" value={ov.total_patients ? (ov.total_screenings / ov.total_patients).toFixed(1) : '--'} sub="Average" color="#334155" />
          </Card>

          {/* Classification Pie */}
          <Card title="Classification Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={(ov.classification_distribution || []).filter(d => d.count > 0)} dataKey="count" nameKey="classification" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name.replace(/_/g, ' ')} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.classification_distribution || []).filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={CLS_COLORS[d.classification] || COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Probability Distribution */}
          <Card title="PNES Probability Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.probability_distribution || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" name="Screenings">
                  {(ov.probability_distribution || []).map((_, i) => <Cell key={i} fill={[COLORS[1], COLORS[2], COLORS[3]][i]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Comorbidity Pie */}
          <Card title="Psychiatric Comorbidity" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={ov.comorbidity_distribution || []} dataKey="count" nameKey="comorbidity" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.comorbidity_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Referral Reason Distribution */}
          <Card title="Referral Reasons" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.referral_reason_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="reason" type="category" width={140} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Semiological Feature Averages */}
          <Card title="Semiological Feature Averages" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ov.semiological_averages || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="feature" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={50} />
                <YAxis domain={[0, 3]} />
                <Tooltip />
                <Bar dataKey="avg_score" fill="#3b82f6" name="Avg Score (0-3)" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Status Distribution */}
          <Card title="Review Status" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={(ov.status_distribution || []).filter(d => d.count > 0)} dataKey="count" nameKey="status" cx="50%" cy="50%" outerRadius={80} label={({ name, percent }) => `${name.replace(/_/g, ' ')} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.status_distribution || []).filter(d => d.count > 0).map((d, i) => (
                    <Cell key={i} fill={STATUS_COLORS[d.status] || COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Monthly Trend */}
          <Card title="Monthly Screening Volume" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={ov.monthly_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="screenings" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ─── BY PATIENT TAB ─── */}
      {tab === 'by_patient' && bd.available && (
        <Card title="Per-Patient PNES Screening Summary" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Patient</th>
                  <th style={{ padding: '8px 12px' }}>Screenings</th>
                  <th style={{ padding: '8px 12px' }}>Latest Classification</th>
                  <th style={{ padding: '8px 12px' }}>Dominant Classification</th>
                  <th style={{ padding: '8px 12px' }}>Avg PNES Prob</th>
                  <th style={{ padding: '8px 12px' }}>Avg Confidence</th>
                </tr>
              </thead>
              <tbody>
                {(bd.per_patient || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{p.screenings}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                        background: p.latest_classification === 'pnes_likely' ? '#fee2e2' : p.latest_classification === 'epilepsy_likely' ? '#dbeafe' : p.latest_classification === 'mixed' ? '#fef3c7' : '#f1f5f9',
                        color: p.latest_classification === 'pnes_likely' ? '#991b1b' : p.latest_classification === 'epilepsy_likely' ? '#1e40af' : p.latest_classification === 'mixed' ? '#92400e' : '#64748b',
                      }}>{p.latest_classification.replace(/_/g, ' ')}</span>
                    </td>
                    <td style={{ padding: '8px 12px' }}>{p.dominant_classification.replace(/_/g, ' ')}</td>
                    <td style={{ padding: '8px 12px', color: p.avg_pnes_probability >= 0.65 ? '#ef4444' : p.avg_pnes_probability >= 0.35 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>{p.avg_pnes_probability}</td>
                    <td style={{ padding: '8px 12px' }}>{p.avg_confidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── RECENT SCREENINGS TAB ─── */}
      {tab === 'recent' && bd.available && (
        <Card title="Recent PNES Screenings (last 20)" span={4}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '6px 8px' }}>Patient</th>
                  <th style={{ padding: '6px 8px' }}>Date</th>
                  <th style={{ padding: '6px 8px' }}>Reason</th>
                  <th style={{ padding: '6px 8px' }}>Classification</th>
                  <th style={{ padding: '6px 8px' }}>PNES Prob</th>
                  <th style={{ padding: '6px 8px' }}>Confidence</th>
                  <th style={{ padding: '6px 8px' }}>Trauma</th>
                  <th style={{ padding: '6px 8px' }}>EEG Normal</th>
                  <th style={{ padding: '6px 8px' }}>Comorbidity</th>
                  <th style={{ padding: '6px 8px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {(bd.recent_screenings || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 600 }}>{s.patient_id}</td>
                    <td style={{ padding: '6px 8px' }}>{s.date}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11 }}>{s.referral_reason}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                        background: s.classification === 'pnes_likely' ? '#fee2e2' : s.classification === 'epilepsy_likely' ? '#dbeafe' : s.classification === 'mixed' ? '#fef3c7' : '#f1f5f9',
                        color: s.classification === 'pnes_likely' ? '#991b1b' : s.classification === 'epilepsy_likely' ? '#1e40af' : s.classification === 'mixed' ? '#92400e' : '#64748b',
                      }}>{s.classification.replace(/_/g, ' ')}</span>
                    </td>
                    <td style={{ padding: '6px 8px', color: s.pnes_probability >= 0.65 ? '#ef4444' : s.pnes_probability >= 0.35 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>{s.pnes_probability ?? '--'}</td>
                    <td style={{ padding: '6px 8px' }}>{s.confidence ?? '--'}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'center' }}>{s.trauma_history ? 'Yes' : 'No'}</td>
                    <td style={{ padding: '6px 8px', textAlign: 'center' }}>{s.eeg_ictal_normal ? 'Yes' : 'No'}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11 }}>{s.psychiatric_comorbidity}</td>
                    <td style={{ padding: '6px 8px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
                        background: s.status === 'confirmed_pnes' ? '#fee2e2' : s.status === 'confirmed_epilepsy' ? '#dcfce7' : s.status === 'reviewed' ? '#dbeafe' : '#fef3c7',
                        color: s.status === 'confirmed_pnes' ? '#991b1b' : s.status === 'confirmed_epilepsy' ? '#166534' : s.status === 'reviewed' ? '#1e40af' : '#92400e',
                      }}>{s.status.replace(/_/g, ' ')}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ─── FEATURE ANALYSIS TAB ─── */}
      {tab === 'features' && bd.available && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Radar chart: semiological features by classification */}
          <Card title="Semiological Features by Classification" span={2}>
            <ResponsiveContainer width="100%" height={350}>
              <RadarChart data={bd.feature_comparison || []}>
                <PolarGrid />
                <PolarAngleAxis dataKey="feature" tick={{ fontSize: 11 }} />
                <PolarRadiusAxis domain={[0, 3]} tick={{ fontSize: 10 }} />
                <Radar name="PNES Likely" dataKey="pnes_likely" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Epilepsy Likely" dataKey="epilepsy_likely" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Mixed" dataKey="mixed" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Indeterminate" dataKey="indeterminate" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.15} strokeWidth={2} />
                <Legend />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </Card>

          {/* Feature comparison table */}
          <Card title="Feature Score Comparison" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'center' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Feature</th>
                    <th style={{ padding: '6px 8px', color: '#ef4444' }}>PNES</th>
                    <th style={{ padding: '6px 8px', color: '#3b82f6' }}>Epilepsy</th>
                    <th style={{ padding: '6px 8px', color: '#f59e0b' }}>Mixed</th>
                    <th style={{ padding: '6px 8px', color: '#94a3b8' }}>Indet.</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.feature_comparison || []).map((f, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', textAlign: 'left', fontWeight: 600 }}>{f.feature}</td>
                      <td style={{ padding: '6px 8px' }}>{f.pnes_likely}</td>
                      <td style={{ padding: '6px 8px' }}>{f.epilepsy_likely}</td>
                      <td style={{ padding: '6px 8px' }}>{f.mixed}</td>
                      <td style={{ padding: '6px 8px' }}>{f.indeterminate}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* EEG Patterns */}
          <Card title="EEG Finding Patterns by Classification" span={1}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                    <th style={{ padding: '6px 8px' }}>Classification</th>
                    <th style={{ padding: '6px 8px' }}>N</th>
                    <th style={{ padding: '6px 8px' }}>Ictal Normal %</th>
                    <th style={{ padding: '6px 8px' }}>Interictal Normal %</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.eeg_patterns || []).map((e, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, color: CLS_COLORS[e.classification] || '#334155' }}>{e.classification.replace(/_/g, ' ')}</td>
                      <td style={{ padding: '6px 8px' }}>{e.total}</td>
                      <td style={{ padding: '6px 8px' }}>{e.ictal_normal_pct}%</td>
                      <td style={{ padding: '6px 8px' }}>{e.interictal_normal_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ─── DEFINITIONS TAB ─── */}
      {tab === 'definitions' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="PNES Screening Glossary" span={2}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px', width: 200 }}>Term</th>
                  <th style={{ padding: '8px 12px' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map((g, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#334155' }}>{g.term}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Classification Tiers">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Classification</th>
                  <th style={{ padding: '8px 12px' }}>Description</th>
                  <th style={{ padding: '8px 12px' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {(defs.classification_tiers || []).map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: CLS_COLORS[t.classification] || '#334155' }}>{t.classification.replace(/_/g, ' ')}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{t.description}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{t.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Scoring Guide (0-3)">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                  <th style={{ padding: '8px 12px' }}>Score</th>
                  <th style={{ padding: '8px 12px' }}>Label</th>
                  <th style={{ padding: '8px 12px' }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {(defs.scoring_guide || []).map((s, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.score}</td>
                    <td style={{ padding: '8px 12px' }}>{s.label}</td>
                    <td style={{ padding: '8px 12px', color: '#475569' }}>{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  )
}
