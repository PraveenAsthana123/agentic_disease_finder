import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const CLASS_COLORS = {
  'PNES likely': '#ef4444',
  'Mixed / Comorbid': '#eab308',
  'Epileptic likely': '#3b82f6',
  'Epileptic confirmed': '#16a34a'
}
const VEEG_COLORS = { urgent: '#ef4444', high: '#f59e0b', routine: '#16a34a' }
const CERTAINTY_COLORS = { documented: '#16a34a', clinically_established: '#3b82f6', probable: '#eab308', possible: '#94a3b8' }
const PIE_COLORS = ['#ef4444', '#eab308', '#3b82f6', '#16a34a', '#8b5cf6', '#f59e0b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

function ClassificationBadge({ classification }) {
  const color = CLASS_COLORS[classification] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{classification || 'Unknown'}</span>
  )
}

function CertaintyBadge({ certainty }) {
  const color = CERTAINTY_COLORS[certainty] || '#94a3b8'
  const label = certainty ? certainty.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

function VeegBadge({ priority }) {
  const color = VEEG_COLORS[priority] || '#94a3b8'
  const label = priority ? priority.charAt(0).toUpperCase() + priority.slice(1) : 'Unknown'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{label}</span>
  )
}

export default function PNESDifferentialDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')
  const [expandedPatient, setExpandedPatient] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/api/pnes-differential/overview`),
          axios.get(`${API_URL}/api/pnes-differential/breakdown`),
          axios.get(`${API_URL}/api/pnes-differential/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading PNES Differential data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'semiology', label: 'Semiology Analysis' },
    { id: 'risk_factors', label: 'Risk Factors' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const kpis = overview?.kpis || {}
  const classificationDist = overview?.classification_distribution || []
  const certaintyDist = overview?.certainty_distribution || []
  const veegDist = overview?.veeg_priority_distribution || []
  const riskFactorFreq = overview?.risk_factor_frequency || []
  const durationHist = overview?.duration_histogram || []
  const probabilityHist = overview?.probability_histogram || []
  const pnesSignsRef = overview?.pnes_signs_reference || []
  const epilepsySignsRef = overview?.epilepsy_signs_reference || []

  const patients = breakdown?.patients || []
  const diagnosticLevels = breakdown?.diagnostic_levels || {}

  const thStyle = { textAlign: 'left', padding: '8px 10px', fontSize: 12, color: '#64748b', borderBottom: '2px solid #e2e8f0', fontWeight: 600 }
  const tdStyle = { padding: '7px 10px', fontSize: 13, borderBottom: '1px solid #f1f5f9' }

  return (
    <div style={{ padding: '24px 32px', background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, margin: '0 0 6px', color: '#0f172a' }}>
        PNES vs Epileptic Seizure Differential
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, margin: '0 0 20px' }}>
        Psychogenic non-epileptic seizure differential analysis: semiology scoring, risk factor profiling, diagnostic certainty classification, and vEEG prioritization
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
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
          {/* 8 KPI cards in 4x2 grid */}
          <Card span={4}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="PNES Likely" value={fmt(kpis.pnes_likely)} color="#ef4444" />
              <KPI label="Mixed / Comorbid" value={fmt(kpis.mixed_comorbid)} color="#eab308" />
              <KPI label="Epileptic Likely" value={fmt(kpis.epileptic_likely)} color="#3b82f6" />
              <KPI label="Urgent vEEG Needed" value={fmt(kpis.urgent_veeg_needed)} color="#ef4444" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Avg PNES Probability" value={`${fmt((kpis.avg_pnes_probability || 0) * 100)}%`} sub="Across all patients" />
              <KPI label="Psychiatric Comorbidity" value={fmt(kpis.psychiatric_comorbidity)} sub="Patients with comorbidity" />
              <KPI label="Documented Certainty" value={fmt(kpis.documented_certainty)} sub="Gold-standard vEEG confirmed" />
              <KPI label="Possible Certainty" value={fmt(kpis.possible_certainty)} sub="Requires further workup" />
            </div>
          </Card>

          {/* Classification Distribution Pie */}
          <Card title="Classification Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={classificationDist} dataKey="count" nameKey="label" cx="50%" cy="50%"
                     outerRadius={85} label={({ label, count }) => `${label}: ${count}`}>
                  {classificationDist.map((d, i) => <Cell key={i} fill={CLASS_COLORS[d.label] || PIE_COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Certainty Distribution Bar */}
          <Card title="Diagnostic Certainty Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={certaintyDist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {certaintyDist.map((d, i) => (
                    <Cell key={i} fill={CERTAINTY_COLORS[d.label] || PIE_COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* vEEG Priority Pie */}
          <Card title="vEEG Priority Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={veegDist} dataKey="count" nameKey="label" cx="50%" cy="50%"
                     outerRadius={85} label={({ label, count }) => `${label}: ${count}`}>
                  {veegDist.map((d, i) => <Cell key={i} fill={VEEG_COLORS[d.label] || PIE_COLORS[i]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* PNES Probability Histogram */}
          <Card title="PNES Probability Distribution" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={probabilityHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} label={{ value: 'Probability', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              PNES probability score distribution across all patients
            </div>
          </Card>

          {/* Event Duration Histogram */}
          <Card title="Average Event Duration Distribution" span={4}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={durationHist}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} label={{ value: 'Seconds', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4, textAlign: 'center' }}>
              PNES events tend to be longer (&gt;2 min); epileptic seizures typically &lt;5 min
            </div>
          </Card>
        </div>
      )}

      {/* ─── Semiology Analysis Tab ─── */}
      {tab === 'semiology' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* PNES-Favoring Signs */}
          <Card title="PNES-Favoring Semiology Signs">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Sign</th>
                  <th style={thStyle}>Weight</th>
                  <th style={thStyle}>Specificity</th>
                </tr>
              </thead>
              <tbody>
                {pnesSignsRef.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.sign}</td>
                    <td style={tdStyle}>
                      <span style={{ fontWeight: 600, color: '#ef4444' }}>{fmt(s.weight)}</span>
                    </td>
                    <td style={tdStyle}>{s.specificity || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Epilepsy-Favoring Signs */}
          <Card title="Epilepsy-Favoring Semiology Signs">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Sign</th>
                  <th style={thStyle}>Weight</th>
                  <th style={thStyle}>Specificity</th>
                </tr>
              </thead>
              <tbody>
                {epilepsySignsRef.map((s, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{s.sign}</td>
                    <td style={tdStyle}>
                      <span style={{ fontWeight: 600, color: '#3b82f6' }}>{fmt(s.weight)}</span>
                    </td>
                    <td style={tdStyle}>{s.specificity || '--'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {/* Sign frequency across patients — PNES signs */}
          <Card title="PNES Sign Frequency Across Patients" span={2}>
            {(() => {
              const signCounts = {}
              patients.forEach(p => {
                (p.pnes_signs || []).forEach(sign => {
                  signCounts[sign] = (signCounts[sign] || 0) + 1
                })
              })
              const signData = Object.entries(signCounts)
                .map(([sign, count]) => ({ sign, count }))
                .sort((a, b) => b.count - a.count)
              return (
                <ResponsiveContainer width="100%" height={Math.max(220, signData.length * 28)}>
                  <BarChart data={signData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="sign" width={220} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )
            })()}
          </Card>

          {/* Sign frequency across patients — Epilepsy signs */}
          <Card title="Epilepsy Sign Frequency Across Patients" span={2}>
            {(() => {
              const signCounts = {}
              patients.forEach(p => {
                (p.epilepsy_signs || []).forEach(sign => {
                  signCounts[sign] = (signCounts[sign] || 0) + 1
                })
              })
              const signData = Object.entries(signCounts)
                .map(([sign, count]) => ({ sign, count }))
                .sort((a, b) => b.count - a.count)
              return (
                <ResponsiveContainer width="100%" height={Math.max(220, signData.length * 28)}>
                  <BarChart data={signData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="sign" width={220} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )
            })()}
          </Card>
        </div>
      )}

      {/* ─── Risk Factors Tab ─── */}
      {tab === 'risk_factors' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Risk Factor Frequency */}
          <Card title="Risk Factor Frequency" span={2}>
            <ResponsiveContainer width="100%" height={Math.max(280, riskFactorFreq.length * 30)}>
              <BarChart data={riskFactorFreq} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="factor" width={250} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(val, name, props) => [`${val} (${fmt(props.payload.pct)}%)`, 'Count']} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Risk Factor Table */}
          <Card title="Risk Factor Details" span={1}>
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Risk Factor</th>
                    <th style={thStyle}>Count</th>
                    <th style={thStyle}>%</th>
                  </tr>
                </thead>
                <tbody>
                  {riskFactorFreq.map((rf, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{rf.factor}</td>
                      <td style={tdStyle}>{fmt(rf.count)}</td>
                      <td style={tdStyle}>
                        <span style={{ fontWeight: 600, color: (rf.pct || 0) > 50 ? '#ef4444' : (rf.pct || 0) > 25 ? '#eab308' : '#16a34a' }}>
                          {fmt(rf.pct)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Psychiatric Comorbidity Breakdown */}
          <Card title="Psychiatric Score Distribution">
            {(() => {
              const phq9Data = []
              const gad7Data = []
              patients.forEach(p => {
                if (p.phq9_score != null) phq9Data.push(p.phq9_score)
                if (p.gad7_score != null) gad7Data.push(p.gad7_score)
              })
              const phq9Bins = [
                { label: 'Minimal (0-4)', count: phq9Data.filter(v => v <= 4).length },
                { label: 'Mild (5-9)', count: phq9Data.filter(v => v >= 5 && v <= 9).length },
                { label: 'Moderate (10-14)', count: phq9Data.filter(v => v >= 10 && v <= 14).length },
                { label: 'Mod-Severe (15-19)', count: phq9Data.filter(v => v >= 15 && v <= 19).length },
                { label: 'Severe (20-27)', count: phq9Data.filter(v => v >= 20).length },
              ]
              const gad7Bins = [
                { label: 'Minimal (0-4)', count: gad7Data.filter(v => v <= 4).length },
                { label: 'Mild (5-9)', count: gad7Data.filter(v => v >= 5 && v <= 9).length },
                { label: 'Moderate (10-14)', count: gad7Data.filter(v => v >= 10 && v <= 14).length },
                { label: 'Severe (15-21)', count: gad7Data.filter(v => v >= 15).length },
              ]
              return (
                <div>
                  <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>PHQ-9 Depression Scores</h4>
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={phq9Bins}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  <h4 style={{ fontSize: 13, color: '#334155', margin: '16px 0 8px' }}>GAD-7 Anxiety Scores</h4>
                  <ResponsiveContainer width="100%" height={160}>
                    <BarChart data={gad7Bins}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" tick={{ fontSize: 10 }} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )
            })()}
          </Card>
        </div>
      )}

      {/* ─── Patient Detail Tab ─── */}
      {tab === 'patients' && (
        <div style={{ display: 'grid', gap: 12 }}>
          {[...patients].sort((a, b) => {
            const order = { 'PNES likely': 0, 'Mixed / Comorbid': 1, 'Epileptic likely': 2, 'Epileptic confirmed': 3 }
            return (order[a.classification] ?? 4) - (order[b.classification] ?? 4)
          }).map((p, i) => {
            const isExpanded = expandedPatient === p.patient_id
            const pnesPct = ((p.pnes_probability || 0) * 100)
            return (
              <Card key={i}>
                <div
                  style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer' }}
                  onClick={() => setExpandedPatient(isExpanded ? null : p.patient_id)}
                >
                  <span style={{ fontSize: 18, transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s', display: 'inline-block' }}>
                    &#9654;
                  </span>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{p.name || p.patient_id}</span>
                    <span style={{ color: '#94a3b8', fontSize: 12, marginLeft: 10 }}>
                      Age {p.age} | {p.gender} | {p.disease}
                    </span>
                  </div>
                  <ClassificationBadge classification={p.classification} />
                  {/* PNES probability mini-bar */}
                  <div style={{ width: 100, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ flex: 1, background: '#e2e8f0', borderRadius: 4, height: 8 }}>
                      <div style={{ width: `${pnesPct}%`, background: pnesPct > 60 ? '#ef4444' : pnesPct > 40 ? '#eab308' : '#3b82f6', borderRadius: 4, height: 8 }} />
                    </div>
                    <span style={{ fontSize: 11, color: '#64748b', minWidth: 32 }}>{fmt(pnesPct)}%</span>
                  </div>
                  <VeegBadge priority={p.veeg_priority} />
                  <CertaintyBadge certainty={p.diagnostic_certainty} />
                </div>

                {isExpanded && (
                  <div style={{ marginTop: 16 }}>
                    {/* Demographics row */}
                    <div style={{ display: 'flex', gap: 20, marginBottom: 16, padding: '10px 14px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: '#475569', flexWrap: 'wrap' }}>
                      <span><strong>Patient ID:</strong> {p.patient_id}</span>
                      <span><strong>Age:</strong> {p.age}</span>
                      <span><strong>Gender:</strong> {p.gender}</span>
                      <span><strong>Disease:</strong> {p.disease}</span>
                      <span><strong>Seizure Count:</strong> {fmt(p.seizure_count)}</span>
                      <span><strong>Events Witnessed:</strong> {fmt(p.events_witnessed)}</span>
                      <span><strong>Events with Aura:</strong> {fmt(p.events_with_aura)}</span>
                      <span><strong>Avg Duration:</strong> {fmt(p.avg_event_duration_sec)} sec</span>
                    </div>

                    {/* Scores row */}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                      <div style={{ padding: '10px 14px', background: '#fef2f2', borderRadius: 8, textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>PNES Semiology</div>
                        <div style={{ fontSize: 20, fontWeight: 700, color: '#ef4444' }}>{fmt(p.pnes_semiology_score)}</div>
                      </div>
                      <div style={{ padding: '10px 14px', background: '#eff6ff', borderRadius: 8, textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Epilepsy Semiology</div>
                        <div style={{ fontSize: 20, fontWeight: 700, color: '#3b82f6' }}>{fmt(p.epilepsy_semiology_score)}</div>
                      </div>
                      <div style={{ padding: '10px 14px', background: '#fef2f2', borderRadius: 8, textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>PNES Probability</div>
                        <div style={{ fontSize: 20, fontWeight: 700, color: '#ef4444' }}>{fmt(pnesPct)}%</div>
                      </div>
                      <div style={{ padding: '10px 14px', background: '#eff6ff', borderRadius: 8, textAlign: 'center' }}>
                        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Epilepsy Probability</div>
                        <div style={{ fontSize: 20, fontWeight: 700, color: '#3b82f6' }}>{fmt((p.epilepsy_probability || 0) * 100)}%</div>
                      </div>
                    </div>

                    {/* PNES Signs Present */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>PNES Signs Present</h4>
                    {(p.pnes_signs || []).length > 0 ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
                        {p.pnes_signs.map((sign, j) => (
                          <span key={j} style={{
                            display: 'inline-block', padding: '3px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                            background: '#fef2f2', color: '#ef4444'
                          }}>{sign}</span>
                        ))}
                      </div>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>No PNES signs identified.</p>
                    )}

                    {/* Epilepsy Signs Present */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Epilepsy Signs Present</h4>
                    {(p.epilepsy_signs || []).length > 0 ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
                        {p.epilepsy_signs.map((sign, j) => (
                          <span key={j} style={{
                            display: 'inline-block', padding: '3px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                            background: '#eff6ff', color: '#3b82f6'
                          }}>{sign}</span>
                        ))}
                      </div>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>No epilepsy signs identified.</p>
                    )}

                    {/* Risk Factors */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Risk Factors ({(p.risk_factors || []).length})</h4>
                    {(p.risk_factors || []).length > 0 ? (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
                        {p.risk_factors.map((rf, j) => (
                          <span key={j} style={{
                            display: 'inline-block', padding: '3px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                            background: '#faf5ff', color: '#8b5cf6'
                          }}>{rf}</span>
                        ))}
                      </div>
                    ) : (
                      <p style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>No risk factors identified.</p>
                    )}

                    {/* Psychiatric Scores */}
                    <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 8px' }}>Psychiatric Assessment Scores</h4>
                    <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 12 }}>
                      <thead>
                        <tr>
                          <th style={thStyle}>Instrument</th>
                          <th style={thStyle}>Score</th>
                          <th style={thStyle}>Severity</th>
                          <th style={thStyle}>Reference</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>PHQ-9 (Depression)</td>
                          <td style={tdStyle}>
                            <span style={{ fontWeight: 600, color: (p.phq9_score || 0) >= 15 ? '#ef4444' : (p.phq9_score || 0) >= 10 ? '#eab308' : '#16a34a' }}>
                              {fmt(p.phq9_score)}
                            </span>
                          </td>
                          <td style={tdStyle}>
                            {p.phq9_score == null ? '--'
                              : p.phq9_score <= 4 ? 'Minimal'
                              : p.phq9_score <= 9 ? 'Mild'
                              : p.phq9_score <= 14 ? 'Moderate'
                              : p.phq9_score <= 19 ? 'Moderately Severe'
                              : 'Severe'}
                          </td>
                          <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>0-27 scale</td>
                        </tr>
                        <tr style={{ background: '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>GAD-7 (Anxiety)</td>
                          <td style={tdStyle}>
                            <span style={{ fontWeight: 600, color: (p.gad7_score || 0) >= 15 ? '#ef4444' : (p.gad7_score || 0) >= 10 ? '#eab308' : '#16a34a' }}>
                              {fmt(p.gad7_score)}
                            </span>
                          </td>
                          <td style={tdStyle}>
                            {p.gad7_score == null ? '--'
                              : p.gad7_score <= 4 ? 'Minimal'
                              : p.gad7_score <= 9 ? 'Mild'
                              : p.gad7_score <= 14 ? 'Moderate'
                              : 'Severe'}
                          </td>
                          <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>0-21 scale</td>
                        </tr>
                        <tr style={{ background: '#fff' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>C-SSRS (Suicidality)</td>
                          <td style={tdStyle}>
                            <span style={{ fontWeight: 600, color: (p.cssrs_score || 0) >= 3 ? '#ef4444' : (p.cssrs_score || 0) >= 1 ? '#eab308' : '#16a34a' }}>
                              {fmt(p.cssrs_score)}
                            </span>
                          </td>
                          <td style={tdStyle}>
                            {p.cssrs_score == null ? '--'
                              : p.cssrs_score === 0 ? 'No risk'
                              : p.cssrs_score <= 2 ? 'Low'
                              : p.cssrs_score <= 4 ? 'Moderate'
                              : 'High'}
                          </td>
                          <td style={{ ...tdStyle, fontSize: 12, color: '#64748b' }}>0-5 scale</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            )
          })}
        </div>
      )}

      {/* ─── Definitions Tab ─── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gap: 16 }}>
          {/* Concepts */}
          {defs.concepts && defs.concepts.length > 0 && (
            <Card title="Key Concepts">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: '25%' }}>Concept</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.concepts.map((c, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>{c.name}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{c.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Diagnostic Levels */}
          {Object.keys(diagnosticLevels).length > 0 && (
            <Card title="Diagnostic Certainty Levels">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: '20%' }}>Level</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(diagnosticLevels).map(([level, desc], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>
                        <CertaintyBadge certainty={level} />
                      </td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Semiology Table */}
          {defs.semiology_table && (
            <Card title="Semiology Reference Table">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                <div>
                  <h4 style={{ fontSize: 13, color: '#ef4444', margin: '0 0 8px' }}>PNES-Favoring Signs</h4>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={thStyle}>Sign</th>
                        <th style={thStyle}>Weight</th>
                        <th style={thStyle}>Specificity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(defs.semiology_table.pnes_favoring || []).map((s, i) => (
                        <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>{s.sign}</td>
                          <td style={tdStyle}>{fmt(s.weight)}</td>
                          <td style={tdStyle}>{s.specificity || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div>
                  <h4 style={{ fontSize: 13, color: '#3b82f6', margin: '0 0 8px' }}>Epilepsy-Favoring Signs</h4>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={thStyle}>Sign</th>
                        <th style={thStyle}>Weight</th>
                        <th style={thStyle}>Specificity</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(defs.semiology_table.epilepsy_favoring || []).map((s, i) => (
                        <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                          <td style={{ ...tdStyle, fontWeight: 600 }}>{s.sign}</td>
                          <td style={tdStyle}>{fmt(s.weight)}</td>
                          <td style={tdStyle}>{s.specificity || '--'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>
          )}

          {/* Management Pathway */}
          {defs.management && defs.management.length > 0 && (
            <Card title="Management Pathway">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: '20%' }}>Phase</th>
                    <th style={thStyle}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.management.map((m, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>{m.phase}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{m.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Quality Metrics */}
          {defs.quality_metrics && defs.quality_metrics.length > 0 && (
            <Card title="Quality Metrics">
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={{ ...thStyle, width: '20%' }}>Metric</th>
                    <th style={{ ...thStyle, width: '20%' }}>Target</th>
                    <th style={thStyle}>Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {defs.quality_metrics.map((qm, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, verticalAlign: 'top' }}>{qm.metric}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#16a34a', fontWeight: 600 }}>{qm.target}</td>
                      <td style={{ ...tdStyle, fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{qm.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}

          {/* Fallback */}
          {!defs.concepts && !defs.semiology_table && !defs.management && !defs.quality_metrics && Object.keys(diagnosticLevels).length === 0 && (
            <Card title="Definitions">
              <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
            </Card>
          )}
        </div>
      )}

      {!defs && tab === 'definitions' && (
        <Card title="Definitions">
          <p style={{ fontSize: 13, color: '#94a3b8' }}>No definition data available from the API.</p>
        </Card>
      )}

      <div style={{ marginTop: 24, padding: 16, background: '#f1f5f9', borderRadius: 8, fontSize: 12, color: '#64748b' }}>
        PNES vs Epileptic Seizure Differential Dashboard — Real clinical.db data ({patients.length} patients) |
        ILAE diagnostic levels, LaFrance &amp; Bhatt semiology criteria
      </div>
    </div>
  )
}
