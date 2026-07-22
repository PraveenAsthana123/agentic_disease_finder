import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, AreaChart, Area
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6','#22c55e','#f97316','#8b5cf6','#ef4444','#eab308','#06b6d4','#ec4899']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}
function fmtPct(v) { return v == null ? '--' : (v * 100).toFixed(1) + '%' }

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
  const colorMap = { ready: '#22c55e', complete: '#22c55e', computed: '#22c55e', 'in-progress': '#eab308', pending: '#94a3b8', partial: '#eab308', failed: '#ef4444', high: '#22c55e', medium: '#eab308', low: '#ef4444', yes: '#22c55e', no: '#ef4444', significant: '#22c55e', 'not-significant': '#94a3b8', ipsilateral: '#3b82f6', contralateral: '#8b5cf6' }
  const color = colorMap[status] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{status}</span>
  )
}

export default function PACDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [o, b, d] = await Promise.all([
          axios.get(`${API_URL}/api/pac/overview`),
          axios.get(`${API_URL}/api/pac/breakdown`),
          axios.get(`${API_URL}/api/pac/definitions`)
        ])
        setOverview(o.data)
        setBreakdown(b.data)
        setDefinitions(d.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading PAC data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'comodulogram', label: 'Comodulogram' },
    { id: 'patient', label: 'Patient Analysis' },
    { id: 'channel', label: 'Channel Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const tabBtn = (id, label) => (
    <button key={id} onClick={() => setTab(id)} style={{
      padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
      background: tab === id ? '#3b82f6' : '#f1f5f9', color: tab === id ? '#fff' : '#64748b'
    }}>{label}</button>
  )

  /* --- Overview data --- */
  const frequencyBandPairs = overview?.frequency_band_pairs || []
  const electrodePairRankings = overview?.electrode_pair_rankings || []
  const pacByCondition = overview?.pac_by_condition || []
  const pipelineStatus = overview?.pipeline_status || {}

  /* --- Comodulogram data --- */
  const comoduloMatrix = breakdown?.comodulogram_matrix || {}
  const temporalPacTrends = breakdown?.temporal_pac_trends || []

  /* --- Patient Analysis data --- */
  const perPatientPac = breakdown?.per_patient_pac || []
  const aedResponseCorrelation = breakdown?.aed_response_correlation || []

  /* --- Channel Detail data --- */
  const channelPairDetail = breakdown?.channel_pair_detail || []

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Phase-Amplitude Coupling Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Cross-frequency coupling analysis — phase-amplitude modulation index, comodulogram, and seizure zone correlation
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => tabBtn(t.id, t.label))}
      </div>

      {/* ======================== OVERVIEW TAB ======================== */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          {/* KPI Row */}
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              <KPI label="Total Recordings" value={fmt(overview.total_recordings)} />
              <KPI label="PAC Analyzed" value={fmt(overview.pac_analyzed)} color="#22c55e" />
              <KPI label="Mean MI" value={fmt(overview.mean_mi)} color="#3b82f6" sub="Modulation Index" />
              <KPI label="Max MI Pair" value={overview.max_mi_pair || '--'} color="#8b5cf6" sub="Best coupled pair" />
              <KPI label="Seizure Zone Corr." value={fmtPct(overview.seizure_zone_correlation)} color="#f97316" sub="Overlap rate" />
            </div>
          </Card>

          {/* Frequency Band Pairs Bar Chart */}
          <Card title="Mean MI by Frequency Band Pair" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={frequencyBandPairs}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="pair" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="mean_mi" name="Mean MI" radius={[4, 4, 0, 0]}>
                  {frequencyBandPairs.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pipeline Status Cards */}
          <Card title="Pipeline Status" span={1}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 10 }}>
              {Object.entries(pipelineStatus.stages || pipelineStatus).map(([stage, status], i) => (
                <div key={i} style={{ padding: 10, background: '#f8fafc', borderRadius: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>{stage.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                  <StatusBadge status={typeof status === 'string' ? status : status?.status || 'pending'} />
                </div>
              ))}
            </div>
          </Card>

          {/* Top 10 Electrode Pair Rankings */}
          <Card title="Top Electrode Pair Rankings (MI)" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={electrodePairRankings.slice(0, 10)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="pair" type="category" tick={{ fontSize: 10 }} width={80} />
                <Tooltip />
                <Bar dataKey="mi" name="MI" radius={[0, 4, 4, 0]}>
                  {electrodePairRankings.slice(0, 10).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* PAC by Condition Bar Chart */}
          <Card title="PAC by Condition (Ictal / Interictal / Postictal)" span={1}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={pacByCondition}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="condition" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="mean_mi" name="Mean MI" radius={[4, 4, 0, 0]}>
                  {pacByCondition.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== COMODULOGRAM TAB ======================== */}
      {tab === 'comodulogram' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Comodulogram Matrix Heatmap */}
          <Card title="Comodulogram Matrix (MI Values)" span={2}>
            {(() => {
              const phaseBands = Object.keys(comoduloMatrix)
              if (phaseBands.length === 0) return <div style={{ color: '#94a3b8', fontSize: 13 }}>No comodulogram data available.</div>
              const ampBands = Object.keys(comoduloMatrix[phaseBands[0]] || {})
              // Compute min/max for color scaling
              let minMI = Infinity, maxMI = -Infinity
              phaseBands.forEach(pb => ampBands.forEach(ab => {
                const val = comoduloMatrix[pb]?.[ab]
                if (val != null) { if (val < minMI) minMI = val; if (val > maxMI) maxMI = val }
              }))
              const getColor = (val) => {
                if (val == null) return '#f1f5f9'
                const t = maxMI === minMI ? 0.5 : (val - minMI) / (maxMI - minMI)
                // Blue (low) → Purple → Red (high)
                const r = Math.round(59 + t * (239 - 59))
                const g = Math.round(130 - t * 130)
                const b = Math.round(246 - t * (246 - 68))
                return `rgb(${r},${g},${b})`
              }
              return (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ borderCollapse: 'collapse', fontSize: 11, minWidth: 500 }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '6px 10px', background: '#f8fafc', color: '#64748b', textAlign: 'left', borderBottom: '1px solid #e2e8f0', whiteSpace: 'nowrap' }}>Phase \ Amp</th>
                        {ampBands.map(ab => (
                          <th key={ab} style={{ padding: '6px 10px', background: '#f8fafc', color: '#64748b', textAlign: 'center', borderBottom: '1px solid #e2e8f0', whiteSpace: 'nowrap' }}>{ab}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {phaseBands.map(pb => (
                        <tr key={pb}>
                          <td style={{ padding: '6px 10px', fontWeight: 600, color: '#334155', background: '#f8fafc', whiteSpace: 'nowrap', borderRight: '1px solid #e2e8f0' }}>{pb}</td>
                          {ampBands.map(ab => {
                            const val = comoduloMatrix[pb]?.[ab]
                            const bg = getColor(val)
                            return (
                              <td key={ab} title={`Phase: ${pb} | Amp: ${ab} | MI: ${val != null ? val.toFixed(4) : '--'}`} style={{
                                padding: '8px 14px', textAlign: 'center', background: bg,
                                color: val != null && (val - minMI) / (maxMI - minMI || 1) > 0.5 ? '#fff' : '#1e293b',
                                fontWeight: 600, cursor: 'default', border: '1px solid rgba(255,255,255,0.3)'
                              }}>
                                {val != null ? val.toFixed(3) : '--'}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>Color scale: blue (low MI) → red (high MI). Hover cells for exact values.</div>
                </div>
              )
            })()}
          </Card>

          {/* Temporal PAC Trends — MI approaching seizure */}
          <Card title="Temporal PAC Trends — MI Approaching Seizure" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={temporalPacTrends}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time_label" tick={{ fontSize: 11 }} label={{ value: 'Time relative to seizure onset', position: 'insideBottom', offset: -4, fontSize: 11, fill: '#94a3b8' }} height={40} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: 'MI', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }} />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="mi" name="Modulation Index" stroke={COLORS[0]} fill={COLORS[0] + '33'} strokeWidth={2} dot={{ r: 3 }} />
                {temporalPacTrends[0]?.mi_theta_gamma != null && (
                  <Area type="monotone" dataKey="mi_theta_gamma" name="Theta-Gamma MI" stroke={COLORS[2]} fill={COLORS[2] + '22'} strokeWidth={2} dot={{ r: 3 }} />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== PATIENT ANALYSIS TAB ======================== */}
      {tab === 'patient' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Per-Patient PAC Table */}
          <Card title="Per-Patient PAC Summary" span={2}>
            <div style={{ maxHeight: 360, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient ID</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Dominant Coupling Pair</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Mean MI</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seizure Zone Overlap</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Lateralization</th>
                  </tr>
                </thead>
                <tbody>
                  {perPatientPac.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{p.dominant_coupling_pair || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(p.mean_mi)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={p.seizure_zone_overlap != null ? (p.seizure_zone_overlap ? 'yes' : 'no') : 'pending'} />
                      </td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                        <StatusBadge status={p.lateralization || 'pending'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* AED Response Correlation — Pre vs Post MI */}
          <Card title="AED Response Correlation — Pre vs Post Treatment MI" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={aedResponseCorrelation}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="medication" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: 'MI', angle: -90, position: 'insideLeft', fontSize: 11, fill: '#94a3b8' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="pre_mi" name="Pre-Treatment MI" fill={COLORS[4]} radius={[4, 4, 0, 0]} />
                <Bar dataKey="post_mi" name="Post-Treatment MI" fill={COLORS[1]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* ======================== CHANNEL DETAIL TAB ======================== */}
      {tab === 'channel' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Channel Pair Detail — PAC Statistics">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 10px', color: '#64748b' }}>Pair</th>
                    <th style={{ textAlign: 'center', padding: '6px 10px', color: '#64748b' }}>Phase Band</th>
                    <th style={{ textAlign: 'center', padding: '6px 10px', color: '#64748b' }}>Amp Band</th>
                    <th style={{ textAlign: 'center', padding: '6px 10px', color: '#64748b' }}>MI</th>
                    <th style={{ textAlign: 'center', padding: '6px 10px', color: '#64748b' }}>p-Value</th>
                    <th style={{ textAlign: 'center', padding: '6px 10px', color: '#64748b' }}>Significant</th>
                  </tr>
                </thead>
                <tbody>
                  {channelPairDetail.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600, fontFamily: 'monospace', fontSize: 11 }}>{c.pair}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{c.phase_band || '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{c.amp_band || '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(c.mi)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center', fontFamily: 'monospace', fontSize: 11 }}>{c.p_value != null ? c.p_value.toFixed(4) : '--'}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>
                        <StatusBadge status={c.significant ? 'significant' : 'not-significant'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ======================== DEFINITIONS TAB ======================== */}
      {tab === 'definitions' && definitions && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {(definitions.definitions || []).map((d, i) => (
            <Card key={i} span={1}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#1e293b' }}>{d.term}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
                {d.clinical_relevance && (
                  <div style={{ fontSize: 11, color: '#64748b', background: '#f8fafc', borderRadius: 6, padding: '6px 10px', borderLeft: '3px solid #3b82f6', marginTop: 4 }}>
                    <strong style={{ color: '#334155' }}>Clinical Relevance:</strong> {d.clinical_relevance}
                  </div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
