import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, LineChart, Line, RadarChart, Radar, PolarGrid,
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

const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#f97316', '#84cc16', '#14b8a6', '#a855f7']
const GRADE_COLORS = { Good: '#10b981', Fair: '#f59e0b', Poor: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'channels', label: 'Channel Detail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function EEGAcquisitionDashboard() {
  const [tab, setTab] = useState('overview')
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [definitions, setDefinitions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      axios.get(`${API_URL}/api/eeg-acquisition/overview`),
      axios.get(`${API_URL}/api/eeg-acquisition/breakdown`),
      axios.get(`${API_URL}/api/eeg-acquisition/definitions`),
    ]).then(([ov, br, df]) => {
      setOverview(ov.data)
      setBreakdown(br.data)
      setDefinitions(df.data)
      setLoading(false)
    }).catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG acquisition data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#1e293b' }}>EEG Acquisition Quality</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
        Electrode impedance, signal quality, recording parameters — 10-20 system channel QC for EEG technicians
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderBottom: '2px solid #e2e8f0' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 20px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t.id ? '#fff' : 'transparent',
            color: tab === t.id ? '#2563eb' : '#64748b',
            borderBottom: tab === t.id ? '2px solid #2563eb' : '2px solid transparent',
            marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && overview && <OverviewTab data={overview} />}
      {tab === 'channels' && breakdown && <ChannelDetailTab data={breakdown} />}
      {tab === 'definitions' && definitions && <DefinitionsTab data={definitions} />}
    </div>
  )
}

function OverviewTab({ data }) {
  const recTypeDist = Object.entries(data.recording_type_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const impDist = Object.entries(data.channel_quality_summary?.impedance_grade_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const qualDist = Object.entries(data.channel_quality_summary?.quality_grade_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const montageDist = Object.entries(data.montage_distribution || {}).map(([k, v]) => ({ name: k, count: v }))
  const sampleRateDist = Object.entries(data.sampling_rate_distribution || {}).map(([k, v]) => ({ name: `${k} Hz`, count: v }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 16 }}>
      {/* KPI row */}
      <Card><KPI label="Total Studies" value={data.total_studies} color="#3b82f6" /></Card>
      <Card><KPI label="Total Patients" value={data.total_patients} color="#8b5cf6" /></Card>
      <Card><KPI label="Avg Duration (min)" value={data.duration_stats?.avg_min} color="#06b6d4" /></Card>
      <Card><KPI label="Good Impedance %" value={`${data.pct_good_impedance}%`} color={data.pct_good_impedance >= 50 ? '#10b981' : '#ef4444'} /></Card>
      <Card><KPI label="Good Quality %" value={`${data.pct_good_quality}%`} color={data.pct_good_quality >= 50 ? '#10b981' : '#f59e0b'} /></Card>
      <Card><KPI label="Avg SNR (dB)" value={data.avg_snr_db} color="#f97316" /></Card>

      {/* Recording type pie chart */}
      <Card title="Recording Type Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={recTypeDist} dataKey="count" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              labelLine={{ strokeWidth: 1 }} style={{ fontSize: 10 }}>
              {recTypeDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Impedance grade pie chart */}
      <Card title="Impedance Grade Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={impDist} dataKey="count" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              labelLine={{ strokeWidth: 1 }} style={{ fontSize: 10 }}>
              {impDist.map((entry, i) => <Cell key={i} fill={GRADE_COLORS[entry.name] || COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Quality grade pie chart */}
      <Card title="Signal Quality Grade Distribution" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie data={qualDist} dataKey="count" nameKey="name" cx="50%" cy="50%"
              outerRadius={100} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              labelLine={{ strokeWidth: 1 }} style={{ fontSize: 10 }}>
              {qualDist.map((entry, i) => <Cell key={i} fill={GRADE_COLORS[entry.name] || COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      {/* Montage distribution bar chart */}
      <Card title="Montage Distribution" span={3}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={montageDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" name="Studies" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Sampling rate distribution bar chart */}
      <Card title="Sampling Rate Distribution" span={3}>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={sampleRateDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" name="Studies" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      {/* Monthly trend line chart */}
      <Card title="Monthly Study Volume" span={6}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data.monthly_trend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} name="Studies" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function ChannelDetailTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* Per-channel stats table */}
      <Card title="Per-Channel Quality (10-20 System)" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Channel</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg Impedance (k&#937;)</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg SNR (dB)</th>
                <th style={{ padding: '6px 10px', textAlign: 'center' }}>Good</th>
                <th style={{ padding: '6px 10px', textAlign: 'center' }}>Fair</th>
                <th style={{ padding: '6px 10px', textAlign: 'center' }}>Poor</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Impedance Grade</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_channel_stats || []).map((ch, i) => {
                const total = (ch.good_count || 0) + (ch.fair_count || 0) + (ch.poor_count || 0)
                const goodPct = total > 0 ? ((ch.good_count / total) * 100).toFixed(0) : 0
                const fairPct = total > 0 ? ((ch.fair_count / total) * 100).toFixed(0) : 0
                const poorPct = total > 0 ? ((ch.poor_count / total) * 100).toFixed(0) : 0
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{ch.channel}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right',
                      color: ch.avg_impedance <= 5 ? '#10b981' : ch.avg_impedance <= 10 ? '#f59e0b' : '#ef4444',
                      fontWeight: 600
                    }}>{ch.avg_impedance}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', color: '#64748b' }}>{ch.avg_snr}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{ch.good_count}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#f59e0b', fontWeight: 600 }}>{ch.fair_count}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{ch.poor_count}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', width: 120 }}>
                        <div style={{ background: '#10b981', width: `${goodPct}%` }} />
                        <div style={{ background: '#f59e0b', width: `${fairPct}%` }} />
                        <div style={{ background: '#ef4444', width: `${poorPct}%` }} />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Per-patient summary */}
      <Card title="Per-Patient Recording Summary" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Type</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Duration (min)</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Rate (Hz)</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Montage</th>
                <th style={{ padding: '6px 10px', textAlign: 'center' }}>Good</th>
                <th style={{ padding: '6px 10px', textAlign: 'center' }}>Fair</th>
                <th style={{ padding: '6px 10px', textAlign: 'center' }}>Poor</th>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Impedance Profile</th>
              </tr>
            </thead>
            <tbody>
              {(data.per_patient_summary || []).map((p, i) => {
                const total = (p.good_impedance || 0) + (p.fair_impedance || 0) + (p.poor_impedance || 0)
                const goodPct = total > 0 ? ((p.good_impedance / total) * 100).toFixed(0) : 0
                const fairPct = total > 0 ? ((p.fair_impedance / total) * 100).toFixed(0) : 0
                const poorPct = total > 0 ? ((p.poor_impedance / total) * 100).toFixed(0) : 0
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{p.patient_id}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={{
                        background: '#eff6ff', color: '#2563eb',
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600
                      }}>{p.recording_type}</span>
                    </td>
                    <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.duration_min}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'right', color: '#64748b' }}>{p.sampling_rate}</td>
                    <td style={{ padding: '6px 10px', color: '#64748b' }}>{p.montage}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#10b981', fontWeight: 600 }}>{p.good_impedance}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#f59e0b', fontWeight: 600 }}>{p.fair_impedance}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center', color: '#ef4444', fontWeight: 600 }}>{p.poor_impedance}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', width: 100 }}>
                        <div style={{ background: '#10b981', width: `${goodPct}%` }} />
                        <div style={{ background: '#f59e0b', width: `${fairPct}%` }} />
                        <div style={{ background: '#ef4444', width: `${poorPct}%` }} />
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Poor quality channels */}
      {(data.poor_quality_channels || []).length > 0 && (
        <Card title="Poor Quality Channels (Flagged)" span={4}>
          <div style={{ background: '#fef2f2', borderRadius: 8, padding: 12 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Patient</th>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Channel</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>Impedance (k&#937;)</th>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Impedance Grade</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>SNR (dB)</th>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Quality Grade</th>
                </tr>
              </thead>
              <tbody>
                {data.poor_quality_channels.map((ch, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #fecaca' }}>
                    <td style={{ padding: '4px 8px', fontWeight: 500 }}>{ch.patient_id}</td>
                    <td style={{ padding: '4px 8px', fontWeight: 600 }}>{ch.channel}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', color: '#ef4444', fontWeight: 600 }}>{ch.impedance_kohm}</td>
                    <td style={{ padding: '4px 8px' }}>
                      <span style={{
                        background: (GRADE_COLORS[ch.impedance_grade] || '#64748b') + '20',
                        color: GRADE_COLORS[ch.impedance_grade] || '#64748b',
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600
                      }}>{ch.impedance_grade}</span>
                    </td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', color: '#64748b' }}>{ch.snr_db}</td>
                    <td style={{ padding: '4px 8px' }}>
                      <span style={{
                        background: (GRADE_COLORS[ch.quality_grade] || '#64748b') + '20',
                        color: GRADE_COLORS[ch.quality_grade] || '#64748b',
                        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600
                      }}>{ch.quality_grade}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Recording type detail */}
      <Card title="Recording Type Detail" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left' }}>Recording Type</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Count</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg Duration (min)</th>
                <th style={{ padding: '6px 10px', textAlign: 'right' }}>Avg Impedance (k&#937;)</th>
              </tr>
            </thead>
            <tbody>
              {(data.recording_type_detail || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>
                    <span style={{
                      background: '#eff6ff', color: '#2563eb',
                      padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600
                    }}>{r.recording_type}</span>
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', color: '#64748b' }}>{r.avg_duration}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right',
                    color: r.avg_impedance <= 5 ? '#10b981' : r.avg_impedance <= 10 ? '#f59e0b' : '#ef4444',
                    fontWeight: 600
                  }}>{r.avg_impedance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Recent studies */}
      <Card title="Recent Studies" span={4}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f1f5f9' }}>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Date</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Patient</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Type</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Duration (min)</th>
                <th style={{ padding: '6px 8px', textAlign: 'right' }}>Rate (Hz)</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Montage</th>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Technician Notes</th>
              </tr>
            </thead>
            <tbody>
              {(data.recent_studies || []).map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                  <td style={{ padding: '6px 8px', color: '#64748b', fontSize: 11 }}>{s.study_date}</td>
                  <td style={{ padding: '6px 8px', fontWeight: 500 }}>{s.patient_id}</td>
                  <td style={{ padding: '6px 8px' }}>
                    <span style={{
                      background: '#eff6ff', color: '#2563eb',
                      padding: '1px 6px', borderRadius: 8, fontSize: 10, fontWeight: 600
                    }}>{s.recording_type}</span>
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right' }}>{s.duration_min}</td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: '#64748b' }}>{s.sampling_rate}</td>
                  <td style={{ padding: '6px 8px', color: '#64748b' }}>{s.montage}</td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8', fontSize: 11, fontStyle: s.technician_notes ? 'italic' : 'normal' }}>
                    {s.technician_notes || '--'}
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

function DefinitionsTab({ data }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {/* Glossary */}
      <Card title="EEG Acquisition Glossary" span={2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {(data.glossary || []).map((g, i) => (
            <div key={i} style={{ padding: '6px 10px', background: i % 2 === 0 ? '#f8fafc' : '#fff', borderRadius: 4 }}>
              <span style={{ fontWeight: 700, fontSize: 12, color: '#334155' }}>{g.term}</span>
              <span style={{ fontSize: 12, color: '#64748b' }}> — {g.definition}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Channel regions */}
      <Card title="10-20 System Channel Regions" span={2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {(data.channel_regions || []).map((r, i) => (
            <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 6, borderLeft: `3px solid ${COLORS[i % COLORS.length]}` }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{r.region}</div>
              <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 2 }}>
                <strong>Channels:</strong> {Array.isArray(r.channels) ? r.channels.join(', ') : r.channels}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{r.description}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Quality thresholds */}
      <Card title="Quality Grading Thresholds" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f1f5f9' }}>
              <th style={{ padding: '6px 10px', textAlign: 'left' }}>Grade</th>
              <th style={{ padding: '6px 10px', textAlign: 'left' }}>Impedance Range</th>
              <th style={{ padding: '6px 10px', textAlign: 'left' }}>SNR Range</th>
            </tr>
          </thead>
          <tbody>
            {(data.quality_thresholds || []).map((q, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                <td style={{ padding: '6px 10px' }}>
                  <span style={{
                    background: (GRADE_COLORS[q.grade] || '#64748b') + '20',
                    color: GRADE_COLORS[q.grade] || '#64748b',
                    padding: '2px 10px', borderRadius: 10, fontSize: 12, fontWeight: 700
                  }}>{q.grade}</span>
                </td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{q.impedance_range}</td>
                <td style={{ padding: '6px 10px', color: '#475569' }}>{q.snr_range}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Recording protocols */}
      <Card title="Recording Protocols" span={2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {(data.recording_protocols || []).map((p, i) => (
            <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{p.type}</div>
              <p style={{ fontSize: 12, color: '#475569', margin: '4px 0' }}>{p.description}</p>
              <p style={{ fontSize: 11, color: '#8b5cf6', margin: '2px 0' }}><strong>Typical Duration:</strong> {p.typical_duration}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
