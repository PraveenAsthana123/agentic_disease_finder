import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'

const API_URL = '/api'

const CLASS_COLORS = {
  Lesional: '#ef4444',
  'Non-Lesional': '#3b82f6',
  Equivocal: '#f59e0b',
  Normal: '#22c55e'
}

const QUALITY_COLORS = {
  Diagnostic: '#22c55e',
  Adequate: '#3b82f6',
  Suboptimal: '#f59e0b',
  'Non-diagnostic': '#ef4444'
}

const LESION_COLORS = [
  '#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#14b8a6', '#64748b'
]

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(2)) : String(v)
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

function ClassBadge({ classification }) {
  const color = CLASS_COLORS[classification] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{classification || 'unknown'}</span>
  )
}

function QualityBadge({ quality }) {
  const color = QUALITY_COLORS[quality] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600,
      background: color + '22', color
    }}>{quality || 'unknown'}</span>
  )
}

export default function MRIReviewDashboard() {
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
          axios.get(`${API_URL}/mri-review/overview`),
          axios.get(`${API_URL}/mri-review/breakdown`),
          axios.get(`${API_URL}/mri-review/definitions`)
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading MRI Review data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview && !breakdown) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No MRI review data available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'scans', label: 'Scans' },
    { id: 'volumetric', label: 'Volumetric' },
    { id: 'concordance', label: 'Concordance' },
    { id: 'definitions', label: 'Definitions' },
  ]

  /* Overview data prep */
  const classDistData = overview?.classification_distribution
    ? Object.entries(overview.classification_distribution).map(([name, value]) => ({ name, value }))
    : []
  const lesionDistData = overview?.lesion_type_distribution
    ? Object.entries(overview.lesion_type_distribution).map(([name, value]) => ({ name, value }))
    : []
  const lobeDistData = overview?.lobe_distribution
    ? Object.entries(overview.lobe_distribution).map(([name, value]) => ({ name, value }))
    : []
  const lateralityData = overview?.laterality_distribution
    ? Object.entries(overview.laterality_distribution).map(([name, value]) => ({ name, value }))
    : []

  /* Breakdown data prep */
  const scans = breakdown?.scans || []
  const volStats = breakdown?.volume_asymmetry_stats || {}
  const abnormalScans = scans.filter(s => s.hippocampal_volume_asymmetry > (volStats.abnormal_threshold || 0.08))
  const concordantScans = scans.filter(s => s.concordant === true)
  const discordantScans = scans.filter(s => s.concordant === false)

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>MRI Brain Review Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Epilepsy pre-surgical evaluation — structural MRI findings, lesion classification, volumetric analysis
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Total Patients" value={fmt(overview.total_patients)} />
              <KPI label="Lesional Rate" value={`${fmt(overview.lesional_rate)}%`} color="#ef4444" />
              <KPI label="Hippocampal Sclerosis" value={fmt(overview.hippocampal_sclerosis_count)} color="#8b5cf6" />
              <KPI label="Volume Abnormal" value={fmt(volStats.abnormal_count)} sub={`threshold: ${volStats.abnormal_threshold}`} color="#f59e0b" />
            </div>
          </Card>

          {/* Pie: Classification */}
          <Card title="Classification Distribution">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={classDistData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {classDistData.map((d, i) => <Cell key={i} fill={CLASS_COLORS[d.name] || '#94a3b8'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {classDistData.map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: CLASS_COLORS[d.name] || '#94a3b8', marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Bar: Lesion Types */}
          <Card title="Lesion Type Distribution" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={lesionDistData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" name="Count">
                  {lesionDistData.map((d, i) => <Cell key={i} fill={LESION_COLORS[i % LESION_COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Bar: Lobe Distribution */}
          <Card title="Lobe Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={lobeDistData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={80} />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Pie: Laterality */}
          <Card title="Laterality" span={2}>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={lateralityData} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  {lateralityData.map((d, i) => <Cell key={i} fill={['#3b82f6', '#ef4444', '#f59e0b'][i] || '#94a3b8'} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 8 }}>
              {lateralityData.map((d, i) => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: ['#3b82f6', '#ef4444', '#f59e0b'][i] || '#94a3b8', marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Tab 2: Scans */}
      {tab === 'scans' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 16 }}>
              <KPI label="Total Scans" value={fmt(breakdown.total_scans)} />
              <KPI label="Concordant (EEG-MRI)" value={fmt(concordantScans.length)} color="#22c55e" />
              <KPI label="Discordant" value={fmt(discordantScans.length)} color="#ef4444" />
            </div>
          </Card>

          <Card title="MRI Scan Details">
            <div style={{ maxHeight: 500, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Age</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Lesion</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Location</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Laterality</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Classification</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Quality</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>HS</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Vol Asym</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Concordant</th>
                  </tr>
                </thead>
                <tbody>
                  {scans.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.age || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.lesion_type || '--'}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.location || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.laterality || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ClassBadge classification={row.classification} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><QualityBadge quality={row.quality} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: row.hippocampal_sclerosis === 'Yes' ? '#8b5cf6' : '#94a3b8', fontWeight: row.hippocampal_sclerosis === 'Yes' ? 600 : 400 }}>{row.hippocampal_sclerosis || '--'}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: row.hippocampal_volume_asymmetry > (volStats.abnormal_threshold || 0.08) ? '#ef4444' : '#475569' }}>{fmt(row.hippocampal_volume_asymmetry)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11 }}>
                        {row.concordant === true
                          ? <span style={{ color: '#22c55e', fontWeight: 600 }}>Yes</span>
                          : row.concordant === false
                            ? <span style={{ color: '#ef4444', fontWeight: 600 }}>No</span>
                            : <span style={{ color: '#94a3b8' }}>--</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Volumetric */}
      {tab === 'volumetric' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              <KPI label="Mean Asymmetry" value={fmt(volStats.mean)} />
              <KPI label="Max Asymmetry" value={fmt(volStats.max)} color="#ef4444" />
              <KPI label="Abnormal Count" value={fmt(volStats.abnormal_count)} color="#f59e0b" />
              <KPI label="Threshold" value={fmt(volStats.abnormal_threshold)} color="#64748b" />
            </div>
          </Card>

          {/* Bar: Volume Asymmetry per Patient */}
          <Card title="Hippocampal Volume Asymmetry by Patient" span={2}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={scans.filter(s => s.hippocampal_volume_asymmetry != null).sort((a, b) => (b.hippocampal_volume_asymmetry || 0) - (a.hippocampal_volume_asymmetry || 0))
                .map(s => ({ name: s.name || s.patient_id, asymmetry: s.hippocampal_volume_asymmetry }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="asymmetry" name="Asymmetry">
                  {scans.filter(s => s.hippocampal_volume_asymmetry != null).sort((a, b) => (b.hippocampal_volume_asymmetry || 0) - (a.hippocampal_volume_asymmetry || 0))
                    .map((s, i) => (
                      <Cell key={i} fill={s.hippocampal_volume_asymmetry > (volStats.abnormal_threshold || 0.08) ? '#ef4444' : '#3b82f6'} />
                    ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Table: Abnormal Volume Asymmetry */}
          <Card title="Abnormal Volume Asymmetry Patients" span={2}>
            {abnormalScans.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No abnormal volume asymmetry detected.</div>
            ) : (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Lesion</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Laterality</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>HS</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Vol Asymmetry</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>T2/FLAIR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {abnormalScans.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.lesion_type || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.laterality || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: row.hippocampal_sclerosis === 'Yes' ? '#8b5cf6' : '#94a3b8' }}>{row.hippocampal_sclerosis || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#ef4444', fontWeight: 600 }}>{fmt(row.hippocampal_volume_asymmetry)}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.t2_flair_signal || '--'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 4: Concordance */}
      {tab === 'concordance' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
              <KPI label="Concordant (EEG-MRI)" value={fmt(concordantScans.length)} color="#22c55e" />
              <KPI label="Discordant" value={fmt(discordantScans.length)} color="#ef4444" />
              <KPI label="Concordance Rate" value={scans.length > 0 ? `${((concordantScans.length / scans.length) * 100).toFixed(1)}%` : '--'} color="#3b82f6" />
            </div>
          </Card>

          {/* Pie: Concordance */}
          <Card title="EEG-MRI Concordance">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={[
                  { name: 'Concordant', value: concordantScans.length },
                  { name: 'Discordant', value: discordantScans.length }
                ]} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={75} paddingAngle={2}>
                  <Cell fill="#22c55e" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          {/* Bar: Concordance by Lesion Type */}
          <Card title="Concordance by Lesion Type">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={(() => {
                const byType = {}
                scans.forEach(s => {
                  const t = s.lesion_type || 'Unknown'
                  if (!byType[t]) byType[t] = { name: t, concordant: 0, discordant: 0 }
                  if (s.concordant === true) byType[t].concordant++
                  else if (s.concordant === false) byType[t].discordant++
                })
                return Object.values(byType)
              })()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={50} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="concordant" fill="#22c55e" name="Concordant" stackId="a" />
                <Bar dataKey="discordant" fill="#ef4444" name="Discordant" stackId="a" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Table: Discordant Cases */}
          <Card title="Discordant Cases — Review Needed" span={2}>
            {discordantScans.length === 0 ? (
              <div style={{ padding: 20, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>No discordant cases.</div>
            ) : (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Lesion</th>
                      <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Location</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Classification</th>
                      <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Seizure Diary</th>
                    </tr>
                  </thead>
                  <tbody>
                    {discordantScans.map((row, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px 8px', fontWeight: 600, fontSize: 12 }}>{row.name || row.patient_id}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.lesion_type || '--'}</td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569' }}>{row.location || '--'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}><ClassBadge classification={row.classification} /></td>
                        <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#475569' }}>{row.seizure_diary_entries || 0} entries</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}

      {/* Tab 5: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          <Card title="Purpose" span={2}>
            <p style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.6 }}>{defs.purpose}</p>
          </Card>

          <Card title="Epilepsy MRI Protocol" span={2}>
            <div style={{ fontSize: 13, color: '#475569' }}>
              <p style={{ margin: '0 0 8px' }}><strong>Field Strength:</strong> {defs.protocol?.field_strength}</p>
              <p style={{ margin: '0 0 4px', fontWeight: 600 }}>Sequences:</p>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {(defs.protocol?.sequences || []).map((seq, i) => (
                  <li key={i} style={{ marginBottom: 4, fontSize: 12 }}>{seq}</li>
                ))}
              </ul>
              {defs.protocol?.reference && (
                <p style={{ margin: '12px 0 0', fontSize: 11, color: '#94a3b8', fontStyle: 'italic' }}>{defs.protocol.reference}</p>
              )}
            </div>
          </Card>

          <Card title="Lesion Types">
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              {(defs.lesion_types || []).map((lt, i) => (
                <div key={i} style={{ padding: '8px 0', borderBottom: i < (defs.lesion_types || []).length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{lt.label} ({lt.code})</span>
                    <span style={{ fontSize: 11, color: '#64748b' }}>{(lt.prevalence * 100).toFixed(0)}% prevalence</span>
                  </div>
                  <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b' }}>{lt.description}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Classification Criteria">
            {defs.classification ? (
              <div style={{ maxHeight: 300, overflow: 'auto' }}>
                {Object.entries(defs.classification).map(([key, val], i) => (
                  <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #f1f5f9' }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{key}</span>
                    <p style={{ margin: '4px 0 0', fontSize: 12, color: '#64748b' }}>{typeof val === 'string' ? val : JSON.stringify(val)}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ margin: 0, fontSize: 13, color: '#94a3b8' }}>No classification criteria available.</p>
            )}
          </Card>

          {defs.clinical_significance && (
            <Card title="Clinical Significance" span={2}>
              {Array.isArray(defs.clinical_significance) ? (
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {defs.clinical_significance.map((item, i) => (
                    <li key={i} style={{ marginBottom: 4, fontSize: 12, color: '#475569' }}>{typeof item === 'string' ? item : JSON.stringify(item)}</li>
                  ))}
                </ul>
              ) : (
                <p style={{ margin: 0, fontSize: 13, color: '#475569' }}>{JSON.stringify(defs.clinical_significance)}</p>
              )}
            </Card>
          )}

          {defs.references && (
            <Card title="References" span={2}>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {(Array.isArray(defs.references) ? defs.references : [defs.references]).map((ref, i) => (
                  <li key={i} style={{ marginBottom: 4, fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>{typeof ref === 'string' ? ref : JSON.stringify(ref)}</li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
