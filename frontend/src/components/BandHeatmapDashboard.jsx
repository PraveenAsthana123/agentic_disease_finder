import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area
} from 'recharts'

const API = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(4)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return (v * 100).toFixed(1) + '%'
}

const TABS = ['Overview', 'Band Distribution', 'Band Ratios', 'Diagnosis Profiles', 'Methodology']

export default function BandHeatmapDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/band-heatmap/overview`),
      axios.get(`${API}/band-heatmap/breakdown`),
      axios.get(`${API}/band-heatmap/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Band Heatmap analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>EEG Band Power Heatmap Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Band power distribution and heatmap visualization — {fmt(kpis.total_subjects)} subjects, {fmt(kpis.bands_analyzed)} bands analyzed
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} ov={ov} />}
      {tab === 1 && <BandDistributionTab data={bd} />}
      {tab === 2 && <BandRatiosTab data={bd} />}
      {tab === 3 && <DiagnosisProfilesTab data={bd} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ kpis, ov }) {
  const bandDistribution = ov.band_distribution || []
  const dominanceCounts = ov.dominance_counts || []
  const heatmapData = ov.heatmap_data || []
  const bands = heatmapData.length > 0 ? Object.keys(heatmapData[0]).filter(k => k !== 'subject' && k !== 'label' && k !== 'id') : []

  function heatColor(val) {
    if (val == null) return '#f8fafc'
    const clamped = Math.min(Math.max(val, 0), 1)
    const r = Math.round(240 - clamped * 200)
    const g = Math.round(240 - clamped * 60)
    const b = Math.round(240 - clamped * 200)
    return `rgb(${r}, ${g}, ${b})`
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Total Subjects" value={fmt(kpis.total_subjects)} />
          <KPI label="Dominant Band" value={kpis.dominant_band || '--'} color="#3b82f6" />
          <KPI label="Abnormal %" value={fmtPct(kpis.abnormal_pct)} color="#ef4444" />
          <KPI label="Mean Entropy" value={fmt(kpis.mean_entropy)} color="#8b5cf6" />
          <KPI label="Bands Analyzed" value={fmt(kpis.bands_analyzed)} />
        </div>
      </Card>

      <Card title="Band Power Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={bandDistribution}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="fraction" name="Fraction" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Dominance Counts">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={dominanceCounts}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" name="Count" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Band Power Heatmap" span={2}>
        {heatmapData.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Subject</th>
                  {bands.map((b, i) => (
                    <th key={i} style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>{b}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {heatmapData.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{row.subject || row.label || row.id || `S${i + 1}`}</td>
                    {bands.map((b, j) => (
                      <td key={j} style={{
                        padding: '6px 10px', textAlign: 'center', fontFamily: 'monospace',
                        background: heatColor(row[b]), fontWeight: 500
                      }}>
                        {row[b] != null ? (typeof row[b] === 'number' ? row[b].toFixed(3) : row[b]) : '--'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>No heatmap data available</div>
        )}
      </Card>
    </div>
  )
}

function BandDistributionTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const bandStats = data.band_stats || []
  const chartData = bandStats.map(b => ({
    band: b.band,
    mean: b.mean,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Mean Power per Band">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="mean" name="Mean Power" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Full Band Statistics">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Band</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Mean</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Std</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Median</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Min</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Max</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Q25</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Q75</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Skewness</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Kurtosis</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>CV</th>
              </tr>
            </thead>
            <tbody>
              {bandStats.map((b, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{b.band}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.mean)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.std)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.median)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.min)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.max)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.q25)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.q75)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.skewness)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.kurtosis)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmt(b.cv)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function BandRatiosTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const ratios = data.ratios || []
  const correlationMatrix = data.correlation_matrix || []
  const bands = correlationMatrix.length > 0 ? Object.keys(correlationMatrix[0]).filter(k => k !== 'band') : []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Band Ratios">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={ratios}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" name="Ratio" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Ratio Details">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Ratio</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Value</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {ratios.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.name}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#8b5cf6', fontWeight: 600 }}>{fmt(r.value)}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{r.description || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Correlation Matrix">
        {correlationMatrix.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Band</th>
                  {bands.map((b, i) => (
                    <th key={i} style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>{b}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{row.band}</td>
                    {bands.map((b, j) => {
                      const corr = row[b]
                      const absCorr = Math.abs(corr || 0)
                      const bg = absCorr > 0.7 ? '#dcfce7' : absCorr > 0.4 ? '#fef9c3' : '#fff'
                      return (
                        <td key={j} style={{
                          padding: '6px 10px', textAlign: 'center', fontFamily: 'monospace',
                          background: bg, color: corr >= 0 ? '#16a34a' : '#dc2626', fontWeight: absCorr > 0.7 ? 600 : 400
                        }}>
                          {corr != null ? (typeof corr === 'number' ? corr.toFixed(3) : corr) : '--'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: 12 }}>No correlation data available</div>
        )}
      </Card>
    </div>
  )
}

function DiagnosisProfilesTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const diagnosisBreakdown = data.diagnosis_breakdown || []
  const bands = diagnosisBreakdown.length > 0
    ? Object.keys(diagnosisBreakdown[0]).filter(k => k !== 'diagnosis' && k !== 'count')
    : []

  const chartData = diagnosisBreakdown.map(d => {
    const entry = { diagnosis: d.diagnosis }
    bands.forEach(b => { entry[b] = d[b] })
    return entry
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Band Power Means Across Diagnoses">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="diagnosis" />
            <YAxis />
            <Tooltip />
            <Legend />
            {bands.map((b, i) => (
              <Bar key={b} dataKey={b} name={b} fill={COLORS[i % COLORS.length]} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-Diagnosis Breakdown">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Diagnosis</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Count</th>
                {bands.map((b, i) => (
                  <th key={i} style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>{b}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {diagnosisBreakdown.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{row.diagnosis}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmt(row.count)}</td>
                  {bands.map((b, j) => (
                    <td key={j} style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(row[b])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions?.available) return <div style={{ color: '#f59e0b' }}>No definitions data</div>

  const bandDefinitions = definitions.band_definitions || []
  const interpretationNotes = definitions.interpretation_notes || []
  const clinicalApplications = definitions.clinical_applications || []
  const references = definitions.references || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Band Definitions">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Band</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Frequency Range (Hz)</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Clinical Significance</th>
              </tr>
            </thead>
            <tbody>
              {bandDefinitions.map((b, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{b.name}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{b.range || '--'}</td>
                  <td style={{ padding: '8px 12px', color: '#475569' }}>{b.clinical_significance || '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Heatmap Interpretation Notes">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {interpretationNotes.map((note, i) => <li key={i} style={{ marginBottom: 6 }}>{note}</li>)}
        </ul>
      </Card>

      <Card title="Clinical Applications">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {clinicalApplications.map((app, i) => <li key={i} style={{ marginBottom: 6 }}>{app}</li>)}
        </ul>
      </Card>

      <Card title="References">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {references.map((ref, i) => <li key={i} style={{ marginBottom: 6 }}>{ref}</li>)}
        </ul>
      </Card>
    </div>
  )
}
