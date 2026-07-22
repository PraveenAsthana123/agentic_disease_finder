import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
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

const TABS = ['Overview', 'Band Analysis', 'Cross-Band Ratios', 'Interference Analysis', 'Methodology']

export default function SpwvdDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/spwvd/overview`),
      axios.get(`${API}/api/spwvd/breakdown`),
      axios.get(`${API}/api/spwvd/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading SPWVD analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  const kpis = ov.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>SPWVD Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Smoothed Pseudo Wigner-Ville Distribution — {fmt(kpis.n_samples)} samples, {fmt(kpis.n_frequency_bins)} frequency bins, {fmtPct(kpis.mean_cross_term_suppression)} cross-term suppression
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab kpis={kpis} bandEnergy={ov.band_energy_distribution} energyConcentration={ov.energy_concentration} freqMarginal={ov.frequency_marginal} crossTerm={ov.cross_term_suppression} />}
      {tab === 1 && <BandAnalysisTab data={bd} />}
      {tab === 2 && <CrossBandRatiosTab data={bd} />}
      {tab === 3 && <InterferenceTab data={bd} crossTerm={ov.cross_term_suppression} resolution={ov.time_frequency_resolution} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ kpis, bandEnergy, energyConcentration, freqMarginal, crossTerm }) {
  const energyData = bandEnergy || []
  const concentrationData = energyConcentration || []
  const marginalData = (freqMarginal || []).filter(d => d.freq_hz <= 100)

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Total Samples" value={fmt(kpis.n_samples)} />
          <KPI label="Frequency Bins" value={fmt(kpis.n_frequency_bins)} />
          <KPI label="Dominant Band" value={kpis.dominant_band || '--'} color="#3b82f6" />
          <KPI label="Cross-Term Suppression" value={fmtPct(kpis.mean_cross_term_suppression)} color="#10b981" />
          <KPI label="Mean Inst. Freq" value={`${kpis.mean_instantaneous_freq_hz || 0} Hz`} color="#8b5cf6" />
          <KPI label="TF Concentration" value={fmt(kpis.tf_concentration_index)} color="#f59e0b" />
        </div>
      </Card>

      <Card title="Band Energy Distribution">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={energyData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="energy" name="Energy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Frequency Marginal Spectrum">
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={marginalData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="freq_hz" label={{ value: 'Hz', position: 'insideBottomRight', offset: -5 }} />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="energy" name="SPWVD Energy" stroke="#8b5cf6" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Energy Concentration Across Samples" span={2}>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={concentrationData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="sample" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="concentration" name="Energy Concentration" stroke="#3b82f6" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function BandAnalysisTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const bandStats = data.band_statistics || []
  const chartData = bandStats.map(b => ({
    band: b.band,
    mean: b.mean,
    std: b.std,
    max: b.max,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="SPWVD Coefficient Statistics per Band (Mean / Std / Max)">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="mean" name="Mean" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="std" name="Std Dev" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="max" name="Max" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Per-Band Statistics">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0' }}>Band</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Freq Range</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Mean</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Std Dev</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Min</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Max</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Energy %</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Skewness</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0' }}>Kurtosis</th>
              </tr>
            </thead>
            <tbody>
              {bandStats.map((b, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{b.band}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', fontFamily: 'monospace' }}>{b.freq_range || '--'}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.mean)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.std)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.min)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.max)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center', color: '#3b82f6', fontWeight: 600 }}>{fmtPct(b.energy_pct)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.skewness)}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{fmt(b.kurtosis)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function CrossBandRatiosTab({ data }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No breakdown data</div>

  const ratios = data.cross_band_ratios || []
  const ratioTrends = data.ratio_trends || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Cross-Band Energy Ratios" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={ratios}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="ratio_name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" name="Ratio" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Ratio Trends Across Samples" span={2}>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={ratioTrends}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="sample" />
            <YAxis />
            <Tooltip />
            <Legend />
            {(ratios || []).slice(0, 5).map((r, i) => (
              <Line key={r.ratio_name} type="monotone" dataKey={r.ratio_name} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={2} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function InterferenceTab({ data, crossTerm, resolution }) {
  if (!data?.available) return <div style={{ color: '#f59e0b' }}>No data</div>

  const coeffDist = data.coefficient_distributions || []
  const instFreq = data.instantaneous_frequency || {}

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Cross-Term Suppression">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <KPI label="Mean Suppression" value={fmtPct(crossTerm?.mean)} color="#10b981" />
          <KPI label="Max Suppression" value={fmtPct(crossTerm?.max)} color="#3b82f6" />
        </div>
        <p style={{ margin: '12px 0 0', fontSize: 12, color: '#64748b' }}>
          {crossTerm?.interpretation || 'Cross-term analysis quantifies interference removed by SPWVD smoothing'}
        </p>
      </Card>

      <Card title="Instantaneous Frequency">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <KPI label="Mean" value={`${instFreq.mean_hz || 0} Hz`} color="#8b5cf6" />
          <KPI label="Std Dev" value={`${instFreq.std_hz || 0} Hz`} />
          <KPI label="Min" value={`${instFreq.min_hz || 0} Hz`} />
          <KPI label="Max" value={`${instFreq.max_hz || 0} Hz`} />
        </div>
      </Card>

      <Card title="Coefficient Distributions per Band" span={2}>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={coeffDist}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="p25" name="25th Pct" fill="#06b6d4" radius={[4, 4, 0, 0]} />
            <Bar dataKey="median" name="Median" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="p75" name="75th Pct" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            <Bar dataKey="max" name="Max" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Resolution Advantage" span={2}>
        <div style={{ fontSize: 13, color: '#475569' }}>
          <p style={{ margin: '0 0 8px' }}><strong>Frequency Resolution:</strong> {resolution?.frequency_resolution_hz || '--'} Hz</p>
          <p style={{ margin: '0 0 8px' }}><strong>Concentration Index:</strong> {fmt(resolution?.concentration_index)}</p>
          <p style={{ margin: '0 0 8px' }}><strong>Spectral Entropy:</strong> {fmt(resolution?.spectral_entropy_bits)} bits</p>
          <p style={{ margin: 0, color: '#64748b', fontSize: 12 }}>{resolution?.advantage_over_stft || ''}</p>
        </div>
      </Card>
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions?.available) return <div style={{ color: '#f59e0b' }}>No definitions data</div>

  const methodology = definitions.methodology || []
  const clinicalRelevance = definitions.clinical_relevance || []
  const waveletNotes = definitions.wavelet_notes || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {methodology.map((m, i) => (
        <Card key={i} title={m.name}>
          <p style={{ margin: '0 0 8px', fontSize: 13, color: '#475569' }}>{m.description}</p>
          {m.parameters && (
            <div style={{ margin: '8px 0', fontSize: 12, color: '#64748b' }}>
              <strong>Parameters:</strong> {m.parameters}
            </div>
          )}
          {m.strengths && m.limitations && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, margin: '12px 0' }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#10b981', marginBottom: 4 }}>Strengths</div>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                  {m.strengths.map((s, j) => <li key={j} style={{ marginBottom: 2 }}>{s}</li>)}
                </ul>
              </div>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#ef4444', marginBottom: 4 }}>Limitations</div>
                <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 12, color: '#475569' }}>
                  {m.limitations.map((l, j) => <li key={j} style={{ marginBottom: 2 }}>{l}</li>)}
                </ul>
              </div>
            </div>
          )}
          {m.reference && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{m.reference}</div>}
        </Card>
      ))}

      <Card title="Clinical Relevance">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {clinicalRelevance.map((cr, i) => <li key={i} style={{ marginBottom: 6 }}>{cr}</li>)}
        </ul>
      </Card>

      <Card title="Implementation Notes">
        <ul style={{ margin: 0, padding: '0 0 0 16px', fontSize: 13, color: '#475569' }}>
          {waveletNotes.map((n, i) => <li key={i} style={{ marginBottom: 6 }}>{n}</li>)}
        </ul>
      </Card>
    </div>
  )
}
