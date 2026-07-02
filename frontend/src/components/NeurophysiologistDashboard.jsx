import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, background: color + '18', color
    }}>{text}</span>
  )
}

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316']

const QUALITY_COLORS = { Good: '#10b981', Fair: '#f59e0b', Poor: '#ef4444', Unknown: '#94a3b8' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'recordings', label: 'Recording Inventory' },
  { id: 'bandpower', label: 'Band Power' },
  { id: 'spectral', label: 'Spectral Features' },
  { id: 'validation', label: 'AI Label Validation' },
  { id: 'seizures', label: 'Seizure Correlates' },
  { id: 'definitions', label: 'Definitions' },
]

export default function NeurophysiologistDashboard() {
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
      axios.get(`${API_URL}/api/neurophysiologist/overview`),
      axios.get(`${API_URL}/api/neurophysiologist/breakdown`),
      axios.get(`${API_URL}/api/neurophysiologist/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading EEG Reviewer data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No neurophysiologist data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Clinical Neurophysiologist / EEG Reviewer</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        EEG recording inventory, background rhythm analysis, band power distribution, signal quality, spectral features, AI label validation
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'recordings' && <RecordingsTab breakdown={breakdown} />}
      {tab === 'bandpower' && <BandPowerTab overview={overview} breakdown={breakdown} />}
      {tab === 'spectral' && <SpectralTab breakdown={breakdown} />}
      {tab === 'validation' && <ValidationTab breakdown={breakdown} />}
      {tab === 'seizures' && <SeizureTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const bandPower = overview.band_power_distribution || []
  const signalQuality = overview.signal_quality_distribution || []
  const rhythm = overview.background_rhythm_distribution || []
  const predictions = overview.prediction_distribution || []
  const activity = overview.daily_activity || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      <Card title="Key Metrics" span={4}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: 12 }}>
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} />)}
        </div>
      </Card>

      <Card title="Band Power Distribution (Relative)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={bandPower}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="power" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Signal Quality" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={signalQuality} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
              {signalQuality.map((e, i) => <Cell key={i} fill={QUALITY_COLORS[e.name] || COLORS[i]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Background Rhythm" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={rhythm} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10 }} width={90} />
            <Tooltip />
            <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="AI Prediction Distribution" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="value" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Pipeline Activity" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={activity}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="events" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function RecordingsTab({ breakdown }) {
  const recordings = breakdown?.recording_inventory || []
  return (
    <Card title={`EEG Recording Inventory (${recordings.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['ID', 'Patient', 'File', 'Disease', 'Channels', 'Rate (Hz)', 'Duration', 'Quality', 'Flat Ch.', 'Dom. Freq', 'Rhythm', 'AI Label', 'Confidence', 'Date'].map(h =>
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {recordings.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}>{r.id}</td>
                <td style={{ padding: '6px', fontWeight: 600 }}>{r.patient_id}</td>
                <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 11 }}>{r.file}</td>
                <td style={{ padding: '6px' }}>{r.disease}</td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{r.n_channels}</td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{r.sampling_rate}</td>
                <td style={{ padding: '6px' }}>{r.duration_hrs ? `${r.duration_hrs}h` : '-'}</td>
                <td style={{ padding: '6px' }}><Badge text={r.signal_quality} color={QUALITY_COLORS[r.signal_quality] || '#94a3b8'} /></td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{r.flat_channels}</td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{r.dominant_freq ? `${r.dominant_freq} Hz` : '-'}</td>
                <td style={{ padding: '6px' }}><Badge text={r.background_rhythm} color="#8b5cf6" /></td>
                <td style={{ padding: '6px' }}><Badge text={r.predicted_label} color={r.predicted_label === 'Epilepsy' ? '#ef4444' : '#10b981'} /></td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{r.confidence != null ? `${(r.confidence * 100).toFixed(0)}%` : '-'}</td>
                <td style={{ padding: '6px', fontSize: 10, color: '#94a3b8' }}>{(r.created_at || '').slice(0, 10)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function BandPowerTab({ overview, breakdown }) {
  const bandPower = overview?.band_power_distribution || []
  const table = breakdown?.band_power_table || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Mean Band Power (All Recordings)" span={2}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={bandPower}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip formatter={v => v.toFixed(4)} />
            <Bar dataKey="power" radius={[4, 4, 0, 0]}>
              {bandPower.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title={`Per-Recording Band Power (${table.length})`} span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Rec #', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'].map(h =>
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {table.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600 }}>{r.patient_id}</td>
                  <td style={{ padding: '6px' }}>{r.recording_id}</td>
                  <td style={{ padding: '6px' }}>{r.delta.toFixed(4)}</td>
                  <td style={{ padding: '6px' }}>{r.theta.toFixed(4)}</td>
                  <td style={{ padding: '6px' }}>{r.alpha.toFixed(4)}</td>
                  <td style={{ padding: '6px' }}>{r.beta.toFixed(4)}</td>
                  <td style={{ padding: '6px' }}>{r.gamma.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function SpectralTab({ breakdown }) {
  const features = breakdown?.spectral_features || []
  const channelStats = breakdown?.channel_stats || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Spectral & Complexity Features (${features.length} recordings)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Rec', 'Spec. Entropy', 'Hjorth Mob.', 'Hjorth Cmplx.', 'Approx Ent.', 'Sample Ent.', 'Hurst', 'DFA \u03b1', 'LZ Cmplx.', 'Kurtosis', 'Skewness'].map(h =>
                  <th key={h} style={{ padding: '6px 4px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569', fontSize: 10 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {features.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '4px', fontWeight: 600 }}>{f.patient_id}</td>
                  <td style={{ padding: '4px' }}>{f.recording_id}</td>
                  <td style={{ padding: '4px' }}>{f.spectral_entropy?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.hjorth_mobility?.toFixed(4) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.hjorth_complexity?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.approx_entropy?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.sample_entropy?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.hurst_exponent?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.dfa_alpha?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.lz_complexity?.toFixed(3) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.kurtosis?.toFixed(2) ?? '-'}</td>
                  <td style={{ padding: '4px' }}>{f.skewness?.toFixed(2) ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title={`Channel Statistics (${channelStats.length} recordings)`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient', 'Rec', 'Channels', 'Mean Ch. Std', 'Max Ch. Std', 'Flat Ch.', 'Sample Channels'].map(h =>
                  <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {channelStats.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600 }}>{c.patient_id}</td>
                  <td style={{ padding: '6px' }}>{c.recording_id}</td>
                  <td style={{ padding: '6px', textAlign: 'center' }}>{c.n_channels}</td>
                  <td style={{ padding: '6px', fontFamily: 'monospace' }}>{c.mean_channel_std}</td>
                  <td style={{ padding: '6px', fontFamily: 'monospace' }}>{c.max_channel_std}</td>
                  <td style={{ padding: '6px', textAlign: 'center' }}>{c.flat_channels}</td>
                  <td style={{ padding: '6px', fontSize: 10, color: '#64748b' }}>{(c.channel_names || []).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function ValidationTab({ breakdown }) {
  const items = breakdown?.ai_validation || []

  return (
    <Card title={`AI Label Validation Queue (${items.length} recordings)`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Patient', 'Rec #', 'AI Label', 'Confidence', 'Signal Quality', 'Class Probabilities', 'Review Status'].map(h =>
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {items.map((v, i) => {
              const probs = v.class_probabilities || {}
              return (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600 }}>{v.patient_id}</td>
                  <td style={{ padding: '6px' }}>{v.recording_id}</td>
                  <td style={{ padding: '6px' }}>
                    <Badge text={v.predicted_label} color={v.predicted_label === 'Epilepsy' ? '#ef4444' : '#10b981'} />
                  </td>
                  <td style={{ padding: '6px' }}>
                    <span style={{ fontWeight: 700, color: v.confidence >= 0.7 ? '#10b981' : v.confidence >= 0.5 ? '#f59e0b' : '#ef4444' }}>
                      {v.confidence != null ? `${(v.confidence * 100).toFixed(0)}%` : '-'}
                    </span>
                  </td>
                  <td style={{ padding: '6px' }}><Badge text={v.signal_quality} color={QUALITY_COLORS[v.signal_quality] || '#94a3b8'} /></td>
                  <td style={{ padding: '6px', fontSize: 11 }}>
                    {Object.entries(probs).map(([cls, prob]) => (
                      <span key={cls} style={{ marginRight: 8 }}>{cls}: {(prob * 100).toFixed(0)}%</span>
                    ))}
                  </td>
                  <td style={{ padding: '6px' }}><Badge text={v.review_status} color="#f59e0b" /></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function SeizureTab({ breakdown }) {
  const seizures = breakdown?.seizure_log || []

  return (
    <Card title={`Seizure Correlates (${seizures.length} events)`}>
      <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
        Seizure events for EEG-clinical correlation. The neurophysiologist reviews these alongside EEG recordings to identify ictal patterns.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Patient', 'Date', 'Duration (s)', 'Severity', 'Location', 'Motor Signs', 'Aura', 'Trigger'].map(h =>
                <th key={h} style={{ padding: '8px 6px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', fontWeight: 600, color: '#475569' }}>{h}</th>
              )}
            </tr>
          </thead>
          <tbody>
            {seizures.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600 }}>{s.patient_id}</td>
                <td style={{ padding: '6px' }}>{s.date}</td>
                <td style={{ padding: '6px', textAlign: 'center' }}>{s.duration_sec ?? '-'}</td>
                <td style={{ padding: '6px' }}>
                  {s.severity && <Badge text={s.severity} color={
                    s.severity === 'severe' ? '#ef4444' : s.severity === 'moderate' ? '#f59e0b' : '#10b981'
                  } />}
                </td>
                <td style={{ padding: '6px' }}>{s.location || '-'}</td>
                <td style={{ padding: '6px' }}>{s.motor_signs != null ? (s.motor_signs ? 'Yes' : 'No') : '-'}</td>
                <td style={{ padding: '6px' }}>{s.aura || '-'}</td>
                <td style={{ padding: '6px' }}>{s.trigger || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function DefinitionsTab({ definitions }) {
  const defs = definitions || []
  return (
    <Card title={`EEG & Neurophysiology Definitions (${defs.length})`}>
      <div style={{ display: 'grid', gap: 12 }}>
        {defs.map((d, i) => (
          <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
            <div style={{ fontWeight: 700, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{d.term}</div>
            <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.5 }}>{d.definition}</div>
          </div>
        ))}
      </div>
    </Card>
  )
}
