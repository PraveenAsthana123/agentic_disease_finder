import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, RadarChart, Radar, PolarGrid,
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
const QUALITY_COLORS = { Good: '#10b981', Fair: '#f59e0b', Poor: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'spectrograms', label: 'Spectrogram Inventory' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'models', label: 'Model Architectures' },
  { id: 'training', label: 'Training Readiness' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function CNNResNetDashboard() {
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
      axios.get(`${API_URL}/api/cnn-resnet/overview`),
      axios.get(`${API_URL}/api/cnn-resnet/breakdown`),
      axios.get(`${API_URL}/api/cnn-resnet/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading CNN/ResNet Spectrogram data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No data'}</div>

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>CNN/ResNet Spectrogram Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG spectrogram generation, CNN/ResNet model architectures, training readiness &amp; frequency band analysis
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', borderRadius: 8, border: 'none', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && renderOverview(overview)}
      {tab === 'spectrograms' && renderSpectrograms(breakdown)}
      {tab === 'patients' && renderPatients(breakdown)}
      {tab === 'models' && renderModels(breakdown)}
      {tab === 'training' && renderTraining(breakdown)}
      {tab === 'pipeline' && renderPipeline(breakdown)}
      {tab === 'definitions' && renderDefinitions(definitions)}
    </div>
  )
}

function renderOverview(ov) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {(ov.kpis || []).map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
      ))}

      <Card title="Band Power Distribution (Mean)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.band_power_chart || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="mean_power" fill="#3b82f6" radius={[4,4,0,0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Signal Quality" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={ov.quality_distribution || []} dataKey="count" nameKey="quality"
              cx="50%" cy="50%" outerRadius={80} label={({ quality, count }) => `${quality}: ${count}`}>
              {(ov.quality_distribution || []).map((d, i) => (
                <Cell key={i} fill={d.color || COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Frequency Band Radar" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <RadarChart data={ov.freq_radar || []}>
            <PolarGrid />
            <PolarAngleAxis dataKey="band" fontSize={10} />
            <PolarRadiusAxis fontSize={9} />
            <Radar name="Power" dataKey="mean_power" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Classification by Disease" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.classification_chart || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill="#10b981" radius={[4,4,0,0]} name="Analyses" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Spectrogram Generation" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ov.daily_activity || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" fontSize={10} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderSpectrograms(bd) {
  const items = bd?.spectrogram_inventory || []
  return (
    <Card title={`Spectrogram Inventory (${items.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Prediction</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Confidence</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Quality</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Channels</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Sample Rate</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Duration (s)</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Date</th>
            </tr>
          </thead>
          <tbody>
            {items.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{r.patient_id}</td>
                <td style={{ padding: 8 }}>{r.disease}</td>
                <td style={{ padding: 8 }}>
                  <Badge text={r.predicted_label || 'N/A'} color="#3b82f6" />
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : '-'}
                </td>
                <td style={{ padding: 8 }}>
                  <Badge text={r.signal_quality || 'N/A'} color={QUALITY_COLORS[r.signal_quality] || '#6b7280'} />
                </td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.n_channels || '-'}</td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{r.sampling_rate || '-'}</td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{r.duration_seconds ? r.duration_seconds.toFixed(1) : '-'}</td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>{r.created_at || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderPatients(bd) {
  const patients = bd?.patient_profiles || []
  return (
    <Card title={`Patient Profiles (${patients.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Spectrograms</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Diseases</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Mean Confidence</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Quality</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Medications</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{p.patient_id}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{p.total_spectrograms}</td>
                <td style={{ padding: 8 }}>
                  {(p.diseases || []).map((d, j) => <Badge key={j} text={d} color="#8b5cf6" />)}
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {p.mean_confidence != null ? `${(p.mean_confidence * 100).toFixed(1)}%` : '-'}
                </td>
                <td style={{ padding: 8 }}>
                  {Object.entries(p.quality_distribution || {}).map(([q, c], j) => (
                    <span key={j} style={{ marginRight: 6 }}>
                      <Badge text={`${q}: ${c}`} color={QUALITY_COLORS[q] || '#6b7280'} />
                    </span>
                  ))}
                </td>
                <td style={{ padding: 8, fontSize: 11, color: '#64748b' }}>
                  {(p.medications || []).join(', ') || 'None'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderModels(bd) {
  const models = bd?.model_architecture || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
      {models.map((m, i) => (
        <Card key={i} title={m.name}>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <div><strong>Type:</strong> {m.type}</div>
            <div><strong>Input Shape:</strong> <code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 4 }}>{m.input_shape}</code></div>
            <div><strong>Layers:</strong> {m.layers}</div>
            <div><strong>Parameters:</strong> {m.parameters?.toLocaleString()}</div>
            <div><strong>Output:</strong> {m.output_classes} classes</div>
            {m.key_features && (
              <div style={{ marginTop: 8 }}>
                <strong>Key Features:</strong>
                <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                  {m.key_features.map((f, j) => <li key={j} style={{ fontSize: 12, color: '#475569' }}>{f}</li>)}
                </ul>
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  )
}

function renderTraining(bd) {
  const readiness = bd?.training_readiness || []
  return (
    <Card title="Training Readiness by Disease">
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Disease</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Spectrograms</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Patients</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Mean Confidence</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Readiness</th>
            </tr>
          </thead>
          <tbody>
            {readiness.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{r.disease}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.spectrogram_count}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.patient_count}</td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {r.mean_confidence != null ? `${(r.mean_confidence * 100).toFixed(1)}%` : '-'}
                </td>
                <td style={{ padding: 8 }}>
                  <Badge
                    text={r.ready ? 'Ready' : 'Needs Data'}
                    color={r.ready ? '#10b981' : '#f59e0b'}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 12, color: '#94a3b8', marginTop: 12 }}>
        Minimum threshold: 10 spectrograms per disease class for training readiness.
      </p>
    </Card>
  )
}

function renderPipeline(bd) {
  const events = bd?.pipeline_log || []
  return (
    <Card title={`Pipeline Event Log (${events.length})`}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 6 }}>Timestamp</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Component</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Action</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Actor</th>
              <th style={{ textAlign: 'left', padding: 6 }}>Detail</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11, color: '#64748b' }}>{e.ts_local || e.ts_utc}</td>
                <td style={{ padding: 6 }}><Badge text={e.component} color="#3b82f6" /></td>
                <td style={{ padding: 6 }}>{e.action}</td>
                <td style={{ padding: 6, fontSize: 11 }}>{e.actor}</td>
                <td style={{ padding: 6, fontSize: 11, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function renderDefinitions(defs) {
  if (!defs) return null
  const sections = [
    { key: 'concepts', title: 'CNN/ResNet Concepts' },
    { key: 'quality_metrics', title: 'Quality Metrics' },
    { key: 'model_variants', title: 'Model Variants' },
    { key: 'compliance', title: 'Compliance References' },
    { key: 'remediation', title: 'Remediation Strategies' },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(400px, 1fr))', gap: 16 }}>
      {sections.map(s => {
        const items = defs[s.key]
        if (!items || !items.length) return null
        return (
          <Card key={s.key} title={s.title}>
            {items.map((d, i) => (
              <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: i < items.length - 1 ? '1px solid #f1f5f9' : 'none' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 4, lineHeight: 1.5 }}>{d.definition}</div>
              </div>
            ))}
          </Card>
        )
      })}
    </div>
  )
}
