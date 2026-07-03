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
const SEVERITY_COLORS = { ok: '#10b981', warning: '#f59e0b', error: '#ef4444' }

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'sequences', label: 'Sequence Inventory' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'models', label: 'Model Architectures' },
  { id: 'training', label: 'Training Readiness' },
  { id: 'seizures', label: 'Seizure Temporal' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function RNNLSTMDashboard() {
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
      axios.get(`${API_URL}/api/rnn-lstm/overview`),
      axios.get(`${API_URL}/api/rnn-lstm/breakdown`),
      axios.get(`${API_URL}/api/rnn-lstm/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading RNN/LSTM Temporal Model data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>{overview?.message || 'No data'}</div>

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, color: '#1e293b' }}>RNN/LSTM Temporal Model Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        EEG temporal sequence analysis, RNN/LSTM/GRU/BiLSTM/Attention architectures, training readiness &amp; seizure temporal patterns
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
      {tab === 'sequences' && renderSequences(breakdown)}
      {tab === 'patients' && renderPatients(breakdown)}
      {tab === 'models' && renderModels(breakdown)}
      {tab === 'training' && renderTraining(breakdown)}
      {tab === 'seizures' && renderSeizures(breakdown)}
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

      <Card title="Temporal Band Power Distribution (Mean)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.band_power_chart || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="band" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="mean_power" fill="#6366f1" radius={[4,4,0,0]} />
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

      <Card title="Seizure Duration Distribution" span={1}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.duration_histogram || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bin" fontSize={10} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill="#ef4444" radius={[4,4,0,0]} name="Events" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Classification by Disease" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={ov.classification_chart || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="predicted_label" fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="count" fill="#10b981" radius={[4,4,0,0]} name="Sequences" />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Sequence Generation" span={2}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ov.daily_activity || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" fontSize={10} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Line type="monotone" dataKey="sequences" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderSequences(bd) {
  const items = bd?.sequence_inventory || []
  return (
    <Card title={`Temporal Sequence Inventory (${items.length})`}>
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
              <th style={{ textAlign: 'right', padding: 8 }}>Samples</th>
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
                  <Badge text={r.predicted_label || 'N/A'} color="#6366f1" />
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : '-'}
                </td>
                <td style={{ padding: 8 }}>
                  <Badge text={r.signal_quality || 'N/A'} color={QUALITY_COLORS[r.signal_quality] || '#6b7280'} />
                </td>
                <td style={{ padding: 8, textAlign: 'right' }}>{r.n_channels || '-'}</td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{r.total_samples ? r.total_samples.toLocaleString() : '-'}</td>
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
    <Card title={`Patient Temporal Profiles (${patients.length})`}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Sequences</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Diseases</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Mean Confidence</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Seizures</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Quality</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Medications</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{p.patient_id}</td>
                <td style={{ padding: 8, textAlign: 'right' }}>{p.total_sequences}</td>
                <td style={{ padding: 8 }}>
                  {(p.diseases || []).map((d, j) => <Badge key={j} text={d} color="#8b5cf6" />)}
                </td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>
                  {p.mean_confidence != null ? `${(p.mean_confidence * 100).toFixed(1)}%` : '-'}
                </td>
                <td style={{ padding: 8, textAlign: 'right' }}>{p.seizure_count || 0}</td>
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
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 16 }}>
      {models.map((m, i) => (
        <Card key={i} title={m.name}>
          <div style={{ fontSize: 13, lineHeight: 1.8 }}>
            <div><strong>Input Shape:</strong> <code style={{ background: '#f1f5f9', padding: '2px 6px', borderRadius: 4 }}>{m.input_shape}</code></div>
            <div><strong>Hidden Size:</strong> {m.hidden_size}</div>
            <div><strong>Layers:</strong> {m.num_layers} recurrent + {m.n_layers - m.num_layers} dense/norm</div>
            <div><strong>Total Params:</strong> {m.total_params?.toLocaleString()}</div>
            <div><strong>Trainable Params:</strong> {m.trainable_params?.toLocaleString()}</div>
            <div><strong>Max Seq Length:</strong> {m.max_seq_len?.toLocaleString()}</div>
            <div><strong>Optimizer:</strong> {m.optimizer}</div>
            <div><strong>Loss:</strong> {m.loss}</div>
            <div><strong>Batch Size:</strong> {m.batch_size}</div>
            <p style={{ fontSize: 12, color: '#64748b', marginTop: 8, lineHeight: 1.5 }}>{m.description}</p>
            <div style={{ marginTop: 8 }}>
              <strong style={{ fontSize: 12 }}>Layer Stack:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                {(m.layers || []).map((l, j) => (
                  <span key={j} style={{
                    padding: '2px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'monospace',
                    background: l.type?.includes('LSTM') || l.type?.includes('GRU') || l.type?.includes('RNN') || l.type?.includes('Attention')
                      ? '#ede9fe' : l.type?.includes('Linear') ? '#dbeafe' : '#f1f5f9',
                    color: '#475569',
                  }}>{l.type}{l.hidden_size ? ` (${l.hidden_size})` : ''}{l.out ? ` (${l.out})` : ''}{l.p ? ` p=${l.p}` : ''}</span>
                ))}
              </div>
            </div>
          </div>
        </Card>
      ))}
    </div>
  )
}

function renderTraining(bd) {
  const tr = bd?.training_readiness || {}
  const perDisease = tr.per_disease || []
  const flags = tr.readiness_flags || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <Card title="Training Readiness Summary" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 16 }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#1e293b' }}>{tr.total_sequences || 0}</div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Total Sequences</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#10b981' }}>{tr.usable_sequences || 0}</div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Good Quality</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700, color: '#3b82f6' }}>{tr.n_classes || 0}</div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Disease Classes</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 700,
              color: tr.class_balance_ratio >= 0.8 ? '#10b981' : tr.class_balance_ratio >= 0.5 ? '#f59e0b' : '#ef4444'
            }}>{tr.class_balance_ratio?.toFixed(2) || '-'}</div>
            <div style={{ fontSize: 12, color: '#64748b' }}>Balance Ratio</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>{tr.balance_status}</div>
          </div>
        </div>
      </Card>

      <Card title="Readiness Flags">
        {flags.map((f, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: SEVERITY_COLORS[f.severity] || '#6b7280',
              flexShrink: 0,
            }} />
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{f.flag}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>{f.detail}</div>
            </div>
          </div>
        ))}
      </Card>

      <Card title="Sequences Per Disease">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={perDisease} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" fontSize={11} />
            <YAxis type="category" dataKey="disease" fontSize={11} width={120} />
            <Tooltip />
            <Bar dataKey="sequences" fill="#8b5cf6" radius={[0,4,4,0]} name="Sequences" />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function renderSeizures(bd) {
  const events = bd?.seizure_temporal || []
  return (
    <Card title={`Seizure Temporal Events (${events.length})`}>
      <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>Patient</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Date</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Time</th>
              <th style={{ textAlign: 'right', padding: 8 }}>Duration (s)</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Severity</th>
              <th style={{ textAlign: 'left', padding: 8 }}>Trigger</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: 8, fontWeight: 500 }}>{e.patient_id}</td>
                <td style={{ padding: 8 }}>{e.event_date || '-'}</td>
                <td style={{ padding: 8, fontFamily: 'monospace', fontSize: 12 }}>{e.event_time || '-'}</td>
                <td style={{ padding: 8, textAlign: 'right', fontFamily: 'monospace' }}>{e.duration_sec || '-'}</td>
                <td style={{ padding: 8 }}>
                  <Badge text={e.severity || 'N/A'}
                    color={e.severity === 'Severe' ? '#ef4444' : e.severity === 'Moderate' ? '#f59e0b' : '#10b981'} />
                </td>
                <td style={{ padding: 8, fontSize: 12, color: '#64748b' }}>{e.trigger || '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
                <td style={{ padding: 6, fontFamily: 'monospace', fontSize: 11, color: '#64748b' }}>{e.ts_utc}</td>
                <td style={{ padding: 6 }}><Badge text={e.component} color="#6366f1" /></td>
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
    { key: 'concepts', title: 'RNN/LSTM Concepts', termKey: 'term', defKey: 'definition' },
    { key: 'quality_metrics', title: 'Quality Metrics', termKey: 'metric', defKey: 'description' },
    { key: 'model_variants', title: 'Model Variants', termKey: 'name', defKey: 'description' },
    { key: 'compliance', title: 'Compliance References', termKey: 'standard', defKey: 'reference' },
    { key: 'remediation', title: 'Remediation Strategies', termKey: 'issue', defKey: 'strategy' },
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
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{d[s.termKey]}</div>
                <div style={{ fontSize: 12, color: '#475569', marginTop: 4, lineHeight: 1.5 }}>{d[s.defKey]}</div>
                {d.target && <div style={{ fontSize: 11, color: '#3b82f6', marginTop: 2 }}>Target: {d.target}</div>}
                {d.params && <div style={{ fontSize: 11, color: '#8b5cf6', marginTop: 2 }}>Params: {d.params}</div>}
              </div>
            ))}
          </Card>
        )
      })}
    </div>
  )
}
