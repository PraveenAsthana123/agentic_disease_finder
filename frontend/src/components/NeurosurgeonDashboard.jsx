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

const CLASSIFICATION_COLORS = {
  LESIONAL: '#ef4444',
  NON_LESIONAL: '#3b82f6',
  EQUIVOCAL: '#f59e0b',
  NORMAL: '#10b981',
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'mri', label: 'MRI Inventory' },
  { id: 'patients', label: 'Surgical Profiles' },
  { id: 'seizures', label: 'Seizure Log' },
  { id: 'eeg', label: 'EEG Summary' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function NeurosurgeonDashboard() {
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
      axios.get(`${API_URL}/api/neurosurgeon/overview`),
      axios.get(`${API_URL}/api/neurosurgeon/breakdown`),
      axios.get(`${API_URL}/api/neurosurgeon/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Neurosurgeon data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No neurosurgeon data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Neurosurgeon / Epilepsy Surgery</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Pre-surgical evaluation, MRI lesion classification, surgical candidacy assessment, seizure burden analysis, EEG concordance
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
      {tab === 'mri' && <MRITab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'seizures' && <SeizureTab breakdown={breakdown} />}
      {tab === 'eeg' && <EEGTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab overview={overview} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = overview.kpis || []
  const lesionDist = overview.lesion_type_distribution || []
  const latDist = overview.laterality_distribution || []
  const locDist = overview.lesion_location_distribution || []
  const sevDist = overview.seizure_severity_distribution || []
  const classDist = overview.mri_classification_distribution || []
  const dailyActivity = overview.daily_activity || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color || COLORS[i % COLORS.length]} /></Card>
      ))}

      {/* Lesion Type Distribution Pie */}
      <Card title="Lesion Type Distribution" span={2}>
        {lesionDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={lesionDist} dataKey="count" nameKey="label" cx="50%" cy="50%" outerRadius={100}
                label={({ label, count }) => `${label} (${count})`}>
                {lesionDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No lesion data</div>}
      </Card>

      {/* Laterality Distribution Bar */}
      <Card title="Laterality Distribution" span={2}>
        {latDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={latDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="laterality" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                {latDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No laterality data</div>}
      </Card>

      {/* Lesion Location Distribution Bar */}
      <Card title="Lesion Location Distribution" span={2}>
        {locDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={locDist} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="location" width={100} tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No location data</div>}
      </Card>

      {/* MRI Classification Distribution */}
      <Card title="MRI Classification" span={2}>
        {classDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={classDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="classification" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {classDist.map((entry, i) => (
                  <Cell key={i} fill={CLASSIFICATION_COLORS[entry.classification] || COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No classification data</div>}
      </Card>

      {/* Seizure Severity Distribution */}
      <Card title="Seizure Severity Distribution" span={2}>
        {sevDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={sevDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="severity" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {sevDist.map((entry, i) => {
                  const c = entry.severity === 'Severe' ? '#ef4444' : entry.severity === 'Moderate' ? '#f59e0b' : '#10b981'
                  return <Cell key={i} fill={c} />
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No seizure data</div>}
      </Card>

      {/* Daily Activity Line */}
      <Card title="Daily Pipeline Activity" span={2}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No activity data</div>}
      </Card>
    </div>
  )
}

function MRITab({ breakdown }) {
  const inventory = breakdown?.mri_inventory || []
  const [filter, setFilter] = useState('')
  const filtered = filter
    ? inventory.filter(r => (r.patient_id || '').toLowerCase().includes(filter.toLowerCase()) ||
        (r.lesion_label || '').toLowerCase().includes(filter.toLowerCase()))
    : inventory

  return (
    <Card title={`MRI Inventory (${filtered.length} scans)`}>
      <input
        placeholder="Filter by patient or lesion..."
        value={filter} onChange={e => setFilter(e.target.value)}
        style={{ width: '100%', padding: '8px 12px', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 13, marginBottom: 12 }}
      />
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: '8px 6px' }}>Patient</th>
              <th style={{ padding: '8px 6px' }}>Lesion Type</th>
              <th style={{ padding: '8px 6px' }}>Location</th>
              <th style={{ padding: '8px 6px' }}>Laterality</th>
              <th style={{ padding: '8px 6px' }}>Classification</th>
              <th style={{ padding: '8px 6px' }}>HS</th>
              <th style={{ padding: '8px 6px' }}>Vol Asym</th>
              <th style={{ padding: '8px 6px' }}>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}><strong>{r.patient_id}</strong></td>
                <td style={{ padding: '6px' }}>
                  <Badge text={r.lesion_label || r.lesion_type || '—'} color={r.classification === 'LESIONAL' ? '#ef4444' : '#3b82f6'} />
                </td>
                <td style={{ padding: '6px' }}>{r.lesion_location || '—'}</td>
                <td style={{ padding: '6px' }}>{r.laterality || '—'}</td>
                <td style={{ padding: '6px' }}>
                  <Badge text={r.classification || '—'} color={CLASSIFICATION_COLORS[r.classification] || '#64748b'} />
                </td>
                <td style={{ padding: '6px' }}>{r.hippocampal_sclerosis || '—'}</td>
                <td style={{ padding: '6px' }}>{r.hippocampal_volume_asymmetry != null ? r.hippocampal_volume_asymmetry.toFixed(2) : '—'}</td>
                <td style={{ padding: '6px' }}>{r.radiologist_confidence || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function PatientsTab({ breakdown }) {
  const profiles = breakdown?.patient_surgical_profiles || []
  return (
    <Card title={`Patient Surgical Profiles (${profiles.length} patients)`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: '8px 6px' }}>Patient</th>
              <th style={{ padding: '8px 6px' }}>MRI Count</th>
              <th style={{ padding: '8px 6px' }}>Lesion Types</th>
              <th style={{ padding: '8px 6px' }}>Locations</th>
              <th style={{ padding: '8px 6px' }}>Laterality</th>
              <th style={{ padding: '8px 6px' }}>HS</th>
              <th style={{ padding: '8px 6px' }}>Surgical Candidate</th>
              <th style={{ padding: '8px 6px' }}>Seizures</th>
            </tr>
          </thead>
          <tbody>
            {profiles.map((p, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}><strong>{p.patient_id}</strong></td>
                <td style={{ padding: '6px' }}>{p.mri_count}</td>
                <td style={{ padding: '6px' }}>
                  {(p.lesion_types || []).map((lt, j) => <Badge key={j} text={lt} color={COLORS[j % COLORS.length]} />)}
                </td>
                <td style={{ padding: '6px' }}>{(p.locations || []).join(', ') || '—'}</td>
                <td style={{ padding: '6px' }}>{(p.lateralities || []).join(', ') || '—'}</td>
                <td style={{ padding: '6px' }}>{p.has_hippocampal_sclerosis ? 'Yes' : 'No'}</td>
                <td style={{ padding: '6px' }}>
                  <Badge text={p.surgical_candidate ? 'Yes' : 'No'} color={p.surgical_candidate ? '#10b981' : '#94a3b8'} />
                </td>
                <td style={{ padding: '6px' }}>
                  {p.seizure_count || 0}
                  {p.seizure_severities && p.seizure_severities.length > 0 && (
                    <span style={{ fontSize: 10, color: '#94a3b8', marginLeft: 4 }}>
                      ({p.seizure_severities.join(', ')})
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function SeizureTab({ breakdown }) {
  const log = breakdown?.seizure_log || []
  return (
    <Card title={`Seizure Log (${log.length} events)`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: '8px 6px' }}>Patient</th>
              <th style={{ padding: '8px 6px' }}>Date</th>
              <th style={{ padding: '8px 6px' }}>Duration (s)</th>
              <th style={{ padding: '8px 6px' }}>Severity</th>
              <th style={{ padding: '8px 6px' }}>Location</th>
              <th style={{ padding: '8px 6px' }}>Motor Signs</th>
              <th style={{ padding: '8px 6px' }}>Injury</th>
              <th style={{ padding: '8px 6px' }}>Aura</th>
              <th style={{ padding: '8px 6px' }}>Trigger</th>
            </tr>
          </thead>
          <tbody>
            {log.map((s, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}><strong>{s.patient_id}</strong></td>
                <td style={{ padding: '6px' }}>{s.event_date || '—'}</td>
                <td style={{ padding: '6px' }}>{s.duration_sec || '—'}</td>
                <td style={{ padding: '6px' }}>
                  {s.severity ? <Badge text={s.severity}
                    color={s.severity === 'Severe' ? '#ef4444' : s.severity === 'Moderate' ? '#f59e0b' : '#10b981'} /> : '—'}
                </td>
                <td style={{ padding: '6px' }}>{s.location || '—'}</td>
                <td style={{ padding: '6px' }}>{s.motor_signs || '—'}</td>
                <td style={{ padding: '6px' }}>{s.injury || '—'}</td>
                <td style={{ padding: '6px' }}>{s.aura || '—'}</td>
                <td style={{ padding: '6px' }}>{s.trigger || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function EEGTab({ breakdown }) {
  const eeg = breakdown?.eeg_summary || []
  return (
    <Card title={`EEG Analysis Summary (${eeg.length} analyses)`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: '8px 6px' }}>Patient</th>
              <th style={{ padding: '8px 6px' }}>Disease</th>
              <th style={{ padding: '8px 6px' }}>Prediction</th>
              <th style={{ padding: '8px 6px' }}>Confidence</th>
              <th style={{ padding: '8px 6px' }}>Signal Quality</th>
              <th style={{ padding: '8px 6px' }}>Date</th>
            </tr>
          </thead>
          <tbody>
            {eeg.map((e, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px' }}><strong>{e.patient_id}</strong></td>
                <td style={{ padding: '6px' }}>{e.disease || '—'}</td>
                <td style={{ padding: '6px' }}>
                  <Badge text={e.predicted_label || '—'} color="#3b82f6" />
                </td>
                <td style={{ padding: '6px' }}>
                  {e.confidence != null ? `${(e.confidence * 100).toFixed(0)}%` : '—'}
                </td>
                <td style={{ padding: '6px' }}>
                  <Badge text={e.signal_quality || '—'}
                    color={e.signal_quality === 'Good' ? '#10b981' : e.signal_quality === 'Fair' ? '#f59e0b' : '#ef4444'} />
                </td>
                <td style={{ padding: '6px' }}>{e.created_at ? e.created_at.slice(0, 10) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function PipelineTab({ overview }) {
  const events = overview?.pipeline_events || []
  return (
    <Card title={`Pipeline Events (last ${events.length})`}>
      <div style={{ maxHeight: 500, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
              <th style={{ padding: '8px 6px' }}>Time</th>
              <th style={{ padding: '8px 6px' }}>Action</th>
              <th style={{ padding: '8px 6px' }}>Entity</th>
              <th style={{ padding: '8px 6px' }}>Detail</th>
            </tr>
          </thead>
          <tbody>
            {events.map((e, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', whiteSpace: 'nowrap' }}>{e.created_at || e.ts || '—'}</td>
                <td style={{ padding: '6px' }}><Badge text={e.action || '—'} color="#3b82f6" /></td>
                <td style={{ padding: '6px' }}>{e.entity_type || '—'}</td>
                <td style={{ padding: '6px', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}>{e.detail || e.details || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div>
  const concepts = definitions.concepts || []
  const metrics = definitions.quality_metrics || []
  const procedures = definitions.surgical_procedures || []
  const refs = definitions.compliance_refs || []
  const remediation = definitions.remediation_strategies || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Neurosurgical Concepts">
        {concepts.map((c, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{c.name || c.title}</div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{c.description || c.text}</div>
          </div>
        ))}
      </Card>

      <Card title="Surgical Procedures">
        {procedures.map((p, i) => (
          <div key={i} style={{ marginBottom: 14, paddingBottom: 12, borderBottom: '1px solid #f1f5f9' }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{p.name || p.title}</div>
            <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{p.description || p.text}</div>
          </div>
        ))}
      </Card>

      <Card title="Quality Metrics">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {metrics.map((m, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: '#334155' }}>{m.name || m.metric}</div>
              <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{m.description || m.target || m.text}</div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Compliance References">
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          {refs.map((r, i) => (
            <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
              <strong>{r.name || r.ref}</strong>: {r.description || r.scope || r.text}
            </li>
          ))}
        </ul>
      </Card>

      <Card title="Remediation Strategies">
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          {remediation.map((r, i) => (
            <li key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>
              <strong>{r.name || r.strategy}</strong>: {r.description || r.text}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
