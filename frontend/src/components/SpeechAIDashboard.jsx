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

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'transcriptions', label: 'Transcription Inventory' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function SpeechAIDashboard() {
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
      axios.get(`${API_URL}/api/speech-ai/overview`),
      axios.get(`${API_URL}/api/speech-ai/breakdown`),
      axios.get(`${API_URL}/api/speech-ai/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Speech AI data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No speech AI data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Speech AI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Clinician-AI conversation transcription, speech analytics, and pipeline health monitoring
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
      {tab === 'transcriptions' && <TranscriptionsTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const kpis = [
    { label: 'Total Transcriptions', value: overview.total_transcriptions ?? 0, color: '#3b82f6' },
    { label: 'Clinician Utterances', value: overview.clinician_utterances ?? 0, color: '#10b981' },
    { label: 'AI Responses', value: overview.ai_responses ?? 0, color: '#8b5cf6' },
    { label: 'Audio Files', value: overview.total_audio_files ?? 0, color: '#f59e0b' },
    { label: 'Patients Covered', value: overview.patients_covered ?? 0, color: '#ec4899' },
    { label: 'Mean Confidence', value: overview.mean_confidence != null ? overview.mean_confidence.toFixed(3) : 'N/A', color: '#06b6d4' },
    { label: 'Pipeline Events', value: overview.pipeline_events ?? 0, color: '#ef4444' },
    { label: 'Mean Transcript Length', value: overview.mean_transcript_length != null ? overview.mean_transcript_length.toFixed(0) : 'N/A', color: '#f97316' },
  ]

  const interactionDist = overview.interaction_type_distribution || []
  const lengthDist = overview.transcript_length_distribution || []
  const dailyActivity = overview.daily_activity || []
  const pipelineHealth = overview.speech_pipeline_health || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
      ))}

      {/* Interaction Type Distribution Pie */}
      <Card title="Interaction Type Distribution" span={2}>
        {interactionDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={interactionDist} dataKey="count" nameKey="label" cx="50%" cy="50%" outerRadius={100}
                label={({ label, count }) => `${label} (${count})`}>
                {interactionDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No interaction data</div>}
      </Card>

      {/* Transcript Length Distribution Bar */}
      <Card title="Transcript Length Distribution" span={2}>
        {lengthDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={lengthDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="tier" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No transcript length data</div>}
      </Card>

      {/* Daily Activity Line Chart */}
      <Card title="Daily Activity" span={2}>
        {dailyActivity.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={dailyActivity}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No daily activity data</div>}
      </Card>

      {/* Pipeline Health Bar */}
      <Card title="Speech Pipeline Health" span={2}>
        {pipelineHealth.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={pipelineHealth} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="component" type="category" tick={{ fontSize: 10 }} width={120} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No pipeline health data</div>}
      </Card>
    </div>
  )
}

function TranscriptionsTab({ breakdown }) {
  const inventory = breakdown?.transcription_inventory || []
  const roleStats = breakdown?.role_stats || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Role Stats */}
      <Card title={`Role Statistics (${roleStats.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Role</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Label</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Count</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Mean Length</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Total Chars</th>
              </tr>
            </thead>
            <tbody>
              {roleStats.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}><Badge text={r.role} color={COLORS[i % COLORS.length]} /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.label}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.mean_length != null ? r.mean_length.toFixed(0) : 'N/A'}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{r.total_chars != null ? r.total_chars.toLocaleString() : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Full Transcription Inventory */}
      <Card title={`Transcription Inventory (${inventory.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Role</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Interaction Type</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Text Preview</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Length</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Length Tier</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {inventory.map((t, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{t.id}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={t.role} color={t.role === 'clinician' ? '#10b981' : t.role === 'ai' ? '#8b5cf6' : '#3b82f6'} />
                  </td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{t.interaction_type || 'N/A'}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 350, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {t.text ? (t.text.length > 80 ? t.text.slice(0, 80) + '...' : t.text) : ''}
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{t.length}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={t.length_tier || 'N/A'} color={
                      t.length_tier === 'long' ? '#ef4444' : t.length_tier === 'medium' ? '#f59e0b' : '#10b981'
                    } />
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{t.ts_local || t.ts_utc || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.patient_profiles || []
  const audioFiles = breakdown?.audio_files || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Patient Profile Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
        {patients.map((p, i) => (
          <Card key={i} title={`Patient ${p.patient_id}`}>
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
              <div><span style={{ fontSize: 12, color: '#64748b' }}>Uploads: </span><strong>{p.n_uploads || 0}</strong></div>
              <div><span style={{ fontSize: 12, color: '#64748b' }}>Analyses: </span><strong>{p.n_analyses || 0}</strong></div>
              {p.mean_confidence != null && (
                <div><span style={{ fontSize: 12, color: '#64748b' }}>Avg Confidence: </span>
                  <Badge text={p.mean_confidence.toFixed(3)} color={p.mean_confidence >= 0.7 ? '#10b981' : '#f59e0b'} />
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {(Array.isArray(p.diseases) ? p.diseases : (p.diseases ? [p.diseases] : [])).map((d, di) => (
                <Badge key={di} text={d} color={COLORS[di % COLORS.length]} />
              ))}
            </div>
          </Card>
        ))}
        {patients.length === 0 && <Card span={2}><div style={{ color: '#94a3b8', fontSize: 13 }}>No patient data</div></Card>}
      </div>

      {/* Audio Files Inventory */}
      <Card title={`Audio Files (${audioFiles.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>File Name</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Disease</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Created</th>
              </tr>
            </thead>
            <tbody>
              {audioFiles.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{f.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={f.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{f.file_name}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={f.disease || 'N/A'} color="#8b5cf6" />
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{f.created_at ? f.created_at.slice(0, 10) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function PipelineTab({ breakdown }) {
  const events = breakdown?.pipeline_events || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title={`Speech Pipeline Events (${events.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Component</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Action</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Actor</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Detail</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{e.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={e.component} color="#8b5cf6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{e.action}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{e.actor || 'system'}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 350, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || ''}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{e.ts_local || e.ts_utc || ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function DefinitionsTab({ definitions }) {
  if (!definitions) return <Card><div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div></Card>

  const concepts = definitions.concepts || []
  const qualityMetrics = definitions.quality_metrics || []
  const compliance = definitions.compliance || []
  const remediation = definitions.remediation || []
  const clinicalRelevance = definitions.clinical_relevance || ''

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Speech AI Concepts */}
      <Card title={`Speech AI Concepts (${concepts.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {concepts.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Quality Metrics */}
      <Card title={`Quality Metrics (${qualityMetrics.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Metric</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {qualityMetrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{m.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{m.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Clinical Relevance */}
      {clinicalRelevance && (
        <Card title="Clinical Relevance">
          <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.6, margin: 0 }}>{clinicalRelevance}</p>
        </Card>
      )}

      {/* Compliance References */}
      <Card title={`Compliance References (${compliance.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Standard</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Reference</th>
            </tr>
          </thead>
          <tbody>
            {compliance.map((c, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{c.name}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{c.reference}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      {/* Remediation Strategies */}
      <Card title={`Remediation Strategies (${remediation.length})`}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Strategy</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
            </tr>
          </thead>
          <tbody>
            {remediation.map((r, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{r.strategy}</td>
                <td style={{ padding: '8px 12px', color: '#64748b' }}>{r.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
