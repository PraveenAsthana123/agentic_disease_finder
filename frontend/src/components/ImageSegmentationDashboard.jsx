import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

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
  { id: 'segments', label: 'Segment Inventory' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ImageSegmentationDashboard() {
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
      axios.get(`${API}/api/image-segmentation/overview`),
      axios.get(`${API}/api/image-segmentation/breakdown`),
      axios.get(`${API}/api/image-segmentation/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Image Segmentation data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No segmentation data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Image Segmentation AI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        EEG trace digitization from images — segmentation tasks, ROI extraction, quality assessment
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

      {tab === 'overview' && <OverviewTab overview={overview} breakdown={breakdown} />}
      {tab === 'segments' && <SegmentsTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview, breakdown }) {
  const kpis = overview.kpis || []
  const segDist = overview.segment_class_distribution || []
  const qualDist = overview.quality_distribution || []
  const timeline = breakdown?.timeline || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
      ))}

      {/* Segment Class Distribution Pie */}
      <Card title="Segment Class Distribution" span={2}>
        {segDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={segDist} dataKey="count" nameKey="class" cx="50%" cy="50%" outerRadius={100}
                label={({ class: c, count }) => `${c} (${count})`}>
                {segDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No segment data</div>}
      </Card>

      {/* Quality Distribution Bar */}
      <Card title="Image Quality Distribution" span={2}>
        {qualDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={qualDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="quality" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No quality data</div>}
      </Card>

      {/* Segment Class Bar */}
      <Card title="Segments by Class" span={2}>
        {segDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={segDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      {/* Timeline */}
      <Card title="Segmentation Activity Timeline" span={2}>
        {timeline.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={timeline}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="events" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No timeline data</div>}
      </Card>
    </div>
  )
}

function SegmentsTab({ breakdown }) {
  const segments = breakdown?.segment_details || []
  const files = breakdown?.file_inventory || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Segmentation Results Table */}
      <Card title={`Segmentation Results (${segments.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Segment Class</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Quality</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Classification</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Location</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((s, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{s.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={s.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{s.segment_class || 'N/A'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={s.quality || 'N/A'} color={s.quality === 'Diagnostic' ? '#10b981' : s.quality === 'Adequate' ? '#f59e0b' : '#94a3b8'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={s.classification || 'N/A'} color={s.classification === 'LESIONAL' ? '#ef4444' : '#10b981'} />
                  </td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{s.location || 'N/A'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{s.created_at ? s.created_at.slice(0, 10) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* File Inventory */}
      <Card title={`EEG File Inventory (${files.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>File</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Disease</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Department</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Uploaded</th>
              </tr>
            </thead>
            <tbody>
              {files.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{f.file_name}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={f.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px' }}>{f.disease}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{f.department}</td>
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

function PatientsTab({ breakdown }) {
  const patients = breakdown?.patient_profiles || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {patients.map((p, i) => (
        <Card key={i} title={`Patient ${p.patient_id}`}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Images: </span><strong>{p.n_uploads || 0}</strong></div>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Segments: </span><strong>{p.n_segments || 0}</strong></div>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Analyses: </span><strong>{p.n_analyses || 0}</strong></div>
            {p.mean_confidence != null && (
              <div><span style={{ fontSize: 12, color: '#64748b' }}>Avg Confidence: </span><strong>{p.mean_confidence}</strong></div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {(p.segment_classes || []).map((cls, ci) => (
              <Badge key={ci} text={cls} color={COLORS[ci % COLORS.length]} />
            ))}
            {(p.qualities || []).map((q, qi) => (
              <Badge key={`q${qi}`} text={q} color={q === 'Diagnostic' ? '#10b981' : '#f59e0b'} />
            ))}
          </div>
        </Card>
      ))}
      {patients.length === 0 && <Card span={2}><div style={{ color: '#94a3b8', fontSize: 13 }}>No patient data</div></Card>}
    </div>
  )
}

function PipelineTab({ breakdown }) {
  const events = breakdown?.pipeline_events || []
  const analyses = breakdown?.analysis_outputs || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Analysis Outputs */}
      <Card title={`Segmentation Analysis Outputs (${analyses.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Prediction</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Confidence</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Signal Quality</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {analyses.map((a, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{a.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={a.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{a.predicted_label}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={a.confidence != null ? a.confidence.toFixed(2) : 'N/A'}
                      color={a.confidence >= 0.7 ? '#10b981' : a.confidence >= 0.5 ? '#f59e0b' : '#ef4444'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={a.signal_quality || 'N/A'} color={a.signal_quality === 'Good' ? '#10b981' : '#f59e0b'} />
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{a.created_at ? a.created_at.slice(0, 10) : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Pipeline Events */}
      <Card title={`Pipeline Events (${events.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
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
                  <td style={{ padding: '6px 10px' }}><Badge text={e.component} color="#8b5cf6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{e.action}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{e.actor || 'system'}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', maxWidth: 350, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail || ''}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{e.ts_utc || e.ts_local || ''}</td>
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
  const sections = definitions?.sections || []
  const compliance = definitions?.compliance || []
  const remediation = definitions?.remediation || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {sections.map((sec, si) => (
        <Card key={si} title={sec.title}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Term</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Description</th>
              </tr>
            </thead>
            <tbody>
              {(sec.items || []).map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600, whiteSpace: 'nowrap' }}>{d.term}</td>
                  <td style={{ padding: '8px 12px', color: '#64748b' }}>{d.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}

      {/* Compliance */}
      <Card title="Compliance References">
        <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.8 }}>
          <ul style={{ margin: 0, paddingLeft: 20 }}>
            {compliance.map((c, i) => (
              <li key={i}><strong>{c.standard}:</strong> {c.note}</li>
            ))}
          </ul>
        </div>
      </Card>

      {/* Remediation */}
      <Card title="Remediation Strategies">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
          {remediation.map((r, i) => (
            <div key={i} style={{ padding: 12, background: '#f8fafc', borderRadius: 8, fontSize: 13 }}>
              <div style={{ fontWeight: 600, marginBottom: 4, color: '#334155' }}>{r.strategy}</div>
              <div style={{ color: '#64748b', fontSize: 12 }}>{r.description}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
