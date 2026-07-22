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
  { id: 'detections', label: 'Detection Inventory' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'pipeline', label: 'Pipeline Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function ObjectDetectionDashboard() {
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
      axios.get(`${API}/api/object-detection/overview`),
      axios.get(`${API}/api/object-detection/breakdown`),
      axios.get(`${API}/api/object-detection/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Object Detection data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No object detection data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Object Detection AI Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Body-movement and lesion detection from video-EEG and MRI — detection classes, confidence, IoU scores
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
      {tab === 'detections' && <DetectionsTab breakdown={breakdown} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'pipeline' && <PipelineTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

function OverviewTab({ overview, breakdown }) {
  const kpis = overview.kpis || []
  const classDist = Object.entries(overview.class_distribution || {}).map(([name, count]) => ({ name, count }))
  const confDist = Object.entries(overview.confidence_distribution || {}).map(([tier, count]) => ({ tier, count }))
  const iouDist = Object.entries(overview.iou_distribution || {}).map(([quality, count]) => ({ quality, count }))
  const locDist = Object.entries(overview.location_distribution || {}).map(([region, count]) => ({ region, count }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} color={k.color} /></Card>
      ))}

      {/* Detection Class Distribution Pie */}
      <Card title="Detection Class Distribution" span={2}>
        {classDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={classDist} dataKey="count" nameKey="name" cx="50%" cy="50%" outerRadius={100}
                label={({ name, count }) => `${name} (${count})`}>
                {classDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No detection data</div>}
      </Card>

      {/* Confidence Distribution Bar */}
      <Card title="Confidence Tier Distribution" span={2}>
        {confDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={confDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="tier" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No confidence data</div>}
      </Card>

      {/* IoU/Quality Distribution */}
      <Card title="IoU Quality Distribution" span={2}>
        {iouDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={iouDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="quality" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No data</div>}
      </Card>

      {/* Location Heatmap Bar */}
      <Card title="Detection by Brain Region" span={2}>
        {locDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={locDist.slice(0, 10)} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis dataKey="region" type="category" tick={{ fontSize: 10 }} width={120} />
              <Tooltip />
              <Bar dataKey="count" fill="#ef4444" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No location data</div>}
      </Card>
    </div>
  )
}

function DetectionsTab({ breakdown }) {
  const inventory = breakdown?.detection_inventory || []
  const classBreakdown = breakdown?.class_breakdown || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Class Breakdown Stats */}
      <Card title={`Detection Classes (${classBreakdown.length})`}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Class</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Code</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Count</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>%</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Avg Confidence</th>
                <th style={{ padding: '8px 10px', textAlign: 'right', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Avg IoU</th>
              </tr>
            </thead>
            <tbody>
              {classBreakdown.map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{c.class}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={c.code} color={COLORS[i % COLORS.length]} /></td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{c.count}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{c.pct}%</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    <Badge text={c.mean_confidence.toFixed(2)} color={c.mean_confidence >= 0.7 ? '#10b981' : '#f59e0b'} />
                  </td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>
                    <Badge text={c.mean_iou.toFixed(2)} color={c.mean_iou >= 0.7 ? '#10b981' : '#f59e0b'} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Full Detection Inventory */}
      <Card title={`Detection Inventory (${inventory.length})`}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>ID</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Patient</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Detection Class</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Region</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Confidence</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>IoU</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Classification</th>
                <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {inventory.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={d.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.detection_class}</td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{d.region || 'N/A'}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={d.det_confidence.toFixed(2)} color={d.det_confidence >= 0.7 ? '#10b981' : d.det_confidence >= 0.5 ? '#f59e0b' : '#ef4444'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={d.iou_score.toFixed(2)} color={d.iou_score >= 0.7 ? '#10b981' : d.iou_score >= 0.5 ? '#f59e0b' : '#ef4444'} />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge text={d.classification || 'N/A'} color={d.classification === 'LESIONAL' ? '#ef4444' : '#10b981'} />
                  </td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{d.created_at ? d.created_at.slice(0, 10) : ''}</td>
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
  const patients = breakdown?.per_patient || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {patients.map((p, i) => (
        <Card key={i} title={`Patient ${p.patient_id}`}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Frames: </span><strong>{p.n_frames || 0}</strong></div>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Detections: </span><strong>{p.n_detections || 0}</strong></div>
            <div><span style={{ fontSize: 12, color: '#64748b' }}>Analyses: </span><strong>{p.n_analyses || 0}</strong></div>
            {p.mean_confidence != null && (
              <div><span style={{ fontSize: 12, color: '#64748b' }}>Avg Det. Conf: </span><strong>{p.mean_confidence.toFixed(2)}</strong></div>
            )}
            {p.mean_model_conf != null && p.mean_model_conf > 0 && (
              <div><span style={{ fontSize: 12, color: '#64748b' }}>Model Conf: </span><strong>{p.mean_model_conf.toFixed(3)}</strong></div>
            )}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
            {(p.classes || []).map((cls, ci) => (
              <Badge key={ci} text={cls} color={COLORS[ci % COLORS.length]} />
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {(p.regions || []).map((r, ri) => (
              <Badge key={ri} text={r} color="#64748b" />
            ))}
          </div>
        </Card>
      ))}
      {patients.length === 0 && <Card span={2}><div style={{ color: '#94a3b8', fontSize: 13 }}>No patient data</div></Card>}
    </div>
  )
}

function PipelineTab({ breakdown }) {
  const events = breakdown?.pipeline_log || []
  const analyses = breakdown?.analysis_table || []
  const actionChart = breakdown?.pipeline_action_chart || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {/* Pipeline Action Distribution */}
      <Card title="Pipeline Actions by Type">
        {actionChart.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={actionChart}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="action" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No pipeline data</div>}
      </Card>

      {/* Detection Analysis Outputs */}
      <Card title={`Detection Analysis Outputs (${analyses.length})`}>
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
      <Card title={`CV Pipeline Events (${events.length})`}>
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
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{e.ts || ''}</td>
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
                  <td style={{ padding: '8px 12px', color: '#64748b' }}>{d.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      ))}
    </div>
  )
}
