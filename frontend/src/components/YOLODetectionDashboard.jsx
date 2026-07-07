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
  { id: 'detections', label: 'Detections' },
  { id: 'models', label: 'Models' },
  { id: 'patients', label: 'Patients' },
  { id: 'definitions', label: 'Definitions' },
]

export default function YOLODetectionDashboard() {
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
      axios.get(`${API_URL}/api/yolo-detection/overview`),
      axios.get(`${API_URL}/api/yolo-detection/breakdown`),
      axios.get(`${API_URL}/api/yolo-detection/definitions`),
    ])
      .then(([ov, bd, df]) => {
        setOverview(ov.data)
        setBreakdown(bd.data)
        setDefinitions(df.data)
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading YOLO Detection data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#94a3b8' }}>No YOLO detection data available</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>YOLO Object/Movement Detection Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        Seizure-related object and movement detection in video-EEG frames — YOLO model variants, confidence, IoU, and per-patient profiles
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

      {tab === 'overview'    && <OverviewTab overview={overview} breakdown={breakdown} />}
      {tab === 'detections'  && <DetectionsTab breakdown={breakdown} />}
      {tab === 'models'      && <ModelsTab breakdown={breakdown} />}
      {tab === 'patients'    && <PatientsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab definitions={definitions} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Overview
// ---------------------------------------------------------------------------
function OverviewTab({ overview, breakdown }) {
  const kpis = overview.kpis || []
  const classDist = Object.entries(overview.class_distribution || {}).map(([name, count]) => ({ name, count }))
  const modelComparison = overview.model_comparison || []
  const modeReadiness = overview.detection_mode_readiness || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
      {/* KPI row */}
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
      ))}

      {/* Detection Class Distribution */}
      <Card title="Detection Class Distribution" span={3}>
        {classDist.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={classDist}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {classDist.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No class distribution data</div>}
      </Card>

      {/* Model Comparison mAP */}
      <Card title="Model mAP Comparison" span={2}>
        {modelComparison.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={modelComparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="mAP" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model comparison data</div>}
      </Card>

      {/* Detection Mode Readiness */}
      <Card title="Detection Mode Readiness" span={5}>
        {modeReadiness.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
            {modeReadiness.map((mode, i) => (
              <div key={i} style={{
                background: '#f8fafc', borderRadius: 8, padding: 14,
                borderLeft: `4px solid ${mode.ready ? '#10b981' : '#f59e0b'}`
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 4 }}>{mode.name}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>{mode.description}</div>
                <Badge text={mode.ready ? 'Ready' : 'Pending'} color={mode.ready ? '#10b981' : '#f59e0b'} />
              </div>
            ))}
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
            {['Real-Time Frame Inference', 'Seizure Onset Detection', 'Multi-Class Body Movement', 'EEG-Synchronized Video', 'Clinical Report Export'].map((name, i) => (
              <div key={i} style={{
                background: '#f8fafc', borderRadius: 8, padding: 14,
                borderLeft: `4px solid ${i < 3 ? '#10b981' : '#f59e0b'}`
              }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 6 }}>{name}</div>
                <Badge text={i < 3 ? 'Ready' : 'Pending'} color={i < 3 ? '#10b981' : '#f59e0b'} />
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Detections
// ---------------------------------------------------------------------------
function DetectionsTab({ breakdown }) {
  const kpis = breakdown?.detection_kpis || []
  const perClassCounts = breakdown?.per_class_counts || []
  const confHistogram = breakdown?.confidence_histogram || []
  const iouByClass = breakdown?.iou_by_class || []
  const topDetections = breakdown?.top_detections || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      {/* KPIs */}
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
      ))}

      {/* Per-class Detection Counts */}
      <Card title="Per-Class Detection Counts" span={3}>
        {perClassCounts.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perClassCounts}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class_name" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {perClassCounts.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No per-class data</div>}
      </Card>

      {/* Confidence Distribution Histogram */}
      <Card title="Confidence Distribution" span={2}>
        {confHistogram.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={confHistogram}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No confidence histogram data</div>}
      </Card>

      {/* IoU by Class */}
      <Card title="Mean IoU by Class" span={1}>
        {iouByClass.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={iouByClass} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, 1]} />
              <YAxis dataKey="class_name" type="category" tick={{ fontSize: 10 }} width={110} />
              <Tooltip />
              <Bar dataKey="mean_iou" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No IoU data</div>}
      </Card>

      {/* Top Detections Table */}
      <Card title={`Top Detections (${topDetections.length})`} span={3}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Frame ID', 'Patient', 'Class', 'Confidence', 'IoU', 'Model', 'Timestamp'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {topDetections.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{d.frame_id}</td>
                  <td style={{ padding: '6px 10px' }}><Badge text={d.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.class_name}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge
                      text={typeof d.confidence === 'number' ? d.confidence.toFixed(2) : d.confidence}
                      color={d.confidence >= 0.7 ? '#10b981' : d.confidence >= 0.5 ? '#f59e0b' : '#ef4444'}
                    />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge
                      text={typeof d.iou === 'number' ? d.iou.toFixed(2) : d.iou}
                      color={d.iou >= 0.7 ? '#10b981' : d.iou >= 0.5 ? '#f59e0b' : '#ef4444'}
                    />
                  </td>
                  <td style={{ padding: '6px 10px' }}><Badge text={d.model || 'N/A'} color="#8b5cf6" /></td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#94a3b8' }}>{d.timestamp ? d.timestamp.slice(0, 16) : ''}</td>
                </tr>
              ))}
              {topDetections.length === 0 && (
                <tr><td colSpan={7} style={{ padding: 16, textAlign: 'center', color: '#94a3b8' }}>No detections found</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Models
// ---------------------------------------------------------------------------
function ModelsTab({ breakdown }) {
  const kpis = breakdown?.model_kpis || []
  const models = breakdown?.model_architectures || []

  // Build chart data from model architectures
  const perfChartData = models.map(m => ({
    model: m.model,
    mAP: typeof m.mAP === 'number' ? m.mAP : parseFloat(m.mAP) || 0,
    inference_ms: typeof m.inference_ms === 'number' ? m.inference_ms : parseFloat(m.inference_ms) || 0,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      {/* KPIs */}
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
      ))}

      {/* Model Architecture Comparison Table */}
      <Card title="Model Architecture Comparison" span={3}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Model', 'Params (M)', 'GFLOPs', 'mAP', 'Inference (ms)', 'Suitable For'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 700, color: COLORS[i % COLORS.length] }}>{m.model}</td>
                  <td style={{ padding: '6px 10px' }}>{m.params}</td>
                  <td style={{ padding: '6px 10px' }}>{m.GFLOPs}</td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge
                      text={typeof m.mAP === 'number' ? m.mAP.toFixed(3) : m.mAP}
                      color={m.mAP >= 0.7 ? '#10b981' : m.mAP >= 0.5 ? '#f59e0b' : '#ef4444'}
                    />
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge
                      text={`${m.inference_ms} ms`}
                      color={m.inference_ms <= 20 ? '#10b981' : m.inference_ms <= 50 ? '#f59e0b' : '#ef4444'}
                    />
                  </td>
                  <td style={{ padding: '6px 10px', color: '#64748b' }}>{m.suitable_for}</td>
                </tr>
              ))}
              {models.length === 0 && (
                <tr><td colSpan={6} style={{ padding: 16, textAlign: 'center', color: '#94a3b8' }}>No model data available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* mAP Bar Chart */}
      <Card title="mAP by Model" span={2}>
        {perfChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perfChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="mAP" radius={[4, 4, 0, 0]}>
                {perfChartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No model performance data</div>}
      </Card>

      {/* Inference Speed Bar Chart */}
      <Card title="Inference Speed by Model (ms)" span={1}>
        {perfChartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={perfChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="inference_ms" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No inference data</div>}
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Patients
// ---------------------------------------------------------------------------
function PatientsTab({ breakdown }) {
  const kpis = breakdown?.patient_kpis || []
  const patients = breakdown?.patient_profiles || []

  const chartData = patients.map(p => ({
    patient_id: p.patient_id,
    total_detections: p.total_detections || 0,
  }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      {/* KPIs */}
      {kpis.map((k, i) => (
        <Card key={i}><KPI label={k.label} value={k.value} sub={k.sub} color={k.color} /></Card>
      ))}

      {/* Per-Patient Detection Bar Chart */}
      <Card title="Detections per Patient" span={3}>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="patient_id" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="total_detections" radius={[4, 4, 0, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : <div style={{ color: '#94a3b8', fontSize: 13 }}>No patient detection data</div>}
      </Card>

      {/* Per-Patient Detection Profiles Table */}
      <Card title={`Patient Detection Profiles (${patients.length})`} span={3}>
        <div style={{ overflowX: 'auto', maxHeight: 500, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead style={{ position: 'sticky', top: 0, background: '#fff' }}>
              <tr style={{ background: '#f8fafc' }}>
                {['Patient ID', 'Video Recordings', 'Total Detections', 'Dominant Class', 'Detection Rate'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px' }}><Badge text={p.patient_id} color="#3b82f6" /></td>
                  <td style={{ padding: '6px 10px', textAlign: 'right' }}>{p.video_recordings ?? 0}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'right', fontWeight: 600 }}>{p.total_detections ?? 0}</td>
                  <td style={{ padding: '6px 10px' }}>
                    {p.dominant_class
                      ? <Badge text={p.dominant_class} color={COLORS[i % COLORS.length]} />
                      : <span style={{ color: '#94a3b8' }}>N/A</span>}
                  </td>
                  <td style={{ padding: '6px 10px' }}>
                    <Badge
                      text={p.detection_rate != null ? (typeof p.detection_rate === 'number' ? `${(p.detection_rate * 100).toFixed(1)}%` : p.detection_rate) : 'N/A'}
                      color={p.detection_rate >= 0.7 ? '#10b981' : p.detection_rate >= 0.4 ? '#f59e0b' : '#ef4444'}
                    />
                  </td>
                </tr>
              ))}
              {patients.length === 0 && (
                <tr><td colSpan={5} style={{ padding: 16, textAlign: 'center', color: '#94a3b8' }}>No patient profiles available</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tab: Definitions
// ---------------------------------------------------------------------------
function DefinitionsTab({ definitions }) {
  const sections = definitions?.sections || []

  // Fallback: flat terms array grouped into one section
  const flatTerms = definitions?.terms || []
  const grouped = {}
  flatTerms.forEach(t => {
    const cat = t.category || 'General'
    if (!grouped[cat]) grouped[cat] = []
    grouped[cat].push(t)
  })
  const fallbackSections = Object.entries(grouped).map(([title, items]) => ({ title, items }))
  const allSections = sections.length > 0 ? sections : fallbackSections

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      {allSections.map((sec, si) => (
        <Card key={si} title={sec.title}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#64748b', width: 220 }}>Term</th>
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
      {allSections.length === 0 && (
        <Card>
          <div style={{ color: '#94a3b8', fontSize: 13 }}>No definitions available</div>
        </Card>
      )}
    </div>
  )
}
