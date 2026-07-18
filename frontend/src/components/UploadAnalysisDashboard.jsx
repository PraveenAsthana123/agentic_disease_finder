import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function QualityBadge({ quality }) {
  const colors = { Excellent: '#10b981', Good: '#3b82f6', Fair: '#f59e0b', Poor: '#ef4444' }
  const c = colors[quality] || '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: c + '22', color: c, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{quality || '--'}</span>
  )
}

function ConfidenceBadge({ confidence }) {
  const v = parseFloat(confidence)
  const c = v >= 0.8 ? '#10b981' : v >= 0.6 ? '#f59e0b' : '#ef4444'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: c + '22', color: c, fontWeight: 600, fontSize: 12
    }}>{v ? v.toFixed(3) : '--'}</span>
  )
}

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

export default function UploadAnalysisDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, br, df] = await Promise.all([
          axios.get(`${API_URL}/upload-analysis/overview`),
          axios.get(`${API_URL}/upload-analysis/breakdown`),
          axios.get(`${API_URL}/upload-analysis/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) { setError(e.message) }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Upload & Analysis data...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>No data available</div>

  const tabs = ['overview', 'patients', 'alerts', 'definitions']

  return (
    <div style={{ padding: 24, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', background: '#f1f5f9', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Upload & Analysis Tracker</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>EEG file upload pipeline — predictions, confidence, signal quality, completion tracking</p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#64748b'
          }}>{t.charAt(0).toUpperCase() + t.slice(1)}</button>
        ))}
      </div>

      {tab === 'overview' && <OverviewTab overview={overview} />}
      {tab === 'patients' && <PatientsTab breakdown={breakdown} />}
      {tab === 'alerts' && <AlertsTab breakdown={breakdown} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  )
}

function OverviewTab({ overview }) {
  const k = overview.kpis || {}
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
      <Card title="Key Metrics" span={3}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 12 }}>
          <KPI label="Total Uploads" value={k.total_uploads} color="#3b82f6" />
          <KPI label="Analyses Done" value={k.total_analyses} color="#10b981" />
          <KPI label="Patients" value={k.distinct_patients} color="#8b5cf6" />
          <KPI label="Completion Rate" value={k.completion_rate} sub="%" color="#10b981" />
          <KPI label="Avg Confidence" value={k.avg_confidence} color="#f59e0b" />
          <KPI label="Poor Signal Rate" value={k.poor_signal_rate} sub="%" color="#ef4444" />
        </div>
      </Card>

      <Card title="Disease Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={overview.disease_distribution || []} dataKey="count" nameKey="disease" cx="50%" cy="50%" outerRadius={80} label={({ disease, count }) => `${disease} (${count})`}>
              {(overview.disease_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Signal Quality">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.signal_quality_distribution || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="signal_quality" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Confidence Buckets">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.confidence_buckets || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Prediction Labels" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={overview.label_distribution || []} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis type="category" dataKey="predicted_label" tick={{ fontSize: 11 }} width={160} />
            <Tooltip />
            <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Daily Upload Trend">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={overview.daily_upload_trend || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="day" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line type="monotone" dataKey="uploads" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function PatientsTab({ breakdown }) {
  const patients = breakdown?.patient_summary || []
  const recent = breakdown?.recent_uploads || []
  const deptWork = breakdown?.department_workload || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Department Workload">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Department</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Uploads</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Patients</th>
              </tr>
            </thead>
            <tbody>
              {deptWork.map((d, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 500 }}>{d.department}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.uploads}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{d.patients}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Per-Patient Summary">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Patient</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Uploads</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Analysed</th>
                <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Avg Confidence</th>
                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Diseases</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.patient_id}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.total_uploads}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.completed_analyses}</td>
                  <td style={{ padding: '8px 12px', textAlign: 'center' }}><ConfidenceBadge confidence={p.avg_confidence} /></td>
                  <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{p.diseases}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Recent Uploads">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Patient</th>
                <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>File</th>
                <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Disease</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Label</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Confidence</th>
                <th style={{ padding: '6px 10px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Signal</th>
                <th style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#475569' }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 500 }}>{r.patient_id}</td>
                  <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{r.file_name}</td>
                  <td style={{ padding: '6px 10px' }}>{r.disease}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.predicted_label || <span style={{ color: '#94a3b8' }}>pending</span>}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.confidence ? <ConfidenceBadge confidence={r.confidence} /> : '--'}</td>
                  <td style={{ padding: '6px 10px', textAlign: 'center' }}>{r.signal_quality ? <QualityBadge quality={r.signal_quality} /> : '--'}</td>
                  <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{r.created_at?.slice(0, 10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function AlertsTab({ breakdown }) {
  const lowConf = breakdown?.low_confidence_analyses || []
  const pending = breakdown?.pending_analyses || []
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Pending Analyses (No Result Yet)">
        {pending.length === 0 ? <p style={{ color: '#64748b', fontSize: 13 }}>All uploads have been analysed.</p> : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#fef2f2' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Patient</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>File</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Disease</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Department</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#991b1b' }}>Uploaded</th>
                </tr>
              </thead>
              <tbody>
                {pending.map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 500 }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontSize: 11 }}>{p.file_name}</td>
                    <td style={{ padding: '8px 12px' }}>{p.disease}</td>
                    <td style={{ padding: '8px 12px' }}>{p.department}</td>
                    <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{p.created_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card title="Low Confidence Analyses (< 0.6) — Manual Review Recommended">
        {lowConf.length === 0 ? <p style={{ color: '#64748b', fontSize: 13 }}>No low-confidence analyses.</p> : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#fffbeb' }}>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>Patient</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>Disease</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>Label</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>Confidence</th>
                  <th style={{ padding: '8px 12px', textAlign: 'center', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>Signal</th>
                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid #e2e8f0', color: '#92400e' }}>Date</th>
                </tr>
              </thead>
              <tbody>
                {lowConf.map((a, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 500 }}>{a.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{a.disease}</td>
                    <td style={{ padding: '8px 12px' }}>{a.predicted_label}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}><ConfidenceBadge confidence={a.confidence} /></td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}><QualityBadge quality={a.signal_quality} /></td>
                    <td style={{ padding: '8px 12px', fontSize: 11, color: '#64748b' }}>{a.created_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

function DefinitionsTab({ defs }) {
  if (!defs) return null
  const sections = [
    { key: 'signal_quality_criteria', title: 'Signal Quality Criteria', cols: ['level', 'description'] },
    { key: 'confidence_interpretation', title: 'Confidence Interpretation', cols: ['range', 'interpretation'] },
    { key: 'file_types', title: 'Supported File Types', cols: ['ext', 'format'] },
    { key: 'pipeline_stages', title: 'Pipeline Stages', cols: ['stage', 'description'] },
    { key: 'glossary', title: 'Glossary', cols: ['term', 'definition'] },
  ]
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
      {sections.map(s => (
        <Card key={s.key} title={s.title}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(defs[s.key] || []).map((item, i) => (
              <div key={i} style={{ background: '#f8fafc', borderRadius: 8, padding: '10px 14px' }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#1e293b' }}>{item[s.cols[0]]}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{item[s.cols[1]]}</div>
              </div>
            ))}
          </div>
        </Card>
      ))}
    </div>
  )
}
