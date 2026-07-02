import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  AreaChart, Area
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316', '#14b8a6']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
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

function QualityBadge({ quality }) {
  const colors = {
    'Good': { bg: '#ecfdf5', fg: '#065f46' },
    'Fair': { bg: '#fff7ed', fg: '#9a3412' },
    'Poor': { bg: '#fef2f2', fg: '#991b1b' },
  }
  const c = colors[quality] || { bg: '#f1f5f9', fg: '#475569' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.fg
    }}>{quality}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'patients', label: 'Patient Profiles' },
  { id: 'analyses', label: 'Analysis Results' },
  { id: 'activity', label: 'Activity Log' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DataAcquisitionDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/data-acquisition/overview`),
      axios.get(`${API_URL}/data-acquisition/breakdown`),
      axios.get(`${API_URL}/data-acquisition/definitions`),
    ])
      .then(([oRes, bRes, dRes]) => {
        setOverview(oRes.data)
        setBreakdown(bRes.data)
        setDefs(dRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Data Acquisition analytics...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>

  const k = overview?.kpis || {}

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: '#0f172a' }}>Data Acquisition Dashboard</h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        End-to-end file ingestion, signal quality, and analysis pipeline metrics from real clinical.db
      </p>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 24, borderBottom: '2px solid #e2e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', color: tab === t.id ? '#3b82f6' : '#64748b',
            fontWeight: tab === t.id ? 600 : 400, cursor: 'pointer', fontSize: 13, marginBottom: -2
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Total Uploads" value={k.total_uploads} color="#3b82f6" /></Card>
            <Card><KPI label="Unique Files" value={k.unique_files} color="#8b5cf6" /></Card>
            <Card><KPI label="Patients with Data" value={k.patients_with_data} sub={`${k.coverage_pct}% of ${k.total_patients}`} color="#10b981" /></Card>
            <Card><KPI label="Total Analyses" value={k.total_analyses} color="#f59e0b" /></Card>
            <Card><KPI label="Avg Confidence" value={k.avg_confidence ? `${(k.avg_confidence * 100).toFixed(1)}%` : '--'} color="#06b6d4" /></Card>
            <Card><KPI label="Signal Quality Rate" value={`${k.signal_quality_rate}%`} sub="Good quality" color="#10b981" /></Card>
            <Card><KPI label="EEG Upload Events" value={k.eeg_upload_events} sub="Transaction log" color="#64748b" /></Card>
            <Card><KPI label="Ingest Events" value={k.ingest_events} sub="Patient master" color="#f97316" /></Card>
          </div>

          {/* Charts row 1 */}
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card title="Daily Upload Trend">
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={overview?.daily_trend || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Area type="monotone" dataKey="uploads" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
            <Card title="File Format Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview?.format_distribution || []} dataKey="count" nameKey="format" cx="50%" cy="50%" outerRadius={80} label={({ format, count }) => `${format}: ${count}`}>
                    {(overview?.format_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Charts row 2 */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Signal Quality Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview?.quality_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="quality" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Confidence Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={overview?.confidence_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {tab === 'patients' && (
        <Card title="Per-Patient Upload Profiles" span={2}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Uploads</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Unique Files</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Files</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Prediction</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Confidence</th>
                  <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Quality</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Last Upload</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.patient_profiles || []).map((p, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600, color: '#1e293b' }}>{p.patient_id}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.uploads}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.unique_files}</td>
                    <td style={{ padding: '8px 12px', color: '#64748b', fontSize: 12 }}>{(p.files || []).join(', ')}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>{p.latest_analysis?.prediction || '--'}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                      {p.latest_analysis?.confidence != null ? `${(p.latest_analysis.confidence * 100).toFixed(0)}%` : '--'}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                      {p.latest_analysis?.quality ? <QualityBadge quality={p.latest_analysis.quality} /> : '--'}
                    </td>
                    <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{p.last_upload?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'analyses' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
            <Card title="Department Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie data={breakdown?.department_distribution || []} dataKey="count" nameKey="department" cx="50%" cy="50%" outerRadius={70} label={({ department, count }) => `${department}: ${count}`}>
                    {(breakdown?.department_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Disease Distribution">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={breakdown?.disease_distribution || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="disease" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
          <Card title="Recent Analyses (Last 20)">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Disease</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Prediction</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Confidence</th>
                    <th style={{ textAlign: 'center', padding: '8px 12px', color: '#475569' }}>Quality</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>File</th>
                    <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.recent_analyses || []).map((a, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{a.patient_id}</td>
                      <td style={{ padding: '8px 12px' }}>{a.disease}</td>
                      <td style={{ padding: '8px 12px' }}>{a.prediction}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{a.confidence != null ? `${(a.confidence * 100).toFixed(0)}%` : '--'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}><QualityBadge quality={a.quality || 'Unknown'} /></td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{a.file}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{a.timestamp?.slice(0, 16)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'activity' && (
        <Card title="Recent Acquisition Events (Last 30)">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Patient</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Component</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Action</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Actor</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Detail</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px', color: '#475569' }}>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {(breakdown?.recent_events || []).map((e, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 600 }}>{e.patient_id}</td>
                    <td style={{ padding: '8px 12px' }}>{e.component}</td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{
                        padding: '2px 8px', borderRadius: 8, fontSize: 11, fontWeight: 600,
                        background: e.action === 'ingest' ? '#dbeafe' : '#f0fdf4',
                        color: e.action === 'ingest' ? '#1e40af' : '#166534'
                      }}>{e.action}</span>
                    </td>
                    <td style={{ padding: '8px 12px', color: '#64748b' }}>{e.actor}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.detail}</td>
                    <td style={{ padding: '8px 12px', fontSize: 12, color: '#64748b' }}>{e.timestamp?.slice(0, 16)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {tab === 'definitions' && defs?.definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {Object.entries(defs.definitions).map(([section, items]) => (
            <Card key={section} title={section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              <dl style={{ margin: 0 }}>
                {Object.entries(items).map(([term, desc]) => (
                  <div key={term} style={{ marginBottom: 12 }}>
                    <dt style={{ fontWeight: 600, fontSize: 13, color: '#1e293b', marginBottom: 2 }}>{term}</dt>
                    <dd style={{ margin: 0, fontSize: 13, color: '#475569', lineHeight: 1.5 }}>{desc}</dd>
                  </div>
                ))}
              </dl>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
