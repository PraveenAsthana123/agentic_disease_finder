import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const CATEGORY_COLORS = {
  clinical: '#3b82f6',
  administrative: '#f59e0b',
  educational: '#22c55e'
}

const COLORS = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#22c55e',
  '#ef4444', '#06b6d4', '#f97316', '#84cc16', '#6366f1']

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
      <div style={{ fontSize: 28, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function ShareBadge({ shared, downloaded }) {
  if (shared && downloaded) return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: '#dcfce7', color: '#22c55e' }}>Downloaded</span>
  )
  if (shared) return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: '#fef9c3', color: '#ca8a04' }}>Shared</span>
  )
  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: '#f1f5f9', color: '#94a3b8' }}>Not Shared</span>
  )
}

function CategoryBadge({ category }) {
  const color = CATEGORY_COLORS[category] || '#94a3b8'
  return (
    <span style={{ display: 'inline-block', padding: '2px 8px', borderRadius: 6, fontSize: 11, fontWeight: 600, background: color + '22', color }}>{category}</span>
  )
}

export default function PatientDocumentsDashboard() {
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
          axios.get(`${API_URL}/api/patient-documents/overview`),
          axios.get(`${API_URL}/api/patient-documents/breakdown`),
          axios.get(`${API_URL}/api/patient-documents/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (e) {
        setError(e.message)
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Patient Documents data...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'types', label: 'Document Types' },
    { id: 'patients', label: 'Patient Detail' },
    { id: 'definitions', label: 'Definitions' },
  ]

  return (
    <div style={{ padding: '20px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Patient Document Management</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Clinical, administrative, and educational document tracking — upload trends, sharing analytics, per-patient inventory
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '1px solid #e2e8f0', paddingBottom: 0 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 16px', border: 'none', borderBottom: tab === t.id ? '2px solid #3b82f6' : '2px solid transparent',
            background: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t.id ? 600 : 400,
            color: tab === t.id ? '#3b82f6' : '#64748b'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Tab 1: Overview */}
      {tab === 'overview' && overview && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 16 }}>
              {(overview.kpis || []).map((kpi, i) => (
                <KPI key={i} label={kpi.label} value={fmt(kpi.value)} sub={kpi.sub} color={kpi.color} />
              ))}
            </div>
          </Card>

          {/* Pie: Document Type Distribution */}
          <Card title="Document Type Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.type_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={40} outerRadius={80} paddingAngle={2}>
                  {(overview.type_distribution || []).map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', marginTop: 4 }}>
              {(overview.type_distribution || []).map((d, i) => (
                <span key={d.name} style={{ fontSize: 10, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: COLORS[i % COLORS.length], marginRight: 3 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Pie: Category + Sharing Status */}
          <Card title="Category Distribution">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={overview.category_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={35} outerRadius={65} paddingAngle={3}>
                  {(overview.category_distribution || []).map((d, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[d.name] || '#94a3b8'} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 4 }}>
              {(overview.category_distribution || []).map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: CATEGORY_COLORS[d.name] || '#94a3b8', marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Pie: Sharing Status */}
          <Card title="Sharing Status">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={overview.sharing_status || []} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  innerRadius={35} outerRadius={65} paddingAngle={3}>
                  {(overview.sharing_status || []).map((d, i) => (
                    <Cell key={i} fill={d.color || '#94a3b8'} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'center', marginTop: 4 }}>
              {(overview.sharing_status || []).map(d => (
                <span key={d.name} style={{ fontSize: 11, color: '#475569' }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: d.color, marginRight: 4 }} />
                  {d.name}: {d.value}
                </span>
              ))}
            </div>
          </Card>

          {/* Line: Monthly Upload Trend */}
          <Card title="Monthly Upload Trend" span={3}>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={overview.monthly_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="uploads" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} name="Uploads" />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {/* Tab 2: Document Types */}
      {tab === 'types' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Bar: File Size by Type */}
          <Card title="Average File Size by Type (KB)" span={2}>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={breakdown.size_by_type || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="document_type" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="avg_kb" fill="#8b5cf6" name="Avg Size (KB)" />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Size by Type Table */}
          <Card title="Storage by Document Type" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Document Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Count</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Avg KB</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Min KB</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Max KB</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Total MB</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.size_by_type || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{row.document_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{row.count}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.avg_kb)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.min_kb)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(row.max_kb)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 600 }}>{row.total_mb}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Type × Category Matrix */}
          <Card title="Type × Category Matrix" span={2}>
            <div style={{ maxHeight: 300, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Document Type</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.type_category || []).map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px' }}>{row.document_type}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><CategoryBadge category={row.category} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontWeight: 600 }}>{row.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 3: Patient Detail */}
      {tab === 'patients' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          {/* Per-Patient Summary */}
          <Card title="Per-Patient Document Summary">
            <div style={{ maxHeight: 350, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Docs</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Types</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Size (MB)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Shared</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Downloaded</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Share Rate</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>First Upload</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Last Upload</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.per_patient || []).map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.doc_count}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.type_count}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.total_size_mb}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: '#22c55e' }}>{p.shared}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', color: '#3b82f6' }}>{p.downloaded}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{p.share_rate}%</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{p.first_upload}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{p.last_upload}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Recent Documents */}
          <Card title="Recent Documents (last 50)">
            <div style={{ maxHeight: 400, overflow: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Patient</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>Type</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b' }}>File Name</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Category</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Size (KB)</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'center', padding: '6px 8px', color: '#64748b' }}>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.recent_documents || []).map((doc, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 8px', fontWeight: 600 }}>{doc.patient_id}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11 }}>{doc.document_type}</td>
                      <td style={{ padding: '6px 8px', fontSize: 11, color: '#475569', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{doc.document_name}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><CategoryBadge category={doc.category} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}>{fmt(doc.file_size_kb)}</td>
                      <td style={{ padding: '6px 8px', textAlign: 'center' }}><ShareBadge shared={doc.shared} downloaded={doc.downloaded} /></td>
                      <td style={{ padding: '6px 8px', textAlign: 'center', fontSize: 11, color: '#64748b' }}>{doc.upload_date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* Tab 4: Definitions */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
          {/* Document Types */}
          <Card title="Document Type Definitions" span={2}>
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs.document_types || []).map((dt, i) => (
                <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>{dt.type}</span>
                  <span style={{ color: '#64748b', marginLeft: 8 }}>{dt.description}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Categories */}
          <Card title="Document Categories">
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs.categories || []).map((cat, i) => (
                <div key={i} style={{ padding: '10px 14px', background: (cat.color || '#94a3b8') + '11', border: `1px solid ${cat.color || '#94a3b8'}33`, borderRadius: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, color: cat.color }}>{cat.name}</div>
                  <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>{cat.description}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Sharing Workflow */}
          <Card title="Sharing Workflow">
            <p style={{ fontSize: 12, color: '#475569', margin: '0 0 12px' }}>{defs.sharing_workflow?.description}</p>
            <div style={{ display: 'grid', gap: 8 }}>
              {(defs.sharing_workflow?.statuses || []).map((s, i) => (
                <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
                  <span style={{ fontWeight: 600 }}>{s.status}</span>
                  <span style={{ color: '#64748b', marginLeft: 8 }}>{s.description}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Glossary */}
          <Card title="Glossary" span={2}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
              {(defs.glossary || []).map((g, i) => (
                <div key={i} style={{ padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12 }}>
                  <span style={{ fontWeight: 600, color: '#1e293b' }}>{g.term}</span>
                  <span style={{ color: '#64748b', marginLeft: 8 }}>{g.definition}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Clinical Note */}
          {defs.clinical_note && (
            <Card span={2}>
              <div style={{ padding: '12px 16px', background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, fontSize: 13, color: '#1e40af' }}>
                {defs.clinical_note}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}
