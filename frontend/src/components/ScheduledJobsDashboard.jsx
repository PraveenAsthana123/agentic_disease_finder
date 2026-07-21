import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const API = '/api'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#ef4444', '#06b6d4', '#ec4899', '#eab308']
const SCHEDULE_COLORS = { Hourly: '#3b82f6', Daily: '#22c55e', Other: '#f97316' }

const thStyle = { padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', fontSize: 13, fontWeight: 600, color: '#475569', whiteSpace: 'nowrap' }
const tdStyle = { padding: '8px 12px', borderBottom: '1px solid #f1f5f9', fontSize: 13, color: '#334155' }

function Card({ title, children, span }) {
  return (
    <div style={{ background: '#fff', borderRadius: 12, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,.08)', gridColumn: span ? `span ${span}` : undefined }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 28, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function ScheduleBadge({ schedule }) {
  let type = 'Other'
  if (schedule && schedule.toLowerCase().includes('hourly')) type = 'Hourly'
  else if (schedule && schedule.toLowerCase().includes('daily')) type = 'Daily'
  const color = SCHEDULE_COLORS[type] || '#94a3b8'
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{type}</span>
}

function ReportBadge({ exists }) {
  const color = exists ? '#22c55e' : '#ef4444'
  const label = exists ? 'Report OK' : 'No Report'
  return <span style={{ display: 'inline-block', padding: '2px 10px', borderRadius: 9999, fontSize: 11, fontWeight: 600, background: color + '18', color, border: `1px solid ${color}40` }}>{label}</span>
}

export default function ScheduledJobsDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [tab, setTab] = useState('overview')
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/scheduled-jobs/overview`),
      axios.get(`${API}/scheduled-jobs/breakdown`),
      axios.get(`${API}/scheduled-jobs/definitions`),
    ])
      .then(([ov, bd, df]) => { setOverview(ov.data); setBreakdown(bd.data); setDefs(df.data) })
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: '#ef4444' }}>Error: {error}</div>
  if (!overview) return <div style={{ padding: 32, color: '#64748b' }}>Loading Scheduled Jobs...</div>
  if (!overview.available) return <div style={{ padding: 32, color: '#f97316' }}>{overview.note || 'Not available'}</div>

  const tabs = ['overview', 'all-jobs', 'reports', 'cron-tags', 'definitions']
  const k = overview.kpis || {}
  const jobs = overview.jobs_summary || []
  const bdJobs = (breakdown && breakdown.jobs) || []

  return (
    <div style={{ padding: 24, background: '#f8fafc', minHeight: '100vh' }}>
      <h2 style={{ margin: '0 0 4px', fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{overview.title || 'Scheduled Jobs Registry'}</h2>
      <p style={{ margin: '0 0 18px', fontSize: 13, color: '#64748b' }}>Cron and background jobs — schedules, scripts, reports, and coverage</p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{ padding: '7px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 500, background: tab === t ? '#3b82f6' : '#e2e8f0', color: tab === t ? '#fff' : '#475569', transition: 'all .15s' }}>{t.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Key Metrics" span={2}>
            <div style={{ display: 'flex', justifyContent: 'space-around', flexWrap: 'wrap', gap: 16 }}>
              <KPI label="Total Jobs" value={k.total_jobs} />
              <KPI label="Daily Jobs" value={k.daily_jobs} />
              <KPI label="Hourly Jobs" value={k.hourly_jobs} />
              <KPI label="Jobs with Reports" value={k.jobs_with_reports} />
              <KPI label="Cron Tags" value={k.unique_cron_tags} />
              <KPI label="Unique Scripts" value={k.unique_scripts} />
            </div>
          </Card>

          <Card title="Schedule Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={overview.schedule_distribution || []} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name}: ${value}`}>
                  {(overview.schedule_distribution || []).map((e, i) => <Cell key={i} fill={SCHEDULE_COLORS[e.name] || COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Jobs per Schedule Type">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.schedule_distribution || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" allowDecimals={false} />
                <YAxis type="category" dataKey="name" width={70} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                  {(overview.schedule_distribution || []).map((e, i) => <Cell key={i} fill={SCHEDULE_COLORS[e.name] || COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <Card title="All Jobs Summary" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Job</th>
                    <th style={thStyle}>Schedule</th>
                    <th style={thStyle}>Type</th>
                    <th style={thStyle}>Report</th>
                    <th style={thStyle}>Purpose</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((j, i) => (
                    <tr key={j.id} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{j.label}</td>
                      <td style={tdStyle}>{j.schedule}</td>
                      <td style={tdStyle}><ScheduleBadge schedule={j.schedule} /></td>
                      <td style={tdStyle}><ReportBadge exists={j.has_report} /></td>
                      <td style={{ ...tdStyle, maxWidth: 300, whiteSpace: 'normal' }}>{j.purpose}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── All Jobs Tab ── */}
      {tab === 'all-jobs' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 16 }}>
          {bdJobs.map((j, i) => (
            <Card key={j.id} title={j.label}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                <ScheduleBadge schedule={j.schedule} />
                <ReportBadge exists={j.report_exists} />
              </div>
              <div style={{ fontSize: 13, color: '#475569', marginBottom: 8 }}>{j.purpose}</div>
              <div style={{ fontSize: 12, color: '#64748b' }}>
                <div><strong>Schedule:</strong> {j.schedule}</div>
                <div><strong>Script:</strong> <code style={{ background: '#f1f5f9', padding: '1px 5px', borderRadius: 4, fontSize: 11 }}>{j.script}</code></div>
                <div><strong>Cron Tag:</strong> <code style={{ background: '#f1f5f9', padding: '1px 5px', borderRadius: 4, fontSize: 11 }}>{j.cron_tag}</code></div>
                <div><strong>Report:</strong> <code style={{ background: '#f1f5f9', padding: '1px 5px', borderRadius: 4, fontSize: 11 }}>{j.report || 'n/a'}</code></div>
                {j.report_exists && j.report_size_bytes > 0 && (
                  <div><strong>Report Size:</strong> {(j.report_size_bytes / 1024).toFixed(1)} KB</div>
                )}
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* ── Reports Tab ── */}
      {tab === 'reports' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Report Coverage" span={2}>
            <div style={{ display: 'flex', gap: 32, marginBottom: 16 }}>
              <KPI label="With Reports" value={bdJobs.filter(j => j.report_exists).length} />
              <KPI label="Missing Reports" value={bdJobs.filter(j => !j.report_exists).length} />
              <KPI label="Total Report Size" value={`${(bdJobs.reduce((s, j) => s + (j.report_size_bytes || 0), 0) / 1024).toFixed(0)} KB`} />
            </div>
          </Card>
          <Card title="Report Inventory" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Job</th>
                    <th style={thStyle}>Report Path</th>
                    <th style={thStyle}>Status</th>
                    <th style={thStyle}>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {bdJobs.map((j, i) => (
                    <tr key={j.id} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{j.label}</td>
                      <td style={tdStyle}><code style={{ fontSize: 11 }}>{j.report || 'n/a'}</code></td>
                      <td style={tdStyle}><ReportBadge exists={j.report_exists} /></td>
                      <td style={tdStyle}>{j.report_exists ? `${(j.report_size_bytes / 1024).toFixed(1)} KB` : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Cron Tags Tab ── */}
      {tab === 'cron-tags' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Cron Tag Registry" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>#</th>
                    <th style={thStyle}>Cron Tag</th>
                    <th style={thStyle}>Job</th>
                    <th style={thStyle}>Script</th>
                    <th style={thStyle}>Schedule</th>
                  </tr>
                </thead>
                <tbody>
                  {bdJobs.filter(j => j.cron_tag).map((j, i) => (
                    <tr key={j.id} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={tdStyle}>{i + 1}</td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}><code style={{ background: '#dbeafe', padding: '2px 8px', borderRadius: 4, fontSize: 12, color: '#1e40af' }}>{j.cron_tag}</code></td>
                      <td style={tdStyle}>{j.label}</td>
                      <td style={tdStyle}><code style={{ fontSize: 11 }}>{j.script}</code></td>
                      <td style={tdStyle}>{j.schedule}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
          <Card title="Schedule Legend">
            {(defs.schedule_legend || []).map(s => (
              <div key={s.label} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                <span style={{ width: 14, height: 14, borderRadius: 4, background: s.color, display: 'inline-block' }} />
                <strong style={{ fontSize: 13 }}>{s.label}</strong>
                <span style={{ fontSize: 12, color: '#64748b' }}>{s.description}</span>
              </div>
            ))}
          </Card>

          <Card title="Glossary" span={2}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Term</th>
                    <th style={thStyle}>Definition</th>
                  </tr>
                </thead>
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={g.term} style={{ background: i % 2 === 0 ? '#fff' : '#f8fafc' }}>
                      <td style={{ ...tdStyle, fontWeight: 600, whiteSpace: 'nowrap' }}>{g.term}</td>
                      <td style={tdStyle}>{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Clinical Notes" span={2}>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.clinical_notes || []).map((n, i) => (
                <li key={i} style={{ fontSize: 13, color: '#475569', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References" span={2}>
            {(defs.references || []).map((r, i) => (
              <div key={i} style={{ marginBottom: 8, fontSize: 13 }}>
                <strong style={{ color: '#1e293b' }}>{r.ref}</strong>
                <span style={{ color: '#64748b', marginLeft: 8 }}>{r.detail}</span>
              </div>
            ))}
          </Card>
        </div>
      )}
    </div>
  )
}
