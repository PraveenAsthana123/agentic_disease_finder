import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts'

const API = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const RISK_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#10b981' }

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

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function fmtPct(v) {
  if (v == null) return '--'
  return v.toFixed(1) + '%'
}

const TABS = ['Overview', 'Risk Analysis', 'Deploy Timeline', 'Activity Heatmap', 'Methodology']

export default function ChangeManagementDashboard() {
  const [tab, setTab] = useState(0)
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [df, setDf] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      axios.get(`${API}/api/change-management/overview`),
      axios.get(`${API}/api/change-management/breakdown`),
      axios.get(`${API}/api/change-management/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDf(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Change Management analysis...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#f59e0b' }}>{ov?.error || 'No data available'}</div>

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 8px', fontSize: 22, color: '#1e293b' }}>Change Management Dashboard</h2>
      <p style={{ margin: '0 0 20px', fontSize: 13, color: '#64748b' }}>
        Git-based change tracking — {ov.total_changes} changes, {ov.total_deploys} deploys, {fmtPct(ov.deploy_success_rate)} success rate
      </p>

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === i ? '#3b82f6' : '#f1f5f9', color: tab === i ? '#fff' : '#475569',
            fontWeight: tab === i ? 600 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {tab === 0 && <OverviewTab ov={ov} bd={bd} />}
      {tab === 1 && <RiskTab ov={ov} bd={bd} />}
      {tab === 2 && <DeployTab bd={bd} ov={ov} />}
      {tab === 3 && <HeatmapTab bd={bd} />}
      {tab === 4 && <MethodologyTab definitions={df} />}
    </div>
  )
}

function OverviewTab({ ov, bd }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Key Metrics" span={2}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 16 }}>
          <KPI label="Total Changes" value={fmt(ov.total_changes)} />
          <KPI label="Total Deploys" value={fmt(ov.total_deploys)} />
          <KPI label="Deploy Success Rate" value={fmtPct(ov.deploy_success_rate)} color="#10b981" />
          <KPI label="Rollbacks" value={fmt(ov.rollback_count)} color="#ef4444" />
          <KPI label="Avg Files/Change" value={fmt(ov.avg_files_per_change)} />
          <KPI label="Avg Lines/Change" value={fmt(ov.avg_lines_per_change)} />
          <KPI label="Last 24h" value={fmt(ov.changes_last_24h)} color="#3b82f6" />
          <KPI label="Last 7d" value={fmt(ov.changes_last_7d)} color="#8b5cf6" />
        </div>
      </Card>

      <Card title="Change Type Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={ov.change_type_distribution} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={80} label={({ type, percent }) => `${type} ${percent.toFixed(0)}%`}>
              {(ov.change_type_distribution || []).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Change Velocity (cumulative)" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={ov.daily_change_counts}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Recent Changes" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Hash</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Date</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Message</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Type</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Risk</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Files</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Lines</th>
              </tr>
            </thead>
            <tbody>
              {(ov.recent_changes || []).slice(0, 10).map((c, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontFamily: 'monospace', color: '#3b82f6' }}>{c.hash}</td>
                  <td style={{ padding: '6px', color: '#64748b' }}>{c.date?.slice(0, 10)}</td>
                  <td style={{ padding: '6px', maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c.message}</td>
                  <td style={{ padding: '6px', textAlign: 'center' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, background: c.type === 'feature' ? '#dbeafe' : '#fef3c7', color: c.type === 'feature' ? '#1d4ed8' : '#92400e' }}>{c.type}</span>
                  </td>
                  <td style={{ padding: '6px', textAlign: 'center' }}>
                    <span style={{ padding: '2px 8px', borderRadius: 4, fontSize: 11, background: RISK_COLORS[c.risk] + '20', color: RISK_COLORS[c.risk] }}>{c.risk}</span>
                  </td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{c.files_changed}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{fmt(c.lines_changed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function RiskTab({ ov, bd }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Risk Distribution">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={ov.risk_distribution} dataKey="count" nameKey="level" cx="50%" cy="50%" outerRadius={80} label={({ level, percent }) => `${level} ${percent.toFixed(0)}%`}>
              {(ov.risk_distribution || []).map((d, i) => <Cell key={i} fill={RISK_COLORS[d.level] || COLORS[i]} />)}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Risk Trend (daily)">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={bd?.risk_trend || ov.daily_change_counts}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="high" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.4} />
            <Area type="monotone" dataKey="medium" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.4} />
            <Area type="monotone" dataKey="low" stackId="1" stroke="#10b981" fill="#10b981" fillOpacity={0.4} />
            <Legend />
          </AreaChart>
        </ResponsiveContainer>
      </Card>

      <Card title="Impact by Change Type" span={2}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Type</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Changes</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Total Files</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Avg Files</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Insertions</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Deletions</th>
                <th style={{ textAlign: 'right', padding: '8px 6px', color: '#64748b' }}>Avg Lines</th>
              </tr>
            </thead>
            <tbody>
              {(bd?.impact_by_type || []).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontWeight: 600 }}>{r.type}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{fmt(r.changes)}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{fmt(r.total_files)}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{fmt(r.avg_files)}</td>
                  <td style={{ padding: '6px', textAlign: 'right', color: '#10b981' }}>+{fmt(r.total_insertions)}</td>
                  <td style={{ padding: '6px', textAlign: 'right', color: '#ef4444' }}>-{fmt(r.total_deletions)}</td>
                  <td style={{ padding: '6px', textAlign: 'right' }}>{fmt(r.avg_lines)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Change Velocity" span={2}>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={bd?.change_velocity || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Bar yAxisId="left" dataKey="daily" fill="#3b82f6" name="Daily" />
            <Line yAxisId="right" type="monotone" dataKey="cumulative" stroke="#8b5cf6" strokeWidth={2} name="Cumulative" dot={false} />
            <Legend />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}

function DeployTab({ bd, ov }) {
  const timeline = bd?.deploy_timeline || []
  const rollbacks = ov?.rollback_events || []

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Deploy Timeline" span={2}>
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Timestamp</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>Level</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Event</th>
              </tr>
            </thead>
            <tbody>
              {timeline.slice(0, 30).map((e, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 11, color: '#64748b', whiteSpace: 'nowrap' }}>{e.ts}</td>
                  <td style={{ padding: '6px', textAlign: 'center' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: 4, fontSize: 11,
                      background: e.level === 'autobuild' ? '#dbeafe' : e.level === 'git' ? '#d1fae5' : '#f1f5f9',
                      color: e.level === 'autobuild' ? '#1d4ed8' : e.level === 'git' ? '#065f46' : '#475569'
                    }}>{e.level}</span>
                  </td>
                  <td style={{ padding: '6px', maxWidth: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.event}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Rollback Events" span={2}>
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff' }}>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Down At</th>
                <th style={{ textAlign: 'center', padding: '8px 6px', color: '#64748b' }}>HTTP</th>
                <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Recovered At</th>
              </tr>
            </thead>
            <tbody>
              {rollbacks.slice(0, 20).map((r, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 11, color: '#ef4444' }}>{r.ts}</td>
                  <td style={{ padding: '6px', textAlign: 'center', color: '#64748b' }}>{r.http}</td>
                  <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: 11, color: '#10b981' }}>{r.recovery_ts}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function HeatmapTab({ bd }) {
  const heatmap = bd?.hourly_heatmap || {}
  const dayLabels = heatmap.day_labels || []
  const hourLabels = heatmap.hour_labels || []
  const matrix = heatmap.matrix || []
  const maxVal = Math.max(1, ...matrix.flat())

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
      <Card title="Commit Activity Heatmap (day x hour)">
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr>
                <th style={{ padding: '4px 8px', color: '#64748b' }}></th>
                {hourLabels.map(h => (
                  <th key={h} style={{ padding: '4px 6px', color: '#64748b', fontWeight: 400, minWidth: 28, textAlign: 'center' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dayLabels.map((day, di) => (
                <tr key={di}>
                  <td style={{ padding: '4px 8px', color: '#334155', fontWeight: 500, whiteSpace: 'nowrap' }}>{day}</td>
                  {(matrix[di] || []).map((val, hi) => {
                    const intensity = val / maxVal
                    const bg = val === 0 ? '#f8fafc' : `rgba(59, 130, 246, ${0.15 + intensity * 0.7})`
                    const fg = intensity > 0.5 ? '#fff' : '#334155'
                    return (
                      <td key={hi} style={{
                        padding: '4px 6px', textAlign: 'center', borderRadius: 3,
                        background: bg, color: val === 0 ? '#cbd5e1' : fg, fontWeight: val > 0 ? 600 : 400
                      }}>{val}</td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, color: '#64748b' }}>
          <span>Less</span>
          {[0, 0.25, 0.5, 0.75, 1].map((v, i) => (
            <div key={i} style={{ width: 16, height: 16, borderRadius: 3, background: v === 0 ? '#f8fafc' : `rgba(59, 130, 246, ${0.15 + v * 0.7})` }} />
          ))}
          <span>More</span>
        </div>
      </Card>
    </div>
  )
}

function MethodologyTab({ definitions }) {
  if (!definitions?.available) return <div style={{ padding: 20, color: '#f59e0b' }}>No definitions available</div>

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 }}>
      <Card title="Change Management Stages" span={2}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {(definitions.stages || []).map((s, i) => (
            <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <div style={{
                width: 28, height: 28, borderRadius: '50%', background: COLORS[i % COLORS.length],
                display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff',
                fontWeight: 700, fontSize: 13, flexShrink: 0
              }}>{i + 1}</div>
              <div>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#1e293b' }}>{s.stage}</div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{s.description}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Metric Definitions" span={2}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Term</th>
              <th style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>Definition</th>
            </tr>
          </thead>
          <tbody>
            {(definitions.metrics || []).map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                <td style={{ padding: '6px', fontWeight: 600, color: '#334155', whiteSpace: 'nowrap' }}>{m.term}</td>
                <td style={{ padding: '6px', color: '#64748b' }}>{m.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
