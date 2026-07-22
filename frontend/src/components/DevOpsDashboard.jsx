import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4', '#e91e63', '#607d8b']

function fmt(v, decimals = 0) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString(undefined, { maximumFractionDigits: decimals }) : String(v)
}

export default function DevOpsDashboard() {
  const [overview, setOverview] = useState(null)
  const [pipelines, setPipelines] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, pl, df] = await Promise.all([
          axios.get(`${API_URL}/api/devops/overview`),
          axios.get(`${API_URL}/api/devops/pipelines`),
          axios.get(`${API_URL}/api/devops/definitions`)
        ])
        setOverview(ov.data)
        setPipelines(pl.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load DevOps data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9881;</div>
      Loading DevOps / CI-CD data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'DevOps data not available.'}
    </div>
  )

  const s = overview.summary || {}
  const velocity = overview.commit_velocity || []
  const commitTypes = overview.commit_types || []
  const topAuthors = overview.top_authors || []
  const recentDeploys = overview.recent_deploys || []
  const hotFiles = overview.hottest_files || []
  const pipelineList = pipelines?.pipelines || []
  const runningJobs = pipelines?.running_jobs || []
  const healthSnap = pipelines?.health_snapshot || {}
  const definitions = defs?.definitions || []

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'pipelines', label: 'Pipelines' },
    { id: 'activity', label: 'Activity' },
    { id: 'definitions', label: 'Definitions' }
  ]

  const cfrColor = s.change_fail_rate_pct > 20 ? '#f44336' : s.change_fail_rate_pct > 10 ? '#ff9800' : '#4caf50'
  const mttrColor = s.mttr_minutes > 1440 ? '#f44336' : s.mttr_minutes > 360 ? '#ff9800' : '#4caf50'

  const kpis = [
    { label: 'Commits (90d)', value: fmt(s.total_commits_90d), color: COLORS[0] },
    { label: 'Deploys (90d)', value: fmt(s.deploy_count_90d), color: COLORS[2] },
    { label: 'Deploy/Day', value: fmt(s.deploy_freq_per_day, 2), color: COLORS[5] },
    { label: 'Change Fail Rate', value: `${fmt(s.change_fail_rate_pct, 1)}%`, color: cfrColor },
    { label: 'MTTR (min)', value: s.mttr_minutes != null ? fmt(s.mttr_minutes) : 'N/A', color: mttrColor },
    { label: 'Avg Daily', value: fmt(s.avg_daily_commits, 1), color: COLORS[1] },
    { label: 'Branch', value: s.current_branch, color: COLORS[3] },
    { label: 'Unpushed', value: fmt(s.commits_ahead), color: s.commits_ahead > 50 ? '#f44336' : COLORS[7] },
  ]

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
        <span style={{ fontSize: 22 }}>&#9881;</span>
        <h2 style={{ margin: 0, fontSize: 18 }}>DevOps / CI-CD Dashboard</h2>
        <span style={{ fontSize: 11, color: '#94a3b8', marginLeft: 'auto' }}>
          Real git data &middot; {overview.period_days}d window &middot; {overview.generated_at}
        </span>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 16, borderBottom: '2px solid #e5e7eb', paddingBottom: 2 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              padding: '6px 14px', fontSize: 12, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : 'transparent',
              color: tab === t.id ? '#fff' : '#64748b',
              border: 'none', borderRadius: '6px 6px 0 0', cursor: 'pointer'
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <>
          {/* KPI cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: 10, marginBottom: 18 }}>
            {kpis.map((k, i) => (
              <div key={i} style={{
                background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 12px',
                borderLeft: `3px solid ${k.color}`
              }}>
                <div style={{ fontSize: 10, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5 }}>{k.label}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: '#1e293b', marginTop: 2 }}>{k.value}</div>
              </div>
            ))}
          </div>

          {/* Commit Velocity Chart */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Commit Velocity (30 days)</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={velocity}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} tickFormatter={d => d.slice(5)} />
                <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Bar dataKey="commits" fill="#1e88e5" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Commit Types + Top Authors side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 14 }}>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Commit Types (90d)</h3>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie data={commitTypes} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={65} label={({ type, count }) => `${type}: ${count}`}>
                    {commitTypes.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
              <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Top Contributors (90d)</h3>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={topAuthors} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" tick={{ fontSize: 10 }} allowDecimals={false} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Bar dataKey="commits" fill="#7c4dff" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* Pipelines Tab */}
      {tab === 'pipelines' && (
        <>
          {/* Health snapshot */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 14 }}>
            <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 10, color: '#166534' }}>API Status</div>
              <div style={{ fontSize: 16, fontWeight: 700, color: healthSnap.api_status === 'healthy' ? '#16a34a' : '#dc2626' }}>
                {healthSnap.api_status || '--'}
              </div>
            </div>
            <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 10, color: '#166534' }}>DB Status</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{healthSnap.db_status || '--'}</div>
            </div>
            <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 10, color: '#1e40af' }}>Endpoints OK</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{healthSnap.endpoints_ok}/{healthSnap.endpoints_total}</div>
            </div>
            <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 10, color: '#1e40af' }}>Pipelines</div>
              <div style={{ fontSize: 16, fontWeight: 700 }}>{pipelines?.enabled || 0} enabled / {pipelines?.total_pipelines || 0}</div>
            </div>
          </div>

          {/* Pipeline table */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Cron Jobs / Pipelines</h3>
            {pipelineList.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>No cron definitions found in jobs/crons/</div>
            ) : (
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Name</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Schedule</th>
                    <th style={{ textAlign: 'center', padding: '4px 8px', color: '#64748b' }}>Enabled</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Status</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Last Run</th>
                  </tr>
                </thead>
                <tbody>
                  {pipelineList.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '4px 8px', fontWeight: 600 }}>{p.name}</td>
                      <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 10 }}>{p.schedule}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'center' }}>
                        <span style={{ color: p.enabled ? '#16a34a' : '#dc2626', fontWeight: 700 }}>
                          {p.enabled ? 'YES' : 'NO'}
                        </span>
                      </td>
                      <td style={{ padding: '4px 8px' }}>{p.status}</td>
                      <td style={{ padding: '4px 8px', fontSize: 10, color: '#94a3b8' }}>{p.last_run || '--'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Running jobs */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Running Jobs (PID check)</h3>
            {runningJobs.length === 0 ? (
              <div style={{ color: '#94a3b8', fontSize: 12 }}>No PID files found</div>
            ) : (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {runningJobs.map((j, i) => (
                  <div key={i} style={{
                    padding: '6px 12px', borderRadius: 6, fontSize: 11,
                    background: j.alive ? '#dcfce7' : '#fef2f2',
                    border: `1px solid ${j.alive ? '#86efac' : '#fecaca'}`,
                    color: j.alive ? '#166534' : '#991b1b'
                  }}>
                    {j.name} {j.alive ? `(PID ${j.pid})` : '(stopped)'}
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}

      {/* Activity Tab */}
      {tab === 'activity' && (
        <>
          {/* Recent deploys */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14, marginBottom: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Recent Deploys / Feature Commits</h3>
            <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Hash</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '4px 8px', color: '#64748b' }}>Subject</th>
                </tr>
              </thead>
              <tbody>
                {recentDeploys.map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 10 }}>{d.hash}</td>
                    <td style={{ padding: '4px 8px', fontSize: 10, color: '#64748b' }}>{d.date?.slice(0, 10)}</td>
                    <td style={{ padding: '4px 8px' }}>{d.subject}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Hottest files */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
            <h3 style={{ fontSize: 13, margin: '0 0 10px', color: '#334155' }}>Hottest Files (30d churn)</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={hotFiles} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" tick={{ fontSize: 10 }} allowDecimals={false} />
                <YAxis type="category" dataKey="file" tick={{ fontSize: 9 }} width={200}
                  tickFormatter={f => f.length > 35 ? '...' + f.slice(-32) : f} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Bar dataKey="changes" fill="#ff9800" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && (
        <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: 14 }}>
          <h3 style={{ fontSize: 13, margin: '0 0 12px', color: '#334155' }}>Metric Definitions</h3>
          {definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 12, paddingBottom: 10, borderBottom: i < definitions.length - 1 ? '1px solid #e2e8f0' : 'none' }}>
              <div style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{d.term}</div>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>{d.definition}</div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2, fontFamily: 'monospace' }}>Source: {d.source}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
