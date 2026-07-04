import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (window._env_ && window._env_.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const STATUS_COLORS = { healthy: '#10b981', degraded: '#f59e0b', down: '#ef4444', ok: '#10b981', warning: '#f59e0b', critical: '#ef4444' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

function Card({ title, children }) {
  return (
    <div style={{ background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 16, fontWeight: 600 }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color }) {
  return (
    <div style={{ background: '#f8fafc', borderRadius: 8, padding: '14px 18px', textAlign: 'center', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#1e293b' }}>{fmt(value)}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>{label}</div>
    </div>
  )
}

function Badge({ text, color }) {
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 12, fontWeight: 600,
      background: (color || '#64748b') + '22', color: color || '#64748b'
    }}>{text}</span>
  )
}

export default function CloudOpsDashboard() {
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
          axios.get(`${API_URL}/api/cloud-ops/overview`),
          axios.get(`${API_URL}/api/cloud-ops/breakdown`),
          axios.get(`${API_URL}/api/cloud-ops/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load cloud ops data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading Cloud Ops data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Cloud Ops data not available.'}
    </div>
  )

  const k = overview.kpis || {}
  const regionStatusDist = Object.entries(overview.region_status_distribution || {}).map(([name, value]) => ({ name, value }))
  const costByService = overview.cost_by_service || []
  const autoscaleSummary = overview.autoscale_summary || {}
  const autoscalePie = [
    { name: 'Scale Up', value: autoscaleSummary.scale_up || 0 },
    { name: 'Scale Down', value: autoscaleSummary.scale_down || 0 }
  ]
  const regionSummary = overview.region_summary || []

  const resourceUtil = breakdown?.resource_utilisation || []
  const costBreakdown = breakdown?.cost_breakdown || []
  const totalCost = breakdown?.total_cost_usd
  const budgetUsd = breakdown?.budget_usd
  const budgetPct = breakdown?.budget_pct
  const autoscaleEvents = breakdown?.autoscale_events || []
  const autoscalePolicies = breakdown?.autoscale_policies || []
  const uptimeHistory = breakdown?.uptime_history || []
  const uptimeAvg = breakdown?.uptime_avg_30d
  const uptimeSla = breakdown?.uptime_sla_target
  const incidents30d = breakdown?.incidents_30d

  const defRegions = defs?.regions || []
  const defServices = defs?.services || []
  const costThresholds = defs?.cost_thresholds || {}
  const statusLevels = defs?.status_levels || {}
  const costAlertLevels = defs?.cost_alert_levels || {}
  const resourceThresholds = defs?.resource_thresholds || {}
  const clinicalRelevance = defs?.clinical_relevance || ''
  const defAutoscalePolicies = defs?.autoscale_policies || []

  const tabs = ['overview', 'regions', 'cost', 'autoscale', 'definitions']
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', fontWeight: active ? 700 : 400,
    borderBottom: active ? '2px solid #3b82f6' : '2px solid transparent',
    color: active ? '#1e293b' : '#64748b', background: 'none', border: 'none',
    borderBottomWidth: 2, borderBottomStyle: 'solid',
    borderBottomColor: active ? '#3b82f6' : 'transparent'
  })

  return (
    <div style={{ padding: 24 }}>
      <h2 style={{ marginBottom: 16, fontSize: 22, fontWeight: 700 }}>Cloud Ops</h2>
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid #e2e8f0', marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={tabStyle(tab === t)}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 20 }}>
            <KPI label="Total Regions" value={k.total_regions} />
            <KPI label="Healthy Regions" value={k.healthy_regions} color="#10b981" />
            <KPI label="Degraded Regions" value={k.degraded_regions} color="#f59e0b" />
            <KPI label="Avg Uptime %" value={k.avg_uptime_pct != null ? k.avg_uptime_pct + '%' : '--'} />
            <KPI label="Total Cost (USD)" value={k.total_cost_usd != null ? '$' + fmt(k.total_cost_usd) : '--'} />
            <KPI label="Budget %" value={k.budget_pct != null ? k.budget_pct + '%' : '--'} color={k.budget_pct > 80 ? '#ef4444' : '#1e293b'} />
            <KPI label="Mean Latency (ms)" value={k.mean_latency_ms} />
            <KPI label="SLA Met" value={k.sla_met} color="#10b981" />
            <KPI label="Incidents (30d)" value={k.total_incidents_30d} color={k.total_incidents_30d > 0 ? '#ef4444' : '#10b981'} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18 }}>
            <Card title="Region Status Distribution">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={regionStatusDist} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {regionStatusDist.map((e, i) => (
                      <Cell key={i} fill={STATUS_COLORS[e.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Cost by Service">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={costByService}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="service" tick={{ fontSize: 11 }} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="cost_usd" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Autoscale Summary">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie data={autoscalePie} dataKey="value" nameKey="name" cx="50%" cy="50%"
                    outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {autoscalePie.map((e, i) => (
                      <Cell key={i} fill={COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </>
      )}

      {tab === 'regions' && (
        <>
          <Card title={`Region Summary (${regionSummary.length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Region</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Status</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Uptime %</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Latency (ms)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Instances</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Error Rate %</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Meets SLA</th>
                  </tr>
                </thead>
                <tbody>
                  {regionSummary.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.region}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={r.status} color={STATUS_COLORS[r.status]} />
                      </td>
                      <td style={{ padding: '8px 12px' }}>{r.uptime_pct}</td>
                      <td style={{ padding: '8px 12px' }}>{r.latency_ms}</td>
                      <td style={{ padding: '8px 12px' }}>{r.instances}</td>
                      <td style={{ padding: '8px 12px' }}>{r.error_rate_pct}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={r.meets_sla ? 'Yes' : 'No'} color={r.meets_sla ? '#10b981' : '#ef4444'} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`Resource Utilization (${resourceUtil.length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Region</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>CPU %</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Memory %</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Disk %</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Network (Mbps)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>CPU Status</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Memory Status</th>
                  </tr>
                </thead>
                <tbody>
                  {resourceUtil.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.label}</td>
                      <td style={{ padding: '8px 12px' }}>{r.cpu_pct}</td>
                      <td style={{ padding: '8px 12px' }}>{r.memory_pct}</td>
                      <td style={{ padding: '8px 12px' }}>{r.disk_pct}</td>
                      <td style={{ padding: '8px 12px' }}>{r.network_mbps}</td>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={r.cpu_status} color={STATUS_COLORS[r.cpu_status]} />
                      </td>
                      <td style={{ padding: '8px 12px' }}>
                        <Badge text={r.memory_status} color={STATUS_COLORS[r.memory_status]} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'cost' && (
        <>
          <Card title="Budget Progress">
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
                <span>Total Cost: <strong>${fmt(totalCost)}</strong></span>
                <span>Budget: <strong>${fmt(budgetUsd)}</strong></span>
                <span>Usage: <strong>{budgetPct != null ? budgetPct + '%' : '--'}</strong></span>
              </div>
              <div style={{ background: '#e2e8f0', borderRadius: 6, height: 20, overflow: 'hidden' }}>
                <div style={{
                  width: Math.min(budgetPct || 0, 100) + '%',
                  height: '100%',
                  borderRadius: 6,
                  background: budgetPct > 100 ? '#ef4444' : budgetPct > 80 ? '#f59e0b' : '#10b981',
                  transition: 'width 0.3s'
                }} />
              </div>
            </div>
          </Card>

          <Card title={`Cost Breakdown (${costBreakdown.length} services)`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Service</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Cost (USD)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Prev Month (USD)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Delta %</th>
                  </tr>
                </thead>
                <tbody>
                  {costBreakdown.map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.service}</td>
                      <td style={{ padding: '8px 12px' }}>${fmt(c.cost_usd)}</td>
                      <td style={{ padding: '8px 12px' }}>${fmt(c.prev_month_usd)}</td>
                      <td style={{ padding: '8px 12px', color: c.delta_pct < 0 ? '#10b981' : c.delta_pct > 0 ? '#ef4444' : '#64748b', fontWeight: 600 }}>
                        {c.delta_pct > 0 ? '+' : ''}{c.delta_pct}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Cost Alert Levels">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Level</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Threshold (USD)</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(costAlertLevels).map(([level, info], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{level}</td>
                      <td style={{ padding: '8px 12px' }}>${fmt(info.threshold_usd)}</td>
                      <td style={{ padding: '8px 12px' }}>{info.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'autoscale' && (
        <>
          <Card title={`Autoscale Events (${autoscaleEvents.length})`}>
            {autoscaleEvents.length === 0 ? (
              <p style={{ color: '#64748b' }}>No autoscale events recorded.</p>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Policy</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Timestamp (UTC)</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Direction</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>From</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>To</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Trigger Metric</th>
                    </tr>
                  </thead>
                  <tbody>
                    {autoscaleEvents.map((e, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{e.policy}</td>
                        <td style={{ padding: '8px 12px' }}>{e.timestamp_utc}</td>
                        <td style={{ padding: '8px 12px' }}>
                          <Badge text={e.direction} color={e.direction === 'up' ? '#3b82f6' : '#f59e0b'} />
                        </td>
                        <td style={{ padding: '8px 12px' }}>{e.from_count}</td>
                        <td style={{ padding: '8px 12px' }}>{e.to_count}</td>
                        <td style={{ padding: '8px 12px' }}>{e.trigger_metric}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card title="Autoscale Policies">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Name</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Min</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Max</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Metric</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Cooldown (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {autoscalePolicies.map((p, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.name}</td>
                      <td style={{ padding: '8px 12px' }}>{p.min}</td>
                      <td style={{ padding: '8px 12px' }}>{p.max}</td>
                      <td style={{ padding: '8px 12px' }}>{p.metric}</td>
                      <td style={{ padding: '8px 12px' }}>{p.cooldown_s}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`Uptime History (30 days) — Avg: ${uptimeAvg != null ? uptimeAvg + '%' : '--'} | SLA Target: ${uptimeSla != null ? uptimeSla + '%' : '--'} | Incidents: ${incidents30d != null ? incidents30d : '--'}`}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={uptimeHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day_offset" tick={{ fontSize: 11 }} label={{ value: 'Days Ago', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                <YAxis domain={[99, 100]} tick={{ fontSize: 11 }} label={{ value: 'Uptime %', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="uptime_pct" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} activeDot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {tab === 'definitions' && (
        <>
          {clinicalRelevance && (
            <Card title="Clinical Relevance">
              <p style={{ fontSize: 14, color: '#334155', lineHeight: 1.6, margin: 0 }}>{clinicalRelevance}</p>
            </Card>
          )}

          <Card title="Status Levels">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left', width: '20%' }}>Level</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Description</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(statusLevels).map(([level, info], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>
                        <Badge text={level} color={STATUS_COLORS[level]} />
                      </td>
                      <td style={{ padding: '8px 12px' }}>{info.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Resource Thresholds">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Resource</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Warning</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Critical</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Unit</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(resourceThresholds).map(([resource, info], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{resource}</td>
                      <td style={{ padding: '8px 12px' }}>{info.warning}</td>
                      <td style={{ padding: '8px 12px' }}>{info.critical}</td>
                      <td style={{ padding: '8px 12px' }}>{info.unit}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Cost Thresholds">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Parameter</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(costThresholds).map(([key, val], i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{key}</td>
                      <td style={{ padding: '8px 12px' }}>{typeof val === 'number' ? '$' + fmt(val) : fmt(val)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`Regions (${defRegions.length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>ID</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Label</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Provider</th>
                  </tr>
                </thead>
                <tbody>
                  {defRegions.map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.id}</td>
                      <td style={{ padding: '8px 12px' }}>{r.label}</td>
                      <td style={{ padding: '8px 12px' }}>{r.provider}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title={`Services (${defServices.length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>ID</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Label</th>
                    <th style={{ padding: '8px 12px', textAlign: 'left' }}>Icon</th>
                  </tr>
                </thead>
                <tbody>
                  {defServices.map((s, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 600 }}>{s.id}</td>
                      <td style={{ padding: '8px 12px' }}>{s.label}</td>
                      <td style={{ padding: '8px 12px' }}>{s.icon}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {defAutoscalePolicies.length > 0 && (
            <Card title="Autoscale Policies (Reference)">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ background: '#f1f5f9' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Name</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Min</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Max</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Metric</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left' }}>Cooldown (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {defAutoscalePolicies.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #e2e8f0' }}>
                        <td style={{ padding: '8px 12px', fontWeight: 600 }}>{p.name}</td>
                        <td style={{ padding: '8px 12px' }}>{p.min}</td>
                        <td style={{ padding: '8px 12px' }}>{p.max}</td>
                        <td style={{ padding: '8px 12px' }}>{p.metric}</td>
                        <td style={{ padding: '8px 12px' }}>{p.cooldown_s}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
