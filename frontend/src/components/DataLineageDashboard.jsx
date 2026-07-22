import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  LineChart, Line, AreaChart, Area
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#f97316', '#14b8a6']

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

function StageBadge({ stage }) {
  const colors = {
    'Ingestion': { bg: '#dbeafe', fg: '#1e40af' },
    'Processing': { bg: '#fef3c7', fg: '#92400e' },
    'Analysis': { bg: '#dcfce7', fg: '#166534' },
    'Validation': { bg: '#f3e8ff', fg: '#6b21a8' },
    'Monitoring': { bg: '#cffafe', fg: '#155e75' },
    'Training': { bg: '#fce7f3', fg: '#9d174d' },
    'Logging': { bg: '#f1f5f9', fg: '#475569' },
    'Communication': { bg: '#fff7ed', fg: '#9a3412' },
    'Storage': { bg: '#ecfdf5', fg: '#065f46' },
    'Governance': { bg: '#fef2f2', fg: '#991b1b' },
    'Feedback': { bg: '#fffbeb', fg: '#78350f' },
  }
  const c = colors[stage] || { bg: '#f1f5f9', fg: '#475569' }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 11, fontWeight: 600, background: c.bg, color: c.fg
    }}>{stage}</span>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'flow', label: 'Data Flow' },
  { id: 'patients', label: 'Patient Lineage' },
  { id: 'audit', label: 'Audit Trail' },
  { id: 'definitions', label: 'Definitions' },
]

export default function DataLineageDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/data-lineage/overview`),
      axios.get(`${API_URL}/api/data-lineage/breakdown`),
      axios.get(`${API_URL}/api/data-lineage/definitions`),
    ])
      .then(([oRes, bRes, dRes]) => {
        setOverview(oRes.data)
        setBreakdown(bRes.data)
        setDefs(dRes.data)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading Data Lineage...</div>
  if (error) return <div style={{ padding: 40, color: '#ef4444' }}>Error: {error}</div>
  if (!overview?.available) return <div style={{ padding: 40 }}>Lineage data not available</div>

  const k = overview.kpis || {}

  const stageData = (overview.pipeline_stages || []).map((s, i) => ({
    ...s, fill: COLORS[i % COLORS.length]
  }))
  const domainData = (overview.domain_distribution || []).map((d, i) => ({
    ...d, name: d.domain, value: d.count, fill: COLORS[i % COLORS.length]
  }))

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>
        Data Lineage Dashboard
      </h2>
      <p style={{ color: '#64748b', fontSize: 13, marginBottom: 20 }}>
        End-to-end data flow tracking: ingestion → processing → analysis → reporting from transaction_log ({k.total_events} events)
      </p>

      {/* Tab navigation */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
              background: tab === t.id ? '#1e293b' : '#f1f5f9',
              color: tab === t.id ? '#fff' : '#475569',
            }}
          >{t.label}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 16, marginBottom: 24 }}>
            <Card><KPI label="Total Events" value={k.total_events} color="#3b82f6" /></Card>
            <Card><KPI label="Components" value={k.unique_components} color="#8b5cf6" /></Card>
            <Card><KPI label="Action Types" value={k.unique_actions} color="#10b981" /></Card>
            <Card><KPI label="Actors" value={k.unique_actors} color="#f59e0b" /></Card>
            <Card><KPI label="Patients Tracked" value={k.patients_tracked} color="#ec4899" /></Card>
            <Card><KPI label="Span (days)" value={k.span_days} color="#06b6d4" /></Card>
            <Card><KPI label="Avg Daily Events" value={k.avg_daily_events} color="#f97316" /></Card>
            <Card><KPI label="Lineage Edges" value={k.lineage_edges} sub="component transitions" color="#14b8a6" /></Card>
          </div>

          {/* Pipeline stages + domain distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
            <Card title="Pipeline Stage Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={stageData} margin={{ left: 20, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="stage" angle={-30} textAnchor="end" height={80} fontSize={11} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {stageData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
            <Card title="Domain Distribution">
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={domainData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, value }) => `${name}: ${value}`}>
                    {domainData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Daily trend */}
          <Card title="Daily Event Activity">
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={overview.daily_trend || []} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" fontSize={11} />
                <YAxis />
                <Tooltip />
                <Area type="monotone" dataKey="events" stroke="#3b82f6" fill="#dbeafe" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>

          {/* Hourly pattern */}
          <Card title="Hourly Activity Pattern" span={2}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={overview.hourly_pattern || []} margin={{ left: 20, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" fontSize={11} label={{ value: 'Hour of Day', position: 'insideBottom', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="events" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* ── Data Flow Tab ── */}
      {tab === 'flow' && (
        <>
          {/* Lineage edges (component transitions) */}
          <Card title="Component-to-Component Data Flow (top 20 transitions)">
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['From', 'To', 'Transitions', 'Flow'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.lineage_edges || []).map((e, i) => {
                    const maxW = Math.max(...(overview.lineage_edges || []).map(x => x.weight))
                    const pct = maxW > 0 ? (e.weight / maxW * 100) : 0
                    return (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{e.from}</td>
                        <td style={{ padding: '6px', fontWeight: 600 }}>{e.to}</td>
                        <td style={{ padding: '6px' }}>{e.weight}</td>
                        <td style={{ padding: '6px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ flex: 1, background: '#f1f5f9', borderRadius: 4, height: 16 }}>
                              <div style={{ width: `${pct}%`, background: '#3b82f6', borderRadius: 4, height: 16 }} />
                            </div>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Component breakdown with actions */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16, marginTop: 16 }}>
            {(breakdown?.component_breakdown || []).slice(0, 12).map(cb => (
              <Card key={cb.component} title={`${cb.component} (${cb.total} events)`}>
                <div style={{ fontSize: 12, lineHeight: 2 }}>
                  {cb.actions.map(a => (
                    <div key={a.action} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span><StageBadge stage={breakdown?.action_stages?.find(s => s.action === a.action)?.stage || 'Other'} /> {a.action}</span>
                      <span style={{ fontWeight: 600 }}>{a.count}</span>
                    </div>
                  ))}
                </div>
              </Card>
            ))}
          </div>

          {/* Actor distribution */}
          <Card title="Actor Distribution" span={2}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={(overview.actor_distribution || []).slice(0, 10)} layout="vertical" margin={{ left: 100, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="actor" fontSize={11} width={100} />
                <Tooltip />
                <Bar dataKey="events" fill="#f59e0b" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </>
      )}

      {/* ── Patient Lineage Tab ── */}
      {tab === 'patients' && breakdown && (
        <>
          <Card title={`Patient Data Lineage (${(breakdown.patient_lineage || []).length} patients tracked)`}>
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['Patient', 'Events', 'Components', 'Actions', 'First Seen', 'Last Seen', 'Component Chain'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.patient_lineage || []).map(p => (
                    <tr key={p.patient_id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600 }}>{p.patient_id}</td>
                      <td>{p.total_events}</td>
                      <td>{p.components_touched}</td>
                      <td>{p.action_types}</td>
                      <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(p.first_seen || '').slice(0, 16)}</td>
                      <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(p.last_seen || '').slice(0, 16)}</td>
                      <td style={{ fontSize: 11 }}>
                        {(p.component_chain || []).map((c, i) => (
                          <span key={i}>
                            {i > 0 && <span style={{ color: '#94a3b8', margin: '0 4px' }}>&rarr;</span>}
                            <span style={{ color: '#3b82f6' }}>{c}</span>
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── Audit Trail Tab ── */}
      {tab === 'audit' && breakdown && (
        <>
          {/* Action-stage mapping */}
          <Card title="Action-to-Pipeline-Stage Mapping">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
              {(breakdown.action_stages || []).map(a => (
                <div key={a.action} style={{
                  padding: '4px 12px', borderRadius: 8, background: '#f8fafc',
                  border: '1px solid #e2e8f0', fontSize: 12
                }}>
                  <strong>{a.action}</strong> <StageBadge stage={a.stage} /> <span style={{ color: '#94a3b8' }}>({a.count})</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Recent audit trail */}
          <Card title="Recent Audit Trail (last 30 events)">
            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                    {['#', 'Patient', 'Component', 'Action', 'Actor', 'Detail', 'Timestamp'].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '8px 6px', color: '#64748b' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(breakdown.audit_trail || []).map(r => (
                    <tr key={r.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px', fontWeight: 600, color: '#94a3b8' }}>{r.id}</td>
                      <td>{r.patient_id}</td>
                      <td style={{ fontWeight: 600 }}>{r.component}</td>
                      <td><StageBadge stage={breakdown.action_stages?.find(s => s.action === r.action)?.stage || r.action} /></td>
                      <td>{r.actor}</td>
                      <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: '#64748b' }}>{r.detail}</td>
                      <td style={{ fontSize: 11, color: '#94a3b8', whiteSpace: 'nowrap' }}>{(r.timestamp || '').slice(0, 19)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'definitions' && defs?.definitions && (
        <div style={{ display: 'grid', gap: 16 }}>
          {Object.entries(defs.definitions).map(([section, items]) => (
            <Card key={section} title={section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
              <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                {Object.entries(items).map(([term, desc]) => (
                  <div key={term} style={{ marginBottom: 8 }}>
                    <strong style={{ color: '#1e293b' }}>{term}:</strong>{' '}
                    <span style={{ color: '#475569' }}>{desc}</span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
