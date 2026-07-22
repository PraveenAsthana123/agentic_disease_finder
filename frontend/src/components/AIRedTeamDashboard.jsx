import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b']
const SEVERITY_COLORS = { Critical: '#ef4444', High: '#f59e0b', Medium: '#3b82f6', Low: '#10b981' }
const STATUS_COLORS = { active: '#ef4444', mitigated: '#10b981', detected: '#f59e0b', flagged: '#8b5cf6', vulnerable: '#ef4444', tested: '#10b981', untested: '#94a3b8' }

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? v.toLocaleString() : String(v)
}

export default function AIRedTeamDashboard() {
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
          axios.get(`${API_URL}/api/ai-red-team/overview`),
          axios.get(`${API_URL}/api/ai-red-team/breakdown`),
          axios.get(`${API_URL}/api/ai-red-team/definitions`)
        ])
        setOverview(ov.data)
        setBreakdown(br.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load red team data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return (
    <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>&#9878;</div>
      Loading AI Red Team data...
    </div>
  )

  if (error) return (
    <div style={{ padding: 20, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#991b1b' }}>
      Error: {error}
    </div>
  )

  if (!overview?.available) return (
    <div style={{ padding: 20, background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, color: '#92400e' }}>
      {overview?.note || 'Red team data not available.'}
    </div>
  )

  const k = overview.kpis || {}
  const attackDist = overview.attack_type_distribution || []
  const dailyEvents = overview.daily_security_events || []
  const vulnMatrix = overview.vulnerability_matrix || []
  const surfaceMap = overview.attack_surface_map || []
  const advTests = breakdown?.adversarial_tests || []
  const injectionLog = breakdown?.prompt_injection_log || []
  const jailbreakAttempts = breakdown?.jailbreak_attempts || []
  const toolAbuseLog = breakdown?.tool_abuse_log || []
  const blockedEvents = breakdown?.blocked_events || []
  const attackVectors = breakdown?.attack_vectors || []

  const cardStyle = { background: '#fff', borderRadius: 10, boxShadow: '0 1px 4px rgba(0,0,0,0.07)', padding: 20, marginBottom: 18 }
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: '8px 8px 0 0', fontWeight: active ? 700 : 400,
    background: active ? '#991b1b' : '#f1f5f9', color: active ? '#fff' : '#64748b',
    border: 'none', fontSize: 13, marginRight: 4
  })

  const kpiItems = [
    { label: 'Tests Run', value: k.total_tests_run, color: '#3b82f6' },
    { label: 'Adversarial Detected', value: k.adversarial_attempts_detected, color: '#ef4444' },
    { label: 'Jailbreak Attempts', value: k.jailbreak_attempts, color: '#f59e0b' },
    { label: 'Injection Scans', value: k.prompt_injection_scans, color: '#8b5cf6' },
    { label: 'Tool Abuse', value: k.tool_abuse_incidents, color: '#ec4899' },
    { label: 'Blocked Events', value: k.blocked_events, color: '#ef4444' },
    { label: 'Vuln Score', value: k.vulnerability_score != null ? `${k.vulnerability_score}` : '--', color: k.vulnerability_score > 50 ? '#ef4444' : '#10b981' },
    { label: 'Coverage', value: k.coverage_pct != null ? `${k.coverage_pct}%` : '--', color: k.coverage_pct >= 80 ? '#10b981' : '#f59e0b' },
  ]

  const kpiStyle = (color) => ({
    background: `${color}11`, border: `1px solid ${color}33`, borderRadius: 8,
    padding: '14px 18px', textAlign: 'center', minWidth: 110
  })

  const statusBadge = (status) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${STATUS_COLORS[status] || '#94a3b8'}22`, color: STATUS_COLORS[status] || '#64748b'
  })

  const severityBadge = (sev) => ({
    display: 'inline-block', padding: '2px 10px', borderRadius: 12, fontSize: 11, fontWeight: 600,
    background: `${SEVERITY_COLORS[sev] || '#94a3b8'}22`, color: SEVERITY_COLORS[sev] || '#64748b'
  })

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'adversarial', label: 'Adversarial Tests' },
    { id: 'injection', label: 'Prompt Injection' },
    { id: 'tool-abuse', label: 'Tool Abuse' },
    { id: 'definitions', label: 'Definitions' }
  ]

  return (
    <div style={{ padding: '18px 24px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>AI Red Team Dashboard</h2>
      </div>

      {/* Tab bar */}
      <div style={{ marginBottom: 18 }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(tab === t.id)} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {kpiItems.map((ki, i) => (
              <div key={i} style={kpiStyle(ki.color)}>
                <div style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>{ki.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: ki.color }}>
                  {typeof ki.value === 'number' ? fmt(ki.value) : (ki.value || '--')}
                </div>
              </div>
            ))}
          </div>

          {/* Charts row: Attack type distribution + Vulnerability matrix */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Attack Type Distribution</h4>
              {attackDist.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={attackDist} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={85}
                      label={({ name, value }) => `${name}: ${value}`}>
                      {attackDist.map((d, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No attacks detected</div>
              )}
            </div>

            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Vulnerability Matrix</h4>
              {vulnMatrix.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={vulnMatrix}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="severity" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Incidents">
                      {vulnMatrix.map((d, i) => (
                        <Cell key={i} fill={SEVERITY_COLORS[d.severity] || '#94a3b8'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div style={{ color: '#94a3b8', textAlign: 'center', paddingTop: 60 }}>No vulnerability data</div>
              )}
            </div>
          </div>

          {/* Daily security events timeline */}
          {dailyEvents.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Daily Security Events</h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={dailyEvents}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="events" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} name="Events" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Attack surface map */}
          {surfaceMap.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Attack Surface Map</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Events</th>
                      <th style={{ padding: '8px 10px' }}>Exposure</th>
                      <th style={{ padding: '8px 10px' }}>Monitoring</th>
                    </tr>
                  </thead>
                  <tbody>
                    {surfaceMap.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.component}</td>
                        <td style={{ padding: '8px 10px' }}>{fmt(r.events)}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={statusBadge(r.exposure === 'high' ? 'vulnerable' : r.exposure === 'medium' ? 'flagged' : 'tested')}>
                            {r.exposure}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px' }}>
                          {r.has_monitoring ? '\u2705 Active' : '\u274C None'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Attack vectors summary */}
          {attackVectors.length > 0 && (
            <div style={cardStyle}>
              <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Attack Vectors</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
                {attackVectors.map((v, i) => (
                  <div key={i} style={{
                    border: `1px solid ${v.status === 'mitigated' ? '#bbf7d0' : '#fecaca'}`,
                    borderRadius: 8, padding: 14,
                    background: v.status === 'mitigated' ? '#f0fdf4' : '#fef2f2'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <span style={{ fontSize: 16 }}>{v.status === 'mitigated' ? '\u2705' : '\u26A0\uFE0F'}</span>
                      <span style={{ fontWeight: 600, color: '#1e293b', fontSize: 13 }}>{v.category}</span>
                    </div>
                    <div style={{ fontSize: 12, color: '#475569' }}>
                      {v.count} incident{v.count !== 1 ? 's' : ''} &middot; Status: <span style={statusBadge(v.status)}>{v.status}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ADVERSARIAL TESTS TAB */}
      {tab === 'adversarial' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Per-Component Adversarial Test Results ({advTests.length})</h4>
            {advTests.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Total Events</th>
                      <th style={{ padding: '8px 10px' }}>Blocked</th>
                      <th style={{ padding: '8px 10px' }}>Monitored</th>
                      <th style={{ padding: '8px 10px' }}>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {advTests.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600 }}>{r.component}</td>
                        <td style={{ padding: '8px 10px' }}>{fmt(r.total_events)}</td>
                        <td style={{ padding: '8px 10px', color: r.blocked > 0 ? '#ef4444' : '#64748b', fontWeight: r.blocked > 0 ? 700 : 400 }}>{r.blocked}</td>
                        <td style={{ padding: '8px 10px' }}>{r.monitored}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={statusBadge(r.status)}>{r.status}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', padding: 30 }}>No adversarial test data</div>
            )}
          </div>

          {/* Blocked events detail */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Blocked Events ({blockedEvents.length})</h4>
            {blockedEvents.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>ID</th>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Detail</th>
                      <th style={{ padding: '8px 10px' }}>Actor</th>
                      <th style={{ padding: '8px 10px' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {blockedEvents.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.id}</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.patient_id || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>{r.component}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569' }}>{r.detail || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>{r.actor || '--'}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>
                          {r.timestamp ? new Date(r.timestamp).toLocaleString() : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#94a3b8', textAlign: 'center', padding: 30 }}>No blocked events recorded</div>
            )}
          </div>
        </>
      )}

      {/* PROMPT INJECTION TAB */}
      {tab === 'injection' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Prompt Injection Scan Results ({injectionLog.length})</h4>
            {injectionLog.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>ID</th>
                      <th style={{ padding: '8px 10px' }}>Role</th>
                      <th style={{ padding: '8px 10px' }}>Severity</th>
                      <th style={{ padding: '8px 10px' }}>Pattern</th>
                      <th style={{ padding: '8px 10px' }}>Snippet</th>
                      <th style={{ padding: '8px 10px' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {injectionLog.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.id}</td>
                        <td style={{ padding: '8px 10px' }}>{r.role || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={severityBadge(r.severity === 'high' ? 'High' : 'Medium')}>
                            {r.severity || '--'}
                          </span>
                        </td>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace', fontSize: 11 }}>{r.pattern_matched || '--'}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569', maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.snippet || '--'}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>
                          {r.timestamp ? new Date(r.timestamp).toLocaleString() : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#10b981', textAlign: 'center', padding: 30, fontWeight: 600 }}>No prompt injection attempts detected</div>
            )}
          </div>

          {/* Jailbreak attempts */}
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Jailbreak Attempts ({jailbreakAttempts.length})</h4>
            {jailbreakAttempts.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>ID</th>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Action</th>
                      <th style={{ padding: '8px 10px' }}>Detail</th>
                      <th style={{ padding: '8px 10px' }}>Status</th>
                      <th style={{ padding: '8px 10px' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jailbreakAttempts.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.id}</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.patient_id || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>{r.component}</td>
                        <td style={{ padding: '8px 10px' }}>{r.action}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569' }}>{r.detail || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={statusBadge(r.status)}>{r.status}</span>
                        </td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>
                          {r.timestamp ? new Date(r.timestamp).toLocaleString() : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#10b981', textAlign: 'center', padding: 30, fontWeight: 600 }}>No jailbreak attempts detected</div>
            )}
          </div>
        </>
      )}

      {/* TOOL ABUSE TAB */}
      {tab === 'tool-abuse' && (
        <>
          <div style={cardStyle}>
            <h4 style={{ margin: '0 0 12px', color: '#334155' }}>Tool Abuse Log ({toolAbuseLog.length})</h4>
            {toolAbuseLog.length > 0 ? (
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #e2e8f0', textAlign: 'left' }}>
                      <th style={{ padding: '8px 10px' }}>ID</th>
                      <th style={{ padding: '8px 10px' }}>Patient</th>
                      <th style={{ padding: '8px 10px' }}>Component</th>
                      <th style={{ padding: '8px 10px' }}>Action</th>
                      <th style={{ padding: '8px 10px' }}>Actor</th>
                      <th style={{ padding: '8px 10px' }}>Detail</th>
                      <th style={{ padding: '8px 10px' }}>Status</th>
                      <th style={{ padding: '8px 10px' }}>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {toolAbuseLog.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i % 2 ? '#f8fafc' : '#fff' }}>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.id}</td>
                        <td style={{ padding: '8px 10px', fontFamily: 'monospace' }}>{r.patient_id || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>{r.component}</td>
                        <td style={{ padding: '8px 10px' }}>{r.action}</td>
                        <td style={{ padding: '8px 10px' }}>{r.actor || '--'}</td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#475569' }}>{r.detail || '--'}</td>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={statusBadge(r.status)}>{r.status}</span>
                        </td>
                        <td style={{ padding: '8px 10px', fontSize: 12, color: '#64748b' }}>
                          {r.timestamp ? new Date(r.timestamp).toLocaleString() : '--'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ color: '#10b981', textAlign: 'center', padding: 30, fontWeight: 600 }}>No tool abuse incidents detected</div>
            )}
          </div>
        </>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'definitions' && defs?.definitions && (
        <div style={{ ...cardStyle, background: '#fef2f2', border: '1px solid #fecaca' }}>
          <h4 style={{ margin: '0 0 14px', color: '#991b1b' }}>AI Red Team Metric Definitions</h4>
          {defs.definitions.map((d, i) => (
            <div key={i} style={{ marginBottom: 12, paddingBottom: 10, borderBottom: i < defs.definitions.length - 1 ? '1px solid #fde8e8' : 'none' }}>
              <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 2 }}>{d.metric}</div>
              <div style={{ color: '#334155', fontSize: 13, marginBottom: 2 }}>{d.description}</div>
              <div style={{ color: '#64748b', fontSize: 11 }}>Source: {d.source}</div>
              {d.compliance && (
                <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>Compliance: {d.compliance}</div>
              )}
            </div>
          ))}
        </div>
      )}

      <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'right', marginTop: 8 }}>
        Source: clinical.db (transaction_log, conversation_log, hitl_reviews, expert_reviews, clinical_decisions)
      </div>
    </div>
  )
}
