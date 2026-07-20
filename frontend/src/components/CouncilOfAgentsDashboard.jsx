import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = '/api'
const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#64748b', '#84cc16', '#f97316', '#14b8a6', '#a855f7']

function fmt(v) {
  if (v == null) return '--'
  return typeof v === 'number' ? (v % 1 === 0 ? v.toLocaleString() : v.toFixed(1)) : String(v)
}

function RoleBadge({ role }) {
  const r = String(role || '').toLowerCase()
  const color = r === 'author' ? '#3b82f6' : r === 'reviewer' ? '#10b981' : r === 'chair' ? '#8b5cf6' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(role || 'N/A')}</span>
  )
}

function StatusBadge({ status }) {
  const s = String(status || '').toLowerCase()
  const color = s === 'built' ? '#10b981' : s === 'planned' ? '#f59e0b' : s === 'scaffold' ? '#64748b' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(status || 'N/A')}</span>
  )
}

function AgreementBadge({ agreement }) {
  const a = String(agreement || '').toLowerCase()
  const color = a === 'agree' ? '#10b981' : a === 'disagree' ? '#ef4444' : a === 'abstain' ? '#f59e0b' : '#64748b'
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      background: color + '22', color, fontWeight: 600, fontSize: 12, textTransform: 'uppercase'
    }}>{String(agreement || 'N/A')}</span>
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

export default function CouncilOfAgentsDashboard() {
  const [tab, setTab] = useState('overview')
  const [ov, setOv] = useState(null)
  const [bd, setBd] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    Promise.all([
      axios.get(`${API_URL}/council-of-agents/overview`),
      axios.get(`${API_URL}/council-of-agents/breakdown`),
      axios.get(`${API_URL}/council-of-agents/definitions`),
    ])
      .then(([o, b, d]) => { setOv(o.data); setBd(b.data); setDefs(d.data) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading council of agents...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>
  if (!ov?.available) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Council of agents data not available.</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'agents', label: 'Agents' },
    { id: 'sessions', label: 'Sessions' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const s = ov.summary || {}

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#64748b',
            fontWeight: 600, fontSize: 13,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <Card title="Council Summary" span={3}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: '8px 0' }}>
              <KPI label="Total Agents" value={s.total_agents} color="#1e293b" />
              <KPI label="Authors" value={s.authors} color="#3b82f6" />
              <KPI label="Reviewers" value={s.reviewers} color="#10b981" />
              <KPI label="Chairs" value={s.chairs} color="#8b5cf6" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, padding: '8px 0', marginTop: 8 }}>
              <KPI label="Decisions Reviewed" value={s.decisions_reviewed} color="#06b6d4" />
              <KPI label="Consensus Rate" value={s.consensus_rate} sub="%" color="#10b981" />
              <KPI label="Override Rate" value={s.override_rate} sub="%" color="#ef4444" />
              <KPI label="Coverage" value={s.coverage_pct} sub="%" color="#f59e0b" />
            </div>
          </Card>

          <Card title="Role Distribution">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={ov.role_distribution || []} cx="50%" cy="50%" outerRadius={90} dataKey="count" nameKey="role" label={({ role, percent }) => `${role} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.role_distribution || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Decision Quality Trend" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={ov.decision_quality_trend || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="decisions" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Review Status Breakdown">
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={ov.review_status_breakdown || []} cx="50%" cy="50%" outerRadius={90} dataKey="count" nameKey="status" label={({ status, percent }) => `${status} ${(percent * 100).toFixed(0)}%`}>
                  {(ov.review_status_breakdown || []).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Council Events" span={2}>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ov.council_events || []} layout="vertical" margin={{ left: 120 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="action" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </div>
      )}

      {tab === 'agents' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title={`Agent Roster (${(bd.agent_roster || []).length})`}>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['ID', 'Task', 'Module', 'Role', 'Status'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.agent_roster || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600, fontFamily: 'monospace' }}>{r.id}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 300 }}>{r.task}</td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{r.module}</td>
                      <td style={{ padding: '6px 10px' }}><RoleBadge role={r.role} /></td>
                      <td style={{ padding: '6px 10px' }}><StatusBadge status={r.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'sessions' && bd && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Recent Sessions">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Date', 'Total Events', 'Components', 'Action Types', 'Participants'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.recent_sessions || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.date}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.total_events)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.components)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.action_types)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.participants)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Voting History">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Patient', 'AI Prediction', 'Confidence', 'Agreement', 'Final Decision', 'Reviewer', 'Date', 'Source'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.voting_history || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.patient_id}</td>
                      <td style={{ padding: '6px 10px' }}>{r.ai_prediction}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.confidence)}</td>
                      <td style={{ padding: '6px 10px' }}><AgreementBadge agreement={r.agreement} /></td>
                      <td style={{ padding: '6px 10px' }}>{r.final_decision}</td>
                      <td style={{ padding: '6px 10px' }}>{r.reviewer}</td>
                      <td style={{ padding: '6px 10px' }}>{r.date}</td>
                      <td style={{ padding: '6px 10px', fontSize: 11, color: '#64748b' }}>{r.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Review Assignments">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f8fafc' }}>
                    {['Component', 'Review Count', 'Reviewers'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(bd.review_assignments || []).map((r, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 600 }}>{r.component}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.review_count)}</td>
                      <td style={{ padding: '6px 10px', textAlign: 'center' }}>{fmt(r.reviewers)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {tab === 'definitions' && defs && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 16 }}>
          <Card title="Council Definitions">
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f8fafc' }}>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569', width: 220 }}>Term</th>
                  <th style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e2e8f0', color: '#475569' }}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {(defs.definitions || []).map((d, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{d.term}</td>
                    <td style={{ padding: '6px 10px', lineHeight: 1.5 }}>{d.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Compliance References">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.compliance_refs || []).map((r, i) => (
                <li key={i} style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 8, color: '#334155' }}>{r}</li>
              ))}
            </ul>
          </Card>

          <Card title="Remediation Checklist">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {(defs.remediation || []).map((r, i) => (
                <li key={i} style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 8, color: '#334155' }}>{r}</li>
              ))}
            </ul>
          </Card>
        </div>
      )}
    </div>
  )
}
