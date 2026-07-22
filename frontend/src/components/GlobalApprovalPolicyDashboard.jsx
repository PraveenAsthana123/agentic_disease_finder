import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#3b82f6', '#22c55e', '#f97316', '#ef4444', '#8b5cf6', '#14b8a6', '#ec4899', '#eab308']
const DECISION_COLORS = { allow: '#22c55e', require_human_approval: '#f97316', deny: '#ef4444' }
const RISK_COLORS = { low: '#22c55e', medium: '#f97316', high: '#ef4444', critical: '#7f1d1d' }

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 8, padding: 16, marginBottom: 16,
      boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 12px', fontSize: 15, fontWeight: 600, color: '#334155' }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, sub }) {
  return (
    <div style={{ textAlign: 'center', padding: '8px 12px' }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function DecisionBadge({ decision }) {
  const bg = DECISION_COLORS[decision] || '#94a3b8'
  const label = decision === 'require_human_approval' ? 'HITL' : decision
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {label}
    </span>
  )
}

function RiskBadge({ band }) {
  const bg = RISK_COLORS[band] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600, textTransform: 'uppercase'
    }}>
      {band}
    </span>
  )
}

function RoleBadge({ role }) {
  const roleColors = {
    clinical_reviewer: '#3b82f6', data_steward: '#14b8a6', model_owner: '#8b5cf6',
    security_officer: '#ef4444', governance_lead: '#ec4899'
  }
  const bg = roleColors[role] || '#94a3b8'
  return (
    <span style={{
      background: `${bg}22`, color: bg, border: `1px solid ${bg}55`,
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600
    }}>
      {role ? role.replace(/_/g, ' ') : '—'}
    </span>
  )
}

const thStyle = {
  padding: '8px 10px', textAlign: 'left', fontSize: 11, fontWeight: 600,
  color: '#64748b', borderBottom: '2px solid #e2e8f0', position: 'sticky', top: 0, background: '#fff'
}
const tdStyle = { padding: '7px 10px', fontSize: 12, borderBottom: '1px solid #f1f5f9', color: '#334155' }

export default function GlobalApprovalPolicyDashboard() {
  const [overview, setOverview] = useState(null)
  const [breakdown, setBreakdown] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState('overview')

  useEffect(() => {
    Promise.all([
      axios.get(`${API_URL}/api/global-approval-policy/overview`),
      axios.get(`${API_URL}/api/global-approval-policy/breakdown`),
      axios.get(`${API_URL}/api/global-approval-policy/definitions`),
    ]).then(([o, b, d]) => {
      setOverview(o.data)
      setBreakdown(b.data)
      setDefs(d.data)
    }).catch(console.error).finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center', color: '#64748b' }}>Loading Global Approval Policy…</div>
  if (!overview?.available) return <div style={{ padding: 32, color: '#ef4444' }}>global_approval_policy.json not found</div>

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'rules', label: 'Policy Rules' },
    { id: 'roles', label: 'Approval Roles' },
    { id: 'risk', label: 'Risk Bands' },
    { id: 'definitions', label: 'Definitions' },
  ]

  const s = overview.summary

  return (
    <div style={{ padding: '20px 24px', fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#0f172a' }}>🛡️ Global Approval Policy Dashboard</h2>
        <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
          {overview.policy_name} v{overview.version} · Status: {overview.status} · Classification: {overview.classification}
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '6px 14px', fontSize: 12, fontWeight: tab === t.id ? 700 : 500,
            background: tab === t.id ? '#3b82f6' : '#fff', color: tab === t.id ? '#fff' : '#475569',
            border: `1px solid ${tab === t.id ? '#3b82f6' : '#e2e8f0'}`, borderRadius: 6, cursor: 'pointer'
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'overview' && (
        <>
          <Card>
            <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 16 }}>
              <KPI label="Policy Rules" value={s.total_rules} />
              <KPI label="Approval Roles" value={s.total_roles} />
              <KPI label="Risk Bands" value={s.total_risk_bands} />
              <KPI label="Decision Types" value={s.total_decision_types} />
              <KPI label="Domains Covered" value={s.applies_to_domains} />
              <KPI label="Approvable Scopes" value={s.total_approvable_scopes} />
              <KPI label="Role-Gated Rules" value={s.rules_requiring_role} />
              <KPI label="Audit Fields" value={s.audit_required_fields} />
              <KPI label="Retention (days)" value={s.retention_days} sub="~7 years" />
            </div>
          </Card>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Card title="Decision Distribution">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={overview.decision_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name === 'require_human_approval' ? 'HITL' : name}: ${value}`}>
                    {overview.decision_distribution.map((_, i) => (
                      <Cell key={i} fill={DECISION_COLORS[overview.decision_distribution[i].name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            <Card title="Domains Covered">
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {overview.applies_to.map((d, i) => (
                  <span key={i} style={{
                    background: '#eff6ff', color: '#3b82f6', border: '1px solid #bfdbfe',
                    borderRadius: 6, padding: '6px 12px', fontSize: 12, fontWeight: 500
                  }}>
                    {d.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
              <div style={{ marginTop: 16, fontSize: 12, color: '#64748b' }}>
                Default decision: <DecisionBadge decision={overview.default_decision} />
              </div>
            </Card>
          </div>

          <Card title="Policy Rules Summary">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    <th style={thStyle}>Rule ID</th>
                    <th style={thStyle}>Name</th>
                    <th style={thStyle}>Decision</th>
                    <th style={thStyle}>Required Role</th>
                    <th style={thStyle}>Audit</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.rules_table.map((r, i) => (
                    <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                      <td style={{ ...tdStyle, fontFamily: 'monospace', fontWeight: 600 }}>{r.rule_id}</td>
                      <td style={tdStyle}>{r.name}</td>
                      <td style={tdStyle}><DecisionBadge decision={r.decision} /></td>
                      <td style={tdStyle}>{r.required_role ? <RoleBadge role={r.required_role} /> : <span style={{ color: '#94a3b8' }}>—</span>}</td>
                      <td style={tdStyle}>{r.required_audit ? '✓' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}

      {tab === 'rules' && breakdown && (
        <>
          {breakdown.per_rule.map((r, i) => (
            <Card key={i} title={`${r.rule_id}: ${r.name}`}>
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 12 }}>
                <div><span style={{ fontSize: 11, color: '#64748b' }}>Decision: </span><DecisionBadge decision={r.decision} /></div>
                {r.required_role && <div><span style={{ fontSize: 11, color: '#64748b' }}>Required Role: </span><RoleBadge role={r.required_role} /></div>}
                <div><span style={{ fontSize: 11, color: '#64748b' }}>Audit: </span><span style={{ fontSize: 12 }}>{r.required_audit ? '✓ Required' : '—'}</span></div>
              </div>
              <div style={{ background: '#f8fafc', borderRadius: 6, padding: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 6 }}>Match Criteria:</div>
                {Object.entries(r.match_criteria).map(([k, v], j) => (
                  <div key={j} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>
                    <span style={{ fontWeight: 600 }}>{k}: </span>
                    <span style={{ fontFamily: 'monospace', fontSize: 11 }}>
                      {Array.isArray(v) ? v.join(', ') : typeof v === 'boolean' ? (v ? 'true' : 'false') : String(v)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </>
      )}

      {tab === 'roles' && breakdown && (
        <>
          <Card title="Approval Role — Scope Matrix">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={breakdown.per_role} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="role" tick={{ fontSize: 10 }} angle={-20} textAnchor="end" height={60}
                  tickFormatter={v => v.replace(/_/g, ' ')} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip formatter={(v) => [v, 'Scopes']} />
                <Bar dataKey="scope_count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {breakdown.per_role.map((r, i) => (
            <Card key={i} title={r.role.replace(/_/g, ' ')}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>{r.scope_count} approvable scopes</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {r.can_approve.map((scope, j) => (
                  <span key={j} style={{
                    background: '#f0fdf4', color: '#166534', border: '1px solid #bbf7d0',
                    borderRadius: 4, padding: '4px 10px', fontSize: 11, fontWeight: 500
                  }}>
                    {scope.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </Card>
          ))}

          {breakdown.hitl && (
            <Card title="Human-In-The-Loop (HITL) Workflow">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Queue</div>
                  <div style={{ fontSize: 12, fontFamily: 'monospace' }}>{breakdown.hitl.queue}</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Default SLA</div>
                  <div style={{ fontSize: 12 }}>{breakdown.hitl.default_sla_minutes} min</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Critical SLA</div>
                  <div style={{ fontSize: 12, color: '#ef4444', fontWeight: 600 }}>{breakdown.hitl.critical_sla_minutes} min</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Allowed Decisions</div>
                  <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                    {breakdown.hitl.allowed_decisions.map((d, j) => (
                      <span key={j} style={{
                        background: '#eff6ff', color: '#3b82f6', border: '1px solid #bfdbfe',
                        borderRadius: 4, padding: '2px 8px', fontSize: 11
                      }}>{d}</span>
                    ))}
                  </div>
                </div>
              </div>
              <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span style={{ background: '#dbeafe', color: '#1d4ed8', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>Agent Request</span>
                <span style={{ color: '#94a3b8', fontSize: 16 }}>→</span>
                <span style={{ background: '#fef3c7', color: '#92400e', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>Approval Queue</span>
                <span style={{ color: '#94a3b8', fontSize: 16 }}>→</span>
                <span style={{ background: '#dcfce7', color: '#166534', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>Human Review</span>
                <span style={{ color: '#94a3b8', fontSize: 16 }}>→</span>
                <span style={{ background: '#f0fdf4', color: '#15803d', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600 }}>Approved / Denied / Escalated</span>
              </div>
            </Card>
          )}
        </>
      )}

      {tab === 'risk' && breakdown && (
        <>
          <Card title="Risk Band Decision Matrix">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={breakdown.per_risk_band} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="band" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false}
                  tickFormatter={() => ''} />
                <Tooltip formatter={(v, name, props) => [props.payload.default_decision, 'Default Decision']} />
                <Bar dataKey={(d) => 1} fill="#3b82f6" radius={[4, 4, 0, 0]}>
                  {breakdown.per_risk_band.map((entry, i) => (
                    <Cell key={i} fill={RISK_COLORS[entry.band] || COLORS[i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {breakdown.per_risk_band.map((rb, i) => (
            <Card key={i}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                <RiskBadge band={rb.band} />
                <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>
                  Default: <DecisionBadge decision={rb.default_decision} />
                </span>
              </div>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Examples:</div>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {rb.examples.map((ex, j) => (
                  <li key={j} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{ex}</li>
                ))}
              </ul>
            </Card>
          ))}

          <Card title="Audit Requirements">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Required Fields ({breakdown.audit_required_fields.length})</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {breakdown.audit_required_fields.map((f, i) => (
                    <span key={i} style={{
                      background: '#f1f5f9', color: '#475569', borderRadius: 4,
                      padding: '3px 8px', fontSize: 11, fontFamily: 'monospace'
                    }}>{f}</span>
                  ))}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#334155', marginBottom: 8 }}>Approval Fields ({breakdown.audit_approval_fields.length})</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {breakdown.audit_approval_fields.map((f, i) => (
                    <span key={i} style={{
                      background: '#fef3c7', color: '#92400e', borderRadius: 4,
                      padding: '3px 8px', fontSize: 11, fontFamily: 'monospace'
                    }}>{f}</span>
                  ))}
                </div>
              </div>
            </div>
            <div style={{ marginTop: 12, fontSize: 12, color: '#64748b' }}>
              Retention: <strong>{breakdown.retention_days} days</strong> (~{Math.round(breakdown.retention_days / 365)} years)
            </div>
          </Card>

          {breakdown.change_control && (
            <Card title="Change Control">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                {Object.entries(breakdown.change_control).map(([k, v], i) => (
                  <div key={i}>
                    <div style={{ fontSize: 11, color: '#64748b' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#334155' }}>
                      {typeof v === 'boolean' ? (v ? '✓ Yes' : '✗ No') : String(v).replace(/_/g, ' ')}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}

      {tab === 'definitions' && defs && (
        <>
          <Card title="Decision Types">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Decision</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {defs.decision_types.map((d, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><DecisionBadge decision={d.decision} /></td>
                    <td style={tdStyle}>{d.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Approval Role Descriptions">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Role</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {defs.role_descriptions.map((r, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><RoleBadge role={r.role} /></td>
                    <td style={tdStyle}>{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Risk Band Legend">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Band</th>
                  <th style={thStyle}>Description</th>
                </tr>
              </thead>
              <tbody>
                {defs.risk_band_legend.map((rb, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={tdStyle}><RiskBadge band={rb.band} /></td>
                    <td style={tdStyle}>{rb.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Glossary">
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={thStyle}>Term</th>
                  <th style={thStyle}>Definition</th>
                </tr>
              </thead>
              <tbody>
                {defs.glossary.map((g, i) => (
                  <tr key={i} style={{ background: i % 2 ? '#f8fafc' : '#fff' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{g.term}</td>
                    <td style={tdStyle}>{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          <Card title="Clinical Notes">
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              {defs.clinical_notes.map((n, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 6 }}>{n}</li>
              ))}
            </ul>
          </Card>

          <Card title="References">
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              {defs.references.map((r, i) => (
                <li key={i} style={{ fontSize: 12, color: '#334155', marginBottom: 4 }}>{r}</li>
              ))}
            </ol>
          </Card>
        </>
      )}
    </div>
  )
}
