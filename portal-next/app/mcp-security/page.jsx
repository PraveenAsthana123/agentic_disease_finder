'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function McpSecurityDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/mcp-security/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/mcp-security/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/mcp-security/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!overview.available) return <div className="alert alert-warning">{overview.note || 'No security data'}</div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'access', label: 'Access & Audit' },
    { id: 'activity', label: 'Activity & Trends' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const riskColor = (level) => level === 'high' ? 'danger' : level === 'medium' ? 'warning' : 'success';
  const exposureColor = (e) => e === 'high' ? 'danger' : e === 'medium' ? 'warning' : 'info';

  return (
    <div>
      <h3>MCP Security Dashboard</h3>
      <p className="text-muted">Guardrail enforcement, access control audit, actor privilege analysis, and security event patterns from clinical.db</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Transactions', value: s.total_transactions || 0, color: 'primary' },
          { label: 'Guardrail Events', value: s.guardrail_events || 0, color: s.guardrail_events > 0 ? 'warning' : 'secondary' },
          { label: 'Guardrail Rate', value: `${s.guardrail_rate_pct || 0}%`, color: 'info' },
          { label: 'Blocked', value: s.blocked_events || 0, color: s.blocked_events > 0 ? 'danger' : 'success' },
          { label: 'Sign-offs', value: s.signoff_events || 0, color: s.signoff_events > 0 ? 'success' : 'secondary' },
          { label: 'Expert Reviews', value: s.expert_reviews || 0, color: s.expert_reviews > 0 ? 'success' : 'secondary' },
          { label: 'HITL Reviews', value: s.hitl_reviews || 0, color: s.hitl_reviews > 0 ? 'success' : 'secondary' },
          { label: 'Oversight Rate', value: `${s.oversight_rate_pct || 0}%`, color: s.oversight_rate_pct >= 1 ? 'success' : 'warning' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Actor Privilege Matrix</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Actor</th><th>Txns</th><th>Comps</th><th>Patients</th><th>Risk</th></tr></thead>
                  <tbody>
                    {(overview.actor_privileges || []).map(a => (
                      <tr key={a.actor}>
                        <td>
                          <strong>{a.actor}</strong>
                          {a.is_security_actor && <span className="badge bg-info ms-1">security</span>}
                          {a.privileged_actions?.length > 0 && (
                            <div className="text-muted small">{a.privileged_actions.join(', ')}</div>
                          )}
                        </td>
                        <td>{a.transactions}</td>
                        <td>{a.components}</td>
                        <td>{a.patients_accessed}</td>
                        <td><span className={`badge bg-${riskColor(a.risk_level)}`}>{a.risk_level}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Attack Surface (Component Exposure)</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Component</th><th>Actors</th><th>Transactions</th><th>Exposure</th></tr></thead>
                  <tbody>
                    {(overview.attack_surface || []).map(c => (
                      <tr key={c.component}>
                        <td>{c.component}</td>
                        <td>{c.distinct_actors}</td>
                        <td>{c.transactions}</td>
                        <td><span className={`badge bg-${exposureColor(c.exposure)}`}>{c.exposure}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Guardrail Event Log</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Component</th><th>Patient</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(overview.guardrail_log || []).map((e, i) => (
                      <tr key={i}>
                        <td className="small text-nowrap">{e.timestamp?.slice(0, 19)}</td>
                        <td>{e.actor}</td>
                        <td><span className={`badge bg-${e.action === 'blocked' ? 'danger' : e.action === 'sign-off' ? 'success' : 'warning'}`}>{e.action}</span></td>
                        <td>{e.component}</td>
                        <td>{e.patient_id || '-'}</td>
                        <td className="small">{e.detail?.slice(0, 80)}</td>
                      </tr>
                    ))}
                    {(overview.guardrail_log || []).length === 0 && (
                      <tr><td colSpan={6} className="text-muted text-center">No guardrail events</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Security Agent Activity</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Time</th><th>Agent</th><th>Action</th><th>Component</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(overview.security_agent_events || []).map((e, i) => (
                      <tr key={i}>
                        <td className="small text-nowrap">{e.timestamp?.slice(0, 19)}</td>
                        <td><span className="badge bg-info">{e.actor}</span></td>
                        <td>{e.action}</td>
                        <td>{e.component}</td>
                        <td className="small">{e.detail?.slice(0, 100)}</td>
                      </tr>
                    ))}
                    {(overview.security_agent_events || []).length === 0 && (
                      <tr><td colSpan={5} className="text-muted text-center">No security agent events</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Access & Audit tab */}
      {tab === 'access' && breakdown && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Patient Data Access Audit</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Patient</th><th>Actors</th><th>Components</th><th>Events</th><th>Risk</th></tr></thead>
                  <tbody>
                    {(breakdown.patient_access_audit || []).slice(0, 20).map(p => (
                      <tr key={p.patient_id}>
                        <td className="font-monospace small">{p.patient_id}</td>
                        <td>{p.distinct_actors}</td>
                        <td>{p.distinct_components}</td>
                        <td>{p.total_events}</td>
                        <td><span className={`badge bg-${p.access_risk === 'elevated' ? 'warning' : 'success'}`}>{p.access_risk}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Actor-Component Access Matrix</div>
              <div className="card-body p-0" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {(breakdown.actor_component_matrix || []).map(a => (
                  <div key={a.actor} className="px-3 py-2 border-bottom">
                    <strong>{a.actor}</strong>
                    <div className="d-flex flex-wrap gap-1 mt-1">
                      {a.components.map(c => (
                        <span key={c.component} className="badge bg-secondary">{c.component} ({c.count})</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Expert Reviews</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Patient</th><th>Time</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(breakdown.expert_reviews || []).map((r, i) => (
                      <tr key={i}>
                        <td>{r.patient_id}</td>
                        <td className="small text-nowrap">{r.timestamp?.slice(0, 19)}</td>
                        <td className="small">{r.detail?.slice(0, 120)}</td>
                      </tr>
                    ))}
                    {(breakdown.expert_reviews || []).length === 0 && (
                      <tr><td colSpan={3} className="text-muted text-center">No expert reviews</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">HITL Reviews</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Patient</th><th>Time</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(breakdown.hitl_reviews || []).map((r, i) => (
                      <tr key={i}>
                        <td>{r.patient_id}</td>
                        <td className="small text-nowrap">{r.timestamp?.slice(0, 19)}</td>
                        <td className="small">{r.detail?.slice(0, 120)}</td>
                      </tr>
                    ))}
                    {(breakdown.hitl_reviews || []).length === 0 && (
                      <tr><td colSpan={3} className="text-muted text-center">No HITL reviews</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recent Privileged Events</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Time</th><th>Actor</th><th>Action</th><th>Component</th><th>Patient</th><th>Detail</th></tr></thead>
                  <tbody>
                    {(breakdown.recent_privileged_events || []).map((e, i) => (
                      <tr key={i}>
                        <td className="small text-nowrap">{e.timestamp?.slice(0, 19)}</td>
                        <td>{e.actor}</td>
                        <td><span className="badge bg-warning text-dark">{e.action}</span></td>
                        <td>{e.component}</td>
                        <td>{e.patient_id || '-'}</td>
                        <td className="small">{e.detail?.slice(0, 80)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Activity & Trends tab */}
      {tab === 'activity' && breakdown && (
        <div className="row">
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Transaction Volume</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Date</th><th>Total</th><th>Blocked</th><th>Sign-offs</th><th>Decisions</th></tr></thead>
                  <tbody>
                    {(breakdown.daily_security || []).map(d => (
                      <tr key={d.date}>
                        <td>{d.date}</td>
                        <td>{d.total_events}</td>
                        <td>{d.blocked > 0 ? <span className="badge bg-danger">{d.blocked}</span> : 0}</td>
                        <td>{d.signoffs > 0 ? <span className="badge bg-success">{d.signoffs}</span> : 0}</td>
                        <td>{d.human_decisions > 0 ? <span className="badge bg-warning text-dark">{d.human_decisions}</span> : 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hourly Activity Pattern</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Hour (UTC)</th><th>Events</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(breakdown.hourly_pattern || []).map(h => {
                      const maxE = Math.max(...(breakdown.hourly_pattern || []).map(x => x.events), 1);
                      const pct = Math.round(h.events / maxE * 100);
                      return (
                        <tr key={h.hour}>
                          <td>{String(h.hour).padStart(2, '0')}:00</td>
                          <td>{h.events}</td>
                          <td><div className="bg-primary rounded" style={{ width: `${pct}%`, height: 12 }} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Conversation Roles</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Role</th><th>Messages</th><th>Share</th></tr></thead>
                  <tbody>
                    {(() => {
                      const total = (breakdown.conversation_roles || []).reduce((s, r) => s + r.count, 0);
                      return (breakdown.conversation_roles || []).map(r => (
                        <tr key={r.role}>
                          <td>{r.role}</td>
                          <td>{r.count}</td>
                          <td>{total > 0 ? `${(r.count / total * 100).toFixed(1)}%` : '-'}</td>
                        </tr>
                      ));
                    })()}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Actor Action Distribution</div>
              <div className="card-body p-0" style={{ maxHeight: 400, overflowY: 'auto' }}>
                {(breakdown.actor_action_matrix || []).map(a => (
                  <div key={a.actor} className="px-3 py-2 border-bottom">
                    <strong>{a.actor}</strong>
                    <div className="d-flex flex-wrap gap-1 mt-1">
                      {a.actions.map(act => (
                        <span key={act.action} className={`badge bg-${
                          ['blocked','delete'].includes(act.action) ? 'danger' :
                          ['sign-off','human_decision'].includes(act.action) ? 'warning' : 'secondary'
                        }`}>{act.action} ({act.count})</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-lg-8">
            {(defs.definitions || []).map((d, i) => (
              <div key={i} className="card mb-2 shadow-sm">
                <div className="card-body py-2">
                  <strong>{d.term}</strong>
                  <div className="text-muted small">{d.definition}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
