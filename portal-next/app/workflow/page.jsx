'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

export default function WorkflowDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [expandedRole, setExpandedRole] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/workflow/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/workflow/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/workflow/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!overview.available) return <div className="alert alert-warning">{overview.note || 'No workflow data'}</div>;

  const s = overview.summary || {};
  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'roles', label: 'Role Workflows' },
    { id: 'activity', label: 'Activity & Trends' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>Workflow Dashboard</h3>
      <p className="text-muted">Consultant workflow execution: role coverage, phase/step progress, sign-off gates, and activity patterns from clinical.db + consultant_workflows.json</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Roles Defined', value: s.total_roles || 0, color: 'primary' },
          { label: 'Total Phases', value: s.total_phases || 0, color: 'info' },
          { label: 'Total Steps', value: s.total_steps || 0, color: 'secondary' },
          { label: 'Sign-off Gates', value: s.total_signoffs || 0, color: 'warning' },
          { label: 'Workflow Events', value: s.workflow_events || 0, color: 'success' },
          { label: 'Expert Reviews', value: s.expert_reviews || 0, color: s.expert_reviews > 0 ? 'success' : 'secondary' },
          { label: 'HITL Reviews', value: s.hitl_reviews || 0, color: s.hitl_reviews > 0 ? 'success' : 'secondary' },
          { label: 'Human Rate', value: `${s.human_rate_pct || 0}%`, color: s.human_rate_pct >= 50 ? 'success' : 'warning' },
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
              <div className="card-header fw-bold">Role Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Role</th><th>Phases</th><th>Steps</th><th>Sign-offs</th></tr></thead>
                  <tbody>
                    {(overview.roles || []).map(r => (
                      <tr key={r.role_key}>
                        <td><strong>{r.name}</strong><br /><span className="text-muted small">{r.summary?.slice(0, 60)}</span></td>
                        <td>{r.phases}</td>
                        <td>{r.steps}</td>
                        <td>{r.signoffs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Actor Workload</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Actor</th><th>Events</th><th>Share</th></tr></thead>
                  <tbody>
                    {(overview.actor_distribution || []).map(a => (
                      <tr key={a.actor}>
                        <td>{a.actor}</td>
                        <td>{a.events}</td>
                        <td>
                          <div className="progress" style={{ height: '16px' }}>
                            <div className="progress-bar bg-primary" style={{ width: `${Math.round(a.events / Math.max(s.total_events, 1) * 100)}%` }}>
                              {Math.round(a.events / Math.max(s.total_events, 1) * 100)}%
                            </div>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Expert Reviews by Role</div>
              <div className="card-body">
                {Object.keys(overview.expert_by_role || {}).length === 0
                  ? <span className="text-muted">No expert reviews yet</span>
                  : Object.entries(overview.expert_by_role).map(([role, data]) => (
                    <div key={role} className="mb-2">
                      <strong>{role}</strong>: {data.total} reviews
                      {Object.entries(data.verdicts || {}).map(([v, c]) => (
                        <span key={v} className={`badge bg-${v === 'approved' ? 'success' : v === 'rejected' ? 'danger' : 'secondary'} ms-1`}>{v}: {c}</span>
                      ))}
                    </div>
                  ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Conversation Roles</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Role</th><th>Turns</th></tr></thead>
                  <tbody>
                    {(overview.conversation_roles || []).map(r => (
                      <tr key={r.role}><td>{r.role}</td><td>{r.turns}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Roles tab */}
      {tab === 'roles' && breakdown && (
        <div>
          {(breakdown.roles || []).map(role => (
            <div key={role.role_key} className="card shadow-sm mb-3">
              <div className="card-header d-flex justify-content-between align-items-center"
                   style={{ cursor: 'pointer' }}
                   onClick={() => setExpandedRole(expandedRole === role.role_key ? null : role.role_key)}>
                <div>
                  <strong>{role.name}</strong>
                  <span className="text-muted ms-2 small">{role.summary?.slice(0, 80)}</span>
                </div>
                <div>
                  <span className="badge bg-primary me-1">{role.total_phases} phases</span>
                  <span className="badge bg-secondary me-1">{role.total_steps} steps</span>
                  <span className="badge bg-warning me-1">{role.signoff_count} sign-offs</span>
                  {role.events_logged > 0 && <span className="badge bg-success me-1">{role.events_logged} events</span>}
                  <span>{expandedRole === role.role_key ? '\u25B2' : '\u25BC'}</span>
                </div>
              </div>
              {expandedRole === role.role_key && (
                <div className="card-body">
                  <div className="row">
                    <div className="col-md-8">
                      <h6>Phases & Steps</h6>
                      {role.phases.map((phase, pi) => (
                        <div key={pi} className="mb-2">
                          <div className="fw-bold text-primary small">{phase.name} ({phase.step_count} steps)</div>
                          <table className="table table-sm table-bordered mb-1">
                            <thead><tr><th>Step</th><th>Input</th><th>Task</th><th>Output</th></tr></thead>
                            <tbody>
                              {phase.steps.map((st, si) => (
                                <tr key={si}>
                                  <td className="fw-semibold small">{st.step}</td>
                                  <td className="small text-muted">{st.input}</td>
                                  <td className="small">{st.task}</td>
                                  <td className="small text-muted">{st.output}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ))}
                    </div>
                    <div className="col-md-4">
                      <h6>Sign-off Gates</h6>
                      <ul className="list-group list-group-flush">
                        {role.signoffs.map((so, i) => (
                          <li key={i} className="list-group-item py-1 small">{so}</li>
                        ))}
                      </ul>
                      <div className="mt-2 small">
                        <div><strong>Events logged:</strong> {role.events_logged}</div>
                        <div><strong>Expert reviews:</strong> {role.expert_reviews}</div>
                        <div><strong>Conversation turns:</strong> {role.conversation_turns}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Activity tab */}
      {tab === 'activity' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Action Distribution</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Action</th><th>Count</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(overview.action_distribution || []).map(a => (
                      <tr key={a.action}>
                        <td><code>{a.action}</code></td>
                        <td>{a.count}</td>
                        <td>
                          <div className="progress" style={{ height: '14px' }}>
                            <div className="progress-bar bg-info" style={{ width: `${Math.round(a.count / Math.max((overview.action_distribution || [])[0]?.count || 1, 1) * 100)}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Activity Trend</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead><tr><th>Date</th><th>Events</th><th>Bar</th></tr></thead>
                  <tbody>
                    {(overview.daily_trend || []).slice(-14).map(d => (
                      <tr key={d.date}>
                        <td className="small">{d.date}</td>
                        <td>{d.events}</td>
                        <td>
                          <div className="progress" style={{ height: '14px' }}>
                            <div className="progress-bar bg-success" style={{ width: `${Math.round(d.events / Math.max(...(overview.daily_trend || []).map(x => x.events), 1) * 100)}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {breakdown && (
            <>
              <div className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold">Hourly Pattern</div>
                  <div className="card-body p-0">
                    <table className="table table-sm mb-0">
                      <thead><tr><th>Hour</th><th>Events</th><th>Bar</th></tr></thead>
                      <tbody>
                        {(breakdown.hourly_pattern || []).map(h => (
                          <tr key={h.hour}>
                            <td>{String(h.hour).padStart(2, '0')}:00</td>
                            <td>{h.events}</td>
                            <td>
                              <div className="progress" style={{ height: '14px' }}>
                                <div className="progress-bar bg-warning" style={{ width: `${Math.round(h.events / Math.max(...(breakdown.hourly_pattern || []).map(x => x.events), 1) * 100)}%` }} />
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold">Patient Workflow Depth (top 20)</div>
                  <div className="card-body p-0">
                    <table className="table table-sm mb-0">
                      <thead><tr><th>Patient</th><th>Components</th><th>Actions</th><th>Events</th></tr></thead>
                      <tbody>
                        {(breakdown.patient_workflows || []).map(p => (
                          <tr key={p.patient_id}>
                            <td><code>{p.patient_id}</code></td>
                            <td>{p.components}</td>
                            <td>{p.actions}</td>
                            <td>{p.events}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>

              <div className="col-12 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold">Recent Workflow Events</div>
                  <div className="card-body p-0">
                    <table className="table table-sm mb-0">
                      <thead><tr><th>Time</th><th>Component</th><th>Action</th><th>Actor</th><th>Patient</th><th>Detail</th></tr></thead>
                      <tbody>
                        {(breakdown.recent_events || []).map((e, i) => (
                          <tr key={i}>
                            <td className="small text-nowrap">{e.ts?.slice(0, 16)}</td>
                            <td><code>{e.component}</code></td>
                            <td><span className="badge bg-secondary">{e.action}</span></td>
                            <td>{e.actor}</td>
                            <td><code>{e.patient_id || '-'}</code></td>
                            <td className="small text-muted">{(e.detail || '').slice(0, 60)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Metric Definitions</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th>Metric</th><th>Definition</th></tr></thead>
              <tbody>
                {(defs.definitions || []).map(d => (
                  <tr key={d.metric}><td className="fw-semibold">{d.metric}</td><td>{d.definition}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
