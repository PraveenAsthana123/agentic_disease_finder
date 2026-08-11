'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPICard({ label, value, sub, color = 'primary' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center py-3">
          <div className={`display-6 fw-bold text-${color}`}>{value}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="xsmall text-muted mt-1" style={{ fontSize: '0.72rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ label, val, max, color = 'primary', unit = '' }) {
  const pct = max > 0 ? Math.round((val / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span className="text-truncate" style={{ maxWidth: 180 }}>{label}</span>
        <span className="fw-bold ms-2">{val.toLocaleString()}{unit}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${Math.max(pct, 2)}%` }} />
      </div>
    </div>
  );
}

export default function ExecutiveAIDashboard() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/executive-ai/overview`).then(r => r.json()),
      fetch(`${API}/api/executive-ai/breakdown`).then(r => r.json()),
      fetch(`${API}/api/executive-ai/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Load failed: {err}</div>;
  if (!ov)  return <div className="text-muted p-4">Loading Executive AI Dashboard…</div>;

  const s        = ov.summary || {};
  const actors   = ov.actor_distribution || [];
  const comps    = ov.top_components || [];
  const actions  = ov.action_distribution || [];
  const dayTrend = ov.daily_throughput || [];
  const deptUtil = ov.department_ai_utilization || [];
  const convRole = ov.conversation_roles || [];

  const compDetail  = bd?.component_detail || [];
  const deptComp    = bd?.department_component_cross || [];
  const weeklyVol   = bd?.weekly_volume || [];
  const expertRevs  = bd?.recent_expert_reviews || [];
  const hitlRevs    = bd?.recent_hitl_reviews || [];

  const definitions = defs?.definitions || [];

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'operations',  label: '⚙️ Operations' },
    { id: 'oversight',   label: '👁️ Oversight' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const maxActorOps = Math.max(...actors.map(a => a.operations), 1);
  const maxCompOps  = Math.max(...comps.map(c => c.operations), 1);
  const maxActionOps = Math.max(...actions.slice(0, 10).map(a => a.operations || a.count || 0), 1);

  return (
    <div>
      <h3>🤖 Executive AI Dashboard</h3>
      <p className="text-muted small">AI operations across all clinical components — penetration, automation, oversight, and throughput from real transaction_log data.</p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <KPICard label="Total AI Operations" value={s.total_ai_operations?.toLocaleString() ?? '—'} color="primary" />
            <KPICard label="Patients Reached" value={s.total_patients ?? '—'} sub={`${s.ai_penetration_pct ?? 0}% penetration`} color="success" />
            <KPICard label="Automation Rate" value={`${s.automation_rate_pct ?? 0}%`} sub="system-actor ops" color="info" />
            <KPICard label="HITL Reviews" value={s.hitl_reviews ?? 0} sub={`+ ${s.expert_reviews ?? 0} expert reviews`} color="warning" />
          </div>

          <div className="row mt-2">
            <KPICard label="Conversations" value={s.total_conversations?.toLocaleString() ?? '—'} color="secondary" />
            <KPICard label="Clinical Assessments" value={s.total_assessments?.toLocaleString() ?? '—'} color="secondary" />
            <KPICard label="Human-initiated Ops" value={s.human_ops?.toLocaleString() ?? '—'} color="secondary" />
            <KPICard label="System-initiated Ops" value={s.system_ops?.toLocaleString() ?? '—'} color="secondary" />
          </div>

          <div className="row mt-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Actor Distribution</div>
                <div className="card-body">
                  {actors.map((a, i) => (
                    <Bar key={i} label={a.actor} val={a.operations} max={maxActorOps}
                         color={i === 0 ? 'info' : i === 1 ? 'secondary' : 'primary'} />
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Top AI Components</div>
                <div className="card-body">
                  {comps.slice(0, 10).map((c, i) => (
                    <Bar key={i} label={c.component} val={c.operations} max={maxCompOps}
                         color={i === 0 ? 'primary' : i < 3 ? 'info' : 'secondary'} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row mt-1">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Conversation Roles</div>
                <div className="card-body">
                  {convRole.length ? convRole.map((r, i) => (
                    <div key={i} className="d-flex justify-content-between py-1 border-bottom small">
                      <span className="text-capitalize">{r.role}</span>
                      <span className="badge bg-secondary">{r.count?.toLocaleString()}</span>
                    </div>
                  )) : <div className="text-muted small">No conversation data</div>}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Daily AI Throughput (recent)</div>
                <div className="card-body" style={{ maxHeight: 200, overflowY: 'auto' }}>
                  {dayTrend.slice(-10).reverse().map((d, i) => (
                    <div key={i} className="d-flex justify-content-between py-1 border-bottom small">
                      <span>{d.date}</span>
                      <span className="fw-bold">{d.operations?.toLocaleString()} ops</span>
                      <span className="text-muted">{d.active_components} components</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* OPERATIONS */}
      {tab === 'operations' && (
        <>
          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Component Detail</div>
                <div className="card-body p-0">
                  <div style={{ maxHeight: 400, overflowY: 'auto' }}>
                    <table className="table table-sm table-hover mb-0">
                      <thead className="table-light sticky-top">
                        <tr>
                          <th>Component</th>
                          <th className="text-end">Ops</th>
                          <th className="text-end">Patients</th>
                          <th>Last Active</th>
                        </tr>
                      </thead>
                      <tbody>
                        {compDetail.map((c, i) => (
                          <tr key={i}>
                            <td><span className="badge bg-light text-dark border">{c.component}</span></td>
                            <td className="text-end fw-bold">{c.ops?.toLocaleString()}</td>
                            <td className="text-end">{c.patients_touched}</td>
                            <td className="small text-muted">{c.last_seen ? c.last_seen.slice(0, 10) : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Top Actions (API endpoints)</div>
                <div className="card-body">
                  {actions.slice(0, 12).map((a, i) => (
                    <Bar key={i} label={a.action || a.endpoint || '—'}
                         val={a.operations || a.count || 0}
                         max={maxActionOps}
                         color={i < 3 ? 'primary' : 'secondary'} />
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Weekly Volume</div>
                <div className="card-body" style={{ maxHeight: 280, overflowY: 'auto' }}>
                  {weeklyVol.length ? weeklyVol.map((w, i) => (
                    <div key={i} className="d-flex justify-content-between py-1 border-bottom small">
                      <span>Week {w.week || w.period}</span>
                      <span className="fw-bold">{(w.operations || w.count || 0).toLocaleString()} ops</span>
                    </div>
                  )) : <div className="text-muted small">No weekly data available</div>}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Department AI Utilization</div>
                <div className="card-body">
                  {deptUtil.length ? deptUtil.map((d, i) => (
                    <div key={i} className="d-flex justify-content-between py-1 border-bottom small">
                      <span className="text-capitalize">{d.department || '(unassigned)'}</span>
                      <span className="badge bg-primary">{d.operations?.toLocaleString() || d.count}</span>
                    </div>
                  )) : <div className="text-muted small">No department utilization data</div>}
                </div>
              </div>
            </div>
          </div>

          {deptComp.length > 0 && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">Department × Component Cross-tab</div>
              <div className="card-body p-0">
                <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light sticky-top">
                      <tr>
                        <th>Department</th>
                        <th>Component</th>
                        <th className="text-end">Ops</th>
                      </tr>
                    </thead>
                    <tbody>
                      {deptComp.map((r, i) => (
                        <tr key={i}>
                          <td className="text-capitalize">{r.department || '—'}</td>
                          <td><span className="badge bg-light text-dark border">{r.component}</span></td>
                          <td className="text-end">{(r.ops || r.operations || 0).toLocaleString()}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* OVERSIGHT */}
      {tab === 'oversight' && (
        <>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">📋 Expert Reviews ({expertRevs.length})</div>
                <div className="card-body p-0" style={{ maxHeight: 350, overflowY: 'auto' }}>
                  {expertRevs.length ? (
                    <table className="table table-sm table-hover mb-0">
                      <thead className="table-light sticky-top">
                        <tr>
                          <th>Patient</th>
                          <th>Reviewer</th>
                          <th>Verdict</th>
                          <th>Date</th>
                        </tr>
                      </thead>
                      <tbody>
                        {expertRevs.map((r, i) => (
                          <tr key={i}>
                            <td>{r.patient_id || '—'}</td>
                            <td className="small">{r.reviewer || '—'}</td>
                            <td>
                              <span className={`badge bg-${r.verdict === 'approved' ? 'success' : r.verdict === 'overridden' ? 'danger' : 'secondary'}`}>
                                {r.verdict || '—'}
                              </span>
                            </td>
                            <td className="small text-muted">{r.date ? r.date.slice(0, 10) : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : <div className="text-muted p-3 small">No expert reviews recorded</div>}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">👤 HITL Reviews ({hitlRevs.length})</div>
                <div className="card-body p-0" style={{ maxHeight: 350, overflowY: 'auto' }}>
                  {hitlRevs.length ? (
                    <table className="table table-sm table-hover mb-0">
                      <thead className="table-light sticky-top">
                        <tr>
                          <th>Patient</th>
                          <th>Reviewer</th>
                          <th>Outcome</th>
                          <th>Date</th>
                        </tr>
                      </thead>
                      <tbody>
                        {hitlRevs.map((r, i) => (
                          <tr key={i}>
                            <td>{r.patient_id || '—'}</td>
                            <td className="small">{r.reviewer || '—'}</td>
                            <td>
                              <span className={`badge bg-${r.outcome === 'approved' ? 'success' : r.outcome === 'override' ? 'warning' : 'secondary'}`}>
                                {r.outcome || '—'}
                              </span>
                            </td>
                            <td className="small text-muted">{r.date ? r.date.slice(0, 10) : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : <div className="text-muted p-3 small">No HITL reviews recorded</div>}
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">AI Governance Summary</div>
            <div className="card-body">
              <div className="row">
                {[
                  { label: 'Total AI Operations', val: s.total_ai_operations?.toLocaleString(), color: 'primary' },
                  { label: 'Human-initiated', val: s.human_ops?.toLocaleString(), color: 'success' },
                  { label: 'System-initiated', val: s.system_ops?.toLocaleString(), color: 'info' },
                  { label: 'Feedback Count', val: s.feedback_count ?? 0, color: 'warning' },
                ].map((item, i) => (
                  <div key={i} className="col-md-3 col-6 text-center mb-3">
                    <div className={`h4 fw-bold text-${item.color}`}>{item.val}</div>
                    <div className="small text-muted">{item.label}</div>
                  </div>
                ))}
              </div>
              <div className="alert alert-info small mt-2 mb-0">
                <strong>Oversight rate: {s.oversight_rate_pct ?? 0}%</strong> — ratio of human review actions (expert + HITL reviews)
                to total AI operations. Automation rate: <strong>{s.automation_rate_pct ?? 0}%</strong> of operations
                are fully system-driven with no human trigger.
              </div>
            </div>
          </div>
        </>
      )}

      {/* DEFINITIONS */}
      {tab === 'definitions' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Metric Definitions</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr>
                  <th style={{ width: 220 }}>Metric</th>
                  <th>Description</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {definitions.map((d, i) => (
                  <tr key={i}>
                    <td><strong>{d.metric}</strong></td>
                    <td className="small">{d.description}</td>
                    <td className="small text-muted font-monospace">{d.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
