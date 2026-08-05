'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {
  completed: 'success',
  running:   'primary',
  failed:    'danger',
  queued:    'warning',
  cancelled: 'secondary',
};
const STATUS_ICON = {
  completed: '✅',
  running:   '⚙️',
  failed:    '❌',
  queued:    '🕐',
  cancelled: '🚫',
};
const MODE_COLOR   = { autonomous: 'info', supervised: 'warning', manual: 'secondary' };
const PRIORITY_COLOR = { critical: 'danger', high: 'warning', medium: 'primary', low: 'secondary' };
const TRIGGER_ICON = { cron: '⏱️', api: '🔌', user: '👤', chain: '🔗', event: '⚡' };

function KpiCard({ label, value, color = 'primary', sub }) {
  return (
    <div className="col">
      <div className={`card border-${color} h-100`}>
        <div className="card-body text-center py-2">
          <div className={`fs-4 fw-bold text-${color}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="small text-muted mt-1">{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function ProgressBar({ pct, color = 'success' }) {
  const p = Math.min(100, Math.max(0, pct || 0));
  return (
    <div className="progress" style={{ height: 8 }}>
      <div className={`progress-bar bg-${color}`} style={{ width: `${p}%` }} />
    </div>
  );
}

function StatusBadge({ status }) {
  const color = STATUS_COLOR[status] || 'secondary';
  const icon  = STATUS_ICON[status]  || '•';
  return <span className={`badge bg-${color}`}>{icon} {status}</span>;
}

export default function OpenClawPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [err,  setErr]  = useState(null);
  const [agentFilter, setAgentFilter] = useState('All');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/openclaw/overview`).then(r => r.json()),
      fetch(`${API}/api/openclaw/breakdown`).then(r => r.json()),
      fetch(`${API}/api/openclaw/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err)  return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov)  return <div className="text-muted p-4">Loading OpenClaw Agent Executions…</div>;

  const s = ov;
  const agentDist   = bd?.agent_distribution    || [];
  const statusDist  = bd?.status_distribution   || [];
  const modeDist    = bd?.mode_distribution      || [];
  const triggerDist = bd?.trigger_distribution   || [];
  const priorityDist= bd?.priority_distribution  || [];
  const topFailing  = bd?.top_failing_agents     || [];
  const perAgent    = bd?.per_agent              || {};

  const agentNames = ['All', ...agentDist.map(a => a.agent_name)];
  const filteredPerAgent = agentFilter === 'All'
    ? Object.entries(perAgent)
    : Object.entries(perAgent).filter(([k]) =>
        k === agentFilter.toLowerCase().replace(/ /g, '_')
      );

  const TABS = [
    { id: 'overview',    label: '📊 Overview' },
    { id: 'agents',      label: '🤖 By Agent' },
    { id: 'executions',  label: '📋 Executions' },
    { id: 'analytics',   label: '📈 Analytics' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const avgDurMin = s.avg_duration_seconds ? (s.avg_duration_seconds / 60).toFixed(1) : '—';

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <h4 className="mb-0">🦾 OpenClaw Agent Execution Dashboard</h4>
        <span className="badge bg-info">Live</span>
        <span className="badge bg-secondary">{s.total_executions} executions</span>
      </div>
      <p className="text-muted small mb-3">
        Autonomous multi-agent execution registry — tracks every agent run across all modes (autonomous / supervised / manual),
        trigger types, step progress, token usage, and failure patterns from <code>openclaw_executions</code> ({s.total_executions} rows).
      </p>

      {/* KPI Row */}
      <div className="row row-cols-2 row-cols-md-4 row-cols-lg-8 g-2 mb-4">
        <KpiCard label="Total Executions"  value={s.total_executions}   color="primary" />
        <KpiCard label="Completed"         value={s.completed}          color="success" />
        <KpiCard label="Running"           value={s.running}            color="primary" />
        <KpiCard label="Failed"            value={s.failed}             color="danger"  />
        <KpiCard label="Queued"            value={s.queued}             color="warning" />
        <KpiCard label="Completion Rate"   value={`${s.completion_rate}%`} color={s.completion_rate >= 70 ? 'success' : 'warning'} />
        <KpiCard label="Avg Duration"      value={`${avgDurMin}m`}     color="info"    />
        <KpiCard label="Total Tokens"      value={s.total_tokens?.toLocaleString()} color="secondary" />
      </div>

      {/* Step Completion */}
      <div className="card mb-4">
        <div className="card-body py-2">
          <div className="d-flex justify-content-between align-items-center mb-1">
            <span className="small fw-semibold">Avg Step Completion</span>
            <span className="small fw-bold text-primary">{s.avg_steps_completion_pct}%</span>
          </div>
          <ProgressBar pct={s.avg_steps_completion_pct} color="primary" />
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Status Distribution */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Status Distribution</div>
              <div className="card-body p-2">
                {statusDist.map(r => (
                  <div key={r.status} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>
                        <span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'} me-1`}>
                          {STATUS_ICON[r.status] || '•'}
                        </span>
                        {r.status}
                      </span>
                      <span className="fw-bold">{r.count}</span>
                    </div>
                    <ProgressBar
                      pct={(r.count / s.total_executions) * 100}
                      color={STATUS_COLOR[r.status] || 'secondary'}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Mode Distribution */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Execution Mode</div>
              <div className="card-body p-2">
                {modeDist.map(r => (
                  <div key={r.execution_mode} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className={`text-${MODE_COLOR[r.execution_mode] || 'secondary'}`}>
                        {r.execution_mode}
                      </span>
                      <span className="fw-bold">{r.count} ({((r.count/s.total_executions)*100).toFixed(0)}%)</span>
                    </div>
                    <ProgressBar
                      pct={(r.count / s.total_executions) * 100}
                      color={MODE_COLOR[r.execution_mode] || 'secondary'}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Trigger Distribution */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Trigger Type</div>
              <div className="card-body p-2">
                {triggerDist.map(r => (
                  <div key={r.triggered_by} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span>{TRIGGER_ICON[r.triggered_by] || '•'} {r.triggered_by}</span>
                      <span className="fw-bold">{r.count}</span>
                    </div>
                    <ProgressBar
                      pct={(r.count / s.total_executions) * 100}
                      color="info"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Priority Distribution */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Priority Breakdown</div>
              <div className="card-body p-2">
                {priorityDist.map(r => (
                  <div key={r.priority} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className={`badge bg-${PRIORITY_COLOR[r.priority] || 'secondary'}`}>{r.priority}</span>
                      <span className="fw-bold">{r.count}</span>
                    </div>
                    <ProgressBar
                      pct={(r.count / s.total_executions) * 100}
                      color={PRIORITY_COLOR[r.priority] || 'secondary'}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top Agents */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Top Agents by Execution Count</div>
              <div className="card-body p-2">
                <table className="table table-sm table-hover mb-0">
                  <thead><tr><th>Agent</th><th className="text-end">Runs</th></tr></thead>
                  <tbody>
                    {agentDist.slice(0, 10).map(a => (
                      <tr key={a.agent_name}>
                        <td className="small">{a.agent_name}</td>
                        <td className="text-end"><span className="badge bg-primary">{a.count}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Top Failing Agents */}
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold text-danger">Top Failing Agents</div>
              <div className="card-body p-2">
                {topFailing.length === 0
                  ? <div className="text-muted small">No failures recorded</div>
                  : (
                    <table className="table table-sm table-hover mb-0">
                      <thead><tr><th>Agent</th><th className="text-end">Failures</th></tr></thead>
                      <tbody>
                        {topFailing.map(a => (
                          <tr key={a.agent_name}>
                            <td className="small">{a.agent_name}</td>
                            <td className="text-end"><span className="badge bg-danger">{a.failure_count}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── BY AGENT TAB ── */}
      {tab === 'agents' && (
        <div>
          <div className="mb-3">
            <label className="form-label small fw-semibold">Filter by Agent</label>
            <select
              className="form-select form-select-sm w-auto"
              value={agentFilter}
              onChange={e => setAgentFilter(e.target.value)}
            >
              {agentNames.map(a => <option key={a}>{a}</option>)}
            </select>
          </div>
          <div className="row g-3">
            {agentDist
              .filter(a => agentFilter === 'All' || a.agent_name === agentFilter)
              .map(a => {
                const key = a.agent_name.toLowerCase().replace(/ /g, '_');
                const execs = perAgent[key] || [];
                const completed = execs.filter(e => e.status === 'completed').length;
                const failed    = execs.filter(e => e.status === 'failed').length;
                const compRate  = execs.length > 0 ? ((completed / execs.length) * 100).toFixed(0) : 0;
                return (
                  <div key={a.agent_name} className="col-md-4">
                    <div className="card h-100">
                      <div className="card-header py-2 d-flex justify-content-between align-items-center">
                        <span className="small fw-semibold">🤖 {a.agent_name}</span>
                        <span className="badge bg-secondary">{a.count} runs</span>
                      </div>
                      <div className="card-body p-2">
                        <div className="row text-center g-1 mb-2">
                          <div className="col-4">
                            <div className="text-success fw-bold">{completed}</div>
                            <div className="text-muted" style={{ fontSize: 10 }}>done</div>
                          </div>
                          <div className="col-4">
                            <div className="text-danger fw-bold">{failed}</div>
                            <div className="text-muted" style={{ fontSize: 10 }}>failed</div>
                          </div>
                          <div className="col-4">
                            <div className="text-primary fw-bold">{compRate}%</div>
                            <div className="text-muted" style={{ fontSize: 10 }}>success</div>
                          </div>
                        </div>
                        <ProgressBar pct={Number(compRate)} color={Number(compRate) >= 70 ? 'success' : 'warning'} />
                        {execs.length > 0 && (
                          <div className="mt-2">
                            {execs.slice(0, 3).map(e => (
                              <div key={e.execution_id} className="d-flex justify-content-between align-items-center py-1 border-bottom small">
                                <span className="text-truncate me-2" style={{ maxWidth: 160 }}
                                  title={e.task_description}>{e.task_description}</span>
                                <StatusBadge status={e.status} />
                              </div>
                            ))}
                            {execs.length > 3 && (
                              <div className="text-muted small mt-1">+{execs.length - 3} more</div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* ── EXECUTIONS TAB ── */}
      {tab === 'executions' && (
        <div className="card">
          <div className="card-header py-2 small fw-semibold">All Execution Records</div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-dark">
                  <tr>
                    <th>Execution ID</th>
                    <th>Agent</th>
                    <th>Task</th>
                    <th>Status</th>
                    <th>Mode</th>
                    <th>Priority</th>
                    <th>Steps</th>
                    <th>Duration</th>
                    <th>Trigger</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.values(perAgent).flat().map(e => (
                    <tr key={e.execution_id}>
                      <td className="small font-monospace" style={{ fontSize: 11 }}>
                        {e.execution_id?.slice(0, 18)}…
                      </td>
                      <td className="small">{e.agent_name}</td>
                      <td className="small text-truncate" style={{ maxWidth: 200 }} title={e.task_description}>
                        {e.task_description}
                      </td>
                      <td><StatusBadge status={e.status} /></td>
                      <td>
                        <span className={`badge bg-${MODE_COLOR[e.execution_mode] || 'secondary'}`}>
                          {e.execution_mode}
                        </span>
                      </td>
                      <td>
                        <span className={`badge bg-${PRIORITY_COLOR[e.priority] || 'secondary'}`}>
                          {e.priority}
                        </span>
                      </td>
                      <td className="small">{e.steps_progress}</td>
                      <td className="small">
                        {e.duration_seconds != null
                          ? `${(e.duration_seconds / 60).toFixed(1)}m`
                          : '—'}
                      </td>
                      <td className="small">
                        {TRIGGER_ICON[e.triggered_by] || ''} {e.triggered_by}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── ANALYTICS TAB ── */}
      {tab === 'analytics' && (
        <div className="row g-3">
          {/* Token Usage Analysis */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Token Usage Summary</div>
              <div className="card-body">
                <div className="row text-center g-2">
                  <div className="col-6">
                    <div className="fs-5 fw-bold text-primary">{s.total_tokens?.toLocaleString()}</div>
                    <div className="small text-muted">Total Tokens</div>
                  </div>
                  <div className="col-6">
                    <div className="fs-5 fw-bold text-info">
                      {s.total_executions > 0
                        ? Math.round(s.total_tokens / s.total_executions).toLocaleString()
                        : '—'}
                    </div>
                    <div className="small text-muted">Avg per Execution</div>
                  </div>
                </div>
                <hr className="my-2" />
                <div className="small text-muted">
                  <strong>Completion Rate:</strong> {s.completion_rate}%
                  {s.completion_rate < 70 && (
                    <span className="text-warning ms-2">⚠ Below 70% threshold</span>
                  )}
                </div>
                <div className="small text-muted mt-1">
                  <strong>Avg Step Completion:</strong> {s.avg_steps_completion_pct}%
                </div>
                <div className="small text-muted mt-1">
                  <strong>Cancelled Executions:</strong> {s.cancelled} ({((s.cancelled/s.total_executions)*100).toFixed(1)}%)
                </div>
              </div>
            </div>
          </div>

          {/* Execution Efficiency */}
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header py-2 small fw-semibold">Execution Efficiency Matrix</div>
              <div className="card-body">
                <table className="table table-sm mb-0">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th className="text-end">Value</th>
                      <th className="text-end">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="small">Completion Rate</td>
                      <td className="text-end small">{s.completion_rate}%</td>
                      <td className="text-end">
                        <span className={`badge bg-${s.completion_rate >= 70 ? 'success' : 'danger'}`}>
                          {s.completion_rate >= 70 ? '✅' : '⚠️'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="small">Failure Rate</td>
                      <td className="text-end small">
                        {((s.failed / s.total_executions) * 100).toFixed(1)}%
                      </td>
                      <td className="text-end">
                        <span className={`badge bg-${(s.failed / s.total_executions) < 0.15 ? 'success' : 'danger'}`}>
                          {(s.failed / s.total_executions) < 0.15 ? '✅' : '⚠️'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="small">Avg Duration</td>
                      <td className="text-end small">{avgDurMin}m</td>
                      <td className="text-end">
                        <span className={`badge bg-${Number(avgDurMin) < 10 ? 'success' : 'warning'}`}>
                          {Number(avgDurMin) < 10 ? '✅' : '⚠️'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="small">Autonomous Rate</td>
                      <td className="text-end small">
                        {modeDist.find(m => m.execution_mode === 'autonomous')?.count || 0} / {s.total_executions}
                      </td>
                      <td className="text-end">
                        <span className="badge bg-info">ℹ️</span>
                      </td>
                    </tr>
                    <tr>
                      <td className="small">Cron-Triggered</td>
                      <td className="text-end small">
                        {triggerDist.find(t => t.triggered_by === 'cron')?.count || 0}
                      </td>
                      <td className="text-end">
                        <span className="badge bg-success">✅</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Agents at a Glance */}
          <div className="col-12">
            <div className="card">
              <div className="card-header py-2 small fw-semibold">All Agents — Execution Count</div>
              <div className="card-body p-2">
                <div className="row g-1">
                  {agentDist.map(a => (
                    <div key={a.agent_name} className="col-6 col-md-4 col-lg-3">
                      <div className="d-flex justify-content-between align-items-center py-1 px-2 border rounded-1 small">
                        <span className="text-truncate me-1">🤖 {a.agent_name}</span>
                        <span className="badge bg-primary">{a.count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div className="row g-3">
          {Object.entries(defs).map(([section, entries]) => (
            <div key={section} className="col-md-6">
              <div className="card h-100">
                <div className="card-header py-2 small fw-semibold text-capitalize">
                  {section.replace(/_/g, ' ')}
                </div>
                <div className="card-body p-2">
                  {typeof entries === 'object' && !Array.isArray(entries)
                    ? Object.entries(entries).map(([k, v]) => (
                        <div key={k} className="mb-2 border-bottom pb-1">
                          <div className="small fw-semibold text-primary">{k}</div>
                          <div className="small text-muted">{v}</div>
                        </div>
                      ))
                    : <div className="small text-muted">{JSON.stringify(entries)}</div>
                  }
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
