'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 text-${color || 'primary'}`}>{value}</div>
          <div className="text-muted small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const map = { healthy: 'success', degraded: 'warning', down: 'danger' };
  return (
    <span className={`badge bg-${map[status] || 'secondary'} text-capitalize`}>
      {status === 'healthy' ? '✓' : status === 'degraded' ? '⚠' : '✗'} {status}
    </span>
  );
}

function UptimeBar({ pct, status }) {
  const color = pct >= 90 ? 'success' : pct >= 70 ? 'warning' : 'danger';
  return (
    <div className="d-flex align-items-center gap-2">
      <div className="progress flex-grow-1" style={{ height: 16 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }}>
          <span className="small px-1">{pct.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}

function ResourceBar({ pct, label }) {
  const color = pct >= 80 ? 'danger' : pct >= 60 ? 'warning' : 'success';
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small">
        <span>{label}</span>
        <span className={`text-${color}`}>{pct?.toFixed(1)}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function SystemHealthDashboard() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [filterComponent, setFilterComponent] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/system-health/overview`).then(r => r.json()),
      fetch(`${API}/api/system-health/breakdown`).then(r => r.json()),
      fetch(`${API}/api/system-health/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = overview.kpis || {};
  const statusDist = overview.status_distribution || {};
  const components = overview.component_summary || [];
  const resDist = overview.resource_distribution || {};
  const timeline = overview.timeline || [];

  const allChecks = breakdown?.all_checks || [];
  const components_list = [...new Set(allChecks.map(c => c.component))].sort();

  const filteredChecks = allChecks.filter(c =>
    (filterComponent === 'all' || c.component === filterComponent) &&
    (filterStatus === 'all' || c.status === filterStatus)
  );

  const uptimePct = kpis.overall_uptime_pct || 0;
  const uptimeColor = uptimePct >= 90 ? 'success' : uptimePct >= 75 ? 'warning' : 'danger';

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'log', label: 'Health Log' },
    { id: 'defs', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>🖥️ System Health Monitor</h3>
      <p className="text-muted">
        Infrastructure health checks across {kpis.components_monitored} components — real{' '}
        <code>system_health_log</code> table, {kpis.total_checks} checks, CPU/memory/disk/response tracking.
      </p>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Overall Uptime" value={`${uptimePct.toFixed(1)}%`} color={uptimeColor} sub={`${statusDist.healthy || 0} healthy / ${(statusDist.degraded || 0) + (statusDist.down || 0)} issues`} />
        <KPI label="Total Checks" value={kpis.total_checks} color="primary" sub={`${kpis.components_monitored} components monitored`} />
        <KPI label="Avg Response" value={`${kpis.avg_response_ms?.toLocaleString()} ms`} color={kpis.avg_response_ms > 2000 ? 'danger' : kpis.avg_response_ms > 500 ? 'warning' : 'success'} sub="round-trip latency" />
        <KPI label="Total Errors" value={kpis.total_errors} color={kpis.total_errors > 50 ? 'danger' : kpis.total_errors > 10 ? 'warning' : 'success'} sub="across all checks" />
      </div>

      {/* Status Alert */}
      {(statusDist.down > 0 || statusDist.degraded > 0) && (
        <div className={`alert alert-${statusDist.down > 0 ? 'danger' : 'warning'} mb-3`}>
          <strong>{statusDist.down > 0 ? '🔴 Components Down' : '⚠️ Degraded Components'}:</strong>{' '}
          {components.filter(c => c.current_status !== 'healthy').map(c => (
            <span key={c.component} className="badge bg-danger me-1">{c.component}</span>
          ))}
          {' '}detected in health log. Review below.
        </div>
      )}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* OVERVIEW TAB */}
      {tab === 'overview' && (
        <div>
          <div className="row">
            {/* Status Distribution */}
            <div className="col-md-4 mb-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Status Distribution</div>
                <div className="card-body">
                  <div className="row text-center mb-3">
                    {[
                      { label: 'Healthy', count: statusDist.healthy || 0, color: 'success' },
                      { label: 'Degraded', count: statusDist.degraded || 0, color: 'warning' },
                      { label: 'Down', count: statusDist.down || 0, color: 'danger' },
                    ].map(s => (
                      <div key={s.label} className="col">
                        <div className={`badge bg-${s.color} fs-5 mb-1 d-block`}>{s.count}</div>
                        <div className="small text-muted">{s.label}</div>
                      </div>
                    ))}
                  </div>
                  <div className="progress" style={{ height: 20 }}>
                    <div className="progress-bar bg-success" style={{ width: `${((statusDist.healthy || 0) / kpis.total_checks) * 100}%` }} title="Healthy" />
                    <div className="progress-bar bg-warning" style={{ width: `${((statusDist.degraded || 0) / kpis.total_checks) * 100}%` }} title="Degraded" />
                    <div className="progress-bar bg-danger" style={{ width: `${((statusDist.down || 0) / kpis.total_checks) * 100}%` }} title="Down" />
                  </div>
                  <div className="d-flex justify-content-between text-muted small mt-1">
                    <span>✓ Healthy</span><span>⚠ Degraded</span><span>✗ Down</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Resource Overview */}
            <div className="col-md-4 mb-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Avg Resource Utilization</div>
                <div className="card-body">
                  <ResourceBar pct={kpis.avg_cpu_pct} label="CPU" />
                  <ResourceBar pct={kpis.avg_memory_pct} label="Memory" />
                  <ResourceBar pct={kpis.avg_disk_pct} label="Disk" />
                  <hr />
                  <div className="small text-muted">Resource level counts (low/moderate/high):</div>
                  {['cpu', 'memory', 'disk'].map(r => (
                    <div key={r} className="d-flex gap-2 mt-1">
                      <span className="text-capitalize small" style={{ width: 60 }}>{r}:</span>
                      <span className="badge bg-success">{resDist[r]?.low || 0}</span>
                      <span className="badge bg-warning text-dark">{resDist[r]?.moderate || 0}</span>
                      <span className="badge bg-danger">{resDist[r]?.high || 0}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Response Time */}
            <div className="col-md-4 mb-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Response Time Overview</div>
                <div className="card-body">
                  <div className="text-center mb-3">
                    <div className={`display-6 fw-bold text-${kpis.avg_response_ms > 2000 ? 'danger' : kpis.avg_response_ms > 500 ? 'warning' : 'success'}`}>
                      {kpis.avg_response_ms?.toLocaleString()} ms
                    </div>
                    <div className="text-muted small">average across all checks</div>
                  </div>
                  <div className="small text-muted mb-2">Slowest components (by avg response):</div>
                  {[...components].sort((a, b) => b.avg_response_ms - a.avg_response_ms).slice(0, 4).map(c => (
                    <div key={c.component} className="d-flex justify-content-between align-items-center mb-1">
                      <span className="small">{c.component}</span>
                      <span className={`badge bg-${c.avg_response_ms > 5000 ? 'danger' : c.avg_response_ms > 1000 ? 'warning' : 'success'}`}>
                        {c.avg_response_ms?.toLocaleString()} ms
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Component Summary Table */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">
              Component Health Summary
              <span className="badge bg-secondary ms-2">{components.length} components</span>
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>Component</th>
                      <th>Current Status</th>
                      <th>Uptime %</th>
                      <th>Checks</th>
                      <th>Healthy</th>
                      <th>Degraded</th>
                      <th>Down</th>
                      <th>Avg Response</th>
                      <th>Avg CPU</th>
                      <th>Avg Mem</th>
                      <th>Errors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {components.map((c, i) => (
                      <tr key={i} className={c.current_status === 'down' ? 'table-danger' : c.current_status === 'degraded' ? 'table-warning' : ''}>
                        <td><strong>{c.component}</strong></td>
                        <td><StatusBadge status={c.current_status} /></td>
                        <td style={{ minWidth: 140 }}><UptimeBar pct={c.uptime_pct} /></td>
                        <td>{c.checks}</td>
                        <td><span className="badge bg-success">{c.healthy}</span></td>
                        <td><span className="badge bg-warning text-dark">{c.degraded}</span></td>
                        <td><span className="badge bg-danger">{c.down}</span></td>
                        <td>
                          <span className={`badge bg-${c.avg_response_ms > 5000 ? 'danger' : c.avg_response_ms > 1000 ? 'warning' : 'success'}`}>
                            {c.avg_response_ms?.toLocaleString()} ms
                          </span>
                        </td>
                        <td>{c.avg_cpu?.toFixed(1)}%</td>
                        <td>{c.avg_mem?.toFixed(1)}%</td>
                        <td>
                          {c.error_count > 0
                            ? <span className="badge bg-danger">{c.error_count}</span>
                            : <span className="badge bg-success">0</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Recent Check Timeline */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Recent Check Timeline (last 10)</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm">
                  <thead className="table-light">
                    <tr>
                      <th>Timestamp</th>
                      <th>Component</th>
                      <th>Status</th>
                      <th>Response ms</th>
                      <th>CPU %</th>
                      <th>Mem %</th>
                      <th>Disk %</th>
                      <th>Errors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...timeline].sort((a, b) => b.timestamp?.localeCompare(a.timestamp)).slice(0, 10).map((c, i) => (
                      <tr key={i} className={c.status === 'down' ? 'table-danger' : c.status === 'degraded' ? 'table-warning' : ''}>
                        <td className="text-muted small">{c.timestamp?.replace('T', ' ')}</td>
                        <td>{c.component}</td>
                        <td><StatusBadge status={c.status} /></td>
                        <td>{c.response_time_ms?.toLocaleString()}</td>
                        <td>{c.cpu_pct?.toFixed(1)}</td>
                        <td>{c.memory_pct?.toFixed(1)}</td>
                        <td>{c.disk_pct?.toFixed(1)}</td>
                        <td>{c.error_count > 0 ? <span className="badge bg-danger">{c.error_count}</span> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HEALTH LOG TAB */}
      {tab === 'log' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <select className="form-select form-select-sm" style={{ width: 180 }} value={filterComponent} onChange={e => setFilterComponent(e.target.value)}>
              <option value="all">All Components</option>
              {components_list.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <select className="form-select form-select-sm" style={{ width: 160 }} value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
              <option value="all">All Statuses</option>
              <option value="healthy">Healthy</option>
              <option value="degraded">Degraded</option>
              <option value="down">Down</option>
            </select>
            <span className="text-muted small align-self-center">{filteredChecks.length} of {allChecks.length} checks</span>
          </div>
          <div className="card shadow-sm">
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-hover">
                  <thead className="table-light">
                    <tr>
                      <th>#</th>
                      <th>Timestamp</th>
                      <th>Component</th>
                      <th>Status</th>
                      <th>Response ms</th>
                      <th>CPU %</th>
                      <th>Memory %</th>
                      <th>Disk %</th>
                      <th>Errors</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredChecks.map((c, i) => (
                      <tr key={i} className={c.status === 'down' ? 'table-danger' : c.status === 'degraded' ? 'table-warning' : ''}>
                        <td className="text-muted small">{c.check_id}</td>
                        <td className="text-muted small">{c.timestamp?.replace('T', ' ')}</td>
                        <td><strong>{c.component}</strong></td>
                        <td><StatusBadge status={c.status} /></td>
                        <td>
                          <span className={`badge bg-${c.response_time_ms > 10000 ? 'danger' : c.response_time_ms > 1000 ? 'warning' : 'success'}`}>
                            {c.response_time_ms?.toLocaleString()}
                          </span>
                        </td>
                        <td>{c.cpu_pct?.toFixed(1)}</td>
                        <td>{c.memory_pct?.toFixed(1)}</td>
                        <td>{c.disk_pct?.toFixed(1)}</td>
                        <td>{c.error_count > 0 ? <span className="badge bg-danger">{c.error_count}</span> : <span className="text-muted">—</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'defs' && (
        <div>
          {defs?.title && (
            <div className="alert alert-info mb-4">
              <strong>{defs.title}</strong>
              {defs.source && <div className="small mt-1">Source: {defs.source}</div>}
            </div>
          )}
          <div className="row">
            {(defs?.glossary || []).map((t, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card shadow-sm h-100">
                  <div className="card-body">
                    <h6 className="card-title text-primary">{t.term}</h6>
                    <p className="card-text small text-muted mb-0">{t.definition}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          {defs?.thresholds && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">SLA Thresholds</div>
              <div className="card-body">
                <table className="table table-sm">
                  <thead><tr><th>Metric</th><th>Healthy</th><th>Degraded</th><th>Down</th></tr></thead>
                  <tbody>
                    {Object.entries(defs.thresholds).map(([k, v]) => (
                      <tr key={k}>
                        <td className="text-capitalize">{k.replace(/_/g, ' ')}</td>
                        <td>{v.healthy}</td>
                        <td>{v.degraded}</td>
                        <td>{v.down}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="alert alert-secondary small mt-4">
        Source: <code>clinical.db → system_health_log</code> ({kpis.total_checks} rows) — real infrastructure health checks.
        Components: {components.map(c => c.component).join(', ')}.
      </div>
    </div>
  );
}
