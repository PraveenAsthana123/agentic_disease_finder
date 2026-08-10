'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview' },
  { id: 'breakdown',  label: 'Issue Breakdown' },
  { id: 'definitions',label: 'Definitions' },
];

const SEV_COLOR  = { P1: 'danger', P2: 'warning', P3: 'info', P4: 'secondary' };
const STAT_COLOR = { open: 'danger', running: 'warning', fixed: 'success', closed: 'success' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function SevBadge({ sev }) {
  return <span className={`badge bg-${SEV_COLOR[sev] || 'secondary'}`}>{sev}</span>;
}

function StatBadge({ status }) {
  const label = status === 'open' ? '🔴 open' : status === 'running' ? '🟡 running' : '✅ fixed';
  return <span className={`badge bg-${STAT_COLOR[status] || 'secondary'}`}>{label}</span>;
}

function ComponentPill({ name, status }) {
  const color = status === 'down' ? 'danger' : status === 'degraded' ? 'warning' : 'success';
  return <span className={`badge bg-${color} me-1 mb-1`}>{name}</span>;
}

export default function RCADashboard() {
  const [overview,   setOverview]   = useState(null);
  const [breakdown,  setBreakdown]  = useState(null);
  const [defs,       setDefs]       = useState(null);
  const [tab,        setTab]        = useState('overview');
  const [filterSev,  setFilterSev]  = useState('all');
  const [filterStat, setFilterStat] = useState('all');
  const [err,        setErr]        = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/rca/overview`).then(r => r.json()),
      fetch(`${API}/api/rca/breakdown`).then(r => r.json()),
      fetch(`${API}/api/rca/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err)       return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = overview.kpis || {};

  /* ── filtered issues ── */
  const allIssues = breakdown?.issues || [];
  const filteredIssues = allIssues.filter(i =>
    (filterSev  === 'all' || i.severity === filterSev) &&
    (filterStat === 'all' || i.status   === filterStat)
  );

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h2 className="mb-0 fw-bold">🔍 Root Cause Analysis Center</h2>
          <p className="text-muted mb-0 small">
            Open issues · SLA tracking · component health · fix evidence
            &nbsp;·&nbsp;
            <span className="text-secondary">{overview.generated}</span>
          </p>
        </div>
        {kpi.p1_critical > 0 && (
          <span className="badge bg-danger fs-6 ms-auto">
            ⚠ {kpi.p1_critical} P1 Open
          </span>
        )}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active fw-semibold' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPI row */}
          <div className="row g-3 mb-4">
            <KPI label="Open Issues"      value={kpi.open_issues}        color="danger"   />
            <KPI label="P1 Critical"      value={kpi.p1_critical}        color="danger"   sub="system-down risk" />
            <KPI label="P2 High"          value={kpi.p2_high}            color="warning"  />
            <KPI label="SLA Breached"     value={kpi.sla_breached}       color={kpi.sla_breached > 0 ? 'danger' : 'success'} />
            <KPI label="Fixed"            value={kpi.fixed_issues}       color="success"  />
            <KPI label="In Progress"      value={kpi.in_progress}        color="warning"  />
            <KPI label="Components Down"  value={kpi.components_down}    color={kpi.components_down > 0 ? 'danger' : 'success'} />
            <KPI label="Components OK"    value={kpi.components_healthy} color="success"  sub={`of ${kpi.components_total} total`} />
          </div>

          {/* Two-column: severity + layer */}
          <div className="row g-3 mb-4">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Severity Breakdown</div>
                <div className="card-body">
                  {Object.entries(overview.severity_breakdown || {}).map(([sev, count]) => (
                    <div key={sev} className="d-flex justify-content-between align-items-center mb-2">
                      <SevBadge sev={sev} />
                      <span className="fw-bold">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Status Summary</div>
                <div className="card-body">
                  {Object.entries(overview.status_summary || {}).map(([stat, count]) => (
                    <div key={stat} className="d-flex justify-content-between align-items-center mb-2">
                      <StatBadge status={stat} />
                      <span className="fw-bold">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Layer Breakdown</div>
                <div className="card-body">
                  {Object.entries(overview.layer_breakdown || {}).map(([layer, info]) => (
                    <div key={layer} className="d-flex justify-content-between align-items-center mb-1">
                      <span className="small">{info.icon} {layer}</span>
                      <span className="small">
                        <span className="text-danger me-1">{info.open} open</span>
                        <span className="text-success">{info.fixed} fixed</span>
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Component health */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Component Health</div>
            <div className="card-body">
              {(overview.component_summary?.down || []).length > 0 && (
                <div className="mb-2">
                  <span className="text-danger fw-semibold me-2">DOWN:</span>
                  {overview.component_summary.down.map(c => <ComponentPill key={c} name={c} status="down" />)}
                </div>
              )}
              {(overview.component_summary?.degraded || []).length > 0 && (
                <div className="mb-2">
                  <span className="text-warning fw-semibold me-2">DEGRADED:</span>
                  {overview.component_summary.degraded.map(c => <ComponentPill key={c} name={c} status="degraded" />)}
                </div>
              )}
              {(overview.component_summary?.healthy || []).length > 0 && (
                <div className="mb-0">
                  <span className="text-success fw-semibold me-2">HEALTHY:</span>
                  {overview.component_summary.healthy.map(c => <ComponentPill key={c} name={c} status="healthy" />)}
                </div>
              )}
            </div>
          </div>

          {/* Top open issues */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Top Open Issues</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>ID</th><th>Sev</th><th>Layer</th><th>Symptom</th><th>Fix</th><th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.top_open || []).map(issue => (
                    <tr key={issue.id}>
                      <td className="fw-mono small">{issue.id}</td>
                      <td><SevBadge sev={issue.severity} /></td>
                      <td className="small">{_LAYER_ICON(issue.layer)} {issue.layer}</td>
                      <td className="small" style={{ maxWidth: 260 }}>{issue.symptom}</td>
                      <td className="small text-muted" style={{ maxWidth: 200 }}>{issue.fix}</td>
                      <td><StatBadge status={issue.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── BREAKDOWN ── */}
      {tab === 'breakdown' && (
        <>
          {/* Filters */}
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <select className="form-select form-select-sm w-auto"
              value={filterSev} onChange={e => setFilterSev(e.target.value)}>
              <option value="all">All Severities</option>
              {['P1','P2','P3','P4'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <select className="form-select form-select-sm w-auto"
              value={filterStat} onChange={e => setFilterStat(e.target.value)}>
              <option value="all">All Statuses</option>
              {['open','running','fixed'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <span className="text-muted small align-self-center">
              {filteredIssues.length} of {allIssues.length} issues
            </span>
          </div>

          {/* Issue table */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">RCA Issue Register</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>ID</th><th>Sev</th><th>Layer</th><th>Symptom</th>
                    <th>Guidance / Fix</th><th>Status</th><th>Scanned</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredIssues.length === 0 ? (
                    <tr><td colSpan={7} className="text-center text-muted py-3">No issues match filters</td></tr>
                  ) : filteredIssues.map(issue => (
                    <tr key={issue.id}>
                      <td className="small text-muted fw-semibold">{issue.id}</td>
                      <td><SevBadge sev={issue.severity} /></td>
                      <td className="small">{issue.layer_icon} {issue.layer}</td>
                      <td className="small" style={{ maxWidth: 240 }}>{issue.symptom}</td>
                      <td className="small text-muted" style={{ maxWidth: 200 }}>{issue.fix}</td>
                      <td><StatBadge status={issue.status} /></td>
                      <td className="small text-muted">{(issue.scanned_at || '').slice(0, 16)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Component health detail */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Component Health Detail</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Component</th><th>Status</th><th>Response (ms)</th>
                    <th>CPU %</th><th>Mem %</th><th>Errors</th><th>Last Check</th>
                  </tr>
                </thead>
                <tbody>
                  {(breakdown?.components || []).map(c => {
                    const statColor = c.status === 'down' ? 'danger' : c.status === 'degraded' ? 'warning' : 'success';
                    return (
                      <tr key={c.component}>
                        <td className="fw-semibold small">{c.component}</td>
                        <td><span className={`badge bg-${statColor}`}>{c.status}</span></td>
                        <td className="small">{c.response_time_ms}</td>
                        <td className="small">{c.cpu_pct}%</td>
                        <td className="small">{c.memory_pct}%</td>
                        <td className="small">{c.error_count}</td>
                        <td className="small text-muted">{(c.last_check || '').slice(0, 16)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* SLA rules */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">SLA Rules</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Event</th><th>SLA</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.sla_rules || []).map((r, i) => (
                    <tr key={i}>
                      <td className="small fw-semibold">{r.event}</td>
                      <td><span className="badge bg-primary">{r.sla}</span></td>
                      <td className="small text-muted">{r.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && (
        <>
          <div className="row g-3 mb-4">
            {(defs?.terms || []).map((t, i) => (
              <div key={i} className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header fw-semibold">{t.term}</div>
                  <div className="card-body small">
                    <p className="mb-2">{t.definition}</p>
                    {t.example && (
                      <p className="text-muted mb-2"><em>Example: {t.example}</em></p>
                    )}
                    {t.levels && (
                      <ul className="mb-0">
                        {Object.entries(t.levels).map(([k, v]) => (
                          <li key={k}><SevBadge sev={k} /> — {v}</li>
                        ))}
                      </ul>
                    )}
                    {t.layers && (
                      <div className="d-flex flex-wrap gap-1 mt-1">
                        {Object.entries(t.layers).map(([k, icon]) => (
                          <span key={k} className="badge bg-secondary">{icon} {k}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">RCA Field Reference</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Field</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs?.fields || []).map((f, i) => (
                    <tr key={i}>
                      <td className="small fw-semibold">{f.field}</td>
                      <td className="small text-muted">{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function _LAYER_ICON(layer) {
  const icons = {
    model:'🤖', frontend:'🖥️', data:'🗄️', security:'🔒',
    git:'📦', backend:'⚙️', infra:'🏗️', api:'🔌',
    vector:'🧮', graph:'🕸️',
  };
  return icons[layer] || '📋';
}
