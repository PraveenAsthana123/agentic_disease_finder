'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'checks', label: 'Live Checks' },
  { id: 'definitions', label: 'Definitions' },
];

const STATUS_COLORS = { ok: 'success', warning: 'warning', critical: 'danger', error: 'secondary' };
const STATUS_ICONS = { ok: '\u2705', warning: '\u26a0\ufe0f', critical: '\ud83d\udea8', error: '\u2753' };
const SEV_COLORS = { P1: 'danger', P2: 'warning' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '\u2014'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function StatusBadge({ status }) {
  const color = STATUS_COLORS[status] || 'secondary';
  const icon = STATUS_ICONS[status] || '';
  return <span className={`badge bg-${color}`}>{icon} {status?.toUpperCase()}</span>;
}

function SevBadge({ severity }) {
  const color = SEV_COLORS[severity] || 'secondary';
  return <span className={`badge bg-${color} me-1`}>{severity}</span>;
}

function OverallStatusBanner({ status }) {
  const color = STATUS_COLORS[status] || 'secondary';
  const icon = STATUS_ICONS[status] || '';
  const msgs = {
    ok: 'All systems operational',
    warning: 'Some checks need attention',
    critical: 'Critical issues detected — immediate action required',
    error: 'Monitoring check error — verify dependencies',
  };
  return (
    <div className={`alert alert-${color} d-flex align-items-center mb-4`} role="alert">
      <span className="fs-4 me-2">{icon}</span>
      <div>
        <strong>Overall Status: {status?.toUpperCase()}</strong>
        <div className="small">{msgs[status] || ''}</div>
      </div>
    </div>
  );
}

function HealthTrendBar({ trend }) {
  if (!trend || trend.length === 0) return null;
  const last14 = trend.slice(-14);
  return (
    <div className="card mb-4">
      <div className="card-header fw-bold">Health Trend (last 14 days)</div>
      <div className="card-body">
        <div className="d-flex gap-1 align-items-end" style={{ height: 60 }}>
          {last14.map((d, i) => {
            const hasIssue = d.issues > 0;
            return (
              <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <div
                  style={{
                    width: '100%',
                    height: 40,
                    background: hasIssue ? '#dc3545' : '#198754',
                    borderRadius: 3,
                    opacity: 0.85,
                  }}
                  title={`${d.date}: ${hasIssue ? d.issues + ' issue(s)' : 'healthy'}`}
                />
                <div className="text-muted" style={{ fontSize: '0.55rem', marginTop: 2 }}>
                  {d.date?.slice(5)}
                </div>
              </div>
            );
          })}
        </div>
        <div className="d-flex gap-3 mt-2">
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#198754', borderRadius: 2 }} /> Healthy</span>
          <span><span style={{ display: 'inline-block', width: 12, height: 12, background: '#dc3545', borderRadius: 2 }} /> Issue</span>
        </div>
      </div>
    </div>
  );
}

function CheckSummaryTable({ checks }) {
  if (!checks || checks.length === 0) return <div className="text-muted">No checks available.</div>;
  return (
    <div className="table-responsive">
      <table className="table table-sm table-hover align-middle">
        <thead className="table-dark">
          <tr>
            <th>Check</th>
            <th>Layer</th>
            <th>Severity</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {checks.map((c, i) => (
            <tr key={i}>
              <td className="fw-semibold">{c.check}</td>
              <td><span className="badge bg-secondary">{c.layer}</span></td>
              <td><SevBadge severity={c.severity} /></td>
              <td><StatusBadge status={c.status} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const s = data.summary || {};
  const checks = data.checks_summary || [];
  const trend = data.health_trend || [];

  return (
    <div>
      <OverallStatusBanner status={data.overall_status} />

      <div className="row mb-3">
        <KPI label="Total Checks" value={s.total_checks} color="primary" sub="live detection runs" />
        <KPI label="OK" value={s.ok} color="success" sub="within thresholds" />
        <KPI label="Warning" value={s.warning} color="warning" sub="approaching limit" />
        <KPI label="Critical" value={s.critical} color={s.critical > 0 ? 'danger' : 'success'} sub="immediate action" />
      </div>

      <HealthTrendBar trend={trend} />

      <div className="card mb-4">
        <div className="card-header fw-bold">Check Summary</div>
        <div className="card-body p-0">
          <CheckSummaryTable checks={checks} />
        </div>
      </div>

      <div className="alert alert-info small mb-0">
        <strong>Run at:</strong> {data.run_at ? new Date(data.run_at).toLocaleString() : '\u2014'}
        &nbsp;&mdash;&nbsp;Live detection against real clinical.db, ChromaDB, and conversation logs.
      </div>
    </div>
  );
}

function CheckDetail({ check }) {
  const [open, setOpen] = useState(false);
  const color = STATUS_COLORS[check.status] || 'secondary';
  const details = check.details || {};

  return (
    <div className={`card mb-3 border-${color}`}>
      <div
        className={`card-header bg-${color} bg-opacity-10 d-flex justify-content-between align-items-center`}
        style={{ cursor: 'pointer' }}
        onClick={() => setOpen(o => !o)}
      >
        <div>
          <StatusBadge status={check.status} />
          <span className="ms-2 fw-semibold">{check.check}</span>
          <span className="badge bg-light text-dark ms-2">{check.layer}</span>
          <SevBadge severity={check.severity} />
        </div>
        <span>{open ? '\u25b2' : '\u25bc'}</span>
      </div>
      {open && (
        <div className="card-body">
          <div className="row mb-3">
            <div className="col-md-6">
              <h6 className="text-muted">Details</h6>
              <table className="table table-sm table-borderless">
                <tbody>
                  {Object.entries(details).map(([k, v]) => (
                    <tr key={k}>
                      <td className="text-muted small" style={{ width: '45%' }}>{k.replace(/_/g, ' ')}</td>
                      <td className="small fw-semibold">
                        {typeof v === 'boolean' ? (v ? 'Yes' : 'No') :
                          typeof v === 'object' ? JSON.stringify(v).slice(0, 120) :
                          String(v)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="col-md-6">
              {check.threshold && (
                <div className="mb-2">
                  <h6 className="text-muted">Threshold</h6>
                  <p className="small mb-1">{check.threshold}</p>
                </div>
              )}
              {check.remediation && (
                <div>
                  <h6 className="text-muted">Remediation</h6>
                  <p className="small mb-0 text-primary">{check.remediation}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ChecksTab({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const checks = data.checks || [];
  const [filter, setFilter] = useState('all');

  const filtered = filter === 'all' ? checks : checks.filter(c => c.status === filter);

  return (
    <div>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {['all', 'critical', 'warning', 'ok', 'error'].map(f => (
          <button
            key={f}
            className={`btn btn-sm ${filter === f ? 'btn-dark' : 'btn-outline-secondary'}`}
            onClick={() => setFilter(f)}
          >
            {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
            {f !== 'all' && ` (${checks.filter(c => c.status === f).length})`}
          </button>
        ))}
      </div>

      {filtered.length === 0 && <div className="text-muted">No checks match filter.</div>}
      {filtered.map((c, i) => <CheckDetail key={i} check={c} />)}

      <div className="alert alert-info small">
        <strong>Run at:</strong> {data.run_at ? new Date(data.run_at).toLocaleString() : '\u2014'}
        &nbsp;&mdash;&nbsp;Click any check to expand details, thresholds, and remediation steps.
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading...</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const metrics = data.metrics || [];
  const statuses = data.statuses || [];
  const layers = data.layers || [];
  const standards = data.standards || [];

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-bold">Metrics</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Metric</th><th>Description</th><th>Unit</th></tr>
            </thead>
            <tbody>
              {metrics.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold small">{m.name}</td>
                  <td className="small">{m.description}</td>
                  <td><span className="badge bg-secondary">{m.unit}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header fw-bold">Check Statuses</div>
        <div className="card-body p-0">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Status</th><th>Color</th><th>Meaning</th></tr>
            </thead>
            <tbody>
              {statuses.map((s, i) => (
                <tr key={i}>
                  <td><StatusBadge status={s.status} /></td>
                  <td className="small">{s.color}</td>
                  <td className="small">{s.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {layers.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-bold">Monitored Layers</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Layer</th><th>Description</th></tr>
              </thead>
              <tbody>
                {layers.map((l, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-primary">{l.layer}</span></td>
                    <td className="small">{l.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {standards.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-bold">Standards &amp; References</div>
          <div className="card-body p-0">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Standard</th><th>Relevance</th></tr>
              </thead>
              <tbody>
                {standards.map((s, i) => (
                  <tr key={i}>
                    <td className="fw-semibold small">{s.name || s}</td>
                    <td className="small">{s.relevance || ''}</td>
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

export default function ProductionMonitoringPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [checks, setChecks] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/production-monitoring/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/production-monitoring/breakdown`).then(r => r.json()).then(setChecks).catch(() => setChecks({ error: 'Failed to load checks' }));
    fetch(`${API}/api/production-monitoring/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-3">
        <span style={{ fontSize: '2rem' }}>&#x1f6e1;&#xfe0f;</span>
        <div>
          <h2 className="mb-0 fw-bold">Production Monitoring Dashboard</h2>
          <div className="text-muted small">Live agentic-AI system health — 6 checks across Agent / MCP / Vector DB / Retrieval / Planner layers</div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active fw-bold' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewTab data={overview} />}
      {tab === 'checks' && <ChecksTab data={checks} />}
      {tab === 'definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
