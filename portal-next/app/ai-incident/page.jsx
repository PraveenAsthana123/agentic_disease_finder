'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'incidents',   label: 'Incidents Log' },
  { id: 'severity',    label: 'Severity & MTTR' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function SeverityBadge({ level }) {
  const map = { Critical: 'danger', High: 'warning', Medium: 'info', Low: 'success' };
  return <span className={`badge bg-${map[level] || 'secondary'}`}>{level || '—'}</span>;
}

function StatusBadge({ status }) {
  const map = { Resolved: 'success', Open: 'danger', 'In Progress': 'warning' };
  return <span className={`badge bg-${map[status] || 'secondary'}`}>{status || '—'}</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const kpis = data.kpis || {};
  const sevDist = data.severity_distribution || [];
  const catDist = data.category_distribution || [];
  const timeline = data.incident_timeline || [];
  const topCats = data.top_categories || [];
  const totalSev = sevDist.reduce((s, x) => s + x.count, 0);
  const totalCat = catDist.reduce((s, x) => s + x.count, 0);
  const sevColors = { Critical: 'danger', High: 'warning', Medium: 'info', Low: 'success' };

  return (
    <div>
      {/* KPI Row */}
      <div className="row mb-3">
        <KPI label="Total Incidents" value={kpis.total_incidents} color="primary" sub="all time" />
        <KPI label="Open" value={kpis.open_incidents} color="danger" sub="unresolved" />
        <KPI label="Resolved" value={kpis.resolved_incidents} color="success" sub={`${data.resolution_rate_pct?.toFixed(0) ?? '—'}% resolution rate`} />
        <KPI label="MTTR" value={kpis.mttr_hours != null ? `${kpis.mttr_hours.toFixed(1)}h` : '—'} color="warning" sub="mean time to resolve" />
      </div>
      <div className="row mb-4">
        <KPI label="Last 30 Days" value={kpis.incidents_30d} color="info" sub="recent incidents" />
        <KPI label="Critical" value={kpis.severity_critical} color="danger" sub="P0 incidents" />
        <KPI label="High" value={kpis.severity_high} color="warning" sub="P1 incidents" />
        <KPI label="Low" value={kpis.severity_low} color="success" sub="informational" />
      </div>

      {/* Severity Distribution */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Severity Distribution</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Severity</th><th>Count</th><th>Share</th><th>Bar</th></tr>
            </thead>
            <tbody>
              {sevDist.map((s, i) => {
                const pct = totalSev > 0 ? (s.count / totalSev) * 100 : 0;
                return (
                  <tr key={i}>
                    <td><SeverityBadge level={s.name} /></td>
                    <td>{s.count}</td>
                    <td>{pct.toFixed(1)}%</td>
                    <td style={{ width: '35%' }}>
                      <div className="progress" style={{ height: '10px' }}>
                        <div className={`progress-bar bg-${sevColors[s.name] || 'secondary'}`} style={{ width: `${pct}%` }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Category Distribution */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Incident Categories</div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead className="table-light">
              <tr><th>Category</th><th>Count</th><th>Share</th><th>Bar</th></tr>
            </thead>
            <tbody>
              {topCats.map((c, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{c.category}</td>
                  <td>{c.count}</td>
                  <td>{c.pct?.toFixed(1)}%</td>
                  <td style={{ width: '35%' }}>
                    <div className="progress" style={{ height: '10px' }}>
                      <div className="progress-bar bg-primary" style={{ width: `${c.pct}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Incident Timeline */}
      {timeline.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Incident Timeline (Recent Days)</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Date</th><th>Incidents</th><th>Resolved</th><th>Open</th><th>Resolution Bar</th></tr>
              </thead>
              <tbody>
                {timeline.map((t, i) => {
                  const open = t.incidents - t.resolved;
                  const pct = t.incidents > 0 ? (t.resolved / t.incidents) * 100 : 0;
                  return (
                    <tr key={i}>
                      <td>{t.date}</td>
                      <td>{t.incidents}</td>
                      <td className="text-success">{t.resolved}</td>
                      <td className={open > 0 ? 'text-danger' : 'text-muted'}>{open}</td>
                      <td style={{ width: '30%' }}>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className="progress-bar bg-success" style={{ width: `${pct}%` }} />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function IncidentsPanel({ data }) {
  const [filter, setFilter] = useState('all');
  const [sevFilter, setSevFilter] = useState('all');
  if (!data) return <div className="text-muted p-3">Loading…</div>;

  const incidents = data.recent_incidents || [];
  const cats = ['all', ...new Set(incidents.map(i => i.category).filter(Boolean))];
  const sevs = ['all', 'Critical', 'High', 'Medium', 'Low'];

  const filtered = incidents.filter(i => {
    const catOk = filter === 'all' || i.category === filter;
    const sevOk = sevFilter === 'all' || i.severity === sevFilter;
    return catOk && sevOk;
  });

  return (
    <div>
      <div className="d-flex gap-2 flex-wrap mb-3 align-items-center">
        <div>
          <label className="small text-muted me-1">Category:</label>
          <select className="form-select form-select-sm d-inline-block w-auto" value={filter} onChange={e => setFilter(e.target.value)}>
            {cats.map(c => <option key={c} value={c}>{c === 'all' ? 'All Categories' : c}</option>)}
          </select>
        </div>
        <div>
          <label className="small text-muted me-1">Severity:</label>
          <select className="form-select form-select-sm d-inline-block w-auto" value={sevFilter} onChange={e => setSevFilter(e.target.value)}>
            {sevs.map(s => <option key={s} value={s}>{s === 'all' ? 'All Severities' : s}</option>)}
          </select>
        </div>
        <span className="badge bg-secondary ms-auto">{filtered.length} of {incidents.length}</span>
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th>
              <th>Timestamp</th>
              <th>Category</th>
              <th>Severity</th>
              <th>Status</th>
              <th>Source</th>
              <th>MTTR (h)</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((inc, i) => (
              <tr key={i}>
                <td className="text-muted small font-monospace">{inc.id}</td>
                <td className="small">{inc.timestamp ? new Date(inc.timestamp).toLocaleString() : '—'}</td>
                <td className="small">{inc.category || '—'}</td>
                <td><SeverityBadge level={inc.severity} /></td>
                <td><StatusBadge status={inc.status} /></td>
                <td className="small text-muted">{inc.source || '—'}</td>
                <td className="small">{inc.resolution_time_hrs != null ? inc.resolution_time_hrs.toFixed(1) : '—'}</td>
                <td className="small text-muted" style={{ maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                  title={inc.description}>{inc.description || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filtered.length === 0 && <div className="text-muted text-center py-3">No incidents match the selected filters.</div>}
    </div>
  );
}

function SeverityPanel({ overview }) {
  if (!overview) return <div className="text-muted p-3">Loading…</div>;
  const kpis = overview.kpis || {};
  const sevDist = overview.severity_distribution || [];
  const total = sevDist.reduce((s, x) => s + x.count, 0);

  const sevDetails = [
    { level: 'Critical', count: kpis.severity_critical, color: 'danger', response: '< 1 hour', meaning: 'Patient safety events or expert disagreement with high-confidence AI. Requires immediate triage.' },
    { level: 'High', count: kpis.severity_high, color: 'warning', response: '< 4 hours', meaning: 'HITL override decisions, low AI confidence (<0.5) on clinical decisions, or system crashes.' },
    { level: 'Medium', count: kpis.severity_medium, color: 'info', response: '< 24 hours', meaning: 'Transaction log errors, failures, blocked operations, and governance events.' },
    { level: 'Low', count: kpis.severity_low, color: 'success', response: '< 72 hours', meaning: 'Informational events, timeouts, minor anomalies, and non-critical alerts.' },
  ];

  return (
    <div>
      {/* MTTR Summary */}
      <div className="card mb-4">
        <div className="card-header fw-semibold">Mean Time to Resolve (MTTR)</div>
        <div className="card-body">
          <div className="row text-center">
            <div className="col-md-4">
              <div className="h2 fw-bold text-warning">{kpis.mttr_hours != null ? `${kpis.mttr_hours.toFixed(1)}h` : '—'}</div>
              <div className="text-muted small">Overall MTTR</div>
            </div>
            <div className="col-md-4">
              <div className="h2 fw-bold text-success">{overview.resolution_rate_pct?.toFixed(1)}%</div>
              <div className="text-muted small">Resolution Rate</div>
            </div>
            <div className="col-md-4">
              <div className="h2 fw-bold text-danger">{kpis.open_incidents}</div>
              <div className="text-muted small">Open Incidents</div>
            </div>
          </div>
        </div>
      </div>

      {/* Severity Tier Cards */}
      <div className="row mb-4">
        {sevDetails.map((s, i) => {
          const pct = total > 0 ? ((s.count || 0) / total * 100).toFixed(1) : 0;
          return (
            <div key={i} className="col-md-6 mb-3">
              <div className={`card border-${s.color} h-100`}>
                <div className={`card-header bg-${s.color} text-white fw-semibold d-flex justify-content-between`}>
                  <span>{s.level}</span>
                  <span className="badge bg-white text-dark">{s.count ?? 0} incidents ({pct}%)</span>
                </div>
                <div className="card-body">
                  <div className="mb-2">
                    <span className="fw-semibold text-muted small">Target Response:</span>
                    <span className="ms-2 small">{s.response}</span>
                  </div>
                  <div className="small text-muted">{s.meaning}</div>
                  <div className="progress mt-2" style={{ height: '8px' }}>
                    <div className={`progress-bar bg-${s.color}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted p-3">Loading…</div>;
  const sevLevels = data.severity_levels || [];
  const categories = data.incident_categories || [];
  const metrics = data.key_metrics || [];

  return (
    <div>
      {/* Severity Levels */}
      {sevLevels.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Severity Level Definitions</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Level</th><th>Description</th></tr>
              </thead>
              <tbody>
                {sevLevels.map((s, i) => (
                  <tr key={i}>
                    <td><SeverityBadge level={s.level} /></td>
                    <td className="small">{s.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Incident Categories */}
      {categories.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Incident Category Definitions</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Category</th><th>Description</th></tr>
              </thead>
              <tbody>
                {categories.map((c, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{c.category}</td>
                    <td className="small">{c.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Key Metrics */}
      {metrics.length > 0 && (
        <div className="card mb-4">
          <div className="card-header fw-semibold">Key Metrics</div>
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0">
              <thead className="table-light">
                <tr><th>Metric</th><th>Description</th></tr>
              </thead>
              <tbody>
                {metrics.map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{m.metric}</td>
                    <td className="small">{m.description}</td>
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

export default function AIIncidentPage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/ai-incident/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load' }));
    fetch(`${API}/api/ai-incident/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load' }));
    fetch(`${API}/api/ai-incident/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load' }));
  }, []);

  return (
    <div>
      <h2 className="mb-1">AI Incident Management</h2>
      <p className="text-muted small mb-3">
        175 incidents · 35 open · 140 resolved · MTTR 105.8h · 6 categories · real transaction_log + clinical alert sources
      </p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel data={overview} />}
      {tab === 'incidents'   && <IncidentsPanel data={breakdown} />}
      {tab === 'severity'    && <SeverityPanel overview={overview} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
