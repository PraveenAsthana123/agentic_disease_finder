'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'breakdown', label: 'All Jobs' },
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

function DistBar({ items, colorFn }) {
  const total = items.reduce((a, b) => a + b.value, 0);
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.sort((a, b) => b.value - a.value).map(({ name, value }) => {
          const pct = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
          const color = colorFn ? colorFn(name) : 'primary';
          return (
            <tr key={name}>
              <td className="text-nowrap small fw-semibold" style={{ width: '40%' }}>{name}</td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 10 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                </div>
              </td>
              <td className="small text-end">{value} <span className="text-muted">({pct}%)</span></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ReportBadge({ exists }) {
  return exists
    ? <span className="badge bg-success">Report</span>
    : <span className="badge bg-secondary">No Report</span>;
}

function OverviewPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  const k = data.kpis || {};
  const schedColor = n => ({ Daily: 'success', Hourly: 'info', Other: 'warning' }[n] || 'secondary');

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Jobs" value={k.total_jobs} color="primary" />
        <KPI label="Daily Jobs" value={k.daily_jobs} color="success" sub="run once or twice/day" />
        <KPI label="Hourly Jobs" value={k.hourly_jobs} color="info" sub="run every hour" />
        <KPI label="Jobs with Reports" value={k.jobs_with_reports} color="warning" sub="latest report on disk" />
      </div>
      <div className="row mb-3">
        <KPI label="Unique Cron Tags" value={k.unique_cron_tags} color="primary" sub="AGENTICFINDER-* entries" />
        <KPI label="Unique Scripts" value={k.unique_scripts} color="secondary" sub="distinct Python scripts" />
      </div>

      <div className="row mb-3">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Schedule Distribution</div>
            <div className="card-body p-2">
              {data.schedule_distribution && data.schedule_distribution.length > 0
                ? <DistBar items={data.schedule_distribution} colorFn={schedColor} />
                : <span className="text-muted">No distribution data</span>}
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Jobs Summary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Job</th>
                    <th>Schedule</th>
                    <th>Report</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.jobs_summary || []).map(j => (
                    <tr key={j.id}>
                      <td className="small">{j.label}</td>
                      <td className="small text-muted">{j.schedule}</td>
                      <td><ReportBadge exists={j.has_report} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;
  const [search, setSearch] = useState('');
  const jobs = (data.jobs || []).filter(j =>
    !search || j.label.toLowerCase().includes(search.toLowerCase()) ||
    j.cron_tag.toLowerCase().includes(search.toLowerCase()) ||
    j.purpose.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div className="mb-3">
        <input
          className="form-control"
          placeholder="Search jobs by name, tag, or purpose…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover align-middle">
          <thead className="table-dark">
            <tr>
              <th>Job</th>
              <th>Schedule</th>
              <th>Script</th>
              <th>Cron Tag</th>
              <th>Report</th>
              <th>Size</th>
              <th>Purpose</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map(j => (
              <tr key={j.id}>
                <td className="fw-semibold small">{j.label}</td>
                <td className="small text-muted">{j.schedule}</td>
                <td className="small font-monospace text-info">{j.script}</td>
                <td className="small"><code>{j.cron_tag}</code></td>
                <td><ReportBadge exists={j.report_exists} /></td>
                <td className="small text-end">
                  {j.report_exists ? `${(j.report_size_bytes / 1024).toFixed(1)} KB` : '—'}
                </td>
                <td className="small">{j.purpose}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-muted small mt-2">{jobs.length} of {(data.jobs || []).length} jobs shown</div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  if (data.error) return <div className="alert alert-warning">{data.error}</div>;

  return (
    <div className="row">
      <div className="col-md-4 mb-3">
        <div className="card h-100">
          <div className="card-header fw-semibold">Schedule Legend</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <tbody>
                {(data.schedule_legend || []).map(l => (
                  <tr key={l.label}>
                    <td>
                      <span className="badge" style={{ backgroundColor: l.color }}>{l.label}</span>
                    </td>
                    <td className="small">{l.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="col-md-8 mb-3">
        <div className="card h-100">
          <div className="card-header fw-semibold">Glossary</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead className="table-light">
                <tr><th>Term</th><th>Definition</th></tr>
              </thead>
              <tbody>
                {(data.glossary || []).map(g => (
                  <tr key={g.term}>
                    <td className="fw-semibold small text-nowrap">{g.term}</td>
                    <td className="small">{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="col-md-6 mb-3">
        <div className="card h-100">
          <div className="card-header fw-semibold">Clinical Notes</div>
          <ul className="list-group list-group-flush">
            {(data.clinical_notes || []).map((n, i) => (
              <li key={i} className="list-group-item small">{n}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="col-md-6 mb-3">
        <div className="card h-100">
          <div className="card-header fw-semibold">References</div>
          <ul className="list-group list-group-flush">
            {(data.references || []).map((r, i) => (
              <li key={i} className="list-group-item small">
                <strong>{r.ref}</strong> — {r.detail}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

export default function ScheduledJobsDashboard() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/scheduled-jobs/overview`).then(r => r.json()).then(setOverview).catch(() => setOverview({ error: 'Failed to load overview' }));
    fetch(`${API}/api/scheduled-jobs/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => setBreakdown({ error: 'Failed to load breakdown' }));
    fetch(`${API}/api/scheduled-jobs/definitions`).then(r => r.json()).then(setDefinitions).catch(() => setDefinitions({ error: 'Failed to load definitions' }));
  }, []);

  return (
    <div className="container-fluid py-4">
      <h2 className="mb-1">Scheduled Jobs Dashboard</h2>
      <p className="text-muted mb-3">Background job registry — cron-scheduled pipelines, training, validation, drift, and governance jobs</p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel data={overview} />}
      {tab === 'breakdown' && <BreakdownPanel data={breakdown} />}
      {tab === 'definitions' && <DefinitionsPanel data={definitions} />}
    </div>
  );
}
