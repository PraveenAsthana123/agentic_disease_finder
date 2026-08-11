'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'all-issues', label: 'All Issues' },
  { id: 'per-surface', label: 'Per Surface' },
  { id: 'definitions', label: 'Definitions' },
];

const SEV_COLOR = { P0: 'dark', P1: 'danger', P2: 'warning', P3: 'info' };
const SURFACE_ICON = {
  model: '🤖', backend: '🖥️', data: '🗄️', security: '🔒',
  git: '📦', frontend: '🖼️', ops: '⚙️',
};

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

function SevBadge({ sev }) {
  const color = SEV_COLOR[sev] || 'secondary';
  return <span className={`badge bg-${color} me-1`}>{sev}</span>;
}

function IssueCard({ issue }) {
  const color = SEV_COLOR[issue.severity] || 'secondary';
  const icon = SURFACE_ICON[issue.surface] || '📌';
  return (
    <div className={`card mb-3 border-${color}`}>
      <div className="card-header d-flex justify-content-between align-items-center py-2">
        <span className="fw-semibold">
          <span className="me-2">{icon}</span>
          {issue.surface?.toUpperCase()} — <SevBadge sev={issue.severity} />
        </span>
        <span className="badge bg-secondary">{issue.status}</span>
      </div>
      <div className="card-body py-2">
        <p className="mb-1 fw-semibold small">{issue.issue}</p>
        <p className="mb-1 text-muted small"><strong>Guidance:</strong> {issue.guidance}</p>
        {issue.scanned_at && (
          <p className="mb-0 text-muted" style={{ fontSize: '0.7rem' }}>Scanned: {issue.scanned_at}</p>
        )}
      </div>
    </div>
  );
}

function OverviewPanel({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;

  const sevDist = ov.severity_distribution || [];
  const surfDist = ov.surface_distribution || [];
  const timeline = ov.scan_timeline || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Issues" value={ov.total_issues} color="primary" />
        <KPI label="Open Issues" value={ov.open_count} color={ov.open_count > 0 ? 'danger' : 'success'} sub={`${ov.open_rate}% open rate`} />
        <KPI label="P1 Critical Open" value={ov.critical_open} color={ov.critical_open > 0 ? 'danger' : 'success'} sub="requires immediate action" />
        <KPI label="Last Scan" value={ov.last_scan ? ov.last_scan.slice(0, 10) : '—'} color="info" sub={ov.last_scan || ''} />
      </div>

      <div className="row mb-4">
        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Severity Distribution</div>
            <div className="card-body">
              {sevDist.map(s => (
                <div key={s.name} className="d-flex justify-content-between align-items-center mb-2">
                  <SevBadge sev={s.name} />
                  <div className="flex-grow-1 mx-2">
                    <div className="progress" style={{ height: 14 }}>
                      <div
                        className={`progress-bar bg-${SEV_COLOR[s.name] || 'secondary'}`}
                        style={{ width: `${Math.round((s.value / Math.max(ov.total_issues, 1)) * 100)}%` }}
                      />
                    </div>
                  </div>
                  <span className="fw-bold small">{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card h-100">
            <div className="card-header fw-semibold">Surface Distribution</div>
            <div className="card-body">
              {surfDist.map(s => (
                <div key={s.name} className="d-flex justify-content-between align-items-center mb-2">
                  <span className="small">{SURFACE_ICON[s.name] || '📌'} {s.name}</span>
                  <div className="flex-grow-1 mx-2">
                    <div className="progress" style={{ height: 14 }}>
                      <div
                        className="progress-bar bg-primary"
                        style={{ width: `${Math.round((s.value / Math.max(ov.total_issues, 1)) * 100)}%` }}
                      />
                    </div>
                  </div>
                  <span className="fw-bold small">{s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {ov.open_count > 0 && (
        <div className="alert alert-warning d-flex align-items-start gap-2">
          <span style={{ fontSize: '1.2rem' }}>⚠️</span>
          <div>
            <strong>{ov.open_count} open advisor issue{ov.open_count !== 1 ? 's' : ''}</strong> require attention.
            {ov.critical_open > 0 && (
              <span className="text-danger ms-2 fw-bold">
                {ov.critical_open} P1 critical — immediate action required.
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function AllIssuesPanel({ bd }) {
  const [filter, setFilter] = useState('all');
  if (!bd) return <div className="text-muted">Loading…</div>;

  const issues = filter === 'open' ? (bd.open_issues || []) : (bd.all_issues || []);

  return (
    <div>
      <div className="d-flex gap-2 mb-3">
        <button className={`btn btn-sm ${filter === 'all' ? 'btn-primary' : 'btn-outline-primary'}`} onClick={() => setFilter('all')}>
          All ({(bd.all_issues || []).length})
        </button>
        <button className={`btn btn-sm ${filter === 'open' ? 'btn-danger' : 'btn-outline-danger'}`} onClick={() => setFilter('open')}>
          Open ({(bd.open_issues || []).length})
        </button>
      </div>

      {issues.length === 0 ? (
        <div className="alert alert-success">No issues match the selected filter.</div>
      ) : (
        issues.map(issue => <IssueCard key={issue.id} issue={issue} />)
      )}
    </div>
  );
}

function PerSurfacePanel({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;

  const summary = bd.surface_summary || [];
  const sevMatrix = bd.surface_severity || [];

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Surface Summary</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-hover table-sm mb-0">
              <thead className="table-light">
                <tr>
                  <th>Surface</th>
                  <th>Total</th>
                  <th>Open</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {summary.map(s => (
                  <tr key={s.surface}>
                    <td><span className="me-1">{SURFACE_ICON[s.surface] || '📌'}</span>{s.surface}</td>
                    <td>{s.total}</td>
                    <td>
                      <span className={`badge bg-${s.open_cnt > 0 ? 'danger' : 'success'}`}>{s.open_cnt}</span>
                    </td>
                    <td>
                      {s.open_cnt === 0
                        ? <span className="badge bg-success">Clean</span>
                        : <span className="badge bg-warning text-dark">Needs attention</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold">Surface × Severity Matrix</div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-hover table-sm mb-0">
              <thead className="table-light">
                <tr>
                  <th>Surface</th>
                  <th>P0</th>
                  <th>P1</th>
                  <th>P2</th>
                  <th>P3</th>
                </tr>
              </thead>
              <tbody>
                {sevMatrix.map(row => (
                  <tr key={row.surface}>
                    <td><span className="me-1">{SURFACE_ICON[row.surface] || '📌'}</span>{row.surface}</td>
                    {['P0','P1','P2','P3'].map(sev => (
                      <td key={sev}>
                        {row[sev] ? <SevBadge sev={sev} /> : <span className="text-muted">—</span>}
                        {row[sev] ? <span className="ms-1 small">{row[sev]}</span> : null}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ df }) {
  if (!df) return <div className="text-muted">Loading…</div>;

  const sevTiers = df.severity_tiers || [];
  const surfCats = df.surface_categories || [];
  const statDefs = df.status_definitions || [];
  const agent = df.advisor_agent || {};
  const glossary = df.glossary || [];

  return (
    <div>
      <div className="card mb-4">
        <div className="card-header fw-semibold">Advisor Agent</div>
        <div className="card-body">
          <p className="mb-1"><strong>Description:</strong> {agent.description}</p>
          <p className="mb-0"><strong>Trigger:</strong> {agent.trigger}</p>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header fw-semibold">Severity Tiers</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Tier</th><th>Label</th><th>Description</th></tr>
            </thead>
            <tbody>
              {sevTiers.map(t => (
                <tr key={t.tier}>
                  <td><SevBadge sev={t.tier} /></td>
                  <td className="fw-semibold">{t.label}</td>
                  <td className="small text-muted">{t.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header fw-semibold">Surface Categories</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Surface</th><th>Description</th></tr>
            </thead>
            <tbody>
              {surfCats.map(s => (
                <tr key={s.surface}>
                  <td>{SURFACE_ICON[s.surface] || '📌'} <strong>{s.surface}</strong></td>
                  <td className="small text-muted">{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header fw-semibold">Status Definitions</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Status</th><th>Meaning</th></tr>
            </thead>
            <tbody>
              {statDefs.map(s => (
                <tr key={s.status}>
                  <td><span className="badge bg-secondary">{s.status}</span></td>
                  <td className="small">{s.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-header fw-semibold">Glossary</div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-light">
              <tr><th>Term</th><th>Definition</th></tr>
            </thead>
            <tbody>
              {glossary.map(g => (
                <tr key={g.term}>
                  <td className="fw-semibold">{g.term}</td>
                  <td className="small text-muted">{g.definition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function AdvisorIssuesPage() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [df, setDf] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/advisor-issues/overview`).then(r => r.json()),
      fetch(`${API}/api/advisor-issues/breakdown`).then(r => r.json()),
      fetch(`${API}/api/advisor-issues/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOv(o);
      setBd(b);
      setDf(d);
    }).catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center gap-3 mb-4">
        <div>
          <h2 className="mb-0 fw-bold">🔍 Advisor Issues</h2>
          <div className="text-muted small">
            System health findings from the Advisor Agent — severity, surface, guidance, scan log
          </div>
        </div>
        {ov?.critical_open > 0 && (
          <span className="badge bg-danger ms-auto" style={{ fontSize: '0.9rem' }}>
            {ov.critical_open} P1 Open
          </span>
        )}
      </div>

      {err && <div className="alert alert-danger">API Error: {err}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview' && <OverviewPanel ov={ov} />}
      {tab === 'all-issues' && <AllIssuesPanel bd={bd} />}
      {tab === 'per-surface' && <PerSurfacePanel bd={bd} />}
      {tab === 'definitions' && <DefinitionsPanel df={df} />}
    </div>
  );
}
