'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const ROLE_COLORS = {
  'Neurologist': 'primary',
  'Psychiatrist': 'info',
  'Occupational Therapist': 'success',
  'EEG Technician': 'warning',
  'Clinical Psychologist': 'danger',
  'Radiologist': 'secondary',
  'IoT Engineer': 'primary',
  'AI Control Tower': 'info',
  'AI Security': 'danger',
  'AI Risk': 'warning',
  'AI Federation': 'success',
  'IS SOP': 'secondary',
  'IRB / Governance Reviewer': 'dark',
};
const roleColor = (role) => ROLE_COLORS[role] || 'dark';

const FORMAT_COLORS = { 'PDF': 'danger', 'CSV': 'success', 'PDF/CSV': 'warning', 'dashboard': 'primary', 'MD': 'secondary' };
const fmtColor = (fmt) => FORMAT_COLORS[fmt] || 'secondary';

const TABS = [
  { id: 'overview', label: '📊 Overview' },
  { id: 'kpis', label: '📈 KPIs by Role' },
  { id: 'reports', label: '📄 Reports by Role' },
  { id: 'definitions', label: '📖 Definitions' },
];

export default function RoleDashboardsPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selectedRole, setSelectedRole] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/role-dashboards/overview`).then(r => r.json()),
      fetch(`${API}/api/role-dashboards/breakdown`).then(r => r.json()),
      fetch(`${API}/api/role-dashboards/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => {
        setOverview(o);
        setBreakdown(b);
        setDefs(d);
        if (b?.per_role?.length) setSelectedRole(b.per_role[0].role);
      })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading role dashboards…</div>;

  const k = overview.kpis || {};
  const roles = breakdown?.per_role || [];
  const activeRole = roles.find(r => r.role === selectedRole);

  return (
    <div className="container-fluid py-3">
      <h3>📋 Per-Role Dashboards &amp; Reports</h3>
      <p className="text-muted small">
        KPI dashboard tiles + standard report catalog for each clinical department —{' '}
        {k.total_roles} roles · {k.total_kpis} KPIs · {k.total_reports} reports · all built.
      </p>

      {/* KPI tiles */}
      <div className="row g-2 mb-3">
        {[
          ['Roles', k.total_roles, 'primary'],
          ['Total KPIs', k.total_kpis, 'info'],
          ['KPIs Built', k.kpis_built, 'success'],
          ['Total Reports', k.total_reports, 'warning'],
          ['Reports Built', k.reports_built, 'danger'],
          ['Coverage', `${k.kpis_built && k.total_kpis ? Math.round((k.kpis_built / k.total_kpis) * 100) : 0}%`, 'dark'],
        ].map(([label, val, color]) => (
          <div key={label} className="col-6 col-md-2">
            <div className={`card border-${color} text-center p-2 h-100`}>
              <div className={`fs-5 fw-bold text-${color}`}>{val ?? '—'}</div>
              <div className="small text-muted">{label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab nav */}
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

      {/* Overview tab */}
      {tab === 'overview' && (
        <div>
          <div className="row g-3 mb-4">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-semibold">KPIs per Role</div>
                <div className="card-body p-2">
                  {(overview.kpis_per_role || []).map(r => (
                    <div key={r.role} className="d-flex align-items-center mb-1">
                      <span className="me-2 fs-5">{r.icon}</span>
                      <span className="small me-auto">{r.role}</span>
                      <div className="progress flex-grow-1 mx-2" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${roleColor(r.role)}`}
                          style={{ width: `${(r.count / 5) * 100}%` }}
                        />
                      </div>
                      <span className={`badge bg-${roleColor(r.role)}`}>{r.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-semibold">Reports per Role</div>
                <div className="card-body p-2">
                  {(overview.reports_per_role || []).map(r => (
                    <div key={r.role} className="d-flex align-items-center mb-1">
                      <span className="me-2 fs-5">{r.icon}</span>
                      <span className="small me-auto">{r.role}</span>
                      <div className="progress flex-grow-1 mx-2" style={{ height: 8 }}>
                        <div
                          className={`progress-bar bg-${roleColor(r.role)}`}
                          style={{ width: `${(r.count / 4) * 100}%` }}
                        />
                      </div>
                      <span className={`badge bg-${roleColor(r.role)}`}>{r.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Report Cadence Distribution</div>
                <div className="card-body p-2">
                  <div className="table-responsive">
                    <table className="table table-sm mb-0">
                      <thead className="table-light"><tr><th>Cadence</th><th>Count</th></tr></thead>
                      <tbody>
                        {(overview.cadence_distribution || []).map(c => (
                          <tr key={c.cadence}>
                            <td><code className="small">{c.cadence}</code></td>
                            <td><span className="badge bg-secondary">{c.count}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Report Format Distribution</div>
                <div className="card-body p-2">
                  <div className="d-flex flex-wrap gap-2">
                    {(overview.format_distribution || []).map(f => (
                      <div key={f.format} className={`card border-${fmtColor(f.format)} text-center p-2`} style={{ minWidth: 80 }}>
                        <div className={`fw-bold text-${fmtColor(f.format)}`}>{f.count}</div>
                        <div className="small text-muted">{f.format}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* KPIs by Role tab */}
      {tab === 'kpis' && (
        <div>
          <div className="mb-3 d-flex gap-2 flex-wrap">
            {roles.map(r => (
              <button
                key={r.role}
                className={`btn btn-sm ${selectedRole === r.role ? `btn-${roleColor(r.role)}` : `btn-outline-${roleColor(r.role)}`}`}
                onClick={() => setSelectedRole(r.role)}
              >{r.icon} {r.role}</button>
            ))}
          </div>

          {activeRole && (
            <div className={`card border-${roleColor(activeRole.role)}`}>
              <div className={`card-header bg-${roleColor(activeRole.role)} text-white d-flex justify-content-between align-items-center`}>
                <span className="fw-semibold">{activeRole.icon} {activeRole.role}</span>
                <span>{(activeRole.kpis || []).length} KPIs</span>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>#</th><th>KPI Label</th><th>Data Source</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {(activeRole.kpis || []).map((kpi, i) => (
                        <tr key={i}>
                          <td className="text-muted">{i + 1}</td>
                          <td className="fw-semibold small">{kpi.label}</td>
                          <td><code className="small">{kpi.source}</code></td>
                          <td>
                            <span className={`badge bg-${kpi.status === 'built' ? 'success' : kpi.status === 'partial' ? 'warning' : 'secondary'}`}>
                              {kpi.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          <div className="row g-3 mt-3">
            {roles.map(r => (
              <div key={r.role} className="col-md-4">
                <div
                  className={`card border-${roleColor(r.role)} h-100 ${selectedRole === r.role ? '' : 'opacity-75'}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelectedRole(r.role)}
                >
                  <div className="card-body p-2">
                    <div className="d-flex justify-content-between align-items-center mb-1">
                      <span className="small fw-semibold">{r.icon} {r.role}</span>
                      <span className={`badge bg-${roleColor(r.role)}`}>{(r.kpis || []).length}</span>
                    </div>
                    {(r.kpis || []).map((kpi, i) => (
                      <div key={i} className="small text-muted border-bottom py-1">• {kpi.label}</div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Reports by Role tab */}
      {tab === 'reports' && (
        <div>
          <div className="mb-3 d-flex gap-2 flex-wrap">
            {roles.map(r => (
              <button
                key={r.role}
                className={`btn btn-sm ${selectedRole === r.role ? `btn-${roleColor(r.role)}` : `btn-outline-${roleColor(r.role)}`}`}
                onClick={() => setSelectedRole(r.role)}
              >{r.icon} {r.role}</button>
            ))}
          </div>

          {activeRole && (
            <div className={`card border-${roleColor(activeRole.role)} mb-4`}>
              <div className={`card-header bg-${roleColor(activeRole.role)} text-white d-flex justify-content-between align-items-center`}>
                <span className="fw-semibold">{activeRole.icon} {activeRole.role}</span>
                <span>{(activeRole.reports || []).length} reports</span>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Report Name</th><th>Cadence</th><th>Format</th><th>Status</th></tr>
                    </thead>
                    <tbody>
                      {(activeRole.reports || []).map((rpt, i) => (
                        <tr key={i}>
                          <td className="fw-semibold small">{rpt.name}</td>
                          <td><code className="small">{rpt.cadence}</code></td>
                          <td>
                            <span className={`badge bg-${fmtColor(rpt.format)}`}>{rpt.format}</span>
                          </td>
                          <td>
                            <span className={`badge bg-${rpt.status === 'built' ? 'success' : 'secondary'}`}>
                              {rpt.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          <h5 className="mt-2">All Reports by Role</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered table-hover">
              <thead className="table-light">
                <tr><th>Role</th><th>Report</th><th>Cadence</th><th>Format</th><th>Status</th></tr>
              </thead>
              <tbody>
                {roles.flatMap(r =>
                  (r.reports || []).map((rpt, i) => (
                    <tr key={`${r.role}-${i}`}>
                      {i === 0
                        ? <td rowSpan={(r.reports || []).length} className="align-middle">
                            <span className={`badge bg-${roleColor(r.role)}`}>{r.icon} {r.role}</span>
                          </td>
                        : null}
                      <td className="small">{rpt.name}</td>
                      <td><code className="small">{rpt.cadence}</code></td>
                      <td><span className={`badge bg-${fmtColor(rpt.format)}`}>{rpt.format}</span></td>
                      <td><span className={`badge bg-${rpt.status === 'built' ? 'success' : 'secondary'}`}>{rpt.status}</span></td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Status Legend</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>Status</th><th>Meaning</th></tr></thead>
                    <tbody>
                      {(defs.status_legend || []).map(s => (
                        <tr key={s.status}>
                          <td>
                            <span className={`badge bg-${s.status === 'built' ? 'success' : s.status === 'partial' ? 'warning' : 'secondary'}`}>
                              {s.status}
                            </span>
                          </td>
                          <td className="small">{s.meaning}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card">
                <div className="card-header fw-semibold">Cadence Legend</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light"><tr><th>Cadence</th><th>Meaning</th></tr></thead>
                    <tbody>
                      {(defs.cadence_legend || []).map(c => (
                        <tr key={c.cadence}>
                          <td><code className="small">{c.cadence}</code></td>
                          <td className="small">{c.meaning}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {defs.format_legend && (
            <div className="card mt-3">
              <div className="card-header fw-semibold">Format Legend</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Format</th><th>Meaning</th></tr></thead>
                  <tbody>
                    {(defs.format_legend || []).map(f => (
                      <tr key={f.format}>
                        <td><span className={`badge bg-${fmtColor(f.format)}`}>{f.format}</span></td>
                        <td className="small">{f.meaning}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="card mt-3">
            <div className="card-header fw-semibold">Source</div>
            <div className="card-body small text-muted">
              <p className="mb-1"><strong>Config:</strong> <code>config/role_dashboards.json</code></p>
              <p className="mb-0">{overview.note}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
