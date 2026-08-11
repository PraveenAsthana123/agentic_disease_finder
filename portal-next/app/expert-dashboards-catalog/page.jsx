'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const PRIORITY_COLOR = {
  P0: '#dc2626', P1: '#ea580c', P2: '#d97706',
  P3: '#2563eb', 'N/A': '#6b7280',
};
const STATUS_COLOR = {
  built: '#16a34a', partial: '#f97316', planned: '#3b82f6',
};

function StatCard({ label, value, sub, color = '#3b82f6' }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-3">
          <div className="h2 fw-bold mb-0" style={{ color }}>{value}</div>
          <div className="small fw-semibold">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function PriBadge({ p }) {
  const c = PRIORITY_COLOR[p] || '#6b7280';
  return (
    <span className="badge ms-1" style={{ background: c, fontSize: 10 }}>{p}</span>
  );
}

function StatusDot({ s }) {
  const c = STATUS_COLOR[s] || '#9ca3af';
  return <span style={{ display:'inline-block', width:8, height:8, borderRadius:'50%', background:c, marginRight:5 }} />;
}

export default function ExpertDashboardsCatalogPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [err, setErr]   = useState(null);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('All');
  const [priFilter, setPriFilter]   = useState('All');
  const [expandedRoles, setExpandedRoles] = useState({});

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/expert-dashboards-catalog/overview`).then(r => r.json()),
      fetch(`${API}/api/expert-dashboards-catalog/breakdown`).then(r => r.json()),
      fetch(`${API}/api/expert-dashboards-catalog/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!ov)  return <div className="p-4 text-muted">Loading Expert Dashboards Catalog…</div>;

  const kpi = ov.kpis;
  const TABS = ['overview', 'by-role', 'directory', 'p0-essentials', 'definitions'];

  /* ── directory filter ── */
  const allDashboards = ov.dashboards_table || [];
  const roles = ['All', ...new Set(allDashboards.map(d => d.role).filter(Boolean))].sort();
  const priorities = ['All', 'P0', 'P1', 'P2', 'P3', 'N/A'];

  const filtered = allDashboards.filter(d => {
    const matchSearch = !search || d.name.toLowerCase().includes(search.toLowerCase()) ||
      (d.role || '').toLowerCase().includes(search.toLowerCase());
    const matchRole = roleFilter === 'All' || d.role === roleFilter;
    const matchPri  = priFilter === 'All' || d.priority === priFilter;
    return matchSearch && matchRole && matchPri;
  });

  const toggleRole = (role) => setExpandedRoles(prev => ({ ...prev, [role]: !prev[role] }));

  const Bar = ({ items, colorFn, labelKey = 'name', valueKey = 'value' }) => {
    const max = Math.max(...items.map(i => i[valueKey]), 1);
    return (
      <div>
        {items.slice(0, 12).map((item, i) => (
          <div key={i} className="mb-1">
            <div className="d-flex justify-content-between small mb-0">
              <span className="text-truncate" style={{ maxWidth: 180 }}>{item[labelKey]}</span>
              <span className="fw-bold">{item[valueKey]}</span>
            </div>
            <div className="progress" style={{ height: 8, borderRadius: 4 }}>
              <div className="progress-bar"
                style={{ width: `${(item[valueKey] / max) * 100}%`, background: colorFn ? colorFn(item) : '#3b82f6', borderRadius: 4 }} />
            </div>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="container-fluid py-4 px-3">
      <h4 className="fw-bold mb-1">📚 Expert Dashboards Catalog</h4>
      <p className="text-muted small mb-3">
        {kpi.total_dashboards} dashboards · {kpi.built} built · {kpi.unique_roles} roles ·{' '}
        {kpi.total_endpoints} API endpoints · Updated {ov.updated_at}
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}
              style={{ textTransform: 'capitalize' }}
            >
              {t.replace(/-/g, ' ')}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <StatCard label="Total Dashboards" value={kpi.total_dashboards} sub="All roles combined" color="#3b82f6" />
            <StatCard label="Built & Live"      value={kpi.built}           sub="Verified endpoints 200" color="#16a34a" />
            <StatCard label="Unique Roles"      value={kpi.unique_roles}    sub="Clinical specialties" color="#8b5cf6" />
            <StatCard label="API Endpoints"     value={kpi.total_endpoints} sub={`${kpi.dashboards_with_endpoints} dashboards with endpoints`} color="#0891b2" />
          </div>

          <div className="row mt-2">
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-semibold small">📊 Status Distribution</div>
                <div className="card-body">
                  {(ov.charts?.status_distribution || []).map((s, i) => (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <StatusDot s={s.name} />
                      <span className="small text-capitalize">{s.name}</span>
                      <span className="ms-auto fw-bold">{s.value}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-semibold small">🎯 Priority Breakdown</div>
                <div className="card-body small">
                  {(ov.charts?.priority_distribution || []).map((p, i) => (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <PriBadge p={p.name} />
                      <div className="progress flex-grow-1 ms-2" style={{ height: 10, borderRadius: 5 }}>
                        <div className="progress-bar" style={{
                          width: `${(p.value / kpi.total_dashboards) * 100}%`,
                          background: PRIORITY_COLOR[p.name] || '#6b7280',
                          borderRadius: 5
                        }} />
                      </div>
                      <span className="ms-2 fw-bold">{p.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header py-2 fw-semibold small">📚 Tech Libraries ({kpi.libraries_count})</div>
                <div className="card-body p-0">
                  <ul className="list-group list-group-flush small">
                    {(ov.libraries || []).map((lib, i) => (
                      <li key={i} className="list-group-item py-1">
                        <span className="fw-semibold">{lib.library}</span>
                        <span className="text-muted ms-1">— {lib.purpose}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mt-2">
            <div className="card-header py-2 fw-semibold small">👥 Dashboards per Role (top 15)</div>
            <div className="card-body">
              <Bar
                items={(ov.charts?.dashboards_per_role || []).slice(0, 15)}
                colorFn={() => '#3b82f6'}
              />
            </div>
          </div>
        </>
      )}

      {/* ── BY ROLE ── */}
      {tab === 'by-role' && (
        <div>
          <p className="text-muted small mb-3">{(bd?.per_role || []).length} role groups · click to expand</p>
          {(bd?.per_role || []).map((group, gi) => (
            <div key={gi} className="card shadow-sm mb-2">
              <div
                className="card-header py-2 d-flex justify-content-between align-items-center"
                style={{ cursor: 'pointer', background: expandedRoles[group.role] ? '#e0e7ff' : undefined }}
                onClick={() => toggleRole(group.role)}
              >
                <span className="fw-semibold small">
                  {expandedRoles[group.role] ? '▾' : '▸'} {group.role}
                </span>
                <span className="badge bg-primary">{group.count}</span>
              </div>
              {expandedRoles[group.role] && (
                <div className="card-body p-0">
                  <table className="table table-sm table-hover mb-0 small">
                    <thead className="table-light">
                      <tr>
                        <th>Dashboard</th>
                        <th>Priority</th>
                        <th>Status</th>
                        <th>Endpoints</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(group.dashboards || []).map((d, di) => (
                        <tr key={di}>
                          <td>{d.name}</td>
                          <td><PriBadge p={d.priority || 'N/A'} /></td>
                          <td>
                            <StatusDot s={d.status} />
                            <span className="text-capitalize">{d.status}</span>
                          </td>
                          <td>
                            {(d.endpoints || []).map((ep, ei) => (
                              <div key={ei}>
                                <code style={{ fontSize: 10 }}>{ep}</code>
                              </div>
                            ))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── DIRECTORY ── */}
      {tab === 'directory' && (
        <>
          <div className="row g-2 mb-3">
            <div className="col-md-5">
              <input
                className="form-control form-control-sm"
                placeholder="Search dashboards by name or role…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-3">
              <select className="form-select form-select-sm" value={roleFilter}
                onChange={e => setRoleFilter(e.target.value)}>
                {roles.map(r => <option key={r}>{r}</option>)}
              </select>
            </div>
            <div className="col-md-2">
              <select className="form-select form-select-sm" value={priFilter}
                onChange={e => setPriFilter(e.target.value)}>
                {priorities.map(p => <option key={p}>{p}</option>)}
              </select>
            </div>
            <div className="col-md-2 d-flex align-items-center">
              <span className="text-muted small">{filtered.length} / {allDashboards.length} shown</span>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-body p-0">
              <div style={{ maxHeight: 600, overflowY: 'auto' }}>
                <table className="table table-sm table-hover mb-0 small">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th style={{ width: 30 }}>#</th>
                      <th>Dashboard</th>
                      <th>Role</th>
                      <th>Visualization</th>
                      <th>Priority</th>
                      <th>Status</th>
                      <th>Endpoints</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((d, i) => (
                      <tr key={i}>
                        <td className="text-muted">{i + 1}</td>
                        <td className="fw-semibold">{d.name}</td>
                        <td>{d.role || <span className="text-muted">—</span>}</td>
                        <td className="text-muted" style={{ maxWidth: 150 }}>{d.viz || '—'}</td>
                        <td><PriBadge p={d.priority || 'N/A'} /></td>
                        <td>
                          <StatusDot s={d.status} />
                          <span className="text-capitalize">{d.status}</span>
                        </td>
                        <td className="text-center">
                          <span className="badge bg-secondary">{d.endpoints || 0}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── P0 ESSENTIALS ── */}
      {tab === 'p0-essentials' && (
        <>
          <p className="text-muted small mb-3">
            P0 dashboards are highest clinical priority — required for safe EEG interpretation and clinical decision support.
          </p>
          {(bd?.must_have_p0 || []).map((d, i) => (
            <div key={i} className="card shadow-sm mb-3 border-danger">
              <div className="card-header py-2 bg-danger bg-opacity-10 d-flex justify-content-between">
                <span className="fw-bold">{d.name}</span>
                <span className="badge" style={{ background: STATUS_COLOR[d.status] || '#6b7280' }}>
                  {d.status}
                </span>
              </div>
              {d.note && (
                <div className="card-body py-2 small text-muted">{d.note}</div>
              )}
            </div>
          ))}

          {/* P0 from directory */}
          <div className="card shadow-sm mt-2">
            <div className="card-header py-2 fw-semibold small">All P0 Dashboards</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0 small">
                <thead className="table-light">
                  <tr><th>Dashboard</th><th>Role</th><th>Visualization</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {allDashboards.filter(d => d.priority === 'P0').map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{d.name}</td>
                      <td>{d.role || '—'}</td>
                      <td className="text-muted">{d.viz || '—'}</td>
                      <td>
                        <StatusDot s={d.status} />
                        <span className="text-capitalize">{d.status}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Status Legend</div>
            <div className="card-body">
              {(defs.status_legend || []).map((s, i) => (
                <div key={i} className="d-flex align-items-start mb-2">
                  <StatusDot s={s.status} />
                  <div>
                    <span className="fw-semibold text-capitalize">{s.status}</span>
                    <span className="text-muted ms-2 small">— {s.meaning}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header py-2 fw-semibold small">Glossary</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <tbody>
                  {(defs.glossary || []).map((g, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ width: 130 }}>{g.term}</td>
                      <td className="text-muted">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {(defs.references || []).length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header py-2 fw-semibold small">References</div>
              <ul className="list-group list-group-flush small">
                {defs.references.map((r, i) => (
                  <li key={i} className="list-group-item">{r}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  );
}
