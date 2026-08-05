'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const catColor = c => ({
  'AI Model': 'primary', 'Data Pipeline': 'info', 'Integration': 'warning',
  'Reporting': 'success', 'Security': 'danger', 'UI': 'secondary',
}[c] || 'dark');

const stalenessColor = days => {
  if (days == null) return 'secondary';
  if (days <= 30) return 'success';
  if (days <= 90) return 'warning';
  return 'danger';
};
const stalenessLabel = days => {
  if (days == null) return '—';
  if (days <= 30) return 'Fresh';
  if (days <= 90) return 'Aging';
  return 'Stale';
};

const KPI = ({ label, value, sub, color = 'primary' }) => (
  <div className="col-6 col-md-3 mb-3">
    <div className={`card border-${color} h-100`}>
      <div className="card-body p-3 text-center">
        <div className={`fs-3 fw-bold text-${color}`}>{value}</div>
        <div className="small text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
      </div>
    </div>
  </div>
);

const Bar = ({ label, value, max, color = 'primary' }) => (
  <div className="mb-2">
    <div className="d-flex justify-content-between small mb-1">
      <span>{label}</span>
      <span className="fw-semibold">{value}</span>
    </div>
    <div className="progress" style={{ height: 10 }}>
      <div className={`progress-bar bg-${color}`} style={{ width: `${Math.round((value / Math.max(max, 1)) * 100)}%` }} />
    </div>
  </div>
);

export default function FeatureFlagsDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [filterCat, setFilterCat] = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [sortBy, setSortBy] = useState('days_since_update');
  const [sortDir, setSortDir] = useState('desc');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/feature-flags/overview`).then(r => r.json()),
      fetch(`${API}/api/feature-flags/breakdown`).then(r => r.json()),
      fetch(`${API}/api/feature-flags/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading Feature Flags data…</div>;

  const k = ov.kpis || {};
  const TABS = ['overview', 'flags', 'categories', 'cleanup', 'definitions'];
  const tabLabel = {
    overview: '🚩 Overview',
    flags: '📋 All Flags',
    categories: '🏷️ By Category',
    cleanup: '🧹 Cleanup',
    definitions: '📖 Definitions',
  };

  // Filter + sort all_flags
  const allFlags = (bd?.all_flags || [])
    .filter(f => (!filterCat || f.category === filterCat) && (!filterStatus || (filterStatus === 'enabled' ? f.enabled : !f.enabled)))
    .filter(f => !search || [f.flag_id, f.name, f.description, f.owner, f.category].join(' ').toLowerCase().includes(search.toLowerCase()))
    .sort((a, b) => {
      const av = a[sortBy] ?? 0;
      const bv = b[sortBy] ?? 0;
      if (typeof av === 'number') return sortDir === 'asc' ? av - bv : bv - av;
      return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });

  const toggleSort = col => {
    if (sortBy === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortBy(col); setSortDir('desc'); }
  };
  const sa = col => sortBy === col ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '';

  const catMax = Math.max(...(ov.category_distribution || []).map(c => c.total));
  const ownerMax = Math.max(...(ov.owner_workload || []).map(o => o.total_flags));
  const categories = [...new Set((bd?.all_flags || []).map(f => f.category))].sort();

  return (
    <div>
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <h3 className="mb-0">🚩 Feature Flags Dashboard</h3>
        <span className="badge bg-dark fs-6">{k.total_flags} flags</span>
        {k.stale_flags > 0 && (
          <span className="badge bg-danger">{k.stale_flags} stale</span>
        )}
        <span className="text-muted small ms-auto">Real data · feature_flags {k.total_flags} rows · clinical.db</span>
      </div>

      {/* KPI Row */}
      <div className="row mb-3">
        <KPI label="Total Flags" value={k.total_flags} color="primary" />
        <KPI label="Enabled" value={k.enabled} sub={`${k.disabled} disabled`} color="success" />
        <KPI label="Full Rollout" value={k.full_rollout} sub={`Avg ${k.avg_rollout_pct}% rollout`} color="info" />
        <KPI label="Stale Flags" value={k.stale_flags} sub="90+ days no update" color="danger" />
      </div>
      <div className="row mb-3">
        <KPI label="Owners" value={k.owners} color="secondary" />
        <KPI label="Categories" value={k.categories} color="warning" />
        <KPI label="Rollout Candidates" value={bd?.rollout_candidates?.length ?? '—'} sub="enabled, not at 100%" color="primary" />
        <KPI label="Cleanup Candidates" value={(bd?.disabled_flags?.length ?? 0)} sub="disabled flags" color="danger" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {tabLabel[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Category Distribution</div>
              <div className="card-body">
                {(ov.category_distribution || []).map(c => (
                  <div key={c.category} className="mb-3">
                    <div className="d-flex justify-content-between small mb-1">
                      <span><span className={`badge bg-${catColor(c.category)} me-1`}>{c.category}</span></span>
                      <span className="fw-semibold">{c.total} total ({c.enabled} on / {c.disabled} off)</span>
                    </div>
                    <div className="progress" style={{ height: 10 }}>
                      <div className="progress-bar bg-success" style={{ width: `${Math.round((c.enabled / Math.max(c.total, 1)) * 100)}%` }} />
                      <div className="progress-bar bg-secondary" style={{ width: `${Math.round((c.disabled / Math.max(c.total, 1)) * 100)}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Rollout Tiers</div>
              <div className="card-body">
                {(ov.rollout_tiers || []).map(rt => (
                  <Bar key={rt.tier} label={rt.tier} value={rt.count}
                    max={Math.max(...(ov.rollout_tiers || []).map(r => r.count))}
                    color={rt.tier === 'Full (100%)' ? 'success' : rt.tier.startsWith('Off') ? 'secondary' : 'info'} />
                ))}
              </div>
            </div>
          </div>
          <div className="col-12 mb-4">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Owner Workload</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr><th>Owner</th><th>Total Flags</th><th>Active</th><th>Inactive</th><th>Active Rate</th></tr>
                    </thead>
                    <tbody>
                      {(ov.owner_workload || []).sort((a, b) => b.total_flags - a.total_flags).map((o, i) => {
                        const rate = o.total_flags > 0 ? Math.round((o.active / o.total_flags) * 100) : 0;
                        return (
                          <tr key={i}>
                            <td><code>{o.owner}</code></td>
                            <td>{o.total_flags}</td>
                            <td><span className="badge bg-success">{o.active}</span></td>
                            <td><span className="badge bg-secondary">{o.total_flags - o.active}</span></td>
                            <td>
                              <div className="d-flex align-items-center gap-2">
                                <div className="progress flex-grow-1" style={{ height: 8 }}>
                                  <div className={`progress-bar bg-${rate >= 70 ? 'success' : rate >= 40 ? 'info' : 'warning'}`} style={{ width: `${rate}%` }} />
                                </div>
                                <span className="small">{rate}%</span>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* All Flags Tab */}
      {tab === 'flags' && (
        <div>
          <div className="d-flex gap-2 mb-3 flex-wrap align-items-center">
            <input className="form-control form-control-sm" style={{ maxWidth: 260 }}
              placeholder="Search name, owner, description…"
              value={search} onChange={e => setSearch(e.target.value)} />
            <select className="form-select form-select-sm" style={{ maxWidth: 160 }}
              value={filterCat} onChange={e => setFilterCat(e.target.value)}>
              <option value="">All Categories</option>
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            <select className="form-select form-select-sm" style={{ maxWidth: 140 }}
              value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
              <option value="">All Statuses</option>
              <option value="enabled">Enabled</option>
              <option value="disabled">Disabled</option>
            </select>
            <span className="text-muted small">{allFlags.length} of {bd?.all_flags?.length ?? 0} shown</span>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover table-bordered">
              <thead className="table-dark">
                <tr>
                  {[['flag_id', 'ID'], ['name', 'Name'], ['category', 'Category'],
                    ['enabled', 'Status'], ['rollout_percentage', 'Rollout %'],
                    ['owner', 'Owner'], ['days_since_update', 'Staleness']].map(([col, lbl]) => (
                    <th key={col} style={{ cursor: 'pointer', whiteSpace: 'nowrap' }} onClick={() => toggleSort(col)}>
                      {lbl}{sa(col)}
                    </th>
                  ))}
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {allFlags.map((f, i) => (
                  <tr key={i}>
                    <td><code className="small">{f.flag_id}</code></td>
                    <td><span className="fw-semibold">{f.name}</span></td>
                    <td><span className={`badge bg-${catColor(f.category)}`}>{f.category}</span></td>
                    <td>
                      <span className={`badge bg-${f.enabled ? 'success' : 'secondary'}`}>
                        {f.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </td>
                    <td>
                      <div className="d-flex align-items-center gap-1">
                        <div className="progress" style={{ height: 8, width: 50 }}>
                          <div className={`progress-bar bg-${f.rollout_percentage === 100 ? 'success' : f.rollout_percentage > 0 ? 'info' : 'secondary'}`}
                            style={{ width: `${f.rollout_percentage}%` }} />
                        </div>
                        <span className="small">{f.rollout_percentage}%</span>
                      </div>
                    </td>
                    <td><code className="small">{f.owner}</code></td>
                    <td>
                      <span className={`badge bg-${stalenessColor(f.days_since_update)}`}>
                        {stalenessLabel(f.days_since_update)}
                        {f.days_since_update != null && ` · ${f.days_since_update}d`}
                      </span>
                    </td>
                    <td className="small text-muted" style={{ maxWidth: 200 }}>{f.description}</td>
                  </tr>
                ))}
                {allFlags.length === 0 && (
                  <tr><td colSpan={8} className="text-center text-muted py-3">No flags match your filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Categories Tab */}
      {tab === 'categories' && (
        <div className="row">
          {Object.entries(bd?.by_category || {}).sort((a, b) => b[1].length - a[1].length).map(([cat, flags]) => (
            <div key={cat} className="col-md-6 mb-4">
              <div className="card shadow-sm h-100">
                <div className={`card-header fw-semibold d-flex justify-content-between align-items-center`}>
                  <span><span className={`badge bg-${catColor(cat)} me-2`}>{cat}</span></span>
                  <span className="badge bg-dark">{flags.length} flags</span>
                </div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Flag</th><th>Status</th><th>Rollout</th><th>Owner</th></tr>
                    </thead>
                    <tbody>
                      {flags.map((f, i) => (
                        <tr key={i}>
                          <td className="small"><code>{f.name}</code></td>
                          <td>
                            <span className={`badge bg-${f.enabled ? 'success' : 'secondary'} small`}>
                              {f.enabled ? 'On' : 'Off'}
                            </span>
                          </td>
                          <td className="small">{f.rollout_percentage}%</td>
                          <td className="small text-muted">{f.owner}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Cleanup Tab */}
      {tab === 'cleanup' && (
        <div className="row">
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm border-danger h-100">
              <div className="card-header fw-semibold bg-danger text-white d-flex justify-content-between">
                <span>🧹 Disabled Flags (Cleanup Candidates)</span>
                <span className="badge bg-white text-danger">{bd?.disabled_flags?.length ?? 0}</span>
              </div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Flag</th><th>Category</th><th>Owner</th><th>Last Updated</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.disabled_flags || []).map((f, i) => (
                      <tr key={i}>
                        <td><code className="small">{f.flag_id}</code><div className="small text-muted">{f.name}</div></td>
                        <td><span className={`badge bg-${catColor(f.category)} small`}>{f.category}</span></td>
                        <td className="small">{f.owner}</td>
                        <td>
                          <span className={`badge bg-${stalenessColor(f.days_since_update)} small`}>
                            {f.days_since_update != null ? `${f.days_since_update}d ago` : '—'}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {(bd?.disabled_flags?.length ?? 0) === 0 && (
                      <tr><td colSpan={4} className="text-center text-muted py-2">No disabled flags</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm border-info h-100">
              <div className="card-header fw-semibold bg-info text-white d-flex justify-content-between">
                <span>📈 Rollout Candidates (enabled, &lt;100%)</span>
                <span className="badge bg-white text-info">{bd?.rollout_candidates?.length ?? 0}</span>
              </div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Flag</th><th>Category</th><th>Rollout</th><th>Owner</th></tr>
                  </thead>
                  <tbody>
                    {(bd?.rollout_candidates || []).sort((a, b) => b.rollout_percentage - a.rollout_percentage).map((f, i) => (
                      <tr key={i}>
                        <td><code className="small">{f.flag_id}</code><div className="small text-muted">{f.name}</div></td>
                        <td><span className={`badge bg-${catColor(f.category)} small`}>{f.category}</span></td>
                        <td>
                          <div className="d-flex align-items-center gap-1">
                            <div className="progress" style={{ height: 8, width: 40 }}>
                              <div className="progress-bar bg-info" style={{ width: `${f.rollout_percentage}%` }} />
                            </div>
                            <span className="small">{f.rollout_percentage}%</span>
                          </div>
                        </td>
                        <td className="small">{f.owner}</td>
                      </tr>
                    ))}
                    {(bd?.rollout_candidates?.length ?? 0) === 0 && (
                      <tr><td colSpan={4} className="text-center text-muted py-2">No rollout candidates</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-4">
            <div className="card shadow-sm border-warning">
              <div className="card-header fw-semibold bg-warning d-flex justify-content-between">
                <span>⏰ Stale Flags (90+ days no update)</span>
                <span className="badge bg-dark">{bd?.stale_flags?.length ?? 0}</span>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Flag ID</th><th>Name</th><th>Category</th><th>Status</th><th>Rollout</th><th>Owner</th><th>Days Stale</th></tr>
                    </thead>
                    <tbody>
                      {(bd?.stale_flags || []).sort((a, b) => (b.days_since_update ?? 0) - (a.days_since_update ?? 0)).map((f, i) => (
                        <tr key={i}>
                          <td><code className="small">{f.flag_id}</code></td>
                          <td className="small fw-semibold">{f.name}</td>
                          <td><span className={`badge bg-${catColor(f.category)} small`}>{f.category}</span></td>
                          <td><span className={`badge bg-${f.enabled ? 'success' : 'secondary'} small`}>{f.enabled ? 'Enabled' : 'Disabled'}</span></td>
                          <td className="small">{f.rollout_percentage}%</td>
                          <td className="small">{f.owner}</td>
                          <td><span className="badge bg-danger small">{f.days_since_update ?? '—'}d</span></td>
                        </tr>
                      ))}
                      {(bd?.stale_flags?.length ?? 0) === 0 && (
                        <tr><td colSpan={7} className="text-center text-muted py-2">No stale flags</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Flag Statuses</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.statuses || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td style={{ width: '25%' }}><span className={`badge bg-${k === 'enabled' ? 'success' : 'secondary'}`}>{k}</span></td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Staleness Thresholds</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.staleness_thresholds || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td style={{ width: '25%' }}><span className={`badge bg-${k === 'fresh' ? 'success' : k === 'aging' ? 'warning' : 'danger'}`}>{k}</span></td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Rollout Tiers</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.rollout_tiers || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td style={{ width: '35%' }} className="fw-semibold small">{k}</td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold">Categories</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <tbody>
                    {Object.entries(defs.categories || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td style={{ width: '35%' }}><span className={`badge bg-${catColor(k)}`}>{k}</span></td>
                        <td className="small text-muted">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          {defs.best_practices && (
            <div className="col-12 mb-4">
              <div className="card shadow-sm border-info">
                <div className="card-header fw-semibold bg-info text-white">Best Practices</div>
                <ul className="list-group list-group-flush">
                  {defs.best_practices.map((p, i) => (
                    <li key={i} className="list-group-item small">{p}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
