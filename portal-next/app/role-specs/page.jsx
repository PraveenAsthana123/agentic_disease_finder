'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const PRIORITY_COLOR = { '10/10': 'danger', '9/10': 'warning', '8/10': 'info' };
const STATUS_COLOR   = { built: 'success', partial: 'warning', planned: 'secondary' };

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-3 col-lg mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function HBar({ label, value, max, color }) {
  const pct = max ? Math.round((value / max) * 100) : 0;
  return (
    <div className="d-flex align-items-center mb-2">
      <span className="text-muted small me-2 text-truncate" style={{ minWidth: 180, maxWidth: 220, fontSize: '0.72rem' }}>{label}</span>
      <div className="progress flex-grow-1" style={{ height: 16 }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }}>
          <span style={{ fontSize: '0.68rem' }}>{value}</span>
        </div>
      </div>
    </div>
  );
}

export default function RoleSpecsDashboard() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [sel,  setSel]  = useState(null);
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/role-specs/overview`).then(r => r.json()),
      fetch(`${API}/api/role-specs/breakdown`).then(r => r.json()),
      fetch(`${API}/api/role-specs/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-4"><div className="spinner-border spinner-border-sm me-2" />Loading Role Specifications…</div>;

  const sum = ov.summary || {};
  const secPerRole  = ov.sections_per_role || [];
  const fieldCounts = ov.field_counts      || [];
  const rolesTable  = ov.roles_table       || [];
  const priDist     = ov.priority_distribution || [];
  const perRole     = (bd || {}).per_role  || [];

  const maxSec  = Math.max(...secPerRole.map(r => r.value), 1);
  const maxFld  = Math.max(...fieldCounts.map(r => r.value), 1);

  const TABS = [
    { id: 'overview',  label: 'Overview' },
    { id: 'roles',     label: `Role Details (${sum.total_roles || 0})` },
    { id: 'fields',    label: 'Field Analysis' },
    { id: 'defs',      label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4cb; Role Specification Registry</h3>
      <p className="text-muted small">
        17-role enterprise specification — assessment field counts, workflow sections, build status, and priority
        for every clinical, governance, and research role in the epilepsy AI platform.
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Total Roles"      value={sum.total_roles}           color="primary" />
        <KPI label="Fields (est.)"    value={sum.total_fields_estimate}  color="info"    sub="across all roles" />
        <KPI label="Total Sections"   value={sum.total_sections}         color="warning" />
        <KPI label="Fully Built"      value={sum.built}                  color="success" sub={`of ${sum.total_roles}`} />
        <KPI label="Partial"          value={sum.partial ?? 0}           color="warning" />
        <KPI label="Planned"          value={sum.planned ?? 0}           color="secondary" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'overview' && (
        <div className="row">

          {/* Priority Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0 h-100">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Priority Distribution</div>
              <div className="card-body p-3">
                {priDist.map(p => {
                  const maxP = Math.max(...priDist.map(x => x.value), 1);
                  return (
                    <div key={p.name} className="mb-3">
                      <div className="d-flex justify-content-between mb-1">
                        <span className={`badge bg-${PRIORITY_COLOR[p.name] || 'secondary'}`}>{p.name}</span>
                        <small className="text-muted">{p.value} role{p.value !== 1 ? 's' : ''}</small>
                      </div>
                      <div className="progress" style={{ height: 14 }}>
                        <div className={`progress-bar bg-${PRIORITY_COLOR[p.name] || 'secondary'}`}
                             style={{ width: `${(p.value / maxP) * 100}%` }}>
                          <span style={{ fontSize: '0.65rem' }}>{p.value}</span>
                        </div>
                      </div>
                    </div>
                  );
                })}

                {/* Build status pill row */}
                <hr className="my-3" />
                <div className="d-flex gap-2 flex-wrap">
                  {(ov.status_distribution || []).map(s => (
                    <div key={s.name} className="text-center">
                      <div className={`badge bg-${STATUS_COLOR[s.name] || 'secondary'} px-3 py-2`}>
                        {s.value} {s.name}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Sections per Role */}
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-primary text-white py-2 small fw-bold">Sections per Role (Assessment Coverage)</div>
              <div className="card-body p-3">
                {secPerRole.map(r => (
                  <HBar key={r.name} label={r.name} value={r.value} max={maxSec} color="primary" />
                ))}
                <div className="text-muted small mt-2">Total sections across all roles: {sum.total_sections}</div>
              </div>
            </div>
          </div>

          {/* Roles summary table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">All Roles — Quick Reference</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Role</th>
                        <th className="text-center">Priority</th>
                        <th className="text-center">Fields</th>
                        <th className="text-center">Sections</th>
                        <th className="text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rolesTable.map(r => (
                        <tr key={r.role}>
                          <td className="small fw-bold">{r.role}</td>
                          <td className="text-center">
                            <span className={`badge bg-${PRIORITY_COLOR[r.priority] || 'secondary'}`}>{r.priority}</span>
                          </td>
                          <td className="text-center small text-muted">{r.fields}</td>
                          <td className="text-center small">{r.sections}</td>
                          <td className="text-center">
                            <span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>{r.status}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Role Details ── */}
      {tab === 'roles' && (
        <div>
          {perRole.map(r => (
            <div key={r.role} className="card shadow-sm border-0 mb-2">
              <div
                className="card-header d-flex justify-content-between align-items-center py-2 px-3"
                style={{ cursor: 'pointer', background: sel === r.role ? '#e8f4fd' : '#f8f9fa' }}
                onClick={() => setSel(sel === r.role ? null : r.role)}
              >
                <div className="d-flex align-items-center gap-2">
                  <span className="fw-bold small">{r.role}</span>
                  <span className={`badge bg-${PRIORITY_COLOR[r.priority] || 'secondary'}`}>{r.priority}</span>
                  <span className={`badge bg-${STATUS_COLOR[r.status] || 'secondary'}`}>{r.status}</span>
                </div>
                <div className="d-flex align-items-center gap-3">
                  <span className="small text-muted">{r.fields} fields</span>
                  <span className="small text-muted">{r.section_count} sections</span>
                  <span className="text-primary small">{sel === r.role ? '▲' : '▼'}</span>
                </div>
              </div>

              {sel === r.role && (
                <div className="card-body p-3">
                  {/* Sections pills */}
                  <div className="mb-2">
                    <strong className="small">Workflow Sections:</strong>
                    <div className="d-flex flex-wrap gap-1 mt-1">
                      {(r.sections || []).map(s => (
                        <span key={s} className="badge bg-primary text-white" style={{ fontSize: '0.7rem' }}>{s}</span>
                      ))}
                    </div>
                  </div>

                  {/* Endpoints */}
                  {r.endpoints && r.endpoints.length > 0 && (
                    <div className="mb-2">
                      <strong className="small">API Endpoints:</strong>
                      <div className="d-flex flex-wrap gap-1 mt-1">
                        {r.endpoints.map(e => (
                          <code key={e} className="small text-success">{e}</code>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Note */}
                  {r.note && (
                    <div className="alert alert-light border py-1 px-2 mb-0">
                      <small className="text-muted">{r.note}</small>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Field Analysis ── */}
      {tab === 'fields' && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">
                Assessment Field Counts per Role (estimated midpoints)
              </div>
              <div className="card-body p-3">
                {fieldCounts.map(r => (
                  <div key={r.name} className="d-flex align-items-center mb-2">
                    <span className="text-muted small me-2 text-truncate" style={{ minWidth: 220, maxWidth: 260, fontSize: '0.72rem' }}>{r.name}</span>
                    <div className="progress flex-grow-1" style={{ height: 18 }}>
                      <div className="progress-bar bg-success" style={{ width: `${(r.value / maxFld) * 100}%` }}>
                        <span style={{ fontSize: '0.68rem' }}>{r.range}</span>
                      </div>
                    </div>
                  </div>
                ))}
                <div className="text-muted small mt-3">
                  Total estimated fields across all 17 roles: <strong>{sum.total_fields_estimate}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Field count stats */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-info text-white py-2 small fw-bold">Field Count Statistics</div>
              <div className="card-body p-3">
                {(() => {
                  const vals = fieldCounts.map(r => r.value);
                  const avg  = vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : 0;
                  const min  = Math.min(...vals);
                  const max  = Math.max(...vals);
                  return (
                    <>
                      <div className="d-flex justify-content-between mb-2"><span className="small text-muted">Highest</span><strong className="text-success">{max}</strong></div>
                      <div className="d-flex justify-content-between mb-2"><span className="small text-muted">Average</span><strong className="text-primary">{avg}</strong></div>
                      <div className="d-flex justify-content-between mb-2"><span className="small text-muted">Lowest</span><strong className="text-warning">{min}</strong></div>
                      <div className="d-flex justify-content-between"><span className="small text-muted">Roles</span><strong>{fieldCounts.length}</strong></div>
                    </>
                  );
                })()}
              </div>
            </div>
          </div>

          <div className="col-md-8 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-warning text-dark py-2 small fw-bold">Priority vs Field Count Matrix</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Role</th>
                      <th className="text-center">Priority</th>
                      <th className="text-center">Fields</th>
                      <th className="text-center">Sections</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rolesTable
                      .slice()
                      .sort((a, b) => {
                        const fa = fieldCounts.find(x => x.name === a.role);
                        const fb = fieldCounts.find(x => x.name === b.role);
                        return (fb?.value || 0) - (fa?.value || 0);
                      })
                      .map(r => (
                        <tr key={r.role}>
                          <td className="small">{r.role}</td>
                          <td className="text-center"><span className={`badge bg-${PRIORITY_COLOR[r.priority] || 'secondary'}`}>{r.priority}</span></td>
                          <td className="text-center small fw-bold">{r.fields}</td>
                          <td className="text-center small">{r.sections}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'defs' && defs && (
        <div className="row">
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-success text-white py-2 small fw-bold">Build Status Legend</div>
              <div className="card-body p-3">
                {(defs.status_legend || []).map(s => (
                  <div key={s.status} className="mb-2 d-flex gap-2 align-items-start">
                    <span className={`badge bg-${STATUS_COLOR[s.status] || 'secondary'} mt-1`}>{s.status}</span>
                    <small className="text-muted">{s.description}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-danger text-white py-2 small fw-bold">Priority Legend</div>
              <div className="card-body p-3">
                {(defs.priority_legend || []).map(p => (
                  <div key={p.priority} className="mb-2 d-flex gap-2 align-items-start">
                    <span className={`badge bg-${PRIORITY_COLOR[p.priority] || 'secondary'} mt-1`}>{p.priority}</span>
                    <small className="text-muted">{p.description}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-4 mb-3">
            <div className="card shadow-sm border-0">
              <div className="card-header bg-dark text-white py-2 small fw-bold">Platform Glossary</div>
              <div className="card-body p-2">
                <table className="table table-sm mb-0">
                  <tbody>
                    {(defs.glossary || []).map(g => (
                      <tr key={g.term}>
                        <td className="small fw-bold align-top" style={{ width: '30%' }}>{g.term}</td>
                        <td className="small text-muted">{g.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
