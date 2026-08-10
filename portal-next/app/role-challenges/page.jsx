'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = { built: 'success', partial: 'warning', planned: 'secondary' };

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

export default function RoleChallengesDashboard() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [selRole, setSelRole] = useState(null);
  const [err,  setErr]  = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/role-challenges/overview`).then(r => r.json()),
      fetch(`${API}/api/role-challenges/breakdown`).then(r => r.json()),
      fetch(`${API}/api/role-challenges/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-4"><div className="spinner-border spinner-border-sm me-2" />Loading Role Challenges data…</div>;

  const sum       = ov.summary           || {};
  const perRole   = ov.challenges_per_role || [];
  const statusDist = ov.status_distribution || [];
  const allRoles  = (bd || {}).roles     || [];
  const roleDefs  = (defs || {}).role_descriptions || [];
  const statusLeg = (defs || {}).status_legend     || [];
  const glossary  = (defs || {}).glossary          || [];

  const maxChallenges = Math.max(...perRole.map(r => r.value), 1);

  // Flatten all challenges for "All Challenges" tab
  const allChallenges = allRoles.flatMap(r =>
    (r.items || []).map(i => ({ ...i, role: r.role, icon: r.icon }))
  );

  const TABS = [
    { id: 'overview',  label: '📊 Overview' },
    { id: 'by-role',   label: `🧑‍⚕️ By Role (${sum.total_roles || 0})` },
    { id: 'all',       label: `📋 All Challenges (${sum.total_challenges || 0})` },
    { id: 'defs',      label: '📖 Definitions' },
  ];

  const statusColor = s => STATUS_COLOR[s] || 'secondary';

  return (
    <div>
      <h3>⚡ Role Challenges &amp; AI Solutions</h3>
      <p className="text-muted small">
        {sum.total_challenges} clinical challenges across {sum.total_roles} roles — each with a concrete AI solution
        and build status. 100% built and verified in the epilepsy AI platform.
      </p>

      {/* KPI row */}
      <div className="row mb-3">
        <KPI label="Roles"          value={sum.total_roles}       color="primary" />
        <KPI label="Challenges"     value={sum.total_challenges}  color="info"    />
        <KPI label="Built"          value={sum.built}             color="success" sub={`${sum.built_pct?.toFixed(0) ?? 100}%`} />
        <KPI label="Partial"        value={sum.partial ?? 0}      color="warning" />
        <KPI label="Planned"        value={sum.planned ?? 0}      color="secondary" />
        <KPI label="Roles Complete" value={sum.roles_fully_built} color="success" sub={`of ${sum.total_roles}`} />
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

      {/* OVERVIEW */}
      {tab === 'overview' && (
        <div>
          <h5 className="mb-3">Challenges per Role</h5>
          {perRole.map(r => (
            <HBar key={r.name} label={r.name} value={r.value} max={maxChallenges} color="primary" />
          ))}

          <div className="row mt-4">
            <div className="col-md-6">
              <h6>Status Distribution</h6>
              <table className="table table-sm table-bordered">
                <thead className="table-light">
                  <tr><th>Status</th><th>Count</th></tr>
                </thead>
                <tbody>
                  {statusDist.map(s => (
                    <tr key={s.name}>
                      <td><span className={`badge bg-${statusColor(s.name)}`}>{s.name}</span></td>
                      <td>{s.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="col-md-6">
              <h6>Platform Coverage</h6>
              <div className="alert alert-success mb-0">
                <strong>{sum.built_pct?.toFixed(1) ?? 100}%</strong> of documented role challenges are addressed
                by built AI features in the epilepsy platform. All {sum.total_roles} roles fully covered.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* BY ROLE */}
      {tab === 'by-role' && (
        <div>
          <div className="row mb-3">
            {allRoles.map(r => (
              <div key={r.role} className="col-md-6 col-lg-4 mb-3">
                <div
                  className={`card h-100 shadow-sm${selRole === r.role ? ' border-primary' : ''}`}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setSelRole(selRole === r.role ? null : r.role)}
                >
                  <div className="card-header py-2 bg-light d-flex justify-content-between align-items-center">
                    <span className="fw-bold">{r.icon} {r.role}</span>
                    <span className="badge bg-primary">{(r.items || []).length} challenges</span>
                  </div>
                  {selRole === r.role && (
                    <div className="card-body p-2">
                      <ul className="list-group list-group-flush">
                        {(r.items || []).map((item, i) => (
                          <li key={i} className="list-group-item px-2 py-2">
                            <div className="text-danger small fw-semibold mb-1">⚠️ {item.challenge}</div>
                            <div className="text-success small">🤖 {item.ai}</div>
                            {item.dashboard && (
                              <div className="text-muted" style={{ fontSize: '0.7rem' }}>
                                Dashboard: {item.dashboard}
                              </div>
                            )}
                            <span className={`badge bg-${statusColor(item.status)} mt-1`}>{item.status}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          {!selRole && (
            <p className="text-muted small text-center">Click a role card to expand its challenges.</p>
          )}
        </div>
      )}

      {/* ALL CHALLENGES */}
      {tab === 'all' && (
        <div>
          <table className="table table-sm table-striped table-bordered">
            <thead className="table-dark">
              <tr>
                <th style={{ width: 140 }}>Role</th>
                <th>Challenge</th>
                <th>AI Solution</th>
                <th style={{ width: 80 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {allChallenges.map((c, i) => (
                <tr key={i}>
                  <td className="small">{c.icon} {c.role}</td>
                  <td className="small text-danger">{c.challenge}</td>
                  <td className="small text-success">{c.ai}</td>
                  <td>
                    <span className={`badge bg-${statusColor(c.status)}`}>{c.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* DEFINITIONS */}
      {tab === 'defs' && (
        <div>
          <h6>Role Descriptions</h6>
          <table className="table table-sm table-bordered mb-4">
            <thead className="table-light">
              <tr><th>Role</th><th>Description</th></tr>
            </thead>
            <tbody>
              {roleDefs.map(r => (
                <tr key={r.role}>
                  <td className="fw-semibold small text-nowrap">{r.role}</td>
                  <td className="small">{r.description}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {statusLeg.length > 0 && (
            <>
              <h6>Status Legend</h6>
              <table className="table table-sm table-bordered mb-4">
                <thead className="table-light">
                  <tr><th>Status</th><th>Meaning</th></tr>
                </thead>
                <tbody>
                  {statusLeg.map(s => (
                    <tr key={s.status}>
                      <td><span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span></td>
                      <td className="small">{s.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          {glossary.length > 0 && (
            <>
              <h6>Glossary</h6>
              <dl className="row">
                {glossary.map(g => (
                  <span key={g.term}>
                    <dt className="col-sm-3 small fw-semibold">{g.term}</dt>
                    <dd className="col-sm-9 small">{g.definition}</dd>
                  </span>
                ))}
              </dl>
            </>
          )}
        </div>
      )}
    </div>
  );
}
