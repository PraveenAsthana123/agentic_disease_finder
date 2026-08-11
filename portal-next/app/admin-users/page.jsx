'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',   label: 'Overview'   },
  { id: 'users',      label: 'User Directory' },
  { id: 'compliance', label: 'MFA & Compliance' },
  { id: 'definitions',label: 'Definitions' },
];

const ROLE_COLOR = {
  Researcher: 'primary',
  Neurologist: 'success',
  Admin: 'danger',
  'Data Scientist': 'info',
  'EEG Tech': 'warning',
  Nurse: 'secondary',
};

const STATUS_COLOR = { active: 'success', inactive: 'secondary' };

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

function RoleBadge({ role }) {
  return (
    <span className={`badge bg-${ROLE_COLOR[role] || 'dark'}`}>{role}</span>
  );
}

function StatusBadge({ status }) {
  return (
    <span className={`badge bg-${STATUS_COLOR[status] || 'secondary'}`}>
      {status === 'active' ? '● active' : '○ inactive'}
    </span>
  );
}

function HBar({ items }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i.count));
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="d-flex align-items-center mb-2">
          <div className="text-end me-2 small" style={{ width: 120, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {it.role || it.department}
          </div>
          <div className="flex-grow-1 bg-light rounded" style={{ height: 18 }}>
            <div
              className={`h-100 rounded bg-${ROLE_COLOR[it.role] || 'primary'}`}
              style={{ width: `${mx > 0 ? (it.count / mx) * 100 : 0}%` }}
            />
          </div>
          <div className="ms-2 small fw-semibold" style={{ width: 32 }}>{it.count}</div>
          {it.pct !== undefined && (
            <div className="text-muted small" style={{ width: 44 }}>{it.pct}%</div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function AdminUsersPage() {
  const [overview,  setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,      setDefs]      = useState(null);
  const [tab,       setTab]       = useState('overview');
  const [sortKey,   setSortKey]   = useState('login_count');
  const [sortDir,   setSortDir]   = useState(-1);
  const [filterSt,  setFilterSt]  = useState('all');
  const [filterRole,setFilterRole]= useState('all');
  const [err,       setErr]       = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/admin-users/overview`).then(r => r.json()),
      fetch(`${API}/api/admin-users/breakdown`).then(r => r.json()),
      fetch(`${API}/api/admin-users/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefs(df);
    }).catch(e => setErr(String(e)));
  }, []);

  if (err)      return <div className="alert alert-danger m-4">{err}</div>;
  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpi = overview.kpis || {};

  /* sorted + filtered user list */
  const allUsers = breakdown?.users || [];
  const roles = [...new Set(allUsers.map(u => u.role))].sort();
  const filteredUsers = allUsers
    .filter(u =>
      (filterSt   === 'all' || u.status === filterSt) &&
      (filterRole === 'all' || u.role   === filterRole)
    )
    .sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === 'number') return sortDir * (bv - av);
      return sortDir * String(av).localeCompare(String(bv));
    });

  const mfaMissing = breakdown?.mfa_missing || [];
  const roleMatrix = breakdown?.role_status_matrix || [];

  function thBtn(key, label) {
    const active = sortKey === key;
    return (
      <th
        className="small user-select-none"
        style={{ cursor: 'pointer', whiteSpace: 'nowrap' }}
        onClick={() => { if (active) setSortDir(d => -d); else { setSortKey(key); setSortDir(-1); } }}
      >
        {label} {active ? (sortDir === -1 ? '▼' : '▲') : ''}
      </th>
    );
  }

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h2 className="mb-0 fw-bold">👥 Admin Users Panel</h2>
          <p className="text-muted mb-0 small">
            Platform operators · role distribution · MFA compliance · login activity
            &nbsp;·&nbsp;
            <span className="text-secondary">{overview.generated}</span>
          </p>
        </div>
        {kpi.inactive_users > 0 && (
          <span className="badge bg-secondary fs-6 ms-auto">
            {kpi.inactive_users} inactive
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
          <div className="row g-3 mb-4">
            <KPI label="Total Users"      value={kpi.total_users}       color="primary"  />
            <KPI label="Active"           value={kpi.active_users}      color="success"  />
            <KPI label="Inactive"         value={kpi.inactive_users}    color="secondary"/>
            <KPI label="MFA Enabled"      value={kpi.mfa_enabled}       color="info"     sub={`${kpi.mfa_rate_pct}% rate`} />
            <KPI label="Roles"            value={kpi.total_roles}       color="warning"  />
            <KPI label="Departments"      value={kpi.total_departments} color="dark"     />
            <KPI label="Avg Logins"       value={kpi.avg_logins}        color="primary"  sub="per user" />
          </div>

          <div className="row g-3 mb-4">
            {/* Role distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Role Distribution</div>
                <div className="card-body">
                  <HBar items={overview.role_breakdown} />
                </div>
              </div>
            </div>
            {/* Department distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Department Distribution</div>
                <div className="card-body">
                  {(overview.dept_breakdown || []).map((d, i) => (
                    <div key={i} className="d-flex justify-content-between align-items-center mb-2">
                      <span className="small">{d.department}</span>
                      <span className="badge bg-secondary">{d.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Top user + Status summary */}
          <div className="row g-3">
            <div className="col-md-4">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Status Summary</div>
                <div className="card-body">
                  {Object.entries(overview.status_summary || {}).map(([st, ct]) => (
                    <div key={st} className="d-flex justify-content-between align-items-center mb-2">
                      <StatusBadge status={st} />
                      <span className="fw-bold">{ct}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {overview.top_user && (
              <div className="col-md-8">
                <div className="card shadow-sm h-100 border-primary">
                  <div className="card-header fw-semibold text-primary">Most Active User</div>
                  <div className="card-body">
                    <div className="h5 fw-bold mb-1">{overview.top_user.full_name}</div>
                    <div className="mb-1"><RoleBadge role={overview.top_user.role} /> &nbsp; {overview.top_user.department}</div>
                    <div className="text-muted small">{overview.top_user.login_count} logins · ID: {overview.top_user.user_id}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* ── USER DIRECTORY ── */}
      {tab === 'users' && (
        <>
          <div className="d-flex gap-2 mb-3 flex-wrap">
            <select className="form-select form-select-sm w-auto" value={filterSt} onChange={e => setFilterSt(e.target.value)}>
              <option value="all">All Statuses</option>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
            </select>
            <select className="form-select form-select-sm w-auto" value={filterRole} onChange={e => setFilterRole(e.target.value)}>
              <option value="all">All Roles</option>
              {roles.map(r => <option key={r} value={r}>{r}</option>)}
            </select>
            <span className="text-muted small align-self-center">
              {filteredUsers.length} of {allUsers.length} users
            </span>
          </div>

          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">User Directory</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    {thBtn('user_id',     'ID')}
                    {thBtn('full_name',   'Name')}
                    {thBtn('role',        'Role')}
                    {thBtn('department',  'Department')}
                    {thBtn('status',      'Status')}
                    {thBtn('login_count', 'Logins')}
                    {thBtn('last_login',  'Last Login')}
                    <th className="small">MFA</th>
                    {thBtn('created_at',  'Created')}
                  </tr>
                </thead>
                <tbody>
                  {filteredUsers.map(u => (
                    <tr key={u.user_id}>
                      <td className="small text-muted fw-mono">{u.user_id}</td>
                      <td className="small fw-semibold">{u.full_name}</td>
                      <td><RoleBadge role={u.role} /></td>
                      <td className="small">{u.department}</td>
                      <td><StatusBadge status={u.status} /></td>
                      <td className="small fw-bold text-center">{u.login_count}</td>
                      <td className="small text-muted">{u.last_login}</td>
                      <td className="text-center">{u.mfa_enabled ? '✅' : '⚠️'}</td>
                      <td className="small text-muted">{u.created_at}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Role × Status matrix */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Role × Status Matrix</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th className="small">Role</th>
                    <th className="small text-success">Active</th>
                    <th className="small text-secondary">Inactive</th>
                    <th className="small">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {roleMatrix.map((r, i) => (
                    <tr key={i}>
                      <td><RoleBadge role={r.role} /></td>
                      <td className="small text-success fw-semibold">{r.active || 0}</td>
                      <td className="small text-secondary">{r.inactive || 0}</td>
                      <td className="small fw-bold">{r.total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── MFA & COMPLIANCE ── */}
      {tab === 'compliance' && (
        <>
          <div className="row g-3 mb-4">
            <KPI label="MFA Enabled"    value={kpi.mfa_enabled}    color="success" sub={`${kpi.mfa_rate_pct}% rate`} />
            <KPI label="MFA Missing"    value={mfaMissing.length}  color={mfaMissing.length > 0 ? 'danger' : 'success'} />
            <KPI label="Active Users"   value={kpi.active_users}   color="primary" />
            <KPI label="Total Users"    value={kpi.total_users}    color="dark"    />
          </div>

          <div className="row g-3 mb-4">
            {/* MFA donut-style bar */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">MFA Compliance</div>
                <div className="card-body">
                  <div className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <small className="text-success fw-semibold">MFA Enabled ({kpi.mfa_enabled})</small>
                      <small>{kpi.mfa_rate_pct}%</small>
                    </div>
                    <div className="progress" style={{ height: 20 }}>
                      <div
                        className="progress-bar bg-success"
                        style={{ width: `${kpi.mfa_rate_pct}%` }}
                      >
                        {kpi.mfa_enabled}
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 p-3 bg-light rounded">
                    <div className="small text-muted">
                      <strong>HIPAA §164.312(d)</strong> requires multi-factor authentication
                      for all users with access to electronic Protected Health Information (ePHI).
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Non-compliant list */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold text-danger">
                  MFA Not Enrolled ({mfaMissing.length})
                </div>
                <div className="card-body">
                  {mfaMissing.length === 0 ? (
                    <div className="alert alert-success mb-0">All users have MFA enabled ✅</div>
                  ) : (
                    <div className="table-responsive">
                      <table className="table table-sm mb-0">
                        <thead className="table-light">
                          <tr><th>Name</th><th>Role</th><th>Status</th></tr>
                        </thead>
                        <tbody>
                          {mfaMissing.map((u, i) => (
                            <tr key={i}>
                              <td className="small fw-semibold">{u.full_name}</td>
                              <td><RoleBadge role={u.role} /></td>
                              <td><StatusBadge status={u.status} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Inactive users list */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Inactive Accounts ({kpi.inactive_users})</div>
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>ID</th><th>Name</th><th>Role</th><th>Department</th><th>Last Login</th><th>Logins</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.users || []).filter(u => u.status === 'inactive').map((u, i) => (
                    <tr key={i}>
                      <td className="small text-muted">{u.user_id}</td>
                      <td className="small fw-semibold">{u.full_name}</td>
                      <td><RoleBadge role={u.role} /></td>
                      <td className="small">{u.department}</td>
                      <td className="small text-muted">{u.last_login}</td>
                      <td className="small">{u.login_count}</td>
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
                          <li key={k}><strong>{k}:</strong> {v}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {defs?.compliance_note && (
            <div className="alert alert-info small mb-4">
              <strong>Compliance:</strong> {defs.compliance_note}
            </div>
          )}

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Field Reference</div>
            <div className="table-responsive">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Field</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs?.fields || []).map((f, i) => (
                    <tr key={i}>
                      <td className="small fw-semibold fw-mono">{f.field}</td>
                      <td className="small text-muted">{f.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="text-muted small mt-3">Source: {defs?.source}</div>
        </>
      )}
    </div>
  );
}
