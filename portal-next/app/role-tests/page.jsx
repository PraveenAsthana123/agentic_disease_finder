'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Role Breakdown' },
  { id: 'definitions', label: 'Definitions' },
];

const STATUS_COLOR = { pass: 'success', built: 'success', partial: 'warning', planned: 'secondary', fail: 'danger' };
const DIM_ICON = {
  API: '🔌', Data: '📊', Model: '🧠', Accuracy: '🎯',
  Process: '⚙️', Frontend: '🖥️', Manual: '✋',
};

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

function StatusBadge({ status }) {
  const color = STATUS_COLOR[status] || 'secondary';
  return <span className={`badge bg-${color}`}>{status}</span>;
}

function RoleBar({ role, value, pass, partial, planned, total }) {
  const passW = total ? (pass / total) * 100 : 0;
  const partW = total ? (partial / total) * 100 : 0;
  const planW = total ? (planned / total) * 100 : 0;
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="fw-semibold small">{role}</span>
        <span className="text-muted small">{pass}/{value} passed</span>
      </div>
      <div className="progress" style={{ height: 20 }}>
        <div className="progress-bar bg-success" style={{ width: `${passW}%` }} title={`${pass} pass`} />
        {partial > 0 && <div className="progress-bar bg-warning" style={{ width: `${partW}%` }} title={`${partial} partial`} />}
        {planned > 0 && <div className="progress-bar bg-secondary" style={{ width: `${planW}%` }} title={`${planned} planned`} />}
      </div>
    </div>
  );
}

export default function RoleTestsDashboard() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv]   = useState(null);
  const [bd, setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/role-tests/overview`).then(r => r.json()),
      fetch(`${API}/api/role-tests/breakdown`).then(r => r.json()),
      fetch(`${API}/api/role-tests/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOv(o); setBd(b); setDefs(d);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="p-4 text-center">
      <div className="spinner-border text-primary" /><div className="mt-2 text-muted">Loading role tests…</div>
    </div>
  );
  if (!ov?.available) return <div className="p-4 alert alert-warning">Role Tests data unavailable.</div>;

  const s = ov.summary;

  return (
    <div>
      <h3 className="mb-1">🧪 Role Tests Dashboard</h3>
      <p className="text-muted mb-3">Per-role acceptance tests — API · Data · Model · Accuracy · Process · Manual</p>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <KPI label="Total Roles"      value={s.total_roles}   color="primary" />
            <KPI label="Total Tests"      value={s.total_tests}   color="info"    />
            <KPI label="Tests Passed"     value={s.passed}        color="success" sub={`${s.pass_pct?.toFixed(1)}%`} />
            <KPI label="Roles All-Pass"   value={s.roles_all_pass} color="success" sub={`of ${s.total_roles}`} />
          </div>

          {/* Status distribution */}
          <div className="card mb-3 shadow-sm">
            <div className="card-header fw-semibold">Test Status Distribution</div>
            <div className="card-body">
              <div className="row g-2">
                {ov.status_distribution?.map((item, i) => (
                  <div key={i} className="col-auto">
                    <div className="d-flex align-items-center gap-2 p-2 border rounded">
                      <StatusBadge status={item.name} />
                      <span className="fw-bold">{item.value}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Tests per role bar chart */}
          <div className="card mb-3 shadow-sm">
            <div className="card-header fw-semibold">Tests per Role</div>
            <div className="card-body">
              {ov.tests_per_role?.map((r, i) => (
                <RoleBar key={i} role={r.name} value={r.value} pass={r.pass} partial={r.partial} planned={r.planned} total={r.value} />
              ))}
              <div className="mt-2 d-flex gap-3 small text-muted">
                <span><span className="badge bg-success me-1">■</span>Pass</span>
                <span><span className="badge bg-warning me-1">■</span>Partial</span>
                <span><span className="badge bg-secondary me-1">■</span>Planned</span>
              </div>
            </div>
          </div>

          {/* Dimension coverage */}
          {ov.dim_coverage && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Dimension Coverage</div>
              <div className="card-body">
                <div className="row g-2">
                  {ov.dim_coverage.map((d, i) => (
                    <div key={i} className="col-6 col-md-4 col-lg-3">
                      <div className="border rounded p-2 text-center">
                        <div className="fs-5">{DIM_ICON[d.dim] || '🔹'}</div>
                        <div className="fw-semibold small">{d.dim}</div>
                        <div className="text-success small">{d.pass} pass</div>
                        {d.partial > 0 && <div className="text-warning small">{d.partial} partial</div>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── BREAKDOWN ── */}
      {tab === 'breakdown' && bd?.roles && (
        <div>
          {bd.roles.map((roleObj, ri) => (
            <div key={ri} className="card mb-3 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center">
                <span className="fw-semibold">{roleObj.role}</span>
                <span className="badge bg-primary">{roleObj.tests?.length ?? 0} tests</span>
              </div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th style={{ width: 90 }}>Dimension</th>
                        <th>Test Case</th>
                        <th style={{ width: 80 }}>Status</th>
                        <th>Maps To</th>
                      </tr>
                    </thead>
                    <tbody>
                      {roleObj.tests?.map((t, ti) => (
                        <tr key={ti}>
                          <td>
                            <span className="badge bg-light text-dark border">
                              {DIM_ICON[t.dim] || ''} {t.dim}
                            </span>
                          </td>
                          <td className="small">{t.case}</td>
                          <td><StatusBadge status={t.status} /></td>
                          <td className="text-muted small" style={{ fontSize: 11 }}>{t.maps_to || '—'}</td>
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

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card mb-3 shadow-sm">
            <div className="card-header fw-semibold">Test Dimension Descriptions</div>
            <div className="card-body">
              <div className="row g-3">
                {defs.dimension_descriptions?.map((d, i) => (
                  <div key={i} className="col-md-6">
                    <div className="border rounded p-3 h-100">
                      <div className="fw-semibold mb-1">{DIM_ICON[d.dim] || '🔹'} {d.dim}</div>
                      <div className="text-muted small">{d.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Status Legend</div>
            <div className="card-body">
              <div className="row g-2">
                {defs.status_legend?.map((sl, i) => (
                  <div key={i} className="col-md-6">
                    <div className="d-flex gap-2 align-items-start border rounded p-2">
                      <StatusBadge status={sl.status} />
                      <div className="text-muted small">{sl.description}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
