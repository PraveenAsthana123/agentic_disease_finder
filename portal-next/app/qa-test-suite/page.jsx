'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: '📊 Overview' },
  { id: 'roles',       label: '👥 Per Role' },
  { id: 'dimensions',  label: '🧪 Dimensions' },
  { id: 'definitions', label: '📚 Definitions' },
];

const PASS_COLOR = { pass: 'success', partial: 'warning', planned: 'secondary' };
const PASS_ICON  = { pass: '✅', partial: '🔧', planned: '📅' };

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

function HBar({ label, value, max, color = 'success' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-semibold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function PassBadge({ status }) {
  return (
    <span className={`badge bg-${PASS_COLOR[status] || 'secondary'}`}>
      {PASS_ICON[status] || '?'} {status}
    </span>
  );
}

function OverviewPanel({ overview }) {
  if (!overview) return <div className="text-muted">Loading…</div>;
  const s = overview.summary || {};
  const roles = overview.role_summaries || [];
  const dims = overview.dimension_table || [];

  return (
    <div>
      <div className="row mb-3">
        <KPI label="Total Tests"    value={s.total_tests}      color="primary"  sub="across all roles" />
        <KPI label="Passing"        value={s.pass}             color="success"  sub="fully verified" />
        <KPI label="Coverage"       value={`${s.coverage_pct ?? 0}%`} color={s.coverage_pct >= 80 ? 'success' : 'warning'} sub="pass rate" />
        <KPI label="Roles Covered"  value={s.total_roles}      color="info"     sub={`${s.testing_dimensions} dimensions`} />
      </div>

      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Pass Rate by Role</div>
            <div className="card-body">
              {roles.map((r, i) => (
                <HBar key={i} label={r.role} value={r.pass_rate} max={100}
                  color={r.pass_rate >= 80 ? 'success' : r.pass_rate >= 50 ? 'warning' : 'danger'} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Test Status Distribution</div>
            <div className="card-body">
              {[
                { label: 'Pass',    value: s.pass,    color: 'success' },
                { label: 'Partial', value: s.partial, color: 'warning' },
                { label: 'Planned', value: s.planned, color: 'secondary' },
              ].map((item, i) => (
                <HBar key={i} label={item.label} value={item.value || 0}
                  max={s.total_tests} color={item.color} />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold">9-Dimension Coverage</div>
        <div className="table-responsive">
          <table className="table table-hover table-sm mb-0">
            <thead className="table-dark">
              <tr>
                <th>Dimension</th>
                <th>What is Tested</th>
                <th>How to Run</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {dims.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{d.dimension}</td>
                  <td className="small">{d.tests}</td>
                  <td className="small font-monospace text-muted">{d.how}</td>
                  <td><PassBadge status={d.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function RolesPanel({ breakdown }) {
  if (!breakdown) return <div className="text-muted">Loading…</div>;
  const roles = breakdown.roles || [];
  const [selected, setSelected] = useState(roles[0]?.role || '');

  const role = roles.find(r => r.role === selected) || roles[0];

  return (
    <div>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {roles.map(r => (
          <button key={r.role}
            className={`btn btn-sm ${selected === r.role ? 'btn-primary' : 'btn-outline-primary'}`}
            onClick={() => setSelected(r.role)}>
            {r.role}
          </button>
        ))}
      </div>

      {role && (
        <div className="card shadow-sm">
          <div className="card-header d-flex align-items-center gap-2">
            <span className="fw-semibold">{role.role}</span>
            <span className="badge bg-success ms-auto">{role.tests?.filter(t => t.status === 'pass').length || 0} pass</span>
            <span className="badge bg-warning">{role.tests?.filter(t => t.status === 'partial').length || 0} partial</span>
            <span className="badge bg-secondary">{role.tests?.filter(t => t.status === 'planned').length || 0} planned</span>
          </div>
          <div className="table-responsive">
            <table className="table table-hover table-sm mb-0">
              <thead className="table-light">
                <tr>
                  <th>Dimension</th>
                  <th>Test Case</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {(role.tests || []).map((t, i) => (
                  <tr key={i}>
                    <td><span className="badge bg-light text-dark">{t.dimension}</span></td>
                    <td className="small">{t.case}</td>
                    <td><PassBadge status={t.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* User stories */}
      {breakdown.user_stories?.length > 0 && (
        <div className="mt-4">
          <h6 className="fw-bold mb-3">👤 User Stories</h6>
          <div className="row">
            {breakdown.user_stories.map((s, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card shadow-sm h-100">
                  <div className="card-header">
                    <span className="badge bg-primary">{s.persona}</span>
                  </div>
                  <div className="card-body">
                    <p className="small fst-italic mb-2">"{s.story}"</p>
                    <code className="small">{s.endpoint}</code>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Demo stories */}
      {breakdown.demo_stories?.length > 0 && (
        <div className="mt-2">
          <h6 className="fw-bold mb-3">🎬 Demo Stories</h6>
          <div className="row">
            {breakdown.demo_stories.map((d, i) => (
              <div key={i} className="col-md-4 mb-3">
                <div className="card shadow-sm h-100 border-info">
                  <div className="card-header bg-info text-white fw-semibold small">{d.title}</div>
                  <div className="card-body">
                    <p className="small mb-2">{d.script}</p>
                    <div className="text-muted" style={{ fontSize: '0.75rem' }}>
                      <strong>Shows:</strong> {d.shows}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DimensionsPanel({ overview }) {
  if (!overview) return <div className="text-muted">Loading…</div>;
  const dims = overview.dimension_table || [];

  return (
    <div className="card shadow-sm">
      <div className="card-header fw-semibold">🧪 9-Dimension Testing Framework</div>
      <div className="table-responsive">
        <table className="table table-hover table-sm mb-0">
          <thead className="table-dark">
            <tr>
              <th>#</th>
              <th>Dimension</th>
              <th>What is Tested</th>
              <th>How to Run</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {dims.map((d, i) => (
              <tr key={i}>
                <td className="text-muted">{i + 1}</td>
                <td className="fw-semibold">{d.dimension}</td>
                <td className="small">{d.tests}</td>
                <td className="small font-monospace text-muted">{d.how}</td>
                <td><PassBadge status={d.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsPanel({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  const definitions = defs.definitions || [];
  const methodology = defs.methodology || {};
  const framework   = defs.framework   || {};

  return (
    <div>
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold">📖 Terms & Definitions</div>
            <ul className="list-group list-group-flush">
              {definitions.map((d, i) => (
                <li key={i} className="list-group-item">
                  <strong className="small">{d.term}</strong>
                  <div className="text-muted" style={{ fontSize: '0.78rem' }}>{d.definition}</div>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-6">
          {methodology.name && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-semibold">🔬 Methodology</div>
              <div className="card-body small">
                <p className="mb-2"><strong>{methodology.name}</strong></p>
                <p className="text-muted">{methodology.description}</p>
                {methodology.phases?.length > 0 && (
                  <ol className="mb-0">
                    {methodology.phases.map((p, i) => <li key={i}>{p}</li>)}
                  </ol>
                )}
              </div>
            </div>
          )}
          {framework.name && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">📐 Framework</div>
              <div className="card-body small">
                <p className="mb-2"><strong>{framework.name}</strong> — {framework.description}</p>
                {framework.dimensions?.length > 0 && (
                  <div className="d-flex flex-wrap gap-1">
                    {framework.dimensions.map((d, i) => (
                      <span key={i} className="badge bg-primary">{d}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function QATestSuitePage() {
  const [tab, setTab] = useState('overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/qa-test-suite/overview`).then(r => r.json()),
      fetch(`${API}/api/qa-test-suite/breakdown`).then(r => r.json()),
      fetch(`${API}/api/qa-test-suite/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefs(df);
    }).catch(e => setErr(e.message));
  }, []);

  const s = overview?.summary || {};

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-4 gap-3">
        <div>
          <h2 className="mb-0 fw-bold">🧪 QA Test Suite</h2>
          <div className="text-muted small">
            {s.total_tests} tests · {s.total_roles} roles · {s.testing_dimensions} dimensions · {s.coverage_pct ?? 0}% coverage
          </div>
        </div>
        <span className="badge bg-success ms-auto fs-6">{s.pass} passing</span>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewPanel   overview={overview} />}
      {tab === 'roles'       && <RolesPanel      breakdown={breakdown} />}
      {tab === 'dimensions'  && <DimensionsPanel overview={overview} />}
      {tab === 'definitions' && <DefinitionsPanel defs={defs} />}
    </div>
  );
}
