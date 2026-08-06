'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const ROLE_COLORS = {
  'Neurologist': 'primary',
  'Clinical Neurophysiologist': 'info',
  'EEG Technician': 'success',
  'Patient': 'warning',
  'AI Governance': 'danger',
  'Pharmacist': 'secondary',
};
const roleColor = (role) => ROLE_COLORS[role] || 'dark';

const TABS = [
  { id: 'overview', label: '📊 Overview' },
  { id: 'roles', label: '👥 Roles' },
  { id: 'flows', label: '🔀 Process Flows' },
  { id: 'definitions', label: '📖 Definitions' },
];

export default function RoleProcessFlowsPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selectedRole, setSelectedRole] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/role-process-flows/overview`).then(r => r.json()),
      fetch(`${API}/api/role-process-flows/breakdown`).then(r => r.json()),
      fetch(`${API}/api/role-process-flows/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => {
        setOverview(o);
        setBreakdown(b);
        setDefs(d);
        if (b?.role_details?.length) setSelectedRole(b.role_details[0].role);
      })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading role process flows…</div>;

  const s = overview.summary || {};
  const roles = breakdown?.role_details || [];
  const activeRole = roles.find(r => r.role === selectedRole);

  return (
    <div className="container-fluid py-3">
      <h3>🔀 Role Process Flows</h3>
      <p className="text-muted small">
        End-to-end clinical workflows for each role — {s.total_roles} roles · {s.total_steps} total steps · Mermaid process diagrams.
      </p>

      {/* KPI tiles */}
      <div className="row g-2 mb-3">
        {[
          ['Roles', s.total_roles, 'primary'],
          ['Total Steps', s.total_steps, 'info'],
          ['Avg Steps / Role', s.avg_steps_per_role, 'success'],
          ['Most Steps', s.max_steps_role, 'warning', s.max_steps_count],
          ['Fewest Steps', s.min_steps_role, 'secondary', s.min_steps_count],
          ['With Mermaid', s.with_mermaid, 'danger'],
        ].map(([label, val, color, sub]) => (
          <div key={label} className="col-6 col-md-2">
            <div className={`card border-${color} text-center p-2 h-100`}>
              <div className={`fs-5 fw-bold text-${color}`}>{val ?? '—'}</div>
              {sub !== undefined && <div className="small text-muted">{sub} steps</div>}
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
          <h5>Steps by Role</h5>
          <div className="row g-3 mb-4">
            {(overview.steps_distribution || []).map(r => (
              <div key={r.role} className="col-md-4">
                <div className="card h-100">
                  <div className="card-body">
                    <div className="d-flex justify-content-between align-items-start">
                      <span className={`badge bg-${roleColor(r.role)}`}>{r.role}</span>
                      <span className="fs-5 fw-bold">{r.steps}</span>
                    </div>
                    <div className="progress mt-2" style={{ height: 6 }}>
                      <div
                        className={`progress-bar bg-${roleColor(r.role)}`}
                        style={{ width: `${(r.steps / (s.max_steps_count || 1)) * 100}%` }}
                      />
                    </div>
                    <div className="small text-muted mt-1">{r.steps} steps</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h5>Role Summary Table</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover align-middle">
              <thead className="table-light">
                <tr>
                  <th>Role</th>
                  <th>Steps</th>
                  <th>First Step</th>
                  <th>Last Step</th>
                  <th>Mermaid</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {(overview.role_table || []).map(r => (
                  <tr key={r.role}>
                    <td>
                      <span className={`badge bg-${roleColor(r.role)}`}>{r.role}</span>
                    </td>
                    <td className="fw-semibold">{r.num_steps}</td>
                    <td className="small">{r.first_step}</td>
                    <td className="small">{r.last_step}</td>
                    <td>
                      {r.has_mermaid
                        ? <span className="badge bg-success">✓ Yes</span>
                        : <span className="badge bg-secondary">Default</span>}
                    </td>
                    <td>
                      <button
                        className="btn btn-sm btn-outline-primary"
                        onClick={() => { setSelectedRole(r.role); setTab('flows'); }}
                      >View Flow →</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Roles tab */}
      {tab === 'roles' && (
        <div className="row g-3">
          {roles.map(r => (
            <div key={r.role} className="col-md-6">
              <div className={`card h-100 border-${roleColor(r.role)}`}>
                <div className="card-header d-flex justify-content-between align-items-center">
                  <span className={`fw-semibold text-${roleColor(r.role)}`}>{r.role}</span>
                  <span className={`badge bg-${roleColor(r.role)}`}>{r.num_steps} steps</span>
                </div>
                <div className="card-body p-2">
                  <ol className="mb-0 ps-3">
                    {(r.steps || []).map(step => (
                      <li key={step.n} className="small py-1 border-bottom">
                        <span className="text-muted me-1">#{step.n}</span>
                        {step.step}
                      </li>
                    ))}
                  </ol>
                </div>
                <div className="card-footer text-end">
                  <button
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() => { setSelectedRole(r.role); setTab('flows'); }}
                  >View Mermaid →</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Process Flows tab */}
      {tab === 'flows' && (
        <div>
          <div className="mb-3">
            <label className="form-label fw-semibold">Select Role</label>
            <div className="d-flex gap-2 flex-wrap">
              {roles.map(r => (
                <button
                  key={r.role}
                  className={`btn btn-sm ${selectedRole === r.role ? `btn-${roleColor(r.role)}` : `btn-outline-${roleColor(r.role)}`}`}
                  onClick={() => setSelectedRole(r.role)}
                >{r.role}</button>
              ))}
            </div>
          </div>

          {activeRole && (
            <div className="card">
              <div className={`card-header bg-${roleColor(activeRole.role)} text-white d-flex justify-content-between align-items-center`}>
                <span className="fw-semibold">{activeRole.role}</span>
                <span>{activeRole.num_steps} steps</span>
              </div>
              <div className="card-body">
                <h6 className="mb-2">Mermaid Process Diagram</h6>
                <pre className="bg-light border rounded p-3 small" style={{ maxHeight: 280, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                  {activeRole.mermaid || '(no mermaid diagram for this role)'}
                </pre>

                <h6 className="mt-3 mb-2">Step-by-Step Sequence</h6>
                <div className="d-flex flex-wrap gap-2">
                  {(activeRole.steps || []).map((step, i) => (
                    <div key={i} className="d-flex align-items-center">
                      <span className={`badge bg-${roleColor(activeRole.role)} me-1`}>{step.n}</span>
                      <span className="small">{step.step}</span>
                      {i < activeRole.steps.length - 1 && (
                        <span className="text-muted mx-2">→</span>
                      )}
                    </div>
                  ))}
                </div>

                <div className="row g-2 mt-3">
                  <div className="col-4">
                    <div className={`card border-${roleColor(activeRole.role)} text-center p-2`}>
                      <div className={`fw-bold text-${roleColor(activeRole.role)}`}>{activeRole.num_steps}</div>
                      <div className="small text-muted">Total Steps</div>
                    </div>
                  </div>
                  <div className="col-4">
                    <div className="card border-success text-center p-2">
                      <div className="fw-bold text-success">{activeRole.steps?.[0]?.step || '—'}</div>
                      <div className="small text-muted">Entry Point</div>
                    </div>
                  </div>
                  <div className="col-4">
                    <div className="card border-secondary text-center p-2">
                      <div className="fw-bold text-secondary">{activeRole.steps?.[activeRole.steps.length - 1]?.step || '—'}</div>
                      <div className="small text-muted">Exit Point</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Definitions tab */}
      {tab === 'definitions' && defs && (
        <div>
          <h5>Role Descriptions</h5>
          <div className="table-responsive mb-4">
            <table className="table table-sm table-bordered">
              <thead className="table-light">
                <tr><th>Role</th><th>Description</th></tr>
              </thead>
              <tbody>
                {(defs.role_descriptions || []).map(r => (
                  <tr key={r.role}>
                    <td><span className={`badge bg-${roleColor(r.role)}`}>{r.role}</span></td>
                    <td className="small">{r.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5>Glossary</h5>
          <div className="table-responsive">
            <table className="table table-sm table-bordered">
              <thead className="table-light">
                <tr><th>Term</th><th>Definition</th></tr>
              </thead>
              <tbody>
                {(defs.glossary || []).map(g => (
                  <tr key={g.term}>
                    <td className="fw-semibold text-nowrap"><code>{g.term}</code></td>
                    <td className="small">{g.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
