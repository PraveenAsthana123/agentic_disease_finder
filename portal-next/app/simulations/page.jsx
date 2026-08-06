'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const LAYER_COLORS = {
  data: 'info',
  process: 'primary',
  accuracy: 'warning',
  reporting: 'success',
  backend: 'secondary',
};

const MODE_COLORS = { auto: 'success', manual: 'danger' };

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-3 mb-2">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-2">
          <div className={`h4 mb-0 text-${color || 'primary'}`}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Bar({ items, colorKey }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i.value));
  return (
    <div>
      {items.map((it, i) => (
        <div key={i} className="d-flex align-items-center mb-1">
          <div className="text-end small me-2" style={{ width: 130, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{it.name}</div>
          <div className="flex-grow-1">
            <div className="progress" style={{ height: 18 }}>
              <div className={`progress-bar bg-${it.color || colorKey || 'primary'}`} style={{ width: `${mx ? ((it.value / mx) * 100) : 0}%` }}>
                <span className="small">{it.value}</span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function LayerBadge({ layer }) {
  return <span className={`badge bg-${LAYER_COLORS[layer] || 'secondary'} me-1`}>{layer}</span>;
}

function ModeBadge({ mode }) {
  return <span className={`badge bg-${MODE_COLORS[mode] || 'secondary'} me-1`}>{mode === 'auto' ? '🤖 auto' : '🧑 manual'}</span>;
}

export default function SimulationsPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [selectedRole, setSelectedRole] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/simulations/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/simulations/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/simulations/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview', label: '📊 Overview' },
    { id: 'simulations', label: '🔄 Simulations' },
    { id: 'per-role', label: '🧑 Per Role' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const kpis = overview.kpis || [];
  const roleTable = overview.role_table || [];
  const modeDistribution = overview.mode_distribution || [];
  const layerDistribution = overview.layer_distribution || [];
  const actorDistribution = overview.actor_distribution || [];
  const stepsPerRole = overview.steps_per_role || [];

  const roles = breakdown?.role_details || [];
  const displayRole = selectedRole || (roles.length ? roles[0].role : null);
  const roleDetail = roles.find(r => r.role === displayRole);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <span className="me-2" style={{ fontSize: 28 }}>🔄</span>
        <div>
          <h4 className="mb-0">Process Simulations</h4>
          <div className="text-muted small">Per-role end-to-end pipeline walkthroughs — human + AI steps</div>
        </div>
      </div>

      {/* KPI row */}
      <div className="row mb-3">
        {kpis.map((k, i) => <KPI key={i} {...k} />)}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview */}
      {tab === 'overview' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Mode Distribution</div>
              <div className="card-body">
                <Bar items={modeDistribution.map(d => ({ ...d, color: d.name === 'auto' ? 'success' : 'danger' }))} />
                <div className="mt-2 text-muted small">auto = AI-driven · manual = human action required</div>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Layer Distribution</div>
              <div className="card-body">
                <Bar items={layerDistribution.map(d => ({ ...d, color: LAYER_COLORS[d.name] || 'secondary' }))} />
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Steps Per Role</div>
              <div className="card-body">
                <Bar items={stepsPerRole.map(d => ({ name: `${d.icon} ${d.role}`, value: d.count, color: 'primary' }))} />
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Actor Breakdown</div>
              <div className="card-body">
                <Bar items={actorDistribution.map(d => ({ ...d, color: 'info' }))} />
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Role Summary</div>
              <div className="card-body p-0">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Role</th>
                      <th>Process</th>
                      <th>Steps</th>
                      <th>Auto</th>
                      <th>Manual</th>
                    </tr>
                  </thead>
                  <tbody>
                    {roleTable.map((r, i) => (
                      <tr key={i}>
                        <td><span className="me-1">{r.icon}</span>{r.role}</td>
                        <td><span className="text-muted small">{r.process}</span></td>
                        <td><span className="badge bg-primary">{r.total_steps}</span></td>
                        <td><span className="badge bg-success">{r.auto_steps}</span></td>
                        <td><span className="badge bg-danger">{r.manual_steps}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Simulations — all roles, collapsible */}
      {tab === 'simulations' && (
        <div>
          {roles.map((r, ri) => (
            <div key={ri} className="card shadow-sm mb-3">
              <div className="card-header d-flex align-items-center">
                <span className="me-2" style={{ fontSize: 20 }}>{r.icon}</span>
                <strong>{r.role}</strong>
                <span className="text-muted ms-2 small">— {r.process}</span>
              </div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr>
                      <th style={{ width: 40 }}>#</th>
                      <th>Layer</th>
                      <th>Mode</th>
                      <th>Actor</th>
                      <th>Input</th>
                      <th>Process</th>
                      <th>Output</th>
                      <th>Maps To</th>
                    </tr>
                  </thead>
                  <tbody>
                    {r.steps.map((s, si) => (
                      <tr key={si}>
                        <td className="text-muted">{s.step}</td>
                        <td><LayerBadge layer={s.layer} /></td>
                        <td><ModeBadge mode={s.mode} /></td>
                        <td><span className="small fw-bold">{s.actor}</span></td>
                        <td><span className="small text-muted">{s.input}</span></td>
                        <td><span className="small">{s.process}</span></td>
                        <td><span className="small text-success">{s.output}</span></td>
                        <td><code className="small">{s.maps_to}</code></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Per Role — role picker + step detail */}
      {tab === 'per-role' && (
        <div className="row">
          <div className="col-md-3 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Select Role</div>
              <div className="list-group list-group-flush">
                {roles.map((r, i) => (
                  <button
                    key={i}
                    className={`list-group-item list-group-item-action ${displayRole === r.role ? 'active' : ''}`}
                    onClick={() => setSelectedRole(r.role)}
                  >
                    <span className="me-2">{r.icon}</span>{r.role}
                    <span className={`badge float-end ${displayRole === r.role ? 'bg-light text-dark' : 'bg-primary'}`}>{r.steps.length}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="col-md-9 mb-3">
            {roleDetail && (
              <div className="card shadow-sm">
                <div className="card-header">
                  <strong>{roleDetail.icon} {roleDetail.role}</strong>
                  <span className="text-muted ms-2 small">— {roleDetail.process}</span>
                </div>
                <div className="card-body p-0">
                  {roleDetail.steps.map((s, si) => (
                    <div key={si} className={`d-flex border-bottom p-2 ${si % 2 === 0 ? 'bg-light' : ''}`}>
                      <div style={{ width: 32, fontWeight: 'bold', color: '#888' }}>{s.step}</div>
                      <div className="flex-grow-1">
                        <div className="d-flex align-items-center mb-1">
                          <LayerBadge layer={s.layer} />
                          <ModeBadge mode={s.mode} />
                          <strong className="ms-1 small">{s.actor}</strong>
                        </div>
                        <div className="small mb-1">
                          <span className="text-muted me-1">In:</span>{s.input}
                          <span className="text-muted mx-2">→</span>
                          <span className="text-muted me-1">Process:</span>{s.process}
                          <span className="text-muted mx-2">→</span>
                          <span className="text-success me-1">Out:</span>{s.output}
                        </div>
                        {s.maps_to && (
                          <div className="small"><span className="text-muted">maps_to:</span> <code className="small">{s.maps_to}</code></div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Definitions */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Layers</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Layer</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.layers || []).map((l, i) => (
                      <tr key={i}>
                        <td><LayerBadge layer={l.layer} /></td>
                        <td className="small">{l.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Modes</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Mode</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.modes || []).map((m, i) => (
                      <tr key={i}>
                        <td><ModeBadge mode={m.mode} /></td>
                        <td className="small">{m.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Actors</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light"><tr><th>Actor</th><th>Role</th></tr></thead>
                  <tbody>
                    {(defs.actors || []).map((a, i) => (
                      <tr key={i}>
                        <td><strong className="small">{a.actor}</strong></td>
                        <td className="small text-muted">{a.role}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Glossary</div>
              <div className="card-body">
                {Object.entries(defs.glossary || {}).map(([k, v], i) => (
                  <div key={i} className="mb-1">
                    <strong className="me-2">{k}:</strong>
                    <span className="text-muted small">{v}</span>
                  </div>
                ))}
                {defs.note && <div className="mt-2 text-info small">{defs.note}</div>}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
