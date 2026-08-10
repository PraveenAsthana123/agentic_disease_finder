'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'breakdown',   label: 'Per Submission' },
  { id: 'actors',      label: 'Actors' },
  { id: 'definitions', label: 'Definitions' },
];

const CAT_COLOR = {
  Administrative: 'primary',
  Regulatory: 'success',
  Technical: 'info',
  Clinical: 'warning',
  Quality: 'danger',
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

function CatBar({ category, count, total }) {
  const pct = total ? Math.round((count / total) * 100) : 0;
  const color = CAT_COLOR[category] || 'secondary';
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between align-items-center mb-1">
        <span className="small fw-semibold">{category}</span>
        <span className="text-muted small">{count} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 14 }}>
        <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function RegulatoryAuditTrailDashboard() {
  const [tab, setTab]     = useState('overview');
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [defs, setDefs]   = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/regulatory-audit-trail/overview`).then(r => r.json()),
      fetch(`${API}/api/regulatory-audit-trail/breakdown`).then(r => r.json()),
      fetch(`${API}/api/regulatory-audit-trail/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOv(o); setBd(b); setDefs(d);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="p-4 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted">Loading regulatory audit trail…</div>
    </div>
  );
  if (!ov?.available) return <div className="p-4 alert alert-warning">Regulatory Audit Trail data unavailable.</div>;

  const k = ov.kpis;
  const totalCatActions = (ov.category_distribution || []).reduce((s, c) => s + c.count, 0);

  return (
    <div>
      <h3 className="mb-1">📋 Regulatory Audit Trail Dashboard</h3>
      <p className="text-muted mb-3">
        Complete action log for 16 regulatory submissions — 102 audit events · 11 actors · 5 categories
      </p>

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
        <div>
          {/* KPIs */}
          <div className="row mb-4">
            <KPI label="Total Audit Actions" value={k.total_actions} color="primary" />
            <KPI label="Regulatory Submissions" value={k.total_submissions} color="success" />
            <KPI label="Active Actors" value={k.total_actors} color="info" />
            <KPI label="Action Categories" value={k.total_categories} color="warning" />
          </div>

          <div className="row mb-4">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Category Distribution</div>
                <div className="card-body">
                  {(ov.category_distribution || []).map(c => (
                    <CatBar key={c.category} category={c.category} count={c.count} total={totalCatActions} />
                  ))}
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Action Breakdown</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Action Type</th><th className="text-end">Count</th></tr>
                    </thead>
                    <tbody>
                      {(ov.action_breakdown || []).map(a => (
                        <tr key={a.action}>
                          <td className="small">{a.action}</td>
                          <td className="text-end">
                            <span className="badge bg-secondary">{a.count}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Monthly Timeline */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Monthly Activity Timeline</div>
            <div className="card-body">
              <div className="d-flex align-items-end gap-1" style={{ height: 120, overflowX: 'auto' }}>
                {(ov.monthly_timeline || []).map(m => {
                  const maxCount = Math.max(...(ov.monthly_timeline || []).map(x => x.count));
                  const h = maxCount ? Math.round((m.count / maxCount) * 100) : 0;
                  return (
                    <div key={m.month} className="d-flex flex-column align-items-center" style={{ minWidth: 32 }} title={`${m.month}: ${m.count} actions`}>
                      <span className="text-muted" style={{ fontSize: 10 }}>{m.count}</span>
                      <div className="bg-primary rounded-top" style={{ width: 20, height: `${h}%`, minHeight: 2 }} />
                      <span className="text-muted" style={{ fontSize: 9, writingMode: 'vertical-rl', marginTop: 2 }}>{m.month.slice(2)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Most Active */}
          <div className="alert alert-info">
            <strong>Most Active Submission:</strong> {k.most_active_submission} — {k.most_active_submission_count} audit actions
          </div>

          {/* Top Actors */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Top Actors by Activity</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead className="table-light">
                  <tr><th>#</th><th>Actor</th><th className="text-end">Actions</th></tr>
                </thead>
                <tbody>
                  {(ov.actor_activity || []).slice(0, 5).map((a, i) => (
                    <tr key={a.actor}>
                      <td className="text-muted small">{i + 1}</td>
                      <td className="small fw-semibold">{a.actor}</td>
                      <td className="text-end"><span className="badge bg-primary">{a.count}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── PER SUBMISSION ── */}
      {tab === 'breakdown' && bd?.available && (
        <div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Audit Actions per Submission</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Submission ID</th>
                      <th className="text-end">Actions</th>
                      <th className="text-end">Actors</th>
                      <th className="text-end">Categories</th>
                      <th>First Action</th>
                      <th>Last Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.per_submission || []).map(s => (
                      <tr key={s.submission_id}>
                        <td className="fw-semibold small text-primary">{s.submission_id}</td>
                        <td className="text-end"><span className="badge bg-primary">{s.action_count}</span></td>
                        <td className="text-end text-muted small">{s.actor_count}</td>
                        <td className="text-end text-muted small">{s.category_count}</td>
                        <td className="text-muted small">{s.first_action?.slice(0, 10)}</td>
                        <td className="text-muted small">{s.last_action?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Recent Actions */}
          {bd.recent_actions && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Recent Audit Events</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-hover mb-0">
                    <thead className="table-light">
                      <tr>
                        <th>Submission</th>
                        <th>Action</th>
                        <th>Actor</th>
                        <th>Category</th>
                        <th>Timestamp</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.recent_actions || []).map((r, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold text-primary">{r.submission_id}</td>
                          <td className="small">{r.action}</td>
                          <td className="small text-muted">{r.actor}</td>
                          <td>
                            <span className={`badge bg-${CAT_COLOR[r.category] || 'secondary'}`} style={{ fontSize: 10 }}>
                              {r.category}
                            </span>
                          </td>
                          <td className="small text-muted">{r.timestamp?.slice(0, 16)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── ACTORS ── */}
      {tab === 'actors' && bd?.available && (
        <div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Actor Activity Summary</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Actor</th>
                      <th className="text-end">Actions</th>
                      <th className="text-end">Submissions</th>
                      <th className="text-end">Action Types</th>
                      <th>Last Activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.per_actor || []).map(a => (
                      <tr key={a.actor}>
                        <td className="fw-semibold small">{a.actor}</td>
                        <td className="text-end"><span className="badge bg-primary">{a.action_count}</span></td>
                        <td className="text-end text-muted small">{a.submission_count}</td>
                        <td className="text-end text-muted small">{a.action_types}</td>
                        <td className="text-muted small">{a.last_activity?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Actor bar chart */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Actions per Actor</div>
            <div className="card-body">
              {(bd.per_actor || []).map(a => {
                const maxA = Math.max(...(bd.per_actor || []).map(x => x.action_count));
                const pct = maxA ? Math.round((a.action_count / maxA) * 100) : 0;
                return (
                  <div key={a.actor} className="mb-2">
                    <div className="d-flex justify-content-between mb-1">
                      <span className="small fw-semibold">{a.actor}</span>
                      <span className="text-muted small">{a.action_count} actions</span>
                    </div>
                    <div className="progress" style={{ height: 14 }}>
                      <div className="progress-bar bg-success" style={{ width: `${pct}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs?.available && (
        <div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Action Types</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Action</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs.action_types || []).map(a => (
                    <tr key={a.action}>
                      <td className="small fw-semibold">{a.action}</td>
                      <td className="small text-muted">{a.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Audit Categories</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Category</th><th>Description</th></tr>
                </thead>
                <tbody>
                  {(defs.categories || []).map(c => (
                    <tr key={c.category}>
                      <td>
                        <span className={`badge bg-${CAT_COLOR[c.category] || 'secondary'}`}>{c.category}</span>
                      </td>
                      <td className="small text-muted">{c.description}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {defs.glossary && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Glossary</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead className="table-light">
                    <tr><th>Term</th><th>Definition</th></tr>
                  </thead>
                  <tbody>
                    {(defs.glossary || []).map(g => (
                      <tr key={g.term}>
                        <td className="small fw-semibold">{g.term}</td>
                        <td className="small text-muted">{g.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
