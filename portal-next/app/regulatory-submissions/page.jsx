'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'reviewers',   label: 'Reviewer Workload' },
  { id: 'products',    label: 'Per Product' },
  { id: 'submissions', label: 'All Submissions' },
  { id: 'definitions', label: 'Definitions' },
];

const STATUS_COLOR = {
  'Approved': 'success',
  'Under Review': 'primary',
  'Submitted': 'info',
  'Pre-submission': 'secondary',
  'Additional Info Requested': 'warning',
  'Rejected': 'danger',
};

const RISK_COLOR = {
  'Class I': 'success',
  'Class IIa': 'info',
  'Class IIb': 'warning',
  'Class III': 'danger',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center py-3">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function BarList({ items, keyField, valueField, colorFn, maxVal }) {
  if (!items?.length) return <p className="text-muted small">No data.</p>;
  const max = maxVal ?? Math.max(...items.map(i => i[valueField] || 0));
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.map((item, i) => {
          const pct = max > 0 ? ((item[valueField] / max) * 100).toFixed(0) : 0;
          const col = colorFn ? colorFn(item[keyField]) : 'primary';
          return (
            <tr key={i}>
              <td className="small fw-semibold" style={{ width: '40%' }}>{item[keyField]}</td>
              <td style={{ width: '45%' }}>
                <div className="progress" style={{ height: 14 }}>
                  <div className={`progress-bar bg-${col}`} style={{ width: `${pct}%` }}>
                    <span className="small">{item[valueField]}</span>
                  </div>
                </div>
              </td>
              <td className="small text-end text-muted">{item[valueField]}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function Badge({ val, map }) {
  const col = map?.[val] || 'secondary';
  return <span className={`badge bg-${col}`}>{val || '—'}</span>;
}

export default function RegulatorySubmissionsPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch(`${API}/api/regulatory-submissions/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/regulatory-submissions/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/regulatory-submissions/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  const kpis = ov?.kpis || {};
  const filteredSubs = (bd?.submission_list || []).filter(s =>
    !search || (s.product || '').toLowerCase().includes(search.toLowerCase()) ||
    (s.pathway || '').toLowerCase().includes(search.toLowerCase()) ||
    (s.status || '').toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="container-fluid py-4">
      <div className="d-flex align-items-center mb-3 gap-3">
        <h2 className="mb-0 fw-bold">📋 Regulatory Submissions</h2>
        <span className="badge bg-secondary">{kpis.total_submissions ?? '—'} submissions</span>
        <span className="badge bg-info">{kpis.total_products ?? '—'} products</span>
        <span className="badge bg-primary">{kpis.total_pathways ?? '—'} pathways</span>
      </div>

      {/* KPI strip */}
      <div className="row mb-4">
        <KPI label="Total Submissions"    value={kpis.total_submissions}   color="primary" />
        <KPI label="Products"             value={kpis.total_products}       color="info" />
        <KPI label="Approved"             value={kpis.approved_count}       color="success" />
        <KPI label="Approval Rate"        value={kpis.approval_rate != null ? `${kpis.approval_rate}%` : '—'} color="success" />
        <KPI label="Avg Validation Score" value={kpis.avg_validation_score != null ? (kpis.avg_validation_score * 100).toFixed(1) + '%' : '—'} color="warning" />
        <KPI label="Reviewers"            value={kpis.total_reviewers}      color="secondary" />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview */}
      {tab === 'overview' && (
        <div className="row g-4">
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header fw-semibold">Status Distribution</div>
              <div className="card-body">
                <BarList
                  items={ov?.status_distribution || []}
                  keyField="status"
                  valueField="count"
                  colorFn={s => STATUS_COLOR[s] || 'secondary'}
                />
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header fw-semibold">Pathway Distribution</div>
              <div className="card-body">
                <BarList
                  items={ov?.pathway_distribution || []}
                  keyField="pathway"
                  valueField="count"
                />
              </div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="card h-100">
              <div className="card-header fw-semibold">Risk Class Distribution</div>
              <div className="card-body">
                <BarList
                  items={ov?.risk_distribution || []}
                  keyField="risk_class"
                  valueField="count"
                  colorFn={r => RISK_COLOR[r] || 'secondary'}
                />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Phase Distribution</div>
              <div className="card-body">
                <BarList
                  items={ov?.phase_distribution || []}
                  keyField="phase"
                  valueField="count"
                />
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Submissions per Product</div>
              <div className="card-body">
                <BarList
                  items={ov?.product_breakdown || []}
                  keyField="product"
                  valueField="count"
                />
              </div>
            </div>
          </div>
          {ov?.submission_timeline?.length > 0 && (
            <div className="col-12">
              <div className="card">
                <div className="card-header fw-semibold">Submission Timeline (by Month)</div>
                <div className="card-body">
                  <div className="table-responsive">
                    <table className="table table-sm">
                      <thead><tr><th>Month</th><th>Count</th><th>Bar</th></tr></thead>
                      <tbody>
                        {ov.submission_timeline.map((row, i) => {
                          const max = Math.max(...ov.submission_timeline.map(r => r.count));
                          const pct = max > 0 ? (row.count / max * 100).toFixed(0) : 0;
                          return (
                            <tr key={i}>
                              <td className="small">{row.month}</td>
                              <td className="small">{row.count}</td>
                              <td style={{ width: '60%' }}>
                                <div className="progress" style={{ height: 12 }}>
                                  <div className="progress-bar bg-primary" style={{ width: `${pct}%` }} />
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
          )}
        </div>
      )}

      {/* Reviewer Workload */}
      {tab === 'reviewers' && (
        <div className="card">
          <div className="card-header fw-semibold">Reviewer Workload</div>
          <div className="card-body">
            {!(bd?.reviewer_workload?.length) ? (
              <p className="text-muted">No reviewer data.</p>
            ) : (
              <div className="table-responsive">
                <table className="table table-striped table-sm align-middle">
                  <thead className="table-dark">
                    <tr>
                      <th>Reviewer</th>
                      <th>Total Submissions</th>
                      <th>Approved</th>
                      <th>Approval Rate</th>
                      <th>Avg Validation Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.reviewer_workload.map((r, i) => {
                      const rate = r.total > 0 ? ((r.approved / r.total) * 100).toFixed(0) : 0;
                      const score = r.avg_validation_score != null ? (r.avg_validation_score * 100).toFixed(1) + '%' : '—';
                      return (
                        <tr key={i}>
                          <td className="fw-semibold">{r.reviewer}</td>
                          <td>{r.total}</td>
                          <td><span className="badge bg-success">{r.approved}</span></td>
                          <td>{rate}%</td>
                          <td>{score}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Per Product */}
      {tab === 'products' && (
        <div className="card">
          <div className="card-header fw-semibold">Per-Product Summary</div>
          <div className="card-body">
            {!(bd?.per_product?.length) ? (
              <p className="text-muted">No product data.</p>
            ) : (
              <div className="table-responsive">
                <table className="table table-striped table-sm align-middle">
                  <thead className="table-dark">
                    <tr>
                      <th>Product</th>
                      <th>Submissions</th>
                      <th>Pathways</th>
                      <th>Statuses</th>
                      <th>Avg Validation Score</th>
                      <th>Overdue</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bd.per_product.map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{p.product}</td>
                        <td>{p.submissions}</td>
                        <td className="small text-muted">{p.pathways}</td>
                        <td className="small">{p.statuses}</td>
                        <td>{p.avg_validation_score != null ? (p.avg_validation_score * 100).toFixed(1) + '%' : '—'}</td>
                        <td>{p.overdue_count > 0 ? <span className="badge bg-danger">{p.overdue_count} overdue</span> : <span className="text-success small">None</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {bd?.overdue_submissions?.length > 0 && (
              <div className="mt-4">
                <h6 className="fw-bold text-danger">⚠ Overdue Submissions</h6>
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead><tr><th>Product</th><th>Pathway</th><th>Submitted</th><th>Status</th><th>Days Overdue</th></tr></thead>
                    <tbody>
                      {bd.overdue_submissions.map((s, i) => (
                        <tr key={i} className="table-danger">
                          <td>{s.product}</td>
                          <td>{s.pathway}</td>
                          <td className="small">{s.submitted_date}</td>
                          <td><Badge val={s.status} map={STATUS_COLOR} /></td>
                          <td><span className="badge bg-danger">{s.days_overdue}d</span></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* All Submissions */}
      {tab === 'submissions' && (
        <div className="card">
          <div className="card-header d-flex justify-content-between align-items-center">
            <span className="fw-semibold">All Submissions ({filteredSubs.length})</span>
            <input
              className="form-control form-control-sm w-auto"
              placeholder="Search product / pathway / status…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ minWidth: 240 }}
            />
          </div>
          <div className="card-body p-0">
            <div className="table-responsive">
              <table className="table table-sm table-hover mb-0 align-middle">
                <thead className="table-dark">
                  <tr>
                    <th>Product</th>
                    <th>Pathway</th>
                    <th>Status</th>
                    <th>Risk Class</th>
                    <th>Phase</th>
                    <th>Reviewer</th>
                    <th>Submitted</th>
                    <th>Val. Score</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredSubs.length === 0 ? (
                    <tr><td colSpan={8} className="text-center text-muted py-3">No submissions match.</td></tr>
                  ) : filteredSubs.map((s, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{s.product}</td>
                      <td className="small">{s.pathway}</td>
                      <td><Badge val={s.status} map={STATUS_COLOR} /></td>
                      <td><Badge val={s.risk_class} map={RISK_COLOR} /></td>
                      <td className="small">{s.phase}</td>
                      <td className="small">{s.reviewer}</td>
                      <td className="small text-muted">{s.submitted_date || '—'}</td>
                      <td>{s.validation_score != null ? (s.validation_score * 100).toFixed(0) + '%' : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Definitions */}
      {tab === 'definitions' && defs && (
        <div className="row g-4">
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Regulatory Pathways</div>
              <div className="card-body">
                <dl className="row mb-0">
                  {(defs.pathways || []).map((p, i) => (
                    <div key={i} className="mb-2">
                      <dt className="small fw-bold">{p.name}</dt>
                      <dd className="small text-muted mb-1">{p.description}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="card h-100">
              <div className="card-header fw-semibold">Status Definitions</div>
              <div className="card-body">
                <dl className="row mb-0">
                  {(defs.statuses || []).map((s, i) => (
                    <div key={i} className="mb-2">
                      <dt className="small fw-bold">
                        <Badge val={s.name} map={STATUS_COLOR} /> {s.name}
                      </dt>
                      <dd className="small text-muted mb-1">{s.description}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </div>
          </div>
          {defs.risk_classes && (
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-semibold">Risk Classifications</div>
                <div className="card-body">
                  {defs.risk_classes.map((r, i) => (
                    <div key={i} className="mb-2">
                      <span className={`badge bg-${RISK_COLOR[r.name] || 'secondary'} me-2`}>{r.name}</span>
                      <span className="small text-muted">{r.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
          {defs.metrics && (
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-semibold">Metric Glossary</div>
                <div className="card-body">
                  <dl className="mb-0">
                    {defs.metrics.map((m, i) => (
                      <div key={i} className="mb-2">
                        <dt className="small fw-bold">{m.name}</dt>
                        <dd className="small text-muted mb-1">{m.description}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
