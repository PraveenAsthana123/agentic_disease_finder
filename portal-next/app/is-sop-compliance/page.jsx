'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const STATUS_COLOR = {
  published: 'success',
  draft: 'secondary',
  under_review: 'warning',
  retired: 'dark',
};

const FINDING_COLOR = {
  compliant: 'success',
  minor_nonconformance: 'warning',
  major_nonconformance: 'danger',
  observation: 'info',
};

const SEVERITY_COLOR = {
  low: 'success',
  medium: 'warning',
  high: 'danger',
  critical: 'dark',
};

const AUDIT_STATUS_COLOR = {
  open: 'danger',
  in_progress: 'warning',
  closed: 'success',
};

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-2">
      <div className={`card border-${color || 'primary'} text-center h-100`}>
        <div className="card-body py-2 px-1">
          <div className={`h4 fw-bold mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.68rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function Bar({ items, labelKey = 'status', countKey = 'count', colorFn }) {
  const mx = Math.max(...(items || []).map(i => i[countKey] || 0), 1);
  return (
    <div>
      {(items || []).map((it, i) => {
        const val = it[countKey] ?? 0;
        const label = it[labelKey] ?? '?';
        const pct = Math.round((val / mx) * 100);
        const color = colorFn ? colorFn(it, label) : 'primary';
        return (
          <div key={i} className="d-flex align-items-center mb-1 gap-2">
            <div className="text-end small text-muted" style={{ width: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.75rem' }}>
              {label}
            </div>
            <div className="flex-grow-1">
              <div className="progress" style={{ height: 16 }}>
                <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }}>
                  <span className="small px-1">{val}</span>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StatusBadge({ status, colorMap }) {
  const color = colorMap[status] || 'secondary';
  return <span className={`badge bg-${color}`}>{status?.replace(/_/g, ' ')}</span>;
}

export default function ISSopComplianceDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [catFilter, setCatFilter] = useState('All');
  const [sopSearch, setSopSearch] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/is-sop-compliance/overview`).then(r => r.json()),
      fetch(`${API}/api/is-sop-compliance/breakdown`).then(r => r.json()),
      fetch(`${API}/api/is-sop-compliance/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">{err}</div>;
  if (!ov) return <div className="p-4 text-muted">Loading IS-SOP Compliance dashboard…</div>;

  // Filter SOPs
  let procs = bd?.procedures || [];
  if (catFilter !== 'All') procs = procs.filter(p => p.category === catFilter);
  if (sopSearch) {
    const s = sopSearch.toLowerCase();
    procs = procs.filter(p =>
      (p.sop_id || '').toLowerCase().includes(s) ||
      (p.title || '').toLowerCase().includes(s) ||
      (p.owner || '').toLowerCase().includes(s)
    );
  }
  const categories = ['All', ...new Set((bd?.procedures || []).map(p => p.category).filter(Boolean))];

  return (
    <div className="container-fluid py-3">
      <h4 className="fw-bold mb-1">
        <span className="me-2">📋</span>IS-SOP Compliance Dashboard
      </h4>
      <p className="text-muted small mb-3">
        {ov.total_sops} SOPs · {ov.total_audits} audits · avg compliance {ov.avg_compliance_score}%
        · {ov.open_findings} open findings · {ov.overdue_reviews} overdue reviews
        · Source: clinical.db — is_sop_procedures + is_sop_audits
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {['overview', 'procedures', 'audit findings', 'definitions'].map(t => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link${tab === t ? ' active' : ''}`}
              onClick={() => setTab(t)}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          {/* KPIs */}
          <div className="row mb-3">
            <KPI label="Total SOPs" value={ov.total_sops} color="primary" />
            <KPI label="Total Audits" value={ov.total_audits} color="info" />
            <KPI label="Avg Compliance" value={`${ov.avg_compliance_score}%`} color={ov.avg_compliance_score >= 80 ? 'success' : 'warning'} />
            <KPI label="Compliant Rate" value={`${ov.compliance_rate}%`} color={ov.compliance_rate >= 50 ? 'success' : 'danger'} />
            <KPI label="Open Findings" value={ov.open_findings} color="danger" />
            <KPI label="Overdue Reviews" value={ov.overdue_reviews} color="warning" />
          </div>

          {ov.open_findings > 0 && (
            <div className="alert alert-warning py-2 small mb-3">
              <strong>⚠ {ov.open_findings} open findings</strong> and <strong>{ov.overdue_reviews} SOPs with overdue reviews</strong> require attention.
              Check the Audit Findings tab for details and corrective actions.
            </div>
          )}

          <div className="row">
            {/* SOP Status Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">SOP Status Distribution</div>
                <div className="card-body">
                  <Bar
                    items={ov.sop_status_distribution}
                    labelKey="status"
                    colorFn={(it) => STATUS_COLOR[it.status] || 'secondary'}
                  />
                </div>
              </div>
            </div>

            {/* Category Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">SOPs by Category</div>
                <div className="card-body">
                  <Bar items={ov.category_distribution} labelKey="category" colorFn={() => 'primary'} />
                </div>
              </div>
            </div>

            {/* Finding Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Audit Finding Distribution</div>
                <div className="card-body">
                  <Bar
                    items={ov.finding_distribution}
                    labelKey="type"
                    colorFn={(it) => FINDING_COLOR[it.type] || 'secondary'}
                  />
                </div>
              </div>
            </div>

            {/* Severity Distribution */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Finding Severity Distribution</div>
                <div className="card-body">
                  <Bar
                    items={ov.severity_distribution}
                    labelKey="severity"
                    colorFn={(it) => SEVERITY_COLOR[it.severity] || 'secondary'}
                  />
                </div>
              </div>
            </div>

            {/* Audit Status */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Audit Status Distribution</div>
                <div className="card-body">
                  <Bar
                    items={ov.audit_status_distribution}
                    labelKey="status"
                    colorFn={(it) => AUDIT_STATUS_COLOR[it.status] || 'secondary'}
                  />
                </div>
              </div>
            </div>

            {/* Standards Coverage */}
            <div className="col-md-6 mb-3">
              <div className="card h-100">
                <div className="card-header py-2 fw-semibold small">Regulatory Standards Coverage</div>
                <div className="card-body">
                  <Bar
                    items={ov.standards_coverage}
                    labelKey="standard"
                    colorFn={() => 'info'}
                  />
                </div>
              </div>
            </div>

            {/* Monthly Audit Trend */}
            {ov.monthly_trend && (
              <div className="col-12 mb-3">
                <div className="card">
                  <div className="card-header py-2 fw-semibold small">Monthly Audit &amp; Finding Trend</div>
                  <div className="card-body">
                    <div className="table-responsive">
                      <table className="table table-sm small text-center mb-0">
                        <thead className="table-light">
                          <tr>
                            <th>Month</th>
                            {ov.monthly_trend.map((m, i) => <th key={i}>{m.month}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          <tr>
                            <td className="fw-semibold text-start">Audits</td>
                            {ov.monthly_trend.map((m, i) => (
                              <td key={i}><span className="badge bg-info">{m.audits}</span></td>
                            ))}
                          </tr>
                          <tr>
                            <td className="fw-semibold text-start">Findings</td>
                            {ov.monthly_trend.map((m, i) => (
                              <td key={i}><span className={`badge bg-${m.findings >= 4 ? 'danger' : m.findings >= 2 ? 'warning' : 'success'}`}>{m.findings}</span></td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* ── PROCEDURES ── */}
      {tab === 'procedures' && (
        <>
          <div className="row mb-2 g-2 align-items-end">
            <div className="col-md-4">
              <input
                className="form-control form-control-sm"
                placeholder="Search SOP ID / title / owner…"
                value={sopSearch}
                onChange={e => setSopSearch(e.target.value)}
              />
            </div>
            <div className="col-md-4">
              <select className="form-select form-select-sm" value={catFilter} onChange={e => setCatFilter(e.target.value)}>
                {categories.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="col-md-4 text-muted small">
              Showing {procs.length} / {bd?.procedures?.length || 0} SOPs
            </div>
          </div>

          <div className="table-responsive">
            <table className="table table-sm table-hover small align-middle">
              <thead className="table-dark">
                <tr>
                  <th>SOP ID</th>
                  <th>Title</th>
                  <th>Status</th>
                  <th>Category</th>
                  <th>Version</th>
                  <th>Owner</th>
                  <th>Compliance</th>
                  <th>Last Reviewed</th>
                  <th>Next Due</th>
                  <th>Standards</th>
                </tr>
              </thead>
              <tbody>
                {procs.map((p, i) => {
                  const overdue = p.next_review_due && new Date(p.next_review_due) < new Date();
                  return (
                    <tr key={i} className={overdue && p.status !== 'retired' ? 'table-warning' : ''}>
                      <td className="fw-semibold">{p.sop_id}</td>
                      <td>{p.title}</td>
                      <td><StatusBadge status={p.status} colorMap={STATUS_COLOR} /></td>
                      <td className="text-muted">{p.category}</td>
                      <td>{p.version}</td>
                      <td className="text-muted">{p.owner}</td>
                      <td>
                        <span className={`badge bg-${p.compliance_score >= 90 ? 'success' : p.compliance_score >= 70 ? 'warning' : 'danger'}`}>
                          {p.compliance_score}%
                        </span>
                      </td>
                      <td>{p.last_reviewed || '—'}</td>
                      <td>
                        {p.next_review_due || '—'}
                        {overdue && p.status !== 'retired' && <span className="ms-1 badge bg-danger">Overdue</span>}
                      </td>
                      <td>
                        {(p.applicable_standards || []).map((s, j) => (
                          <span key={j} className="badge bg-secondary me-1" style={{ fontSize: '0.65rem' }}>{s}</span>
                        ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── AUDIT FINDINGS ── */}
      {tab === 'audit findings' && (
        <>
          <div className="table-responsive">
            <table className="table table-sm table-hover small align-middle">
              <thead className="table-dark">
                <tr>
                  <th>Audit ID</th>
                  <th>SOP</th>
                  <th>Date</th>
                  <th>Finding Type</th>
                  <th>Severity</th>
                  <th>Status</th>
                  <th>Auditor</th>
                  <th>Description</th>
                  <th>Corrective Action</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.audits || []).map((a, i) => (
                  <tr key={i} className={a.severity === 'critical' ? 'table-danger' : a.severity === 'high' ? 'table-warning' : ''}>
                    <td className="fw-semibold">{a.audit_id}</td>
                    <td>{a.sop_id}</td>
                    <td>{a.audit_date}</td>
                    <td><StatusBadge status={a.finding_type} colorMap={FINDING_COLOR} /></td>
                    <td><StatusBadge status={a.severity} colorMap={SEVERITY_COLOR} /></td>
                    <td><StatusBadge status={a.status} colorMap={AUDIT_STATUS_COLOR} /></td>
                    <td className="text-muted" style={{ fontSize: '0.72rem' }}>{a.auditor}</td>
                    <td style={{ maxWidth: 200, fontSize: '0.72rem' }}>{a.finding_description}</td>
                    <td style={{ maxWidth: 200, fontSize: '0.72rem' }} className="text-muted">{a.corrective_action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-lg-6 mb-3">
            <div className="card">
              <div className="card-header fw-semibold">Key Concepts</div>
              <div className="card-body p-0">
                <table className="table table-sm small mb-0">
                  <thead className="table-light"><tr><th>Term</th><th>Definition</th></tr></thead>
                  <tbody>
                    {(defs.concepts || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold" style={{ width: 160 }}>{c.term}</td>
                        <td>{c.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="col-lg-6 mb-3">
            <div className="card mb-3">
              <div className="card-header fw-semibold">Severity Levels</div>
              <div className="card-body p-0">
                <table className="table table-sm small mb-0">
                  <thead className="table-light"><tr><th>Level</th><th>Description</th><th>Action</th></tr></thead>
                  <tbody>
                    {(defs.severity_levels || []).map((s, i) => (
                      <tr key={i}>
                        <td><span className={`badge bg-${SEVERITY_COLOR[s.level] || 'secondary'}`}>{s.level}</span></td>
                        <td>{s.description}</td>
                        <td className="text-muted">{s.action}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card">
              <div className="card-header fw-semibold">SOP Categories</div>
              <div className="card-body p-0">
                <table className="table table-sm small mb-0">
                  <thead className="table-light"><tr><th>Category</th><th>Description</th></tr></thead>
                  <tbody>
                    {(defs.sop_categories || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold">{c.category || c.name}</td>
                        <td className="text-muted">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {defs.data_sources && (
            <div className="col-12">
              <div className="card">
                <div className="card-header fw-semibold">Data Sources</div>
                <div className="card-body small">
                  {Array.isArray(defs.data_sources)
                    ? <ul className="mb-0 ps-3">{defs.data_sources.map((s, i) => <li key={i}>{s}</li>)}</ul>
                    : <p className="mb-0">{defs.data_sources}</p>
                  }
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
