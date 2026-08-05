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
  resolved: 'success',
};

export default function ISSopDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);
  const [catFilter, setCatFilter] = useState('All');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/is-sop/overview`).then(r => r.json()),
      fetch(`${API}/api/is-sop/breakdown`).then(r => r.json()),
      fetch(`${API}/api/is-sop/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading IS-SOP Compliance Dashboard...</div>;

  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'procedures', label: 'SOP Procedures' },
    { id: 'audits', label: 'Audit Findings' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const scoreColor = (s) => s >= 90 ? 'success' : s >= 75 ? 'warning' : 'danger';

  const cats = ['All', ...new Set((bd?.procedures || []).map(p => p.category))].sort();
  const filtered = catFilter === 'All'
    ? (bd?.procedures || [])
    : (bd?.procedures || []).filter(p => p.category === catFilter);

  return (
    <div className="p-3">
      <h3>&#x1f4d1; IS-SOP Compliance Dashboard</h3>
      <p className="text-muted small">
        Information Security Standard Operating Procedures — 20 SOPs across 6 domains,
        30 audits, compliance scoring per HIPAA, 21 CFR Part 11, IEC 62443, NIST CSF, ISO 27001.
      </p>

      {/* KPI cards */}
      <div className="row g-2 mb-3">
        {[
          { label: 'Total SOPs', val: ov.total_sops, color: 'primary' },
          { label: 'Published', val: ov.published, color: 'success' },
          { label: 'Under Review', val: ov.under_review, color: 'warning' },
          { label: 'Avg Compliance', val: `${ov.avg_compliance?.toFixed(1)}%`, color: scoreColor(ov.avg_compliance) },
          { label: 'Total Audits', val: ov.total_audits, color: 'info' },
          { label: 'Open Findings', val: ov.open_findings, color: ov.open_findings > 0 ? 'danger' : 'success' },
          { label: 'Closed Findings', val: ov.closed_findings, color: 'secondary' },
          { label: 'Reviews Overdue', val: ov.overdue, color: ov.overdue > 0 ? 'danger' : 'success' },
        ].map(k => (
          <div key={k.label} className="col-6 col-md-3">
            <div className={`card border-${k.color} text-center p-2 h-100`}>
              <div className={`fs-4 fw-bold text-${k.color}`}>{k.val}</div>
              <div className="small text-muted">{k.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Compliance by Category */}
      <div className="row g-2 mb-3">
        {(ov.compliance_by_category || []).map(cat => (
          <div key={cat.category} className="col-6 col-md-4">
            <div className="card p-2 h-100">
              <div className="small fw-bold mb-1">{cat.category}</div>
              <div className="d-flex align-items-center gap-2">
                <div className="progress flex-grow-1" style={{ height: 10 }}>
                  <div
                    className={`progress-bar bg-${scoreColor(cat.avg_score)}`}
                    style={{ width: `${cat.avg_score}%` }}
                  />
                </div>
                <span className={`badge bg-${scoreColor(cat.avg_score)}`}>{cat.avg_score?.toFixed(0)}%</span>
              </div>
              <div className="text-muted" style={{ fontSize: '0.7rem' }}>{cat.count} SOP{cat.count !== 1 ? 's' : ''}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Status distribution badges */}
      <div className="mb-3 d-flex flex-wrap gap-2">
        {(ov.status_distribution || []).map(s => (
          <span key={s.status} className={`badge bg-${STATUS_COLOR[s.status] || 'secondary'} px-3 py-2`}>
            {s.status?.replace(/_/g, ' ')}: {s.count}
          </span>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div>
          <h6 className="text-muted mb-2">Compliance Score by Category</h6>
          <div className="table-responsive">
            <table className="table table-sm table-striped">
              <thead><tr>
                <th>Category</th>
                <th>SOP Count</th>
                <th>Avg Compliance</th>
                <th>Grade</th>
              </tr></thead>
              <tbody>
                {(ov.compliance_by_category || []).sort((a, b) => b.avg_score - a.avg_score).map(cat => (
                  <tr key={cat.category}>
                    <td>{cat.category}</td>
                    <td>{cat.count}</td>
                    <td>
                      <div className="d-flex align-items-center gap-2">
                        <div className="progress flex-grow-1" style={{ height: 8 }}>
                          <div className={`progress-bar bg-${scoreColor(cat.avg_score)}`} style={{ width: `${cat.avg_score}%` }} />
                        </div>
                        <span>{cat.avg_score?.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td>
                      <span className={`badge bg-${scoreColor(cat.avg_score)}`}>
                        {cat.avg_score >= 90 ? 'Excellent' : cat.avg_score >= 75 ? 'Adequate' : 'At Risk'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Findings summary */}
          {ov.findings_by_type && (
            <div className="mt-3">
              <h6 className="text-muted">Audit Findings Breakdown</h6>
              <div className="d-flex flex-wrap gap-2">
                {(ov.findings_by_type || []).map(f => (
                  <div key={f.type} className={`card border-${FINDING_COLOR[f.type] || 'secondary'} p-2 text-center`} style={{ minWidth: 130 }}>
                    <div className={`fs-5 fw-bold text-${FINDING_COLOR[f.type] || 'secondary'}`}>{f.count}</div>
                    <div className="small">{f.type?.replace(/_/g, ' ')}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* SOP Procedures Tab */}
      {tab === 'procedures' && (
        <div>
          <div className="mb-2 d-flex gap-2 align-items-center">
            <span className="text-muted small">Filter by category:</span>
            {cats.map(c => (
              <button
                key={c}
                className={`btn btn-sm ${catFilter === c ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setCatFilter(c)}
              >
                {c}
              </button>
            ))}
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>ID</th>
                  <th>Title</th>
                  <th>Category</th>
                  <th>Status</th>
                  <th>Version</th>
                  <th>Owner</th>
                  <th>Compliance</th>
                  <th>Next Review</th>
                  <th>Standards</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => (
                  <tr key={p.sop_id}>
                    <td><code>{p.sop_id}</code></td>
                    <td style={{ maxWidth: 220 }}><small>{p.title}</small></td>
                    <td><small>{p.category}</small></td>
                    <td>
                      <span className={`badge bg-${STATUS_COLOR[p.status] || 'secondary'}`}>
                        {p.status?.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td>v{p.version}</td>
                    <td><small>{p.owner}</small></td>
                    <td>
                      <span className={`badge bg-${scoreColor(p.compliance_score)}`}>
                        {p.compliance_score}%
                      </span>
                    </td>
                    <td><small>{p.next_review_due}</small></td>
                    <td>
                      {(p.applicable_standards || []).map(s => (
                        <span key={s} className="badge bg-light text-dark me-1" style={{ fontSize: '0.65rem' }}>{s}</span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-muted small mt-1">{filtered.length} of {bd?.procedures?.length || 0} SOPs shown</div>
        </div>
      )}

      {/* Audit Findings Tab */}
      {tab === 'audits' && (
        <div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Audit ID</th>
                  <th>SOP</th>
                  <th>Date</th>
                  <th>Auditor</th>
                  <th>Finding</th>
                  <th>Severity</th>
                  <th>Status</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {(bd?.audits || []).map(a => (
                  <tr key={a.audit_id}>
                    <td><code>{a.audit_id}</code></td>
                    <td><code>{a.sop_id}</code></td>
                    <td><small>{a.audit_date}</small></td>
                    <td><small>{a.auditor}</small></td>
                    <td>
                      <span className={`badge bg-${FINDING_COLOR[a.finding_type] || 'secondary'}`} style={{ fontSize: '0.65rem' }}>
                        {a.finding_type?.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${SEVERITY_COLOR[a.severity] || 'secondary'}`}>
                        {a.severity}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${AUDIT_STATUS_COLOR[a.status] || 'secondary'}`}>
                        {a.status}
                      </span>
                    </td>
                    <td style={{ maxWidth: 260 }}><small>{a.finding_description}</small></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-muted small mt-1">{bd?.audits?.length || 0} audit records</div>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && (
        <div>
          {(defs?.concepts || []).map(c => (
            <div key={c.name} className="card mb-2 border-0 shadow-sm">
              <div className="card-body py-2">
                <div className="fw-bold small">{c.name}</div>
                <div className="text-muted" style={{ fontSize: '0.82rem' }}>{c.description}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
