'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s => ({
  'Approved': 'success', 'Submitted': 'primary', 'Under Review': 'warning',
  'Pre-submission': 'secondary', 'Additional Info Requested': 'danger',
}[s] || 'secondary');

const consentColor = s => ({
  granted: 'success', pending: 'warning', declined: 'danger',
  withdrawn: 'secondary', expired: 'info',
}[s] || 'secondary');

const studyStatusColor = s => {
  if (!s) return 'secondary';
  if (s.includes('Completed') || s.includes('Passed')) return 'success';
  if (s.includes('Progress')) return 'primary';
  if (s.includes('Failed')) return 'danger';
  if (s.includes('Planned')) return 'secondary';
  return 'info';
};

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

function Bar({ items, labelKey, valueKey, colorFn }) {
  if (!items || !items.length) return null;
  const mx = Math.max(...items.map(i => i[valueKey]));
  return (
    <div>
      {items.map(item => (
        <div key={item[labelKey]} className="mb-2">
          <div className="d-flex justify-content-between small mb-1">
            <span>{item[labelKey]}</span>
            <span className="fw-bold">{item[valueKey]}</span>
          </div>
          <div className="progress" style={{ height: 10 }}>
            <div
              className={`progress-bar bg-${colorFn ? colorFn(item[labelKey]) : 'primary'}`}
              style={{ width: `${(item[valueKey] / mx) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default function IRBReviewerDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/irb-reviewer/overview`).then(r => r.json()),
      fetch(`${API}/api/irb-reviewer/breakdown`).then(r => r.json()),
      fetch(`${API}/api/irb-reviewer/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="text-center p-5"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis;
  const TABS = [
    { id: 'overview', label: 'Overview' },
    { id: 'consent', label: 'Consent' },
    { id: 'regulatory', label: 'Regulatory' },
    { id: 'validation', label: 'Validation Studies' },
    { id: 'audit', label: 'Audit Trail' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: 28 }} className="me-2">⚖️</span>
        <div>
          <h4 className="mb-0">IRB / Governance Reviewer</h4>
          <div className="text-muted small">Consent · Regulatory Submissions · Validation Evidence · Audit Trail</div>
        </div>
      </div>

      {/* Tab Nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <>
          <div className="row">
            <KPI label="Total Consents" value={kpis.total_consents} color="primary" sub={`${kpis.consented_patients} patients`} />
            <KPI label="Consent Grant Rate" value={`${kpis.consent_grant_rate_pct}%`} color="success" sub={`${kpis.consent_granted} granted`} />
            <KPI label="Regulatory Submissions" value={kpis.total_regulatory_submissions} color="info" sub={`${kpis.submissions_approved} approved`} />
            <KPI label="Avg Validation Score" value={`${kpis.avg_validation_score_pct}%`} color="warning" sub="composite 0–100" />
          </div>
          <div className="row">
            <KPI label="Validation Studies" value={kpis.total_validation_studies} color="secondary" sub={`${kpis.studies_completed} completed/passed`} />
            <KPI label="Avg Sensitivity" value={`${kpis.avg_sensitivity_pct}%`} color="success" sub="completed studies" />
            <KPI label="Avg Specificity" value={`${kpis.avg_specificity_pct}%`} color="info" sub="completed studies" />
            <KPI label="Avg AUC" value={kpis.avg_auc} color="primary" sub="ROC curve" />
          </div>
          <div className="row">
            <KPI label="Audit Events" value={kpis.total_audit_events} color="secondary" sub={`${kpis.unique_actors} actors`} />
            <KPI label="Pending Consents" value={kpis.consent_pending} color="warning" sub="awaiting signature" />
            <KPI label="Non-Consented" value={kpis.consent_non_consented} color="danger" sub="withdrawn or declined" />
            <KPI label="Submissions In Review" value={kpis.submissions_in_review} color="info" sub="submitted/under review" />
          </div>

          <div className="row mt-3">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Consent by Type</div>
                <div className="card-body p-3">
                  <table className="table table-sm table-hover mb-0">
                    <thead><tr>
                      <th>Type</th><th>Granted</th><th>Pending</th><th>Declined</th><th>Withdrawn</th><th>Total</th>
                    </tr></thead>
                    <tbody>
                      {(ov.consent_by_type || []).map(ct => (
                        <tr key={ct.consent_type}>
                          <td>{ct.consent_type?.replace(/_/g, ' ')}</td>
                          <td><span className="badge bg-success">{ct.granted || 0}</span></td>
                          <td><span className="badge bg-warning text-dark">{ct.pending || 0}</span></td>
                          <td><span className="badge bg-danger">{ct.declined || 0}</span></td>
                          <td><span className="badge bg-secondary">{ct.withdrawn || 0}</span></td>
                          <td className="fw-bold">{ct.total}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-3 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Submission Status</div>
                <div className="card-body p-3">
                  <Bar items={ov.submission_status} labelKey="status" valueKey="count" />
                </div>
              </div>
            </div>
            <div className="col-md-3 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Audit by Category</div>
                <div className="card-body p-3">
                  <Bar items={ov.audit_by_category} labelKey="category" valueKey="count" />
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── CONSENT ── */}
      {tab === 'consent' && bd && (
        <>
          <h6 className="mb-3">Per-Patient Consent Summary</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr>
                <th>Patient</th><th>Granted</th><th>Pending</th><th>Total</th><th>Rate</th>
              </tr></thead>
              <tbody>
                {(bd.consent_per_patient || []).map(p => (
                  <tr key={p.patient_id}>
                    <td><code>{p.patient_id}</code></td>
                    <td><span className="badge bg-success">{p.granted}</span></td>
                    <td><span className="badge bg-warning text-dark">{p.pending}</span></td>
                    <td>{p.total}</td>
                    <td>
                      <div className="progress" style={{ height: 8, width: 80 }}>
                        <div className="progress-bar bg-success"
                          style={{ width: `${(p.granted / p.total) * 100}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── REGULATORY ── */}
      {tab === 'regulatory' && bd && (
        <>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold">By Pathway</div>
                <div className="card-body p-3">
                  <Bar items={bd.by_pathway} labelKey="pathway" valueKey="count" />
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-header fw-bold">By Risk Class</div>
                <div className="card-body p-3">
                  <Bar items={bd.by_risk_class} labelKey="risk_class" valueKey="count" />
                </div>
              </div>
            </div>
          </div>
          <h6>All Regulatory Submissions</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr>
                <th>ID</th><th>Pathway</th><th>Product</th><th>Risk Class</th>
                <th>Phase</th><th>Status</th><th>Val. Score</th><th>Reviewer</th>
              </tr></thead>
              <tbody>
                {(bd.submissions || []).map(s => (
                  <tr key={s.submission_id}>
                    <td><code className="small">{s.submission_id}</code></td>
                    <td className="small">{s.pathway}</td>
                    <td className="small">{s.product_name}</td>
                    <td><span className="badge bg-secondary">{s.risk_class}</span></td>
                    <td className="small">{s.phase}</td>
                    <td><span className={`badge bg-${statusColor(s.status)}`}>{s.status}</span></td>
                    <td>{s.validation_score_pct != null ? `${s.validation_score_pct}%` : '—'}</td>
                    <td className="small">{s.reviewer}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── VALIDATION STUDIES ── */}
      {tab === 'validation' && bd && (
        <>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">By Study Type</div>
                <div className="card-body p-3">
                  <Bar items={bd.by_study_type} labelKey="study_type" valueKey="count" />
                </div>
              </div>
            </div>
          </div>
          <h6>All Validation Studies</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr>
                <th>Study ID</th><th>Type</th><th>Status</th><th>N</th>
                <th>Sens.</th><th>Spec.</th><th>AUC</th><th>PI</th><th>Site</th>
              </tr></thead>
              <tbody>
                {(bd.studies || []).map(s => (
                  <tr key={s.study_id}>
                    <td><code className="small">{s.study_id}</code></td>
                    <td className="small">{s.study_type}</td>
                    <td><span className={`badge bg-${studyStatusColor(s.status)}`}>{s.status}</span></td>
                    <td>{s.sample_size ?? '—'}</td>
                    <td>{s.sensitivity != null ? `${(s.sensitivity * 100).toFixed(1)}%` : '—'}</td>
                    <td>{s.specificity != null ? `${(s.specificity * 100).toFixed(1)}%` : '—'}</td>
                    <td>{s.auc_roc != null ? s.auc_roc.toFixed(3) : '—'}</td>
                    <td className="small">{s.principal_investigator}</td>
                    <td className="small">{s.site}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── AUDIT TRAIL ── */}
      {tab === 'audit' && bd && (
        <>
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Top Actors</div>
                <div className="card-body p-3">
                  <Bar items={bd.top_actors} labelKey="actor" valueKey="events" />
                </div>
              </div>
            </div>
          </div>
          <h6>Recent Audit Events</h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead><tr>
                <th>Submission</th><th>Action</th><th>Actor</th><th>Timestamp</th>
                <th>Category</th><th>Document</th>
              </tr></thead>
              <tbody>
                {(bd.recent_audit || []).map((e, i) => (
                  <tr key={i}>
                    <td><code className="small">{e.submission_id}</code></td>
                    <td className="small">{e.action}</td>
                    <td className="small">{e.actor}</td>
                    <td className="small text-muted">{e.timestamp?.slice(0, 16)}</td>
                    <td><span className="badge bg-info text-dark">{e.category}</span></td>
                    <td className="small text-muted">{e.document_ref}</td>
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
          {[
            ['Consent Statuses', defs.consent_statuses],
            ['Regulatory Pathways', defs.regulatory_pathways],
            ['Risk Classes', defs.risk_classes],
            ['Validation Study Types', defs.validation_study_types],
            ['Audit Categories', defs.audit_categories],
            ['Key Metrics', defs.key_metrics],
          ].map(([title, obj]) => obj && (
            <div key={title} className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold">{title}</div>
                <div className="card-body p-3">
                  {Object.entries(obj).map(([k, v]) => (
                    <div key={k} className="mb-2">
                      <span className="badge bg-secondary me-2">{k.replace(/_/g, ' ')}</span>
                      <span className="text-muted small">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
          <div className="col-12">
            <div className="alert alert-info small">
              <strong>Mission:</strong> {defs.mission}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
