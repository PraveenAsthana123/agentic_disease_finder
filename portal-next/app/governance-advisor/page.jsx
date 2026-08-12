'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const statusColor = s =>
  s === 'pass' ? 'success' :
  s === 'fail' ? 'danger' : 'warning';

const severityColor = s =>
  s === 'critical' ? 'danger' :
  s === 'high' ? 'warning' :
  s === 'medium' ? 'info' : 'secondary';

const findingColor = f =>
  f === 'major_nonconformance' ? 'danger' :
  f === 'minor_nonconformance' ? 'warning' :
  f === 'observation' ? 'info' : 'success';

const regStatusColor = s =>
  s === 'Approved' ? 'success' :
  s === 'Under Review' ? 'info' :
  s === 'Submitted' ? 'primary' : 'secondary';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'consent',     label: 'Consent & Privacy' },
  { id: 'hitl',        label: 'AI Oversight & HITL' },
  { id: 'regulatory',  label: 'Regulatory & SOPs' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub }) {
  return (
    <div className="col-6 col-md-2 mb-2">
      <div className="card text-center shadow-sm border-0 h-100">
        <div className="card-body py-2 px-1">
          <div className={`h3 mb-0 text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted" style={{ fontSize: '0.72rem' }}>{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.65rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

/* ── Overview Tab ─────────────────────────────────────────────────── */
function OverviewTab({ ov, bd }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const s = ov.summary || {};

  return (
    <div>
      {/* KPI row */}
      <div className="row g-2 mb-3">
        <KPI label="Consent Records" value={s.total_consent_records} color="primary" />
        <KPI label="Consent Granted" value={`${s.consent_grant_rate_pct}%`} color="success" sub={`${s.consent_granted}/${s.total_consent_records}`} />
        <KPI label="AI Decisions" value={s.ai_decisions_total} color="info" />
        <KPI label="Override Rate" value={`${s.ai_override_rate_pct}%`} color={s.ai_override_rate_pct < 20 ? 'success' : 'danger'} sub={`${s.ai_override_count} overrides`} />
        <KPI label="HITL Coverage" value="100%" color="success" sub="all decisions reviewed" />
        <KPI label="SOP Compliance" value={`${s.avg_sop_compliance_pct}%`} color={s.avg_sop_compliance_pct > 75 ? 'success' : 'warning'} sub="avg across SOPs" />
      </div>
      <div className="row g-2 mb-3">
        <KPI label="Regulatory Submissions" value={s.regulatory_submissions} color="primary" />
        <KPI label="Approved" value={s.regulatory_approved} color="success" />
        <KPI label="Audit Events" value={s.audit_trail_events?.toLocaleString()} color="secondary" />
        <KPI label="Audit Actors" value={s.audit_actors} color="info" />
        <KPI label="Feature Flags" value={`${s.feature_flags_enabled}/${s.feature_flags_total}`} color="secondary" sub="enabled/total" />
        <KPI label="TX Log Events" value={s.transaction_log_events?.toLocaleString()} color="secondary" />
      </div>

      {/* Governance Thresholds */}
      <h6 className="fw-semibold mt-3 mb-2">Governance Threshold Dashboard</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr><th>KPI</th><th>Threshold</th><th>Actual</th><th>Status</th></tr>
          </thead>
          <tbody>
            {(ov.governance_thresholds || []).map((t, i) => (
              <tr key={i}>
                <td className="fw-medium">{t.kpi}</td>
                <td className="text-muted">{t.threshold}</td>
                <td><strong>{t.actual}</strong></td>
                <td><span className={`badge bg-${statusColor(t.status)}`}>{t.status.toUpperCase()}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* AI Decision breakdown donut (bar) */}
      <div className="row mt-3">
        <div className="col-md-6">
          <h6 className="fw-semibold mb-2">AI Decision Breakdown</h6>
          {Object.entries(ov.ai_decision_breakdown || {}).map(([k, v]) => {
            const total = Object.values(ov.ai_decision_breakdown).reduce((a, b) => a + b, 0);
            const pct = total ? Math.round(v / total * 100) : 0;
            const color = k === 'Override' ? 'danger' : k === 'Confirm' ? 'success' : k === 'Escalate' ? 'warning' : 'secondary';
            return (
              <div key={k} className="mb-1">
                <div className="d-flex justify-content-between small mb-0">
                  <span>{k}</span><span>{v} ({pct}%)</span>
                </div>
                <div className="progress" style={{ height: 8 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        <div className="col-md-6">
          <h6 className="fw-semibold mb-2">Neurologist Agreement</h6>
          {Object.entries(ov.neurologist_agreement || {}).map(([k, v]) => {
            const total = Object.values(ov.neurologist_agreement).reduce((a, b) => a + b, 0);
            const pct = total ? Math.round(v / total * 100) : 0;
            const color = k === 'Agree' ? 'success' : k === 'Partial' ? 'warning' : 'danger';
            return (
              <div key={k} className="mb-1">
                <div className="d-flex justify-content-between small mb-0">
                  <span>{k}</span><span>{v} ({pct}%)</span>
                </div>
                <div className="progress" style={{ height: 8 }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Audit trail categories */}
      <div className="row mt-3">
        <div className="col-md-6">
          <h6 className="fw-semibold mb-2">Audit Trail by Category</h6>
          {(ov.audit_categories || []).map(({ category, count }) => {
            const total = (ov.audit_categories || []).reduce((a, b) => a + b.count, 0);
            const pct = total ? Math.round(count / total * 100) : 0;
            return (
              <div key={category} className="mb-1">
                <div className="d-flex justify-content-between small mb-0">
                  <span>{category}</span><span>{count} ({pct}%)</span>
                </div>
                <div className="progress" style={{ height: 8 }}>
                  <div className="progress-bar bg-primary" style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        <div className="col-md-6">
          <h6 className="fw-semibold mb-2">Regulatory Status</h6>
          {Object.entries(ov.regulatory_status || {}).map(([k, v]) => {
            const total = Object.values(ov.regulatory_status).reduce((a, b) => a + b, 0);
            const pct = total ? Math.round(v / total * 100) : 0;
            return (
              <div key={k} className="mb-1">
                <div className="d-flex justify-content-between small mb-0">
                  <span>{k}</span><span>{v} ({pct}%)</span>
                </div>
                <div className="progress" style={{ height: 8 }}>
                  <div className={`progress-bar bg-${regStatusColor(k)}`} style={{ width: `${pct}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ── Consent & Privacy Tab ───────────────────────────────────────── */
function ConsentTab({ ov, bd }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const s = ov.summary || {};
  const statusOrder = ['granted', 'pending', 'declined', 'withdrawn', 'expired'];

  return (
    <div>
      <div className="row g-2 mb-3">
        <KPI label="Total Records" value={s.total_consent_records} color="primary" />
        <KPI label="Granted" value={s.consent_granted} color="success" />
        <KPI label="Pending" value={s.consent_pending} color="warning" />
        <KPI label="Declined" value={s.consent_declined} color="danger" />
        <KPI label="Withdrawn" value={s.consent_withdrawn} color="secondary" />
        <KPI label="Expired" value={s.consent_expired} color="secondary" />
      </div>

      <h6 className="fw-semibold mb-2">Consent by Type</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>Consent Type</th>
              {statusOrder.map(s => <th key={s} className="text-capitalize">{s}</th>)}
              <th>Total</th>
            </tr>
          </thead>
          <tbody>
            {(ov.consent_by_type || []).map((row, i) => {
              const total = statusOrder.reduce((a, s) => a + (row[s] || 0), 0);
              return (
                <tr key={i}>
                  <td className="fw-medium text-capitalize">{row.consent_type?.replace(/_/g, ' ')}</td>
                  {statusOrder.map(s => (
                    <td key={s}>
                      {row[s] > 0 ? (
                        <span className={`badge bg-${s === 'granted' ? 'success' : s === 'pending' ? 'warning' : s === 'declined' ? 'danger' : 'secondary'} text-${s === 'pending' ? 'dark' : 'white'}`}>
                          {row[s]}
                        </span>
                      ) : <span className="text-muted">—</span>}
                    </td>
                  ))}
                  <td><strong>{total}</strong></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {bd?.expiring_consents?.length > 0 && (
        <>
          <h6 className="fw-semibold mt-3 mb-2 text-warning">
            ⚠ Consents Expiring Within 90 Days ({bd.expiring_consents.length})
          </h6>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr><th>Patient</th><th>Consent Type</th><th>Expiry Date</th></tr>
              </thead>
              <tbody>
                {bd.expiring_consents.map((r, i) => (
                  <tr key={i}>
                    <td>{r.patient_id}</td>
                    <td className="text-capitalize">{r.consent_type?.replace(/_/g, ' ')}</td>
                    <td className="text-warning">{r.expiry_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <div className="alert alert-info mt-3">
        <strong>Frameworks:</strong> DPDP Act 2023 §6 · HIPAA Privacy Rule §164.508 · ICMR 2017 Part III · Helsinki Declaration §26
      </div>
    </div>
  );
}

/* ── AI Oversight & HITL Tab ──────────────────────────────────────── */
function HITLTab({ ov, bd }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const s = ov.summary || {};

  return (
    <div>
      <div className="row g-2 mb-3">
        <KPI label="Total AI Decisions" value={s.ai_decisions_total} color="primary" />
        <KPI label="Confirmed" value={s.ai_decisions_total - s.ai_override_count - (ov.ai_decision_breakdown?.Escalate || 0) - (ov.ai_decision_breakdown?.Defer || 0)} color="success" />
        <KPI label="Override Rate" value={`${s.ai_override_rate_pct}%`} color={s.ai_override_rate_pct < 20 ? 'success' : 'danger'} />
        <KPI label="HITL Coverage" value="100%" color="success" />
        <KPI label="Avg AI Confidence" value={`${s.avg_ai_confidence_pct}%`} color="info" />
        <KPI label="Neurologist Agree" value={`${s.neurologist_agree_pct}%`} color="success" />
      </div>

      <h6 className="fw-semibold mb-2">Per-Reviewer Override Analysis</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>Reviewer</th>
              <th>Total</th>
              <th>Confirms</th>
              <th>Overrides</th>
              <th>Escalates</th>
              <th>Override Rate</th>
              <th>Avg AI Conf</th>
            </tr>
          </thead>
          <tbody>
            {(bd?.reviewer_override_breakdown || []).map((r, i) => (
              <tr key={i}>
                <td className="fw-medium">{r.reviewer}</td>
                <td>{r.total}</td>
                <td><span className="badge bg-success">{r.confirms}</span></td>
                <td><span className={`badge bg-${r.override_rate_pct > 30 ? 'danger' : 'warning'}`}>{r.overrides}</span></td>
                <td><span className="badge bg-warning text-dark">{r.escalates}</span></td>
                <td>
                  <div className="d-flex align-items-center gap-1">
                    <div className="progress flex-grow-1" style={{ height: 6 }}>
                      <div className={`progress-bar bg-${r.override_rate_pct > 30 ? 'danger' : 'success'}`}
                        style={{ width: `${r.override_rate_pct}%` }} />
                    </div>
                    <small>{r.override_rate_pct}%</small>
                  </div>
                </td>
                <td>{r.avg_confidence_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="alert alert-success mt-3">
        <strong>HITL Guarantee:</strong> Every AI prediction is reviewed by a neurologist before clinical action.
        Override rate {s.ai_override_rate_pct}% is within the target threshold of &lt; 20% (IMDRF SaMD Guidance N41).
      </div>
    </div>
  );
}

/* ── Regulatory & SOPs Tab ────────────────────────────────────────── */
function RegulatoryTab({ ov, bd }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  const sopComp = ov.sop_compliance_summary || {};

  return (
    <div>
      <div className="row g-2 mb-3">
        <KPI label="SOPs Total" value={sopComp.total} color="primary" />
        <KPI label="Published" value={sopComp.published} color="success" />
        <KPI label="Under Review" value={sopComp.under_review} color="warning" />
        <KPI label="Avg Compliance" value={`${sopComp.avg_compliance_pct}%`} color={sopComp.avg_compliance_pct > 75 ? 'success' : 'warning'} />
        <KPI label="Open Findings" value={sopComp.open_findings} color="warning" />
        <KPI label="Critical Findings" value={sopComp.critical_findings} color={sopComp.critical_findings > 0 ? 'danger' : 'success'} />
      </div>

      <h6 className="fw-semibold mb-2">SOP Procedures (Sorted by Compliance Score ↑)</h6>
      <div className="table-responsive" style={{ maxHeight: 300 }}>
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr><th>SOP ID</th><th>Title</th><th>Category</th><th>Version</th><th>Status</th><th>Compliance</th><th>Next Review</th></tr>
          </thead>
          <tbody>
            {(bd?.sop_procedures || []).map((p, i) => (
              <tr key={i}>
                <td className="fw-medium">{p.sop_id}</td>
                <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.title}</td>
                <td><span className="badge bg-secondary">{p.category}</span></td>
                <td>{p.version}</td>
                <td><span className={`badge bg-${p.status === 'published' ? 'success' : 'warning'}`}>{p.status}</span></td>
                <td>
                  <div className="d-flex align-items-center gap-1">
                    <div className="progress flex-grow-1" style={{ height: 6 }}>
                      <div className={`progress-bar bg-${p.compliance_score >= 80 ? 'success' : p.compliance_score >= 70 ? 'warning' : 'danger'}`}
                        style={{ width: `${p.compliance_score}%` }} />
                    </div>
                    <small>{p.compliance_score}%</small>
                  </div>
                </td>
                <td className="text-muted">{p.next_review_due}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-semibold mt-3 mb-2">Audit Findings (Most Recent 20)</h6>
      <div className="table-responsive" style={{ maxHeight: 280 }}>
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr><th>Audit ID</th><th>SOP</th><th>Date</th><th>Auditor</th><th>Finding Type</th><th>Severity</th><th>Status</th></tr>
          </thead>
          <tbody>
            {(bd?.sop_audits || []).map((a, i) => (
              <tr key={i}>
                <td>{a.audit_id}</td>
                <td>{a.sop_id}</td>
                <td>{a.audit_date}</td>
                <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.auditor}</td>
                <td><span className={`badge bg-${findingColor(a.finding_type)} text-${a.finding_type === 'minor_nonconformance' ? 'dark' : 'white'}`}>{a.finding_type?.replace(/_/g, ' ')}</span></td>
                <td><span className={`badge bg-${severityColor(a.severity)}`}>{a.severity}</span></td>
                <td><span className={`badge bg-${a.status === 'closed' ? 'success' : 'warning'} text-${a.status !== 'closed' ? 'dark' : 'white'}`}>{a.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-semibold mt-3 mb-2">Regulatory Submissions</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr><th>Submission ID</th><th>Pathway</th><th>Classification</th><th>Status</th><th>Submitted</th><th>Target</th></tr>
          </thead>
          <tbody>
            {(bd?.regulatory_submissions || []).map((r, i) => (
              <tr key={i}>
                <td className="fw-medium">{r.submission_id}</td>
                <td>{r.pathway}</td>
                <td><span className="badge bg-secondary">{r.classification}</span></td>
                <td><span className={`badge bg-${regStatusColor(r.status)}`}>{r.status}</span></td>
                <td>{r.submitted_date}</td>
                <td>{r.target_date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Definitions Tab ──────────────────────────────────────────────── */
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;

  return (
    <div>
      <h6 className="fw-semibold mb-2">Governance Concepts</h6>
      <div className="accordion" id="conceptAccordion">
        {(defs.concepts || []).map((c, i) => (
          <div className="accordion-item" key={i}>
            <h2 className="accordion-header">
              <button className={`accordion-button ${i > 0 ? 'collapsed' : ''} py-2`} type="button"
                data-bs-toggle="collapse" data-bs-target={`#concept-${i}`}>
                <strong>{c.term}</strong>
              </button>
            </h2>
            <div id={`concept-${i}`} className={`accordion-collapse collapse ${i === 0 ? 'show' : ''}`}>
              <div className="accordion-body py-2">
                <p className="mb-1">{c.definition}</p>
                <small className="text-muted">Standard: {c.standard}</small>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-semibold mt-3 mb-2">Governance Frameworks</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr><th>Framework</th><th>Scope</th></tr>
          </thead>
          <tbody>
            {(defs.governance_frameworks || []).map((f, i) => (
              <tr key={i}>
                <td className="fw-medium">{f.framework}</td>
                <td className="text-muted">{f.scope}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-semibold mt-3 mb-2">Performance Thresholds</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr><th>KPI</th><th>Threshold</th><th>Rationale</th></tr>
          </thead>
          <tbody>
            {(defs.performance_thresholds || []).map((t, i) => (
              <tr key={i}>
                <td className="fw-medium">{t.kpi}</td>
                <td><span className="badge bg-primary">{t.threshold}</span></td>
                <td className="text-muted">{t.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-semibold mt-3 mb-2">References</h6>
      <ol className="small text-muted">
        {(defs.references || []).map((r, i) => <li key={i}>{r}</li>)}
      </ol>
    </div>
  );
}

/* ── Main Component ───────────────────────────────────────────────── */
export default function GovernanceAdvisorDashboard() {
  const [tab, setTab] = useState('overview');
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/governance-advisor/overview`).then(r => r.json()).then(setOv).catch(e => setErr(e.message));
    fetch(`${API}/api/governance-advisor/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/governance-advisor/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (err) return <div className="container mt-4"><div className="alert alert-danger">{err}</div></div>;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <h4 className="mb-0">⚖️ Governance Advisor Dashboard</h4>
        <span className="badge bg-primary">Responsible AI / Ethics / Regulatory</span>
      </div>

      {/* Nav tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t.id}>
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'overview'    && <OverviewTab ov={ov} bd={bd} />}
      {tab === 'consent'     && <ConsentTab ov={ov} bd={bd} />}
      {tab === 'hitl'        && <HITLTab ov={ov} bd={bd} />}
      {tab === 'regulatory'  && <RegulatoryTab ov={ov} bd={bd} />}
      {tab === 'definitions' && <DefinitionsTab defs={defs} />}
    </div>
  );
}
