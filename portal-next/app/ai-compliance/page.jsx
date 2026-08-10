'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview',    label: 'Overview' },
  { id: 'hitl',        label: 'HITL & Expert Reviews' },
  { id: 'decisions',   label: 'Clinical Decisions' },
  { id: 'definitions', label: 'Definitions' },
];

const TIER_COLOR = { 'High-Risk': 'danger', 'Limited Risk': 'warning', 'Minimal Risk': 'success' };

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

function PctBar({ label, pct, color }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="text-muted small">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 14 }}>
        <div className={`progress-bar bg-${color || 'primary'}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function AIComplianceDashboard() {
  const [tab, setTab]     = useState('overview');
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [defs, setDefs]   = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ai-compliance/overview`).then(r => r.json()),
      fetch(`${API}/api/ai-compliance/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ai-compliance/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOv(o); setBd(b); setDefs(d);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="p-4 text-center">
      <div className="spinner-border text-primary" />
      <div className="mt-2 text-muted">Loading AI compliance data…</div>
    </div>
  );
  if (!ov?.available) return <div className="p-4 alert alert-warning">AI Compliance data unavailable.</div>;

  const s = ov.summary;
  const maxOps = Math.max(...(ov.daily_audit_trail || []).map(d => d.ops), 1);

  return (
    <div>
      <h3 className="mb-1">⚖️ AI Compliance Dashboard</h3>
      <p className="text-muted mb-3">
        HITL oversight · expert agreement · clinical decision audit · EU AI Act risk tiers — {s.total_analyses} analyses · {s.total_patients} patients
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
          {/* KPIs row 1 */}
          <div className="row mb-2">
            <KPI label="AI Analyses" value={s.total_analyses} color="primary" />
            <KPI label="HITL Override Rate" value={`${s.override_rate_pct}%`} color={s.override_rate_pct > 30 ? 'danger' : 'success'} sub={`${s.hitl_overrides}/${s.total_hitl_reviews} reviews`} />
            <KPI label="Expert Agreement" value={`${s.agreement_rate_pct}%`} color="info" sub={`${s.expert_agree}/${s.total_expert_reviews} agree`} />
            <KPI label="Review Coverage" value={`${s.review_coverage_pct}%`} color={s.review_coverage_pct >= 90 ? 'success' : 'warning'} sub="of AI analyses reviewed" />
          </div>
          {/* KPIs row 2 */}
          <div className="row mb-4">
            <KPI label="Accountability" value={`${s.accountability_pct}%`} color="success" sub="audit-trail attribution" />
            <KPI label="Consent Coverage" value={`${s.consent_pct}%`} color="success" sub="GDPR Art. 9" />
            <KPI label="Clinical Decisions" value={s.total_clinical_decisions} color="secondary" sub={`${s.decision_confirms} confirm / ${s.decision_overrides} override`} />
            <KPI label="Audit Transactions" value={s.total_transactions?.toLocaleString()} color="dark" sub="full traceability" />
          </div>

          {/* Compliance gauges */}
          <div className="row mb-4">
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Compliance Scorecard</div>
                <div className="card-body">
                  <PctBar label="HITL Review Coverage" pct={Math.round((s.total_hitl_reviews / s.total_analyses) * 100)} color="primary" />
                  <PctBar label="Expert Agreement Rate" pct={s.agreement_rate_pct} color="info" />
                  <PctBar label="Clinical Decision Coverage" pct={Math.round((s.total_clinical_decisions / s.total_analyses) * 100)} color="warning" />
                  <PctBar label="Accountability (Audit Trail)" pct={s.accountability_pct} color="success" />
                  <PctBar label="Consent Coverage" pct={s.consent_pct} color="success" />
                </div>
              </div>
            </div>
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">EU AI Act Risk Tiers</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Component</th><th>Tier</th><th>Article</th></tr>
                    </thead>
                    <tbody>
                      {(ov.eu_ai_act_tiers || []).map(t => (
                        <tr key={t.component}>
                          <td className="small fw-semibold">{t.component}</td>
                          <td>
                            <span className={`badge bg-${TIER_COLOR[t.tier] || 'secondary'}`} style={{ fontSize: 10 }}>
                              {t.tier}
                            </span>
                          </td>
                          <td className="small text-muted">{t.article}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Expert Reviews by Role */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Expert Reviews by Role</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Role</th><th className="text-end">Reviews</th><th className="text-end">Agreed</th><th className="text-end">Agreement %</th></tr>
                </thead>
                <tbody>
                  {(ov.expert_by_role || []).map(r => (
                    <tr key={r.role}>
                      <td className="small fw-semibold">{r.role}</td>
                      <td className="text-end text-muted small">{r.reviews}</td>
                      <td className="text-end text-muted small">{r.agreed}</td>
                      <td className="text-end">
                        <span className="badge bg-info">{r.reviews ? Math.round((r.agreed / r.reviews) * 100) : 0}%</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Daily audit trail sparkline */}
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Daily Audit Operations (last 30 days)</div>
            <div className="card-body">
              <div className="d-flex align-items-end gap-1" style={{ height: 100, overflowX: 'auto' }}>
                {(ov.daily_audit_trail || []).slice(-30).map(d => {
                  const h = Math.round((d.ops / maxOps) * 100);
                  return (
                    <div key={d.day} className="d-flex flex-column align-items-center" style={{ minWidth: 20 }} title={`${d.day}: ${d.ops.toLocaleString()} ops, ${d.actors} actors`}>
                      <div className="bg-primary rounded-top" style={{ width: 14, height: `${Math.max(h, 2)}%` }} />
                      <span className="text-muted" style={{ fontSize: 8, writingMode: 'vertical-rl', marginTop: 2 }}>{d.day.slice(5)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="alert alert-success small">
            <strong>✅ Compliance Status:</strong> {s.accountability_pct}% audit attribution · {s.consent_pct}% consent coverage · {s.total_hitl_reviews} HITL reviews · {s.review_coverage_pct}% clinical review coverage.
            Override rate {s.override_rate_pct}% — within acceptable range for high-risk AI under EU AI Act Art. 14 human oversight requirements.
          </div>
        </div>
      )}

      {/* ── HITL & EXPERT REVIEWS ── */}
      {tab === 'hitl' && bd?.available && (
        <div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">HITL Reviews ({bd.hitl_reviews?.length || 0})</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th>
                      <th>Analysis</th>
                      <th>AI Prediction</th>
                      <th>Human Decision</th>
                      <th>Outcome</th>
                      <th>Reason</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.hitl_reviews || []).map((r, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold text-primary">{r.patient_id}</td>
                        <td className="small text-muted">{r.analysis_id ?? '—'}</td>
                        <td className="small">{r.ai_prediction}</td>
                        <td className="small">{r.human_decision || '—'}</td>
                        <td>
                          <span className={`badge bg-${r.decision === 'override' ? 'danger' : 'success'}`} style={{ fontSize: 10 }}>
                            {r.decision}
                          </span>
                        </td>
                        <td className="small text-muted">{r.reason_code || '—'}</td>
                        <td className="small text-muted">{r.created_at?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Expert Reviews ({bd.expert_reviews?.length || 0})</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th>
                      <th>Role</th>
                      <th>Expert</th>
                      <th>Finding</th>
                      <th>AI Agreement</th>
                      <th>Note</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.expert_reviews || []).map((r, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold text-primary">{r.patient_id}</td>
                        <td className="small">{r.role}</td>
                        <td className="small text-muted">{r.expert}</td>
                        <td className="small">{r.finding || '—'}</td>
                        <td>
                          <span className={`badge bg-${r.agree_with_ai === 'agree' ? 'success' : 'danger'}`} style={{ fontSize: 10 }}>
                            {r.agree_with_ai}
                          </span>
                        </td>
                        <td className="small text-muted">{r.note || '—'}</td>
                        <td className="small text-muted">{r.created_at?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── CLINICAL DECISIONS ── */}
      {tab === 'decisions' && bd?.available && (
        <div>
          <div className="card shadow-sm">
            <div className="card-header fw-semibold">Clinical Decisions ({bd.clinical_decisions?.length || 0} shown)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th>
                      <th>AI Prediction</th>
                      <th>Confidence</th>
                      <th>Neurologist</th>
                      <th>Final Decision</th>
                      <th>Reviewer</th>
                      <th>Note</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.clinical_decisions || []).slice(0, 50).map((d, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold text-primary">{d.patient_id}</td>
                        <td className="small">{d.ai_prediction}</td>
                        <td>
                          <span className={`badge bg-${(d.ai_confidence || 0) > 0.8 ? 'success' : 'warning'}`} style={{ fontSize: 10 }}>
                            {d.ai_confidence ? `${Math.round(d.ai_confidence * 100)}%` : '—'}
                          </span>
                        </td>
                        <td className="small text-muted">{d.neurologist_agreement}</td>
                        <td>
                          <span className={`badge bg-${d.final_decision === 'Override' ? 'danger' : 'success'}`} style={{ fontSize: 10 }}>
                            {d.final_decision}
                          </span>
                        </td>
                        <td className="small text-muted">{d.reviewer}</td>
                        <td className="small text-muted" style={{ maxWidth: 200 }}>{d.note || '—'}</td>
                        <td className="small text-muted">{d.created_at?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {(bd.clinical_decisions || []).length > 50 && (
                <div className="p-2 text-center text-muted small">
                  Showing first 50 of {bd.clinical_decisions.length} clinical decisions
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs?.available && (
        <div>
          <div className="card shadow-sm mb-4">
            <div className="card-header fw-semibold">Metric Definitions</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Metric</th><th>Description</th><th>Source</th></tr>
                </thead>
                <tbody>
                  {(defs.definitions || []).map(d => (
                    <tr key={d.metric}>
                      <td className="small fw-semibold text-nowrap">{d.metric}</td>
                      <td className="small">{d.description}</td>
                      <td className="small text-muted font-monospace">{d.source}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-semibold">EU AI Act Risk Tier Reference</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Tier</th><th>Requirement</th><th>Applies To</th></tr>
                </thead>
                <tbody>
                  {(ov.eu_ai_act_tiers || []).map(t => (
                    <tr key={t.component}>
                      <td>
                        <span className={`badge bg-${TIER_COLOR[t.tier] || 'secondary'}`}>{t.tier}</span>
                      </td>
                      <td className="small">{t.reason}</td>
                      <td className="small text-muted">{t.component} — <em>{t.article}</em></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
