'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const agreeColor = v =>
  v === 'agree' ? 'success' : v === 'disagree' ? 'danger' : 'warning';

const decisionColor = d =>
  d === 'Confirm' || d === 'accept' ? 'success'
  : d === 'Override' || d === 'override' ? 'danger'
  : d === 'Escalate' ? 'warning'
  : d === 'Defer' ? 'secondary'
  : 'info';

const pct = (n, t) => (t ? ((n / t) * 100).toFixed(1) : '0.0');

export default function ModelGovernanceDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/model-governance/overview`).then(r => r.json()),
      fetch(`${API}/api/model-governance/breakdown`).then(r => r.json()),
      fetch(`${API}/api/model-governance/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = ov.kpis || {};
  const tabs = [
    { id: 'overview',   label: '📊 Overview' },
    { id: 'consultant', label: '🧑‍⚕️ Consultant Matrix' },
    { id: 'decisions',  label: '⚖️ Decision Chain' },
    { id: 'hitl',       label: '🙋 HITL Reviews' },
    { id: 'definitions',label: '📖 Definitions' },
  ];

  return (
    <div>
      <h3>🏛️ Model Governance Dashboard</h3>
      <p className="text-muted">
        HITL sign-off chain, expert consultant agreement, AI prediction override audit,
        and model lifecycle governance — across {k.total_analyses} AI analyses.
      </p>

      {/* KPI Cards */}
      <div className="row g-2 mb-3">
        {[
          { label: 'AI Analyses', val: k.total_analyses, color: 'primary' },
          { label: 'Clinical Decisions', val: k.total_clinical_decisions, color: 'info' },
          { label: 'HITL Reviews', val: k.total_hitl_reviews, color: 'warning' },
          { label: 'Expert Reviews', val: k.total_expert_reviews, color: 'secondary' },
          { label: 'Sign-Off Rate', val: `${k.sign_off_rate?.toFixed(1)}%`, color: 'success' },
          { label: 'Override Rate', val: `${k.override_rate?.toFixed(1)}%`, color: 'danger' },
          { label: 'Expert Agreement', val: `${k.expert_agreement_rate?.toFixed(1)}%`, color: 'primary' },
          { label: 'Avg Confidence', val: `${((k.avg_confidence || 0) * 100).toFixed(0)}%`, color: 'info' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3">
            <div className={`card border-${c.color} shadow-sm h-100`}>
              <div className="card-body text-center py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.val}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* Overview Tab */}
      {tab === 'overview' && (
        <div className="row g-3">
          {/* Sign-Off Chain */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">⚖️ HITL Sign-Off Chain</div>
              <div className="card-body">
                {(ov.sign_off_chain || []).map(s => (
                  <div key={s.decision} className="d-flex justify-content-between align-items-center mb-2">
                    <span className={`badge bg-${decisionColor(s.decision)} fs-6`}>{s.decision}</span>
                    <strong>{s.count}</strong>
                  </div>
                ))}
                <div className="mt-2 small text-muted">
                  Accept: {ov.sign_off_chain?.find(s => s.decision === 'accept')?.count || 0} ·
                  Override: {ov.sign_off_chain?.find(s => s.decision === 'override')?.count || 0}
                </div>
              </div>
            </div>
          </div>

          {/* Model Lifecycle by Disease */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">🧠 Model Lifecycle by Disease</div>
              <div className="card-body">
                {(ov.model_lifecycle || []).map(m => (
                  <div key={m.disease} className="mb-2">
                    <div className="d-flex justify-content-between small mb-1">
                      <span className="text-capitalize">{m.disease.replace('_', ' ')}</span>
                      <strong>{m.analyses}</strong>
                    </div>
                    <div className="progress" style={{ height: 8 }}>
                      <div
                        className="progress-bar bg-primary"
                        style={{ width: `${pct(m.analyses, k.total_analyses)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Governance Timeline (recent) */}
          <div className="col-md-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">📅 Recent Governance Events</div>
              <div className="card-body p-0">
                <ul className="list-group list-group-flush" style={{ maxHeight: 280, overflowY: 'auto' }}>
                  {(ov.governance_timeline || []).slice(-10).reverse().map((ev, i) => (
                    <li key={i} className="list-group-item py-1 small">
                      <div className="d-flex justify-content-between">
                        <span className="text-muted">{ev.date}</span>
                        <span className={`badge bg-${decisionColor(ev.final_decision || ev.decision || ev.agree_with_ai)}`}>
                          {ev.final_decision || ev.decision || ev.agree_with_ai || ev.type}
                        </span>
                      </div>
                      <div>{ev.patient_id} — <em className="text-muted">{ev.type?.replace('_', ' ')}</em></div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Consultant Matrix Tab */}
      {tab === 'consultant' && bd && (
        <div>
          <h5 className="mb-3">Expert Reviewer Agreement Matrix</h5>
          <div className="table-responsive mb-4">
            <table className="table table-bordered table-hover small">
              <thead className="table-dark">
                <tr>
                  <th>Role</th>
                  <th>Expert</th>
                  <th>Reviews</th>
                  <th>Agree</th>
                  <th>Disagree</th>
                  <th>Agreement Rate</th>
                </tr>
              </thead>
              <tbody>
                {(ov.consultant_matrix || []).map((c, i) => (
                  <tr key={i}>
                    <td>{c.role}</td>
                    <td><strong>{c.expert}</strong></td>
                    <td>{c.total_reviews}</td>
                    <td><span className="badge bg-success">{c.agree}</span></td>
                    <td><span className="badge bg-danger">{c.disagree}</span></td>
                    <td>
                      <div className="progress" style={{ height: 10, minWidth: 80 }}>
                        <div
                          className={`progress-bar bg-${c.agreement_rate >= 70 ? 'success' : 'danger'}`}
                          style={{ width: `${c.agreement_rate}%` }}
                        />
                      </div>
                      <small>{c.agreement_rate?.toFixed(1)}%</small>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h5 className="mb-3">Expert Review Detail</h5>
          <div className="table-responsive">
            <table className="table table-sm table-hover small">
              <thead className="table-secondary">
                <tr>
                  <th>Date</th>
                  <th>Patient</th>
                  <th>Role</th>
                  <th>Expert</th>
                  <th>Finding</th>
                  <th>AI Agreement</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {(bd.expert_detail || []).map((e, i) => (
                  <tr key={i}>
                    <td>{e.created_at?.slice(0, 10)}</td>
                    <td><code>{e.patient_id}</code></td>
                    <td>{e.role}</td>
                    <td>{e.expert}</td>
                    <td className="small">{e.finding}</td>
                    <td>
                      <span className={`badge bg-${agreeColor(e.agree_with_ai)}`}>
                        {e.agree_with_ai}
                      </span>
                    </td>
                    <td className="text-muted">{e.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Decision Chain Tab */}
      {tab === 'decisions' && bd && (
        <div>
          <h5 className="mb-3">Clinical Decision Audit Chain
            <span className="badge bg-secondary ms-2">{(bd.decision_chain || []).length} decisions</span>
          </h5>
          <div className="table-responsive">
            <table className="table table-bordered table-hover small">
              <thead className="table-dark">
                <tr>
                  <th>Date</th>
                  <th>Patient</th>
                  <th>AI Prediction</th>
                  <th>Confidence</th>
                  <th>Neuro Agreement</th>
                  <th>Final Decision</th>
                  <th>Reviewer</th>
                  <th>Artifact Risk</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {(bd.decision_chain || []).map((d, i) => (
                  <tr key={i}>
                    <td>{d.created_at?.slice(0, 10)}</td>
                    <td><code>{d.patient_id}</code></td>
                    <td>{d.ai_prediction}</td>
                    <td>
                      <span className={`badge bg-${d.ai_confidence >= 0.8 ? 'success' : d.ai_confidence >= 0.6 ? 'warning' : 'danger'}`}>
                        {((d.ai_confidence || 0) * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${agreeColor(d.neurologist_agreement?.toLowerCase())}`}>
                        {d.neurologist_agreement}
                      </span>
                    </td>
                    <td>
                      <span className={`badge bg-${decisionColor(d.final_decision)}`}>
                        {d.final_decision}
                      </span>
                    </td>
                    <td>{d.reviewer}</td>
                    <td>
                      {d.artifact_risk && (
                        <span className={`badge bg-${d.artifact_risk === 'None' ? 'secondary' : d.artifact_risk === 'Low' ? 'info' : 'warning'}`}>
                          {d.artifact_risk}
                        </span>
                      )}
                    </td>
                    <td className="text-muted small" style={{ maxWidth: 200 }}>{d.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Role Agreement Matrix */}
          {bd.role_agreement_matrix && (
            <div className="mt-4">
              <h5>Role-Level Agreement Summary</h5>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead className="table-secondary">
                    <tr>
                      <th>Role</th>
                      <th>Reviews</th>
                      <th>Agree</th>
                      <th>Partial</th>
                      <th>Disagree</th>
                      <th>Agree Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.role_agreement_matrix || []).map((r, i) => (
                      <tr key={i}>
                        <td>{r.role || r.neurologist_agreement}</td>
                        <td>{r.total}</td>
                        <td><span className="badge bg-success">{r.agree}</span></td>
                        <td><span className="badge bg-warning text-dark">{r.partial}</span></td>
                        <td><span className="badge bg-danger">{r.disagree}</span></td>
                        <td>
                          <strong>{r.agree_rate?.toFixed(1)}%</strong>
                          <div className="progress mt-1" style={{ height: 6 }}>
                            <div className="progress-bar bg-success" style={{ width: `${r.agree_rate}%` }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* HITL Reviews Tab */}
      {tab === 'hitl' && bd && (
        <div>
          <h5 className="mb-3">Human-in-the-Loop Review Log</h5>
          <div className="row g-3 mb-3">
            <div className="col-md-3">
              <div className="card border-success shadow-sm text-center">
                <div className="card-body py-2">
                  <div className="h4 text-success">{bd.hitl_detail?.filter(h => h.decision === 'accept').length || 0}</div>
                  <div className="small text-muted">Accepted</div>
                </div>
              </div>
            </div>
            <div className="col-md-3">
              <div className="card border-danger shadow-sm text-center">
                <div className="card-body py-2">
                  <div className="h4 text-danger">{bd.hitl_detail?.filter(h => h.decision === 'override').length || 0}</div>
                  <div className="small text-muted">Overridden</div>
                </div>
              </div>
            </div>
            <div className="col-md-3">
              <div className="card border-info shadow-sm text-center">
                <div className="card-body py-2">
                  <div className="h4 text-info">{bd.hitl_detail?.length || 0}</div>
                  <div className="small text-muted">Total HITL</div>
                </div>
              </div>
            </div>
            <div className="col-md-3">
              <div className="card border-primary shadow-sm text-center">
                <div className="card-body py-2">
                  <div className="h4 text-primary">{k.sign_off_rate?.toFixed(0)}%</div>
                  <div className="small text-muted">Sign-Off Rate</div>
                </div>
              </div>
            </div>
          </div>

          <div className="table-responsive">
            <table className="table table-bordered small">
              <thead className="table-dark">
                <tr>
                  <th>Date</th>
                  <th>Patient</th>
                  <th>Analysis ID</th>
                  <th>AI Prediction</th>
                  <th>HITL Decision</th>
                  <th>Human Decision</th>
                  <th>Reason</th>
                  <th>Reviewer</th>
                </tr>
              </thead>
              <tbody>
                {(bd.hitl_detail || []).map((h, i) => (
                  <tr key={i}>
                    <td>{h.created_at?.slice(0, 10)}</td>
                    <td><code>{h.patient_id}</code></td>
                    <td>{h.analysis_id || '—'}</td>
                    <td>{h.ai_prediction}</td>
                    <td>
                      <span className={`badge bg-${decisionColor(h.decision)}`}>{h.decision}</span>
                    </td>
                    <td>{h.human_decision || '—'}</td>
                    <td><code>{h.reason_code || '—'}</code></td>
                    <td>{h.reviewer_id || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Feedback Log */}
          {(bd.feedback_log || []).length > 0 && (
            <div className="mt-4">
              <h5>Feedback Log</h5>
              <div className="table-responsive">
                <table className="table table-sm small">
                  <thead className="table-secondary">
                    <tr><th>Date</th><th>Patient</th><th>Rating</th><th>Comment</th></tr>
                  </thead>
                  <tbody>
                    {(bd.feedback_log || []).map((f, i) => (
                      <tr key={i}>
                        <td>{f.created_at?.slice(0, 10)}</td>
                        <td><code>{f.patient_id}</code></td>
                        <td>{'⭐'.repeat(f.rating || 0)}</td>
                        <td>{f.comment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'definitions' && defs && (
        <div>
          {(defs.sections || []).map((sec, si) => (
            <div key={si} className="mb-4">
              <h5 className="border-bottom pb-1">{sec.title}</h5>
              <div className="row g-2">
                {(sec.items || []).map((item, ii) => (
                  <div key={ii} className="col-md-6">
                    <div className="card shadow-sm h-100">
                      <div className="card-body py-2">
                        <strong>{item.term}</strong>
                        <p className="text-muted small mb-0 mt-1">{item.definition}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          <div className="alert alert-secondary small mt-3">
            <strong>References:</strong> FDA AI/ML-Based SaMD Action Plan 2021 · EU AI Act 2024 (Art. 9, 17) ·
            WHO Ethics & Governance of AI for Health 2021 · ICH E6(R3) GCP 2023 ·
            IMDRF AI/ML SaMD Working Group 2022 · Topol Review 2019.
          </div>
        </div>
      )}
    </div>
  );
}
