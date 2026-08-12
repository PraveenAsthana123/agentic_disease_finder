'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const AGR_COLORS = { Agree: 'success', Partial: 'warning', Disagree: 'danger' };
const DEC_COLORS = { Confirm: 'success', Override: 'danger', Defer: 'warning', Escalate: 'info' };
const CLASS_COLORS = ['primary', 'success', 'info', 'warning', 'danger'];
const HTIL_BADGE = { accept: 'success', override: 'danger' };

export default function PMSPage() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/pms/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/pms/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/pms/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!overview) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const k = overview.kpis || {};
  const tabs = [
    { id: 'overview', label: '&#x1F4CA; Overview' },
    { id: 'agreement', label: '&#x1F91D; AI vs Clinician' },
    { id: 'performance', label: '&#x1F3AF; Per-Class' },
    { id: 'hitl', label: '&#x1F6A8; HITL Vigilance' },
    { id: 'definitions', label: '&#x1F4D6; Definitions' },
  ];

  return (
    <div>
      <h3>&#x1F4CA; Post-Market Surveillance — AI Performance Monitor</h3>
      <p className="text-muted">
        Real-world AI vs clinician agreement, override rates, and vigilance reporting —
        {k.total_decisions} decisions, {k.total_patients} patients, 5 reviewers · EU MDR Art.83 ·
        FDA 21 CFR 803 · ISO 14971 · IEC 62304
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Decisions', value: k.total_decisions, color: 'primary' },
          { label: 'Patients', value: k.total_patients, color: 'info' },
          { label: 'Agreement Rate', value: `${k.agree_rate_pct}%`, color: 'success' },
          { label: 'Override Rate', value: `${k.override_rate_pct}%`, color: 'danger' },
          { label: 'Avg Confidence', value: k.avg_confidence, color: 'warning' },
          { label: 'Overrides', value: k.override_count, color: 'dark' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Alert banner if override rate is high */}
      {k.override_rate_pct > 20 && (
        <div className="alert alert-danger d-flex align-items-center mb-3" role="alert">
          <span className="me-2" style={{fontSize:'1.5rem'}}>&#x26A0;&#xFE0F;</span>
          <div>
            <strong>PMS Vigilance Alert:</strong> Override rate {k.override_rate_pct}% exceeds 20%
            threshold — mandatory root-cause analysis required per EU MDR Art.83 / ISO 14971.
          </div>
        </div>
      )}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link${tab === t.id ? ' active' : ''}`}
              onClick={() => setTab(t.id)}
              dangerouslySetInnerHTML={{ __html: t.label }}
            />
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <div>
          <div className="row mb-3">
            {/* Agreement donut (bar) */}
            <div className="col-md-6">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Neurologist Agreement Distribution</h6>
                  <div className="progress mb-2" style={{height: '28px'}}>
                    {(overview.agreement_distribution || []).map(a => (
                      <div
                        key={a.label}
                        className={`progress-bar bg-${AGR_COLORS[a.label] || 'secondary'}`}
                        style={{width: `${a.pct}%`}}
                        title={`${a.label}: ${a.count} (${a.pct}%)`}
                      >
                        {a.label} {a.pct}%
                      </div>
                    ))}
                  </div>
                  {(overview.agreement_distribution || []).map(a => (
                    <div key={a.label} className="d-flex justify-content-between mb-1">
                      <span><span className={`badge bg-${AGR_COLORS[a.label]}`}>{a.label}</span></span>
                      <span>{a.count} ({a.pct}%)</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Final decision distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm border-0 h-100">
                <div className="card-body">
                  <h6 className="card-title">Final Clinical Decision</h6>
                  {(overview.final_decision_distribution || []).map(d => (
                    <div key={d.decision} className="mb-2">
                      <div className="d-flex justify-content-between mb-1">
                        <span><span className={`badge bg-${DEC_COLORS[d.decision] || 'secondary'}`}>{d.decision}</span></span>
                        <span className="small">{d.count} ({d.pct}%)</span>
                      </div>
                      <div className="progress" style={{height: '8px'}}>
                        <div
                          className={`progress-bar bg-${DEC_COLORS[d.decision] || 'secondary'}`}
                          style={{width: `${d.pct}%`}}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Confidence buckets */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">AI Confidence Distribution</h6>
              <div className="d-flex gap-2 flex-wrap">
                {(overview.confidence_buckets || []).map((b, i) => (
                  <div key={b.bucket} className={`badge bg-${CLASS_COLORS[i % 5]} fs-6 p-2`}>
                    {b.bucket}: {b.count}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Reviewer workload */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Reviewer Workload & Override Rates</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Reviewer</th>
                      <th>Decisions</th>
                      <th>Overrides</th>
                      <th>Override Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(overview.reviewer_workload || []).map(r => (
                      <tr key={r.reviewer}>
                        <td>{r.reviewer}</td>
                        <td>{r.decisions}</td>
                        <td>{r.overrides}</td>
                        <td>
                          <span className={`badge bg-${r.override_rate_pct > 30 ? 'danger' : r.override_rate_pct > 20 ? 'warning' : 'success'}`}>
                            {r.override_rate_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Confidence calibration insight */}
          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card border-success shadow-sm">
                <div className="card-body">
                  <h6>&#x2705; Confidence When Agreed</h6>
                  <div className="h2 text-success">{k.avg_conf_when_agree}</div>
                  <small className="text-muted">Mean AI confidence on cases where neurologist agreed</small>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card border-danger shadow-sm">
                <div className="card-body">
                  <h6>&#x274C; Confidence When Disagreed</h6>
                  <div className="h2 text-danger">{k.avg_conf_when_disagree}</div>
                  <small className="text-muted">Mean AI confidence on cases where neurologist disagreed</small>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── AI VS CLINICIAN TAB ── */}
      {tab === 'agreement' && breakdown && (
        <div>
          {/* Confidence × Agreement crosstab */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Confidence Tier × Agreement Crosstab</h6>
              <p className="text-muted small">
                High-confidence but low-agreement cases are a key PMS calibration signal.
              </p>
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Confidence Tier</th>
                      <th>Agreement</th>
                      <th>Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.confidence_agreement_crosstab || []).map((r, i) => (
                      <tr key={i}>
                        <td>{r.conf_tier}</td>
                        <td><span className={`badge bg-${AGR_COLORS[r.agreement] || 'secondary'}`}>{r.agreement}</span></td>
                        <td>{r.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Artifact risk distribution */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Artifact Risk Assessment Distribution</h6>
              <div className="progress mb-2" style={{height: '24px'}}>
                {(breakdown.artifact_risk_distribution || []).map((a, i) => {
                  const total = breakdown.artifact_risk_distribution.reduce((s, x) => s + x.count, 0);
                  const pct = total ? Math.round(a.count / total * 100) : 0;
                  return (
                    <div key={a.artifact_risk}
                      className={`progress-bar bg-${CLASS_COLORS[i % 5]}`}
                      style={{width: `${pct}%`}}
                      title={`${a.artifact_risk}: ${a.count}`}>
                      {a.artifact_risk} {a.count}
                    </div>
                  );
                })}
              </div>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Artifact Risk</th><th>Cases</th><th>Avg Confidence</th></tr>
                  </thead>
                  <tbody>
                    {(breakdown.artifact_risk_distribution || []).map(a => (
                      <tr key={a.artifact_risk}>
                        <td>{a.artifact_risk}</td>
                        <td>{a.count}</td>
                        <td>{a.avg_confidence}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PER-CLASS PERFORMANCE TAB ── */}
      {tab === 'performance' && breakdown && (
        <div>
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Per-Prediction-Class AI Performance</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Prediction Class</th>
                      <th>Total</th>
                      <th>Avg Conf</th>
                      <th>Agree</th>
                      <th>Partial</th>
                      <th>Disagree</th>
                      <th>Overrides</th>
                      <th>Agree Rate</th>
                      <th>Override Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.per_class_performance || []).map((c, i) => (
                      <tr key={c.ai_prediction}>
                        <td><span className={`badge bg-${CLASS_COLORS[i % 5]}`}>{c.ai_prediction}</span></td>
                        <td>{c.total}</td>
                        <td>{c.avg_confidence}</td>
                        <td className="text-success">{c.agree}</td>
                        <td className="text-warning">{c.partial}</td>
                        <td className="text-danger">{c.disagree}</td>
                        <td>{c.overrides}</td>
                        <td>
                          <span className={`badge bg-${c.agree_rate_pct >= 50 ? 'success' : 'danger'}`}>
                            {c.agree_rate_pct}%
                          </span>
                        </td>
                        <td>
                          <span className={`badge bg-${c.override_rate_pct > 30 ? 'danger' : c.override_rate_pct > 20 ? 'warning' : 'success'}`}>
                            {c.override_rate_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Per patient */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Per-Patient Summary ({(breakdown.per_patient || []).length} patients)</h6>
              <div className="table-responsive" style={{maxHeight: '360px', overflowY: 'auto'}}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th>
                      <th>Decisions</th>
                      <th>Avg Conf</th>
                      <th>Agree</th>
                      <th>Overrides</th>
                      <th>Agree Rate</th>
                      <th>Override Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.per_patient || []).map(p => (
                      <tr key={p.patient_id}>
                        <td>{p.patient_id}</td>
                        <td>{p.total_decisions}</td>
                        <td>{p.avg_confidence}</td>
                        <td>{p.agree}</td>
                        <td>{p.overrides}</td>
                        <td>
                          <span className={`badge bg-${p.agree_rate_pct >= 50 ? 'success' : 'danger'}`}>
                            {p.agree_rate_pct}%
                          </span>
                        </td>
                        <td>
                          <span className={`badge bg-${p.override_rate_pct > 30 ? 'danger' : 'success'}`}>
                            {p.override_rate_pct}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── HITL VIGILANCE TAB ── */}
      {tab === 'hitl' && breakdown && (
        <div>
          <div className="alert alert-info mb-3">
            <strong>Human-In-The-Loop (HITL) Vigilance Log</strong> — {breakdown.hitl_reviews?.length} HITL
            events recorded. All overrides are automatically logged for EU MDR Art.83 / FDA 21 CFR 803
            reporting and feed into the ISO 14971 risk management file.
          </div>

          {/* Vigilance thresholds */}
          {defs && (
            <div className="card shadow-sm border-0 mb-3">
              <div className="card-body">
                <h6 className="card-title">PMS Vigilance Thresholds</h6>
                {(defs.vigilance_thresholds || []).map((v, i) => (
                  <div key={i} className="mb-2 p-2 bg-light rounded">
                    <div className="fw-bold text-danger">&#x26A0;&#xFE0F; {v.signal}</div>
                    <div className="text-muted small">&#x27A1;&#xFE0F; {v.action}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* HITL log */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">HITL Event Log</h6>
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>#</th>
                      <th>Patient</th>
                      <th>Analysis</th>
                      <th>AI Prediction</th>
                      <th>Decision</th>
                      <th>Human Decision</th>
                      <th>Reason Code</th>
                      <th>Reviewer</th>
                      <th>Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.hitl_reviews || []).map(h => (
                      <tr key={h.id}>
                        <td>{h.id}</td>
                        <td>{h.patient_id}</td>
                        <td>{h.analysis_id ?? '—'}</td>
                        <td>{h.ai_prediction ?? '—'}</td>
                        <td>
                          <span className={`badge bg-${HTIL_BADGE[h.decision] || 'secondary'}`}>
                            {h.decision}
                          </span>
                        </td>
                        <td>{h.human_decision ?? '—'}</td>
                        <td>{h.reason_code ?? '—'}</td>
                        <td>{h.reviewer_id ?? '—'}</td>
                        <td className="small">{h.created_at?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && defs && (
        <div>
          {/* Regulatory context */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Regulatory Context</h6>
              {Object.entries(defs.regulatory_context || {}).map(([key, text]) => (
                <div key={key} className="mb-3">
                  <div className="fw-bold text-primary">{key.replace(/_/g, ' ')}</div>
                  <div className="text-muted small">{text}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Key metrics */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Key Metrics</h6>
              {Object.entries(defs.key_metrics || {}).map(([key, text]) => (
                <div key={key} className="mb-2 p-2 bg-light rounded">
                  <div className="fw-bold">{key}</div>
                  <div className="text-muted small">{text}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Glossary */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">Glossary</h6>
              <dl className="row mb-0">
                {Object.entries(defs.glossary || {}).map(([term, defn]) => (
                  <div key={term}>
                    <dt className="col-sm-3">{term}</dt>
                    <dd className="col-sm-9 text-muted small">{defn}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>

          {/* References */}
          <div className="card shadow-sm border-0 mb-3">
            <div className="card-body">
              <h6 className="card-title">References</h6>
              <ol className="mb-0">
                {(defs.references || []).map((r, i) => (
                  <li key={i} className="text-muted small mb-1">{r}</li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
