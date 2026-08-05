'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'hallucinations', label: 'Hallucination Cases' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'reviews', label: 'Expert & HITL Reviews' },
  { id: 'definitions', label: 'Definitions' },
];

function KPI({ label, value, color, sub, badge }) {
  return (
    <div className="col-6 col-md-3 mb-3">
      <div className="card shadow-sm h-100">
        <div className="card-body text-center">
          <div className={`h4 mb-1 fw-bold text-${color || 'primary'}`}>{value ?? '—'}</div>
          <div className="text-muted small">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{sub}</div>}
          {badge && <span className={`badge bg-${badge.color} mt-1`}>{badge.text}</span>}
        </div>
      </div>
    </div>
  );
}

function DistBar({ items, labelKey = 'label', valueKey = 'value', total }) {
  const sum = total || items.reduce((a, b) => a + (b[valueKey] || 0), 0) || 1;
  const COLORS = ['primary', 'success', 'warning', 'danger', 'info', 'secondary'];
  return (
    <table className="table table-sm mb-0">
      <tbody>
        {items.map((item, i) => {
          const label = item[labelKey] || String(i);
          const val = item[valueKey] || 0;
          const pct = ((val / sum) * 100).toFixed(1);
          const color = item.color || COLORS[i % COLORS.length];
          return (
            <tr key={i}>
              <td className="small fw-semibold" style={{ width: '45%' }}>{label}</td>
              <td style={{ width: '40%' }}>
                <div className="progress" style={{ height: '8px' }}>
                  <div className={`progress-bar bg-${color}`} style={{ width: `${pct}%` }} />
                </div>
              </td>
              <td className="small text-end text-muted" style={{ width: '15%' }}>{val} ({pct}%)</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function GateBadge({ status }) {
  const color = status === 'PASS' ? 'success' : 'danger';
  const icon = status === 'PASS' ? '✅' : '🚨';
  return (
    <span className={`badge bg-${color} fs-6 px-3 py-2`}>{icon} Gate {status}</span>
  );
}

export default function GroundingGatePage() {
  const [tab, setTab] = useState('overview');
  const [ovr, setOvr] = useState(null);
  const [brk, setBrk] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/grounding-gate/overview`).then(r => r.json()),
      fetch(`${API}/api/grounding-gate/breakdown`).then(r => r.json()),
      fetch(`${API}/api/grounding-gate/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOvr(o); setBrk(b); setDefs(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /></div>;
  if (err) return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ovr?.kpis || {};

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0">🛡️ Agent Grounding Gate</h4>
          <small className="text-muted">Hallucination detection · Citation verification · Calibration monitoring</small>
        </div>
        {ovr && <GateBadge status={ovr.gate_status} />}
        <span className="text-muted small ms-auto">
          Threshold: {ovr?.gate_threshold_pct}% grounding · Confidence gate: ≥{ovr?.confidence_gate}
        </span>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button
              className={`nav-link ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && ovr && (
        <>
          <div className="row g-2 mb-3">
            <KPI label="Grounding Score"
              value={`${kpi.grounding_score_pct}%`}
              color={kpi.grounding_score_pct >= 75 ? 'success' : kpi.grounding_score_pct >= 60 ? 'warning' : 'danger'}
              sub={`${kpi.total_grounded} / ${kpi.total_decisions} grounded`}
              badge={{ color: ovr.gate_status === 'PASS' ? 'success' : 'danger', text: ovr.gate_status }} />
            <KPI label="Hallucination Rate"
              value={`${kpi.hallucination_rate_pct}%`}
              color={kpi.hallucination_rate_pct > 20 ? 'danger' : kpi.hallucination_rate_pct > 10 ? 'warning' : 'success'}
              sub={`${kpi.total_hallucinations} high-conf + expert disagree`} />
            <KPI label="High-Conf Accuracy"
              value={`${kpi.high_conf_accuracy_pct}%`}
              color={kpi.high_conf_accuracy_pct >= 75 ? 'success' : 'warning'}
              sub={`conf ≥ ${ovr.confidence_gate} (${kpi.total_high_conf} decisions)`} />
            <KPI label="Calibration Error (ECE)"
              value={kpi.calibration_error_ece}
              color={kpi.calibration_error_ece < 0.05 ? 'success' : kpi.calibration_error_ece < 0.1 ? 'warning' : 'danger'}
              sub="0 = perfect · >0.1 = poor" />
          </div>
          <div className="row g-2 mb-3">
            <KPI label="HITL Override Rate"
              value={`${kpi.hitl_override_rate_pct}%`}
              color={kpi.hitl_override_rate_pct < 20 ? 'success' : 'warning'}
              sub="Human overrides of AI decisions" />
            <KPI label="Expert Agreement"
              value={`${kpi.expert_agreement_rate_pct}%`}
              color="info"
              sub="Expert reviews agreeing with AI" />
            <KPI label="Avg Confidence"
              value={kpi.avg_confidence}
              color="primary"
              sub="Mean AI confidence score" />
            <KPI label="Avg AUC (Validation)"
              value={kpi.avg_auc ?? '—'}
              color="primary"
              sub={`Sens ${kpi.avg_sensitivity ?? '—'} · Spec ${kpi.avg_specificity ?? '—'}`} />
          </div>

          <div className="row g-3">
            {/* Agreement Distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header"><strong>Neurologist Agreement Distribution</strong></div>
                <div className="card-body">
                  <DistBar
                    items={(ovr.agreement_distribution || []).map(d => ({
                      ...d,
                      color: d.label === 'Agree' ? 'success' : d.label === 'Partial' ? 'warning' : 'danger',
                    }))}
                    labelKey="label" valueKey="value" />
                </div>
              </div>
            </div>

            {/* Severity Distribution */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header"><strong>Grounding Severity Classes</strong></div>
                <div className="card-body">
                  <DistBar items={ovr.severity_distribution || []} labelKey="label" valueKey="value" />
                  <p className="text-muted small mt-2 mb-0">
                    Critical = high-confidence prediction with expert disagreement (hallucination risk).
                  </p>
                </div>
              </div>
            </div>

            {/* Confidence Histogram */}
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header"><strong>Confidence Score Distribution</strong></div>
                <div className="card-body">
                  <div className="d-flex align-items-end gap-1" style={{ height: '100px' }}>
                    {(ovr.confidence_histogram || []).map((b, i) => {
                      const maxCount = Math.max(...(ovr.confidence_histogram || []).map(x => x.count), 1);
                      const h = Math.round((b.count / maxCount) * 80);
                      const inGate = parseFloat(b.range) >= ovr.confidence_gate;
                      return (
                        <div key={i} className="d-flex flex-column align-items-center flex-grow-1" title={`${b.range}: ${b.count}`}>
                          <small className="text-muted" style={{ fontSize: '0.6rem' }}>{b.count}</small>
                          <div
                            className={`w-100 rounded-top ${inGate ? 'bg-warning' : 'bg-primary'}`}
                            style={{ height: `${h}px` }}
                          />
                          <small style={{ fontSize: '0.55rem', transform: 'rotate(-45deg)', marginTop: '4px' }}>{b.range.split('–')[0]}</small>
                        </div>
                      );
                    })}
                  </div>
                  <p className="text-muted small mt-2 mb-0">
                    🟡 Yellow bars = high-confidence zone (≥{ovr.confidence_gate}). Hallucination risk is highest here when expert disagrees.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className={`alert alert-${ovr.gate_status === 'PASS' ? 'success' : 'danger'} mt-3`}>
            <strong>Gate Status: {ovr.gate_status}</strong> — {ovr.summary}
          </div>
        </>
      )}

      {/* ── HALLUCINATION CASES ── */}
      {tab === 'hallucinations' && brk && (
        <>
          <div className="alert alert-warning">
            <strong>⚠️ Hallucination Definition:</strong> AI confidence ≥ {ovr?.confidence_gate} but neurologist
            partially or fully disagrees. These cases require mandatory human review before clinical use.
          </div>

          {/* Grounding by confidence band */}
          <div className="card shadow-sm mb-3">
            <div className="card-header"><strong>Grounding Rate by Confidence Band</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Confidence Band</th>
                    <th>Total</th>
                    <th>Grounded</th>
                    <th>Grounding %</th>
                    <th>Hallucinations</th>
                  </tr>
                </thead>
                <tbody>
                  {(brk.grounding_by_confidence_band || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{b.band}</td>
                      <td>{b.total}</td>
                      <td className="text-success">{b.grounded}</td>
                      <td>
                        <span className={`badge bg-${b.grounding_pct >= 75 ? 'success' : b.grounding_pct >= 60 ? 'warning' : 'danger'}`}>
                          {b.grounding_pct}%
                        </span>
                      </td>
                      <td className="text-danger fw-bold">{b.hallucinations}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Hallucination cases table */}
          <div className="card shadow-sm">
            <div className="card-header">
              <strong>🚨 Hallucination-Risk Cases ({brk.hallucination_cases?.length || 0})</strong>
            </div>
            <div className="card-body p-0" style={{ maxHeight: '400px', overflowY: 'auto' }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>Patient</th>
                    <th>AI Prediction</th>
                    <th>Confidence</th>
                    <th>Expert Agreement</th>
                    <th>Final Decision</th>
                    <th>Top Channels</th>
                    <th>Note</th>
                  </tr>
                </thead>
                <tbody>
                  {(brk.hallucination_cases || []).map((c, i) => (
                    <tr key={i}>
                      <td className="small fw-semibold">{c.patient_id}</td>
                      <td className="small">{c.ai_prediction}</td>
                      <td>
                        <span className="badge bg-danger">{(c.ai_confidence * 100).toFixed(0)}%</span>
                      </td>
                      <td>
                        <span className={`badge bg-${c.neurologist_agreement === 'Disagree' ? 'danger' : 'warning'}`}>
                          {c.neurologist_agreement}
                        </span>
                      </td>
                      <td className="small">{c.final_decision}</td>
                      <td className="small text-muted">{c.top_channels}</td>
                      <td className="small text-muted" style={{ maxWidth: '200px' }}>{c.note}</td>
                    </tr>
                  ))}
                  {(!brk.hallucination_cases || brk.hallucination_cases.length === 0) && (
                    <tr><td colSpan={7} className="text-center text-success py-3">✅ No hallucination-risk cases detected</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Patient summary */}
          <div className="card shadow-sm mt-3">
            <div className="card-header"><strong>Patient Grounding Summary (Top 15)</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Patient</th><th>Decisions</th><th>Grounded</th><th>Hallucinations</th></tr>
                </thead>
                <tbody>
                  {(brk.patient_grounding_summary || []).map((p, i) => (
                    <tr key={i} className={p.hallucinations > 0 ? 'table-warning' : ''}>
                      <td className="fw-semibold">{p.patient_id}</td>
                      <td>{p.decisions}</td>
                      <td className="text-success">{p.grounded}</td>
                      <td className={p.hallucinations > 0 ? 'text-danger fw-bold' : 'text-muted'}>{p.hallucinations}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* ── CALIBRATION ── */}
      {tab === 'calibration' && brk && (
        <>
          <div className="alert alert-info">
            <strong>Calibration Curve:</strong> A perfectly calibrated model lies on the diagonal (confidence = accuracy).
            Points above = overconfident. Points below = underconfident. ECE = area between curve and diagonal.
          </div>
          <div className="card shadow-sm">
            <div className="card-header"><strong>Confidence vs. Accuracy (Reliability Diagram)</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Confidence Bin</th>
                    <th>Avg Confidence</th>
                    <th>Empirical Accuracy</th>
                    <th>Count</th>
                    <th>Gap (|conf − acc|)</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {(brk.calibration_curve || []).map((b, i) => {
                    const gap = Math.abs(b.avg_confidence - b.accuracy);
                    const status = gap < 0.05 ? 'Well-calibrated' : gap < 0.15 ? 'Moderate' : 'Poorly calibrated';
                    const color = gap < 0.05 ? 'success' : gap < 0.15 ? 'warning' : 'danger';
                    return (
                      <tr key={i}>
                        <td>{b.bin}</td>
                        <td>{b.avg_confidence}</td>
                        <td>{b.accuracy}</td>
                        <td>{b.count}</td>
                        <td>{gap.toFixed(3)}</td>
                        <td><span className={`badge bg-${color}`}>{status}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
          <div className="mt-3">
            <div className="card shadow-sm">
              <div className="card-body">
                <strong>ECE Summary:</strong>{' '}
                <span className={`badge bg-${kpi.calibration_error_ece < 0.05 ? 'success' : kpi.calibration_error_ece < 0.1 ? 'warning' : 'danger'} fs-6`}>
                  ECE = {kpi.calibration_error_ece}
                </span>
                <p className="text-muted small mt-2 mb-0">
                  ECE &lt; 0.05: excellent · 0.05–0.10: acceptable · &gt; 0.10: requires recalibration.
                  Recalibration methods: temperature scaling, isotonic regression, Platt scaling.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── EXPERT & HITL REVIEWS ── */}
      {tab === 'reviews' && brk && (
        <>
          <div className="row g-3">
            {/* Expert Reviews */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header"><strong>Expert Reviews ({brk.expert_reviews?.length || 0})</strong></div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Patient</th><th>Role</th><th>Expert</th><th>Agreement</th><th>Finding</th></tr>
                    </thead>
                    <tbody>
                      {(brk.expert_reviews || []).map((r, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{r.patient_id}</td>
                          <td className="small">{r.role}</td>
                          <td className="small">{r.expert}</td>
                          <td>
                            <span className={`badge bg-${r.agree_with_ai === 'agree' ? 'success' : r.agree_with_ai === 'partial' ? 'warning' : 'danger'}`}>
                              {r.agree_with_ai}
                            </span>
                          </td>
                          <td className="small text-muted" style={{ maxWidth: '150px' }}>{r.finding}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* HITL Reviews */}
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header"><strong>HITL Reviews ({brk.hitl_reviews?.length || 0})</strong></div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Patient</th><th>AI Prediction</th><th>Decision</th><th>Human Decision</th><th>Date</th></tr>
                    </thead>
                    <tbody>
                      {(brk.hitl_reviews || []).map((r, i) => (
                        <tr key={i} className={r.decision === 'override' ? 'table-warning' : ''}>
                          <td className="small fw-semibold">{r.patient_id}</td>
                          <td className="small">{r.ai_prediction}</td>
                          <td>
                            <span className={`badge bg-${r.decision === 'override' ? 'warning' : 'success'}`}>
                              {r.decision}
                            </span>
                          </td>
                          <td className="small">{r.human_decision || '—'}</td>
                          <td className="small text-muted">{(r.created_at || '').slice(0, 10)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && defs && (
        <>
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <h6>Purpose</h6>
              <p className="text-muted small">{defs.purpose}</p>
            </div>
          </div>

          {/* Key Terms */}
          <div className="card shadow-sm mb-3">
            <div className="card-header"><strong>Key Terms</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Term</th><th>Definition</th><th>Formula</th></tr>
                </thead>
                <tbody>
                  {(defs.key_terms || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{t.term}</td>
                      <td className="small text-muted">{t.definition}</td>
                      <td className="small text-info font-monospace">{t.formula || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Gate Levels */}
          <div className="card shadow-sm mb-3">
            <div className="card-header"><strong>Gate Levels</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Level</th><th>Threshold</th><th>Action</th></tr>
                </thead>
                <tbody>
                  {(defs.grounding_levels || []).map((l, i) => (
                    <tr key={i}>
                      <td><span className={`badge bg-${l.color} fs-6`}>{l.level}</span></td>
                      <td className="small fw-semibold">{l.threshold}</td>
                      <td className="small text-muted">{l.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Data Sources + References */}
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header"><strong>Data Sources</strong></div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Table</th><th>Rows</th><th>Key Fields</th></tr>
                    </thead>
                    <tbody>
                      {(defs.data_sources || []).map((s, i) => (
                        <tr key={i}>
                          <td className="small font-monospace">{s.source}</td>
                          <td className="small">{s.rows}</td>
                          <td className="small text-muted">{s.fields}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header"><strong>Standards & References</strong></div>
                <div className="card-body">
                  <p className="fw-semibold small mb-1">Standards Alignment:</p>
                  <ul className="list-unstyled mb-2">
                    {(defs.standards_alignment || []).map((s, i) => (
                      <li key={i} className="small text-muted">• {s}</li>
                    ))}
                  </ul>
                  <p className="fw-semibold small mb-1">References:</p>
                  <ul className="list-unstyled mb-0">
                    {(defs.references || []).map((r, i) => (
                      <li key={i} className="small text-muted">• {r}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
