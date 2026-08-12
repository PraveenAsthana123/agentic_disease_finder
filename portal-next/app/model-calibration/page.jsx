"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010";

const TABS = ["Overview", "Reliability", "Per Model", "Definitions"];

function Badge({ text, color }) {
  const colors = {
    green: "bg-success text-white",
    yellow: "bg-warning text-dark",
    orange: "bg-warning text-dark",
    red: "bg-danger text-white",
    blue: "bg-primary text-white",
    gray: "bg-secondary text-white",
  };
  return (
    <span className={`badge ${colors[color] || colors.gray} ms-1`}>{text}</span>
  );
}

function VerdictBadge({ verdict }) {
  const map = {
    "Well-calibrated": "green",
    Acceptable: "blue",
    Moderate: "yellow",
    Poor: "red",
  };
  return <Badge text={verdict} color={map[verdict] || "gray"} />;
}

function KpiCard({ label, value, sub, color }) {
  const border = {
    green: "border-success",
    blue: "border-primary",
    yellow: "border-warning",
    red: "border-danger",
    gray: "border-secondary",
  };
  return (
    <div className={`card border-2 ${border[color] || border.gray} h-100`}>
      <div className="card-body text-center py-3">
        <div className="fw-bold text-muted small text-uppercase">{label}</div>
        <div className="display-6 fw-bold my-1">{value}</div>
        {sub && <div className="text-muted small">{sub}</div>}
      </div>
    </div>
  );
}

function CalibrationBar({ label, conf, acc, n }) {
  const gap = Math.abs(conf - acc);
  const color = gap < 0.05 ? "#22c55e" : gap < 0.10 ? "#3b82f6" : gap < 0.15 ? "#f59e0b" : "#ef4444";
  return (
    <div className="mb-3">
      <div className="d-flex justify-content-between small mb-1">
        <span className="fw-semibold">{label}</span>
        <span className="text-muted">n={n} · conf={Math.round(conf * 100)}% · proxy={Math.round(acc * 100)}% · gap={Math.round(gap * 100)}%</span>
      </div>
      <div className="position-relative" style={{ height: 18, background: "#e5e7eb", borderRadius: 4 }}>
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${conf * 100}%`, background: "#93c5fd", borderRadius: 4 }} />
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: `${acc * 100}%`, background: color, borderRadius: 4, opacity: 0.7 }} />
      </div>
      <div className="d-flex small text-muted mt-1">
        <span className="me-3"><span style={{ background: "#93c5fd", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} /> Confidence</span>
        <span><span style={{ background: color, display: "inline-block", width: 12, height: 12, borderRadius: 2 }} /> Proxy Acc</span>
      </div>
    </div>
  );
}

function ReliabilityDiagram({ bins }) {
  if (!bins || bins.length === 0) return <p className="text-muted">No data</p>;
  const maxN = Math.max(...bins.map((b) => b.n));
  return (
    <div>
      <p className="small text-muted mb-2">Reliability diagram: predicted confidence vs proxy observed accuracy. Perfect calibration = bars touch the diagonal.</p>
      <div className="d-flex align-items-end" style={{ height: 180, gap: 4, paddingBottom: 24 }}>
        {bins.map((b, i) => {
          const predH = Math.max(4, (b.avg_predicted_confidence || 0) * 160);
          const obsH = Math.max(4, (b.proxy_observed_accuracy || 0) * 160);
          const gap = Math.abs((b.avg_predicted_confidence || 0) - (b.proxy_observed_accuracy || 0));
          const barColor = gap < 0.05 ? "#22c55e" : gap < 0.10 ? "#3b82f6" : gap < 0.15 ? "#f59e0b" : "#ef4444";
          return (
            <div key={i} className="d-flex flex-column align-items-center" style={{ flex: 1 }}>
              <div className="d-flex align-items-end" style={{ height: 160, gap: 2 }}>
                <div title={`Predicted: ${Math.round((b.avg_predicted_confidence || 0) * 100)}%`}
                  style={{ width: 14, height: predH, background: "#93c5fd", borderRadius: "3px 3px 0 0" }} />
                <div title={`Observed: ${Math.round((b.proxy_observed_accuracy || 0) * 100)}%`}
                  style={{ width: 14, height: obsH, background: barColor, borderRadius: "3px 3px 0 0", opacity: 0.85 }} />
              </div>
              <div className="text-center" style={{ fontSize: 9, lineHeight: 1.2, width: 32 }}>{b.bin}</div>
            </div>
          );
        })}
      </div>
      <div className="d-flex small text-muted mt-1">
        <span className="me-3"><span style={{ background: "#93c5fd", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} /> Predicted Conf</span>
        <span><span style={{ background: "#22c55e", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} /> Proxy Obs Acc</span>
      </div>
    </div>
  );
}

export default function ModelCalibrationPage() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [def, setDef] = useState(null);
  const [tab, setTab] = useState(0);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/model-calibration/overview`).then((r) => r.json()),
      fetch(`${API}/api/model-calibration/breakdown`).then((r) => r.json()),
      fetch(`${API}/api/model-calibration/definitions`).then((r) => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); })
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Error: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading calibration data…</div>;

  const s = ov.summary || {};
  const tripod = ov.tripod_ai || {};

  const verdictColor =
    s.calibration_verdict === "Well-calibrated" ? "green"
      : s.calibration_verdict === "Acceptable" ? "blue"
      : s.calibration_verdict === "Moderate" ? "yellow"
      : "red";

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-1">
        <h4 className="mb-0">📐 Model Calibration</h4>
        <VerdictBadge verdict={s.calibration_verdict} />
        {tripod.compliant && <Badge text="TRIPOD-AI ✓" color="green" />}
      </div>
      <p className="text-muted small mb-3">
        Confidence calibration quality across {s.total_model_runs} model runs · {s.total_analyses} clinical analyses ·
        {s.model_types} model types · {s.diseases_covered} diseases ·
        Reference: Collins BMJ 2024 · Van Calster Lancet Digit Health 2019
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? "active" : ""}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-3 mb-4">
            <div className="col-md-2">
              <KpiCard label="ECE" value={s.ece != null ? s.ece.toFixed(3) : "—"} sub="Expected Calibration Error" color={verdictColor} />
            </div>
            <div className="col-md-2">
              <KpiCard label="Brier Score" value={s.brier_score != null ? s.brier_score.toFixed(4) : "—"} sub="Squared conf error" color={s.brier_score < 0.10 ? "green" : "yellow"} />
            </div>
            <div className="col-md-2">
              <KpiCard label="Avg Confidence" value={s.avg_confidence != null ? `${Math.round(s.avg_confidence * 100)}%` : "—"} sub="133 analyses" color="blue" />
            </div>
            <div className="col-md-2">
              <KpiCard label="Best AUC" value={s.best_auc != null ? s.best_auc.toFixed(3) : "—"} sub={s.best_model} color="green" />
            </div>
            <div className="col-md-2">
              <KpiCard label="Model Runs" value={s.total_model_runs} sub={`${s.model_types} types`} color="gray" />
            </div>
            <div className="col-md-2">
              <KpiCard label="Calibration" value={s.calibration_verdict} sub={tripod.ece_threshold} color={verdictColor} />
            </div>
          </div>

          {/* TRIPOD-AI compliance panel */}
          <div className="card mb-4 border-0 shadow-sm">
            <div className="card-header bg-light fw-semibold">TRIPOD-AI Item 22 — Calibration Reporting</div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-6">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td className="fw-semibold">Item 22a</td><td>{tripod.item_22a}</td><td><span className="badge bg-success">✓</span></td></tr>
                      <tr><td className="fw-semibold">Item 22b</td><td>{tripod.item_22b}</td><td><span className="badge bg-success">✓</span></td></tr>
                      <tr><td className="fw-semibold">ECE verdict</td><td>{tripod.ece_threshold}</td><td><VerdictBadge verdict={tripod.status} /></td></tr>
                      <tr><td className="fw-semibold">Overall</td><td>TRIPOD-AI compliance</td><td>{tripod.compliant ? <span className="badge bg-success">PASS</span> : <span className="badge bg-warning">REVIEW</span>}</td></tr>
                    </tbody>
                  </table>
                </div>
                <div className="col-md-6">
                  <p className="small text-muted mb-0">
                    TRIPOD-AI (Collins et al. BMJ 2024) mandates reporting of model calibration for all clinical AI
                    prediction models. ECE &lt; 0.05 is considered well-calibrated for clinical deployment.
                    Brier Score &lt; 0.10 indicates good probabilistic accuracy. This dashboard satisfies Items 22a and 22b.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Calibration buckets */}
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header bg-light fw-semibold">Confidence Bucket Calibration</div>
                <div className="card-body">
                  {(ov.calibration_buckets || []).map((b, i) => (
                    <CalibrationBar key={i} label={b.label} conf={b.avg_conf} acc={b.proxy_acc} n={b.n} />
                  ))}
                  <p className="text-muted small mt-2 mb-0">Blue = predicted confidence · Colored = proxy accuracy · Gap = calibration error</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header bg-light fw-semibold">Disease-Level Confidence</div>
                <div className="card-body">
                  <table className="table table-sm">
                    <thead><tr><th>Disease</th><th>N</th><th>Avg Conf</th></tr></thead>
                    <tbody>
                      {(ov.disease_confidence || []).map((d, i) => (
                        <tr key={i}>
                          <td className="text-capitalize">{d.disease}</td>
                          <td>{d.n_analyses}</td>
                          <td>
                            <div className="d-flex align-items-center gap-2">
                              <div style={{ width: 80, height: 8, background: "#e5e7eb", borderRadius: 4 }}>
                                <div style={{ width: `${(d.avg_confidence || 0) * 100}%`, height: "100%", background: "#3b82f6", borderRadius: 4 }} />
                              </div>
                              <span>{d.avg_confidence != null ? `${Math.round(d.avg_confidence * 100)}%` : "—"}</span>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Model type summary */}
          <div className="card border-0 shadow-sm mt-3">
            <div className="card-header bg-light fw-semibold">Model Type Summary</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Model Type</th><th>Runs</th><th>Avg Accuracy</th><th>Avg AUC</th></tr>
                </thead>
                <tbody>
                  {(ov.model_type_summary || []).map((m, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{m.model_type}</td>
                      <td>{m.n_runs}</td>
                      <td>{m.avg_accuracy != null ? `${Math.round(m.avg_accuracy * 100)}%` : "—"}</td>
                      <td>
                        <div className="d-flex align-items-center gap-2">
                          <div style={{ width: 80, height: 8, background: "#e5e7eb", borderRadius: 4 }}>
                            <div style={{ width: `${(m.avg_auc || 0) * 100}%`, height: "100%", background: "#22c55e", borderRadius: 4 }} />
                          </div>
                          <span>{m.avg_auc != null ? m.avg_auc.toFixed(3) : "—"}</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Reliability ── */}
      {tab === 1 && bd && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-8">
              <div className="card border-0 shadow-sm">
                <div className="card-header bg-light fw-semibold">Reliability Diagram (10 Decile Bins)</div>
                <div className="card-body">
                  <ReliabilityDiagram bins={bd.reliability_bins} />
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card border-0 shadow-sm h-100">
                <div className="card-header bg-light fw-semibold">Calibration Legend</div>
                <div className="card-body">
                  <table className="table table-sm">
                    <tbody>
                      <tr><td><span className="badge bg-success">ECE &lt; 0.05</span></td><td>Well-calibrated</td></tr>
                      <tr><td><span className="badge bg-primary">ECE 0.05–0.10</span></td><td>Acceptable</td></tr>
                      <tr><td><span className="badge bg-warning text-dark">ECE 0.10–0.15</span></td><td>Moderate</td></tr>
                      <tr><td><span className="badge bg-danger">ECE &gt; 0.15</span></td><td>Poor</td></tr>
                    </tbody>
                  </table>
                  <p className="text-muted small mt-2">Bars show confidence bucket gap vs proxy observed accuracy. Perfect calibration: blue = colored bar height.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Disease calibration */}
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header bg-light fw-semibold">Disease-Level Calibration Analysis</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Disease</th><th>N</th><th>Avg Conf</th><th>Proxy AUC</th>
                    <th>Gap</th><th>Conf ≥70%</th><th>Conf ≥80%</th><th>Conf ≥90%</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.disease_calibration || []).map((d, i) => {
                    const thresholds = d.confidence_thresholds || [];
                    const t70 = thresholds.find((t) => t.threshold === 0.7);
                    const t80 = thresholds.find((t) => t.threshold === 0.8);
                    const t90 = thresholds.find((t) => t.threshold === 0.9);
                    const gap = d.calibration_gap || 0;
                    const gapColor = gap < 0.05 ? "text-success" : gap < 0.10 ? "text-primary" : "text-warning";
                    return (
                      <tr key={i}>
                        <td className="text-capitalize fw-semibold">{d.disease}</td>
                        <td>{d.n_analyses}</td>
                        <td>{d.avg_confidence != null ? `${Math.round(d.avg_confidence * 100)}%` : "—"}</td>
                        <td>{d.proxy_auc != null ? d.proxy_auc.toFixed(3) : "—"}</td>
                        <td className={gapColor}>{gap.toFixed(3)}</td>
                        <td>{t70 ? `${t70.n_above} (${t70.pct}%)` : "—"}</td>
                        <td>{t80 ? `${t80.n_above} (${t80.pct}%)` : "—"}</td>
                        <td>{t90 ? `${t90.n_above} (${t90.pct}%)` : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-dataset calibration */}
          <div className="card border-0 shadow-sm">
            <div className="card-header bg-light fw-semibold">Per-Dataset Calibration</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>Dataset</th><th>Runs</th><th>Avg Accuracy</th><th>Avg AUC</th><th>ECE Proxy</th></tr>
                </thead>
                <tbody>
                  {(bd.per_dataset || []).map((d, i) => (
                    <tr key={i}>
                      <td>{d.dataset}</td>
                      <td>{d.n_runs}</td>
                      <td>{d.avg_accuracy != null ? `${Math.round(d.avg_accuracy * 100)}%` : "—"}</td>
                      <td>{d.avg_auc != null ? d.avg_auc.toFixed(3) : "—"}</td>
                      <td>{d.dataset_ece_proxy != null ? d.dataset_ece_proxy.toFixed(4) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Per Model ── */}
      {tab === 2 && bd && (
        <div>
          {/* Per model type calibration */}
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header bg-light fw-semibold">Per Model Type — Calibration Quality</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr>
                    <th>Model Type</th><th>Runs</th><th>Avg Acc</th><th>Avg AUC</th>
                    <th>Avg F1</th><th>Avg Prec</th><th>Avg Recall</th>
                    <th>Cal Gap</th><th>Brier</th><th>Tendency</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.per_model_type || []).map((m, i) => {
                    const tendColor = m.tendency === "Well-calibrated" ? "success" : m.tendency === "Overconfident" ? "warning" : "info";
                    return (
                      <tr key={i}>
                        <td className="fw-semibold">{m.model_type}</td>
                        <td>{m.n_runs}</td>
                        <td>{m.avg_accuracy != null ? `${Math.round(m.avg_accuracy * 100)}%` : "—"}</td>
                        <td>{m.avg_auc != null ? m.avg_auc.toFixed(3) : "—"}</td>
                        <td>{m.avg_f1 != null ? m.avg_f1.toFixed(3) : "—"}</td>
                        <td>{m.avg_precision != null ? m.avg_precision.toFixed(3) : "—"}</td>
                        <td>{m.avg_recall != null ? m.avg_recall.toFixed(3) : "—"}</td>
                        <td>{m.calibration_gap != null ? m.calibration_gap.toFixed(3) : "—"}</td>
                        <td>{m.brier_score != null ? m.brier_score.toFixed(4) : "—"}</td>
                        <td><span className={`badge bg-${tendColor} text-${tendColor === "warning" ? "dark" : "white"}`}>{m.tendency}</span></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Top models */}
          <div className="card border-0 shadow-sm">
            <div className="card-header bg-light fw-semibold">Top 15 Model Runs (by AUC)</div>
            <div className="card-body p-0" style={{ maxHeight: 420, overflowY: "auto" }}>
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light sticky-top">
                  <tr>
                    <th>#</th><th>Model</th><th>Type</th><th>Task</th><th>Dataset</th>
                    <th>Acc</th><th>AUC</th><th>F1</th><th>Prec</th><th>Rec</th>
                    <th>N</th><th>Train(s)</th><th>Inf(ms)</th>
                  </tr>
                </thead>
                <tbody>
                  {(bd.top_models || []).map((m, i) => (
                    <tr key={i}>
                      <td className="text-muted">{i + 1}</td>
                      <td><span className="badge bg-light text-dark border">{m.model_name}</span></td>
                      <td>{m.model_type}</td>
                      <td className="small">{m.task}</td>
                      <td className="small">{m.dataset}</td>
                      <td className="fw-semibold">{m.accuracy != null ? `${Math.round(m.accuracy * 100)}%` : "—"}</td>
                      <td className="text-success fw-semibold">{m.auc != null ? m.auc.toFixed(3) : "—"}</td>
                      <td>{m.f1 != null ? m.f1.toFixed(3) : "—"}</td>
                      <td>{m.precision != null ? m.precision.toFixed(3) : "—"}</td>
                      <td>{m.recall != null ? m.recall.toFixed(3) : "—"}</td>
                      <td>{m.n_samples?.toLocaleString()}</td>
                      <td>{m.training_sec}</td>
                      <td>{m.inference_ms}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 3 && def && (
        <div>
          <div className="row g-3 mb-3">
            {(def.concepts || []).map((c, i) => (
              <div key={i} className="col-md-6">
                <div className="card border-0 shadow-sm h-100">
                  <div className="card-body">
                    <h6 className="card-title fw-bold mb-1">{c.term}</h6>
                    <p className="card-text small text-muted mb-1">{c.definition}</p>
                    {c.reference && <p className="card-text small text-primary mb-0">📖 {c.reference}</p>}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="card border-0 shadow-sm mb-3">
            <div className="card-header bg-light fw-semibold">Calibration Thresholds</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Metric</th><th>Threshold</th><th>Verdict</th></tr></thead>
                <tbody>
                  {(def.thresholds || []).map((t, i) => (
                    <tr key={i}>
                      <td>{t.metric}</td>
                      <td><code>{t.threshold}</code></td>
                      <td>{t.verdict}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card border-0 shadow-sm">
            <div className="card-header bg-light fw-semibold">Standards & Guidelines</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th>Standard</th><th>Requirement</th></tr></thead>
                <tbody>
                  {(def.standards || []).map((s, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{s.standard}</td>
                      <td className="small">{s.requirement || (s.items_required || []).join("; ")}</td>
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
