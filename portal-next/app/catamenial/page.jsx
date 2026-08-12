"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010";

const TABS = ["Overview", "Patients", "AED Interactions", "Hormones", "Definitions"];

function Badge({ text, color }) {
  const colors = {
    green: "bg-success text-white",
    yellow: "bg-warning text-dark",
    red: "bg-danger text-white",
    blue: "bg-primary text-white",
    purple: "bg-purple text-white",
    gray: "bg-secondary text-white",
  };
  return (
    <span className={`badge ${colors[color] || colors.gray} ms-1`}>{text}</span>
  );
}

function RiskBadge({ risk }) {
  const map = { High: "red", Moderate: "yellow", "Low-Moderate": "yellow", Low: "green" };
  return <Badge text={risk} color={map[risk] || "gray"} />;
}

function PhaseBadge({ phase }) {
  const map = { C1: "red", C2: "yellow", C3: "purple", "N/A": "gray" };
  return (
    <span className={`badge ms-1 ${
      phase === "C1" ? "bg-danger" :
      phase === "C2" ? "bg-warning text-dark" :
      phase === "C3" ? "bg-info" : "bg-secondary"
    } text-white`}>{phase}</span>
  );
}

function KpiCard({ label, value, sub, color }) {
  const border = { red: "border-danger", blue: "border-primary", green: "border-success", purple: "border-info" };
  return (
    <div className={`card border-2 ${border[color] || "border-secondary"} h-100`}>
      <div className="card-body text-center py-3">
        <div className={`fs-2 fw-bold text-${color === "purple" ? "info" : color}`}>{value ?? "—"}</div>
        <div className="small text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: "0.7rem" }}>{sub}</div>}
      </div>
    </div>
  );
}

// Simple bar chart using inline CSS
function BarChart({ data, xKey, yKey, color, height = 120 }) {
  if (!data || !data.length) return <div className="text-muted small">No data</div>;
  const max = Math.max(...data.map(d => d[yKey] || 0)) || 1;
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height }}>
      {data.map((d, i) => {
        const pct = ((d[yKey] || 0) / max) * 100;
        return (
          <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center" }}>
            <div
              title={`${d[xKey]}: ${d[yKey]}`}
              style={{
                width: "100%",
                height: `${pct}%`,
                background: color || "#0d6efd",
                borderRadius: "3px 3px 0 0",
                minHeight: pct > 0 ? 4 : 0,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}

export default function CatamenialPage() {
  const [tab, setTab] = useState("Overview");
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/catamenial/overview`).then(r => r.json()),
      fetch(`${API}/api/catamenial/breakdown`).then(r => r.json()),
      fetch(`${API}/api/catamenial/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /><p className="mt-3">Loading catamenial data…</p></div>;
  if (error) return <div className="container py-4"><div className="alert alert-danger">Error: {error}</div></div>;

  const kpis = overview?.kpis || {};

  return (
    <div className="container-fluid py-4">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h2 className="mb-0 fw-bold">Catamenial Epilepsy</h2>
          <div className="text-muted small">
            Seizure clustering by menstrual cycle phase · Duncan C1/C2/C3 classification
          </div>
        </div>
        <span className="badge bg-danger ms-auto">Hormonal Epilepsy</span>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? "active fw-semibold" : ""}`}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === "Overview" && (
        <div>
          {/* KPI row */}
          <div className="row g-3 mb-4">
            <div className="col-6 col-md-3">
              <KpiCard label="Female Patients" value={kpis.female_patients} color="blue" />
            </div>
            <div className="col-6 col-md-3">
              <KpiCard label="Catamenial Identified" value={kpis.catamenial_identified}
                sub={`${kpis.catamenial_rate_pct}% of female patients`} color="red" />
            </div>
            <div className="col-6 col-md-3">
              <KpiCard label="Avg Seizure Reduction" value={kpis.avg_seizure_reduction_pct ? `${kpis.avg_seizure_reduction_pct}%` : "—"}
                sub="with targeted treatment" color="green" />
            </div>
            <div className="col-6 col-md-3">
              <KpiCard label="C1 / C2 / C3"
                value={`${kpis.phase_c1_count}/${kpis.phase_c2_count}/${kpis.phase_c3_count}`}
                sub="perimenstrual / periovulatory / luteal" color="purple" />
            </div>
          </div>

          {/* Cycle day seizure chart + phase distribution */}
          <div className="row g-3 mb-4">
            <div className="col-md-8">
              <div className="card">
                <div className="card-header fw-semibold">Aggregate Seizure Frequency by Cycle Day</div>
                <div className="card-body">
                  <div className="d-flex align-items-end gap-1" style={{ height: 120 }}>
                    {(overview?.seizure_by_cycle_day || []).map((d, i) => {
                      const max = 3.5;
                      const pct = (d.avg_seizures_per_day / max) * 100;
                      const isPerimens = d.cycle_day >= 26 || d.cycle_day <= 3;
                      const isOvul = d.cycle_day >= 10 && d.cycle_day <= 15;
                      const col = isPerimens ? "#dc3545" : isOvul ? "#fd7e14" : "#6c757d";
                      return (
                        <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center" }}>
                          <div title={`Day ${d.cycle_day}: ${d.avg_seizures_per_day}`}
                            style={{ width: "100%", height: `${pct}%`, background: col, borderRadius: "2px 2px 0 0", minHeight: 2 }} />
                          {(d.cycle_day % 7 === 1) && (
                            <div style={{ fontSize: 9, color: "#888" }}>{d.cycle_day}</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  <div className="d-flex gap-3 mt-2" style={{ fontSize: "0.75rem" }}>
                    <span><span style={{ background: "#dc3545", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} className="me-1" />C1 Perimenstrual</span>
                    <span><span style={{ background: "#fd7e14", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} className="me-1" />C2 Periovulatory</span>
                    <span><span style={{ background: "#6c757d", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} className="me-1" />Baseline</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card h-100">
                <div className="card-header fw-semibold">Phase Distribution</div>
                <div className="card-body">
                  {(overview?.phase_distribution || []).map(p => (
                    <div key={p.phase} className="mb-3">
                      <div className="d-flex justify-content-between small mb-1">
                        <span className="fw-semibold">{p.label.split(" ")[0]} <PhaseBadge phase={p.phase} /></span>
                        <span>{p.count} pts</span>
                      </div>
                      <div className="progress" style={{ height: 8 }}>
                        <div className="progress-bar bg-danger"
                          style={{ width: `${(p.count / (kpis.catamenial_identified || 1)) * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Severity + treatments */}
          <div className="row g-3 mb-4">
            <div className="col-md-4">
              <div className="card">
                <div className="card-header fw-semibold">Severity Distribution</div>
                <div className="card-body">
                  {(overview?.severity_distribution || []).map(s => {
                    const col = s.severity === "Severe" ? "bg-danger" : s.severity === "Moderate" ? "bg-warning" : "bg-success";
                    return (
                      <div key={s.severity} className="d-flex align-items-center gap-2 mb-2">
                        <span className={`badge ${col} ${s.severity === "Moderate" ? "text-dark" : "text-white"}`} style={{ width: 80 }}>{s.severity}</span>
                        <div className="progress flex-grow-1" style={{ height: 8 }}>
                          <div className={`progress-bar ${col}`}
                            style={{ width: `${(s.count / (kpis.catamenial_identified || 1)) * 100}%` }} />
                        </div>
                        <span className="small text-muted">{s.count}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
            <div className="col-md-8">
              <div className="card">
                <div className="card-header fw-semibold">Top Treatments</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Treatment</th><th>Evidence</th><th>Patterns</th></tr>
                    </thead>
                    <tbody>
                      {(overview?.top_treatments || []).map((t, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{t.name}</td>
                          <td><span className="badge bg-info text-dark">{t.evidence}</span></td>
                          <td>{t.suitable.split(", ").map(p => <PhaseBadge key={p} phase={p} />)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Biomarkers */}
          <div className="card">
            <div className="card-header fw-semibold">Hormonal Biomarkers — Catamenial Findings</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Biomarker</th><th>Catamenial Finding</th></tr>
                </thead>
                <tbody>
                  {(overview?.biomarkers_summary || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{b.name}</td>
                      <td className="small text-danger">{b.catamenial_finding}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="alert alert-secondary mt-3 small">{overview?.note}</div>
        </div>
      )}

      {/* ── Patients ── */}
      {tab === "Patients" && (
        <div>
          <div className="card mb-3">
            <div className="card-header fw-semibold">
              Per-Patient Catamenial Profile ({breakdown?.patient_table?.length || 0} female patients)
            </div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr>
                      <th>Patient</th><th>Age</th><th>Disease</th><th>Catamenial</th>
                      <th>Phase</th><th>Cycle</th><th>Prog (ng/mL)</th><th>E2 (pg/mL)</th>
                      <th>P/E</th><th>Cluster Score</th><th>Recommended Tx</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown?.patient_table || []).map((p, i) => (
                      <tr key={i}>
                        <td className="small fw-semibold">{p.patient_id}</td>
                        <td className="small">{p.age ?? "—"}</td>
                        <td className="small">{p.disease}</td>
                        <td>
                          {p.catamenial
                            ? <span className="badge bg-danger">Yes</span>
                            : <span className="badge bg-secondary">No</span>}
                        </td>
                        <td><PhaseBadge phase={p.phase} /></td>
                        <td className="small">{p.cycle_regularity}</td>
                        <td className={`small ${p.catamenial && p.progesterone_ng_ml < 3 ? "text-danger fw-bold" : ""}`}>
                          {p.progesterone_ng_ml}
                        </td>
                        <td className="small">{p.estradiol_pg_ml}</td>
                        <td className={`small ${p.pe_ratio < 1.0 ? "text-danger" : "text-success"}`}>
                          {p.pe_ratio}
                        </td>
                        <td>
                          <span className={`badge ${p.cluster_score >= 5 ? "bg-danger" : p.cluster_score >= 3 ? "bg-warning text-dark" : "bg-success"}`}>
                            {p.cluster_score}
                          </span>
                        </td>
                        <td className="small">{p.recommended_treatment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Phase profiles */}
          <div className="row g-3">
            {(breakdown?.phase_profiles || []).map((ph, i) => (
              <div key={i} className="col-md-4">
                <div className="card">
                  <div className={`card-header fw-semibold ${ph.id === "C1" ? "bg-danger text-white" : ph.id === "C2" ? "bg-warning text-dark" : "bg-info text-white"}`}>
                    {ph.label}
                  </div>
                  <div className="card-body small">
                    <div className="mb-2"><strong>Days:</strong> {ph.days}</div>
                    <div className="mb-2">{ph.description}</div>
                    <div className="mb-2"><strong>Mechanism:</strong> {ph.mechanism}</div>
                    <div className="mb-2">
                      <strong>Patients identified:</strong> {ph.n_patients}
                      {ph.n_patients > 0 && <> · avg cluster: <span className="text-danger fw-bold">{ph.avg_cluster_score}</span></>}
                    </div>
                    <div><strong>Treatments:</strong>
                      <ul className="mb-0 ps-3">
                        {(ph.recommended_treatments || []).map((t, j) => <li key={j}>{t}</li>)}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── AED Interactions ── */}
      {tab === "AED Interactions" && (
        <div>
          <div className="alert alert-warning small mb-3">
            <strong>Clinical Alert:</strong> Enzyme-inducing AEDs accelerate estrogen/progesterone
            metabolism — reducing contraceptive efficacy and disrupting hormonal catamenial therapy.
            Review at every appointment.
          </div>
          <div className="card mb-4">
            <div className="card-header fw-semibold">AED — Hormonal Interaction Table</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead className="table-light">
                  <tr><th>AED</th><th>Interaction</th><th>Risk</th><th>Clinical Action</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.aed_hormone_interactions || []).map((row, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{row.aed}</td>
                      <td className="small">{row.interaction}</td>
                      <td><RiskBadge risk={row.risk} /></td>
                      <td className="small">{row.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Treatment catalog */}
          <div className="card">
            <div className="card-header fw-semibold">Catamenial Treatment Catalog</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Treatment</th><th>Evidence</th><th>Mechanism</th><th>Dosing</th><th>Patterns</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.treatment_catalog || []).map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{t.name}</td>
                      <td><span className="badge bg-info text-dark">{t.evidence}</span></td>
                      <td className="small">{t.mechanism}</td>
                      <td className="small text-muted">{t.dosing}</td>
                      <td>{(t.suitable_patterns || []).map(p => <PhaseBadge key={p} phase={p} />)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Hormones ── */}
      {tab === "Hormones" && (
        <div>
          <div className="card mb-4">
            <div className="card-header fw-semibold">Hormonal Seizure-Risk Curve — 28-Day Cycle</div>
            <div className="card-body">
              <div className="row g-2 mb-2">
                <div className="col-12" style={{ fontSize: "0.75rem", color: "#666" }}>
                  Normalized hormone levels (0–1) + seizure risk index (higher = more vulnerable)
                </div>
              </div>
              <div style={{ overflowX: "auto" }}>
                <div className="d-flex align-items-end gap-1" style={{ height: 140, minWidth: 400 }}>
                  {(breakdown?.hormone_seizure_curve || []).map((d, i) => {
                    const riskH = d.seizure_risk_index * 140;
                    const progH = d.progesterone_norm * 140;
                    const e2H = d.estradiol_norm * 140;
                    return (
                      <div key={i} style={{ flex: 1, position: "relative", height: 140 }}
                        title={`Day ${d.cycle_day} | E2: ${d.estradiol_norm} | Prog: ${d.progesterone_norm} | Risk: ${d.seizure_risk_index}`}>
                        {/* Seizure risk */}
                        <div style={{ position: "absolute", bottom: 0, left: "30%", width: "40%",
                          height: riskH, background: "rgba(220,53,69,0.4)", borderRadius: "2px 2px 0 0" }} />
                        {/* E2 */}
                        <div style={{ position: "absolute", bottom: 0, left: "0%", width: "28%",
                          height: e2H, background: "#fd7e14", borderRadius: "2px 2px 0 0", opacity: 0.7 }} />
                        {/* Progesterone */}
                        <div style={{ position: "absolute", bottom: 0, left: "70%", width: "30%",
                          height: progH, background: "#0d6efd", borderRadius: "2px 2px 0 0", opacity: 0.7 }} />
                      </div>
                    );
                  })}
                </div>
                <div className="d-flex justify-content-between mt-1" style={{ fontSize: 9, color: "#888" }}>
                  {[1, 7, 14, 21, 28].map(d => (
                    <span key={d}>Day {d}</span>
                  ))}
                </div>
              </div>
              <div className="d-flex gap-3 mt-2" style={{ fontSize: "0.75rem" }}>
                <span><span style={{ background: "#fd7e14", display: "inline-block", width: 12, height: 12, borderRadius: 2, opacity: 0.7 }} className="me-1" />Estradiol (E2)</span>
                <span><span style={{ background: "#0d6efd", display: "inline-block", width: 12, height: 12, borderRadius: 2, opacity: 0.7 }} className="me-1" />Progesterone</span>
                <span><span style={{ background: "rgba(220,53,69,0.4)", display: "inline-block", width: 12, height: 12, borderRadius: 2 }} className="me-1" />Seizure Risk Index</span>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header fw-semibold">Biomarker Reference Values</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light">
                  <tr><th>Biomarker</th><th>Normal Range</th><th>Catamenial Finding</th></tr>
                </thead>
                <tbody>
                  {(breakdown?.biomarker_reference || []).map((b, i) => (
                    <tr key={i}>
                      <td className="fw-semibold small">{b.name}</td>
                      <td className="small">{b.normal_range}</td>
                      <td className="small text-danger">{b.catamenial_finding}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === "Definitions" && (
        <div>
          <div className="row g-3 mb-4">
            <div className="col-md-8">
              <div className="card">
                <div className="card-header fw-semibold">Clinical Concepts ({definitions?.concepts?.length})</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th style={{ width: "28%" }}>Term</th><th>Definition</th></tr>
                    </thead>
                    <tbody>
                      {(definitions?.concepts || []).map((c, i) => (
                        <tr key={i}>
                          <td className="fw-semibold small">{c.term}</td>
                          <td className="small">{c.definition}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-4">
              <div className="card mb-3">
                <div className="card-header fw-semibold">Guidelines</div>
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead className="table-light">
                      <tr><th>Body</th><th>Year</th><th>Recommendation</th></tr>
                    </thead>
                    <tbody>
                      {(definitions?.guidelines || []).map((g, i) => (
                        <tr key={i}>
                          <td className="small fw-semibold">{g.body}</td>
                          <td className="small">{g.year}</td>
                          <td className="small">{g.recommendation}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card">
                <div className="card-header fw-semibold">Screening Checklist</div>
                <ul className="list-group list-group-flush">
                  {(definitions?.screening_checklist || []).map((item, i) => (
                    <li key={i} className="list-group-item small d-flex gap-2">
                      <span className="text-success fw-bold">✓</span>{item}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header fw-semibold">References</div>
            <ul className="list-group list-group-flush">
              {(definitions?.references || []).map((r, i) => (
                <li key={i} className="list-group-item small">{r}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
