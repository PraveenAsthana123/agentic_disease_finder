"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8010";
const TABS = ["Overview", "Patients & Variants", "Triggers & Contraindications", "Treatments", "Definitions"];

// ─── small reusable components ────────────────────────────────────────────────

function KpiCard({ label, value, sub, color = "primary" }) {
  const colorMap = { primary: "border-primary text-primary", danger: "border-danger text-danger", success: "border-success text-success", warning: "border-warning text-warning", info: "border-info text-info", secondary: "border-secondary text-secondary" };
  const [border, text] = (colorMap[color] || colorMap.primary).split(" ");
  return (
    <div className={`card border-2 ${border} h-100`}>
      <div className="card-body text-center py-3">
        <div className={`fs-2 fw-bold ${text}`}>{value ?? "—"}</div>
        <div className="small fw-semibold text-muted">{label}</div>
        {sub && <div className="text-muted" style={{ fontSize: "0.72rem" }}>{sub}</div>}
      </div>
    </div>
  );
}

function InlineBar({ label, pct, color = "#0d6efd", count }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="text-muted">{count !== undefined ? `${count} (${pct}%)` : `${pct}%`}</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

function SeverityBadge({ severity }) {
  const colors = { ABSOLUTE: "danger", HIGH: "warning", MODERATE: "info" };
  return <span className={`badge bg-${colors[severity] || "secondary"}`}>{severity}</span>;
}

function EvidenceBadge({ evidence }) {
  const lvl = (evidence || "").split(" ")[1];
  const colors = { A: "success", B: "primary", C: "secondary" };
  return <span className={`badge bg-${colors[lvl] || "secondary"} ms-1`}>{evidence}</span>;
}

function FdaBadge({ status }) {
  if (!status) return null;
  const isApproved = status.includes("FDA-approved");
  const isLevelA = status.includes("Level A");
  const color = isApproved ? "success" : isLevelA ? "primary" : "secondary";
  return <span className={`badge bg-${color} ms-1`} style={{ fontSize: "0.68rem" }}>{isApproved ? "FDA-Approved" : isLevelA ? "Level A" : "Off-label"}</span>;
}

// ─── tabs ─────────────────────────────────────────────────────────────────────

function OverviewTab({ ov, bd }) {
  if (!ov) return <div className="text-muted">Loading overview…</div>;
  const k = ov.kpi || {};
  const variantColors = ["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#6b7280"];

  return (
    <div>
      {/* alert banner */}
      <div className="alert alert-danger d-flex align-items-center py-2 mb-3">
        <span className="me-2 fs-5">⚠️</span>
        <span className="small"><strong>CONTRAINDICATION:</strong> Carbamazepine, Lamotrigine, and Phenytoin are <strong>ABSOLUTELY contraindicated</strong> in Dravet syndrome — they paradoxically worsen seizures via Nav1.1 inhibition in GABAergic interneurons.</span>
      </div>

      {/* KPIs */}
      <div className="row g-3 mb-4">
        <div className="col-6 col-md-3"><KpiCard label="Dravet Patients" value={k.total_patients} sub="SCN1A cohort" color="primary" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Avg Onset" value={`${k.avg_onset_months} mo`} sub="median ~5 months" color="info" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Avg Seizures/Month" value={k.avg_seizures_per_month} sub="pharmacoresistant" color="danger" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Responder Rate" value={`${k.responder_pct}%`} sub="≥50% reduction" color="success" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Pharmacoresistant" value={`${k.pharmacoresistant_pct}%`} sub="<50% reduction" color="warning" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Avg SUDEP Risk" value={k.avg_sudep_risk_score} sub="score 1–10" color="danger" /></div>
        <div className="col-6 col-md-3"><KpiCard label="FDA-Approved Tx" value={k.fda_approved_therapies} sub="CBD · fenfluramine · stiripentol" color="success" /></div>
        <div className="col-6 col-md-3"><KpiCard label="Contraindicated AEDs" value={k.contraindicated_aeds} sub="absolute + high risk" color="secondary" /></div>
      </div>

      <div className="row g-3 mb-4">
        {/* SCN1A variant distribution */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>SCN1A Variant Distribution</strong></div>
            <div className="card-body">
              {(ov.scn1a_variant_distribution || []).map((v, i) => (
                <InlineBar key={v.variant} label={v.variant} pct={v.pct} count={v.count} color={variantColors[i % variantColors.length]} />
              ))}
            </div>
          </div>
        </div>

        {/* Seizure frequency histogram */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Seizure Frequency / Month</strong></div>
            <div className="card-body">
              {(ov.seizure_frequency_histogram || []).map(b => (
                <InlineBar key={b.bin} label={`${b.bin} seizures/mo`} pct={Math.round(b.count / (k.total_patients || 1) * 100)} count={b.count} color="#ef4444" />
              ))}
            </div>
          </div>
        </div>

        {/* Comorbidities */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Comorbidity Prevalence</strong></div>
            <div className="card-body">
              {(ov.comorbidity_prevalence || []).map(c => (
                <InlineBar key={c.comorbidity} label={c.comorbidity} pct={c.pct} count={c.count} color="#f59e0b" />
              ))}
            </div>
          </div>
        </div>

        {/* Top triggers */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Top Seizure Triggers</strong></div>
            <div className="card-body">
              {(ov.top_triggers || []).map(t => (
                <div key={t.trigger} className="d-flex justify-content-between align-items-center mb-2">
                  <div>
                    <span className="small fw-semibold">{t.trigger}</span>
                    <EvidenceBadge evidence={t.evidence} />
                  </div>
                  <span className="badge bg-danger">{t.prevalence_pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Treatment use */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Current Treatment Use Distribution</strong></div>
            <div className="card-body">
              {(ov.treatment_use_distribution || []).map(d => (
                <InlineBar key={d.drug} label={d.drug} pct={d.pct} count={d.count} color="#10b981" />
              ))}
            </div>
          </div>
        </div>

        {/* Developmental trajectory */}
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header py-2"><strong>Developmental Trajectory</strong></div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead className="table-light"><tr><th style={{width:"28%"}}>Age</th><th>Dravet Pattern</th></tr></thead>
                <tbody>
                  {(ov.milestone_summary || []).map(m => (
                    <tr key={m.age_window}>
                      <td className="small fw-semibold text-nowrap">{m.age_window}</td>
                      <td className="small text-muted">{m.dravet_pattern}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* References */}
      <div className="card bg-light">
        <div className="card-body py-2">
          <div className="small fw-semibold mb-1">Key References</div>
          {(ov.references || []).map((r, i) => (
            <div key={i} className="text-muted" style={{ fontSize: "0.72rem" }}>• {r}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ bd }) {
  const [search, setSearch] = useState("");
  const [respFilter, setRespFilter] = useState("all");
  const [varFilter, setVarFilter] = useState("all");
  if (!bd) return <div className="text-muted">Loading…</div>;

  const pts = (bd.patient_table || []).filter(p => {
    const matchSearch = !search || p.patient_id.toLowerCase().includes(search.toLowerCase());
    const matchResp = respFilter === "all" || (respFilter === "responder" ? p.responder : !p.responder);
    const matchVar = varFilter === "all" || p.scn1a_variant === varFilter;
    return matchSearch && matchResp && matchVar;
  });

  const variantClasses = [...new Set((bd.patient_table || []).map(p => p.scn1a_variant))].sort();

  return (
    <div>
      <div className="row g-2 mb-3">
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Search patient ID…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={respFilter} onChange={e => setRespFilter(e.target.value)}>
            <option value="all">All patients</option>
            <option value="responder">Responders (≥50%)</option>
            <option value="non-responder">Non-responders</option>
          </select>
        </div>
        <div className="col-md-4">
          <select className="form-select form-select-sm" value={varFilter} onChange={e => setVarFilter(e.target.value)}>
            <option value="all">All SCN1A variants</option>
            {variantClasses.map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      </div>
      <div className="small text-muted mb-2">Showing {pts.length} of {(bd.patient_table || []).length} patients</div>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-dark">
            <tr>
              <th>Patient</th><th>Age</th><th>Onset</th><th>SCN1A Variant</th>
              <th>Sz/Mo</th><th>Regimen</th><th>Reduction</th><th>Responder</th><th>SUDEP</th><th>Comorbidities</th>
            </tr>
          </thead>
          <tbody>
            {pts.map(p => (
              <tr key={p.patient_id}>
                <td className="small">{p.patient_id}</td>
                <td className="small">{p.age}y</td>
                <td className="small">{p.onset_months}mo</td>
                <td className="small text-truncate" style={{ maxWidth: 120 }} title={p.scn1a_variant}>{p.scn1a_variant}</td>
                <td className="small text-center"><span className="badge bg-danger">{p.seizures_per_month}</span></td>
                <td className="small" style={{ maxWidth: 120 }} title={p.current_regimen}>{p.current_regimen}</td>
                <td className="small text-center">
                  <span className={`badge ${p.pct_seizure_reduction >= 50 ? "bg-success" : p.pct_seizure_reduction >= 30 ? "bg-warning text-dark" : "bg-secondary"}`}>
                    {p.pct_seizure_reduction}%
                  </span>
                </td>
                <td className="small text-center">{p.responder ? "✅" : "❌"}</td>
                <td className="small text-center">
                  <span className={`badge ${p.sudep_risk_score >= 7 ? "bg-danger" : p.sudep_risk_score >= 4 ? "bg-warning text-dark" : "bg-success"}`}>
                    {p.sudep_risk_score}/10
                  </span>
                </td>
                <td className="small text-muted text-truncate" style={{ maxWidth: 150 }} title={p.comorbidities}>{p.comorbidities}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* SCN1A variant catalog */}
      <div className="mt-4">
        <h6 className="fw-semibold">SCN1A Variant Catalog</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead className="table-light">
              <tr><th>Class</th><th>Example</th><th>Prevalence</th><th>Phenotype</th><th>Channel Effect</th></tr>
            </thead>
            <tbody>
              {(bd.scn1a_variant_catalog || []).map(v => (
                <tr key={v.class}>
                  <td className="small fw-semibold">{v.class}</td>
                  <td className="small font-monospace">{v.example}</td>
                  <td className="small text-center"><span className="badge bg-info text-dark">{v.pct}%</span></td>
                  <td className="small">{v.phenotype}</td>
                  <td className="small text-muted">{v.channel_effect}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function TriggersTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      {/* Triggers */}
      <h6 className="fw-semibold mb-3">Seizure Triggers — Prevalence & Management</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover">
          <thead className="table-dark">
            <tr><th>Trigger</th><th>Prevalence</th><th>Mechanism</th><th>Management</th><th>Evidence</th></tr>
          </thead>
          <tbody>
            {(bd.trigger_catalog || []).sort((a, b) => b.prevalence_pct - a.prevalence_pct).map(t => (
              <tr key={t.trigger}>
                <td className="small fw-semibold">{t.trigger}</td>
                <td className="small text-center"><span className="badge bg-warning text-dark">{t.prevalence_pct}%</span></td>
                <td className="small text-muted">{t.mechanism}</td>
                <td className="small">{t.management}</td>
                <td><EvidenceBadge evidence={t.evidence} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Contraindicated AEDs */}
      <div className="alert alert-danger py-2 mb-3">
        <strong>⛔ ABSOLUTE / HIGH-RISK CONTRAINDICATIONS</strong> — The following AEDs paradoxically worsen Dravet syndrome seizures by blocking Nav1.1 in GABAergic interneurons. Prescribing must be avoided.
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-bordered table-hover">
          <thead className="table-danger">
            <tr><th>Drug</th><th>Severity</th><th>Mechanism</th><th>Evidence</th></tr>
          </thead>
          <tbody>
            {(bd.contraindicated_aeds || []).map(c => (
              <tr key={c.aed}>
                <td className="small fw-semibold">{c.aed}</td>
                <td><SeverityBadge severity={c.severity} /></td>
                <td className="small text-muted">{c.mechanism}</td>
                <td className="small">{c.evidence}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TreatmentsTab({ bd }) {
  const [selected, setSelected] = useState(null);
  if (!bd) return <div className="text-muted">Loading…</div>;

  return (
    <div>
      <div className="mb-3 small text-muted">Click a row to expand drug detail.</div>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-dark">
            <tr><th>Drug</th><th>Status</th><th>Year</th><th>Dose</th><th>Efficacy</th></tr>
          </thead>
          <tbody>
            {(bd.approved_treatments || []).map(t => (
              <>
                <tr key={t.drug} style={{ cursor: "pointer" }} onClick={() => setSelected(selected === t.drug ? null : t.drug)}>
                  <td className="small fw-semibold">{t.drug}</td>
                  <td><FdaBadge status={t.fda_status} /></td>
                  <td className="small">{t.year}</td>
                  <td className="small text-muted" style={{ maxWidth: 180 }}>{t.dose}</td>
                  <td className="small text-muted text-truncate" style={{ maxWidth: 200 }} title={t.efficacy}>{t.efficacy}</td>
                </tr>
                {selected === t.drug && (
                  <tr key={t.drug + "_detail"}>
                    <td colSpan={5} className="bg-light">
                      <div className="p-2">
                        <div className="mb-1"><strong>FDA Status:</strong> <span className="small text-muted">{t.fda_status}</span></div>
                        <div className="mb-1"><strong>Mechanism:</strong> <span className="small text-muted">{t.moa}</span></div>
                        <div className="mb-1"><strong>Efficacy:</strong> <span className="small text-success">{t.efficacy}</span></div>
                        <div><strong>Safety / Monitoring:</strong> <span className="small text-danger">{t.safety}</span></div>
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>

      {/* Developmental trajectory */}
      <div className="mt-4">
        <h6 className="fw-semibold">Developmental Trajectory Milestones</h6>
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead className="table-light"><tr><th>Age Window</th><th>Typical Expected</th><th>Dravet Pattern</th></tr></thead>
            <tbody>
              {(bd.developmental_trajectory || []).map(m => (
                <tr key={m.age_window}>
                  <td className="small fw-semibold text-nowrap">{m.age_window}</td>
                  <td className="small text-muted">{m.expected}</td>
                  <td className="small">{m.dravet_pattern}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      {/* Concepts */}
      <h6 className="fw-semibold mb-3">Clinical Concepts</h6>
      <div className="row g-3 mb-4">
        {(defs.concepts || []).map(c => (
          <div key={c.term} className="col-md-6">
            <div className="card h-100 border-0 shadow-sm">
              <div className="card-body py-2">
                <div className="fw-semibold small">{c.term}</div>
                <div className="text-muted small mt-1">{c.definition}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Standards */}
      <h6 className="fw-semibold mb-2">Standards & Guidelines</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered">
          <thead className="table-light"><tr><th>Standard</th><th>Reference</th><th>Note</th></tr></thead>
          <tbody>
            {(defs.standards || []).map(s => (
              <tr key={s.standard}>
                <td className="small fw-semibold">{s.standard}</td>
                <td className="small text-muted">{s.reference}</td>
                <td className="small">{s.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Thresholds */}
      <h6 className="fw-semibold mb-2">Key Clinical Thresholds</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered">
          <thead className="table-light"><tr><th>Threshold</th><th>Value</th><th>Clinical Use</th></tr></thead>
          <tbody>
            {(defs.key_thresholds || []).map(t => (
              <tr key={t.threshold}>
                <td className="small fw-semibold">{t.threshold}</td>
                <td className="small text-info">{t.value}</td>
                <td className="small text-muted">{t.clinical_use}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── main page ────────────────────────────────────────────────────────────────

export default function DravetPage() {
  const [tab, setTab] = useState("Overview");
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/dravet/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
  }, []);

  useEffect(() => {
    if (tab === "Patients & Variants" || tab === "Triggers & Contraindications" || tab === "Treatments") {
      if (!breakdown) {
        fetch(`${API}/api/dravet/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(String(e)));
      }
    }
    if (tab === "Definitions" && !definitions) {
      fetch(`${API}/api/dravet/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(String(e)));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      {/* header */}
      <div className="d-flex justify-content-between align-items-start mb-3 flex-wrap gap-2">
        <div>
          <h4 className="mb-0">🧬 Dravet Syndrome Dashboard</h4>
          <div className="text-muted small">SCN1A-Driven Severe Childhood Epilepsy · Pharma­coresistance · FDA-Approved Therapies · SUDEP Risk</div>
        </div>
        <div className="text-end">
          <span className="badge bg-danger me-1">Nav1.1 LoF</span>
          <span className="badge bg-warning text-dark me-1">Thermosensitive</span>
          <span className="badge bg-success">3 FDA-Approved Tx</span>
        </div>
      </div>

      {error && <div className="alert alert-danger small">API error: {error}</div>}

      {/* tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === "Overview" && <OverviewTab ov={overview} bd={breakdown} />}
      {tab === "Patients & Variants" && <PatientsTab bd={breakdown} />}
      {tab === "Triggers & Contraindications" && <TriggersTab bd={breakdown} />}
      {tab === "Treatments" && <TreatmentsTab bd={breakdown} />}
      {tab === "Definitions" && <DefinitionsTab defs={definitions} />}

      <div className="mt-4 text-muted" style={{ fontSize: "0.7rem" }}>
        Data: 41-patient clinical.db epilepsy cohort with deterministic SCN1A variant overlay ·
        Pharmacology: Wirrell 2022 Neurology · CARE1/CARE2 (Devinsky 2017/2018) · STICLO (Chiron 2000) ·
        FDA approvals: Epidiolex 2018 · Diacomit 2018 · Fintepla 2020
      </div>
    </div>
  );
}
