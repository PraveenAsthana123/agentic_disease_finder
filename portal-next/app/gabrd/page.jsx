'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const ACCENT = '#7b2d8b';
const ACCENT_LIGHT = '#f3e8ff';

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-warning py-2 small mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
        <strong>GABRD (1p36.33) — GABA-A δ Subunit · Extrasynaptic Tonic Inhibition:</strong>{' '}
        δ-GABA-A receptors mediate <strong>tonic (sustained) Cl⁻ inhibition</strong> — non-desensitizing, extrasynaptic, high GABA-affinity.{' '}
        GABRD LOF → loss of tonic inhibition → <strong>GGE / GEFS+ spectrum</strong>.{' '}
        <em>δ-GABA-A = primary neurosteroid (ALLO) target → CATAMENIAL C1 perimenstrual seizure clusters.</em>{' '}
        <em>δ-GABA-A = high-affinity ethanol target (3-10 mM) → ALCOHOL-WITHDRAWAL seizure risk.</em>{' '}
        <strong>Ganaxolone (synthetic ALLO) = precision therapy for GABRD δ-GABA-A LOF.</strong>{' '}
        <span className="text-danger fw-bold">ABSOLUTE CI: CBZ/OXC/PHT (GGE aggravation) · Tiagabine (NCSE). LTG: EEG pre-prescribing mandatory. POLG before VPA. VPPP females.</span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={ACCENT} />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Catamenial C1" value={`${kpis.catamenial_pct}%`} color="#e83e8c" />
        <KPI label="Alcohol Misuse" value={`${kpis.alcohol_misuse_pct}%`} color="#fd7e14" />
        <KPI label="Myoclonic Sz" value={`${kpis.myoclonic_pct}%`} color={ACCENT} />
        <KPI label="Absence Sz" value={`${kpis.absence_pct}%`} color="#0d6efd" />
        <KPI label="On VPA" value={`${kpis.on_vpa_pct}%`} color="#198754" />
        <KPI label="On ESM" value={`${kpis.on_esm_pct}%`} color="#0dcaf0" />
        <KPI label="On LEV" value={`${kpis.on_lev_pct}%`} color="#20c997" />
        <KPI label="On KD" value={`${kpis.on_kd_pct}%`} color="#ffc107" />
        <KPI label="Ganaxolone" value={kpis.on_ganaxolone_n} color="#7b2d8b" />
        <KPI label="SUDEP High Risk" value={kpis.sudep_high_risk_n} color="#dc3545" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: ACCENT }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.etiology}</span>
                    <span className="badge" style={{ background: ACCENT }}>{e.n} ({e.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.pct / maxEtio * 100)}%`, background: ACCENT }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: ACCENT }}>
              Treatment Arsenal
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="d-flex justify-content-between align-items-center border-bottom py-1 small">
                  <span className="fw-semibold">{t.drug}</span>
                  <span className="badge text-bg-light text-dark border">{t.level}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
              Monitoring Schedule
            </div>
            <div className="card-body">
              {monitoring.map((m, i) => (
                <div key={i} className="d-flex justify-content-between border-bottom py-1 small">
                  <span>{m.item}</span>
                  <span className="text-muted text-nowrap ms-2">{m.frequency}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
              Lifecycle Windows
            </div>
            <div className="card-body">
              {lifecycle.map((w, i) => (
                <div key={i} className="border-bottom py-1 small">
                  <span className="fw-semibold">{w.window}:</span>{' '}
                  <span className="text-muted">{w.headline}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-danger py-2">⛔ Key Thresholds</div>
        <div className="card-body">
          <div className="row g-2">
            {thresholds.map((t, i) => (
              <div key={i} className="col-md-4">
                <div className="border rounded p-2 small h-100">
                  <div className="fw-semibold">{t.label}</div>
                  <div className="text-primary">{t.value} <span className="text-muted">{t.unit}</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card shadow-sm border-danger mb-3">
        <div className="card-header fw-semibold text-danger py-2">⛔ Contraindications Summary</div>
        <div className="card-body">
          {(data.contraindications_summary || []).map((ci, i) => (
            <span key={i} className="badge bg-danger me-2 mb-1">{ci}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const etiologies = data.etiology_distribution || [];
  const patients = data.patient_sample || [];
  const summary = data.summary || {};

  return (
    <div>
      <div className="row g-3 mb-4">
        {[
          { label: "Drug Resistant", val: `${summary.drug_resistant_pct}%`, color: "#dc3545" },
          { label: "Catamenial C1", val: `${summary.catamenial_pct}%`, color: "#e83e8c" },
          { label: "Alcohol Misuse", val: `${summary.alcohol_misuse_pct}%`, color: "#fd7e14" },
          { label: "POLG Tested", val: `${summary.polg_tested_pct}%`, color: ACCENT },
        ].map((k, i) => (
          <div key={i} className="col-md-3">
            <div className="card text-center shadow-sm">
              <div className="card-body py-2">
                <div className="fw-bold fs-4" style={{ color: k.color }}>{k.val}</div>
                <div className="small text-muted">{k.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {summary.vpa_without_polg_n > 0 && (
        <div className="alert alert-danger small mb-3">
          ⚠️ <strong>{summary.vpa_without_polg_n} patients on VPA without POLG screen</strong> — Alpers-Huttenlocher risk. Screen immediately.
        </div>
      )}

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          Etiology Catalog
        </div>
        <div className="card-body p-0">
          {etiologies.map((e, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-semibold small">{e.etiology}</span>
                <span className="badge" style={{ background: ACCENT }}>{e.n} / {e.pct}%</span>
              </div>
              <div className="small text-muted mb-1">{e.mechanism}</div>
              <div className="small"><span className="fw-semibold">Variants: </span>{e.typical_variants}</div>
              <div className="small"><span className="fw-semibold">EEG: </span>{e.eeg_signature}</div>
              <div className="small text-primary"><span className="fw-semibold">Phenotype: </span>{e.phenotype}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          Patient Sample (n=15 of 40)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0 small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Sex</th><th>Age</th><th>Onset</th>
                <th>Category</th><th>VPA</th><th>ESM</th><th>LEV</th><th>CLB</th>
                <th>Ganax</th><th>KD</th><th>POLG</th><th>Catamen</th><th>Alc</th><th>DRE</th><th>SUDEP↑</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age}y</td>
                  <td>{p.onset_age}y</td>
                  <td><span className="badge bg-secondary text-wrap" style={{ fontSize: '0.65rem', maxWidth: 120 }}>{p.category}</span></td>
                  <td>{p.on_vpa ? '✅' : '—'}</td>
                  <td>{p.on_esm ? '✅' : '—'}</td>
                  <td>{p.on_lev ? '✅' : '—'}</td>
                  <td>{p.on_clb ? '✅' : '—'}</td>
                  <td>{p.on_ganaxolone ? '💜' : '—'}</td>
                  <td>{p.on_kd ? '🥑' : '—'}</td>
                  <td><span className={p.polg_tested === 'Y' ? 'text-success fw-bold' : 'text-danger fw-bold'}>{p.polg_tested}</span></td>
                  <td>{p.catamenial ? '🌙' : '—'}</td>
                  <td>{p.alcohol_misuse ? '⚠️' : '—'}</td>
                  <td>{p.drug_resistant ? '🔴' : '—'}</td>
                  <td>{p.sudep_high_risk ? '💀' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const seizures = data.seizure_detail || [];
  const triggers = data.trigger_detail || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: ACCENT }}>
          Seizure Types — GABRD Epilepsy
        </div>
        <div className="card-body p-0">
          {seizures.map((s, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fw-bold">{s.type}</span>
                <span className="badge" style={{ background: ACCENT }}>{s.prevalence_pct}%</span>
              </div>
              <div className="progress mb-2" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, background: ACCENT }} />
              </div>
              <div className="small text-muted mb-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
              <div className="small alert alert-warning py-1 px-2 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          Seizure Triggers — GABRD Specific
        </div>
        <div className="card-body">
          {triggers.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="d-flex justify-content-between small fw-semibold mb-1">
                <span>{t.trigger}</span>
                <span>{t.pct}%</span>
              </div>
              <div className="progress mb-1" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${t.pct}%`, background: i === 2 ? '#fd7e14' : i === 3 ? '#e83e8c' : ACCENT }} />
              </div>
              <div className="small text-muted">{t.note}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const treatments = data.treatment_detail || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: ACCENT }}>
          Treatment Arsenal — GABRD GGE / GEFS+ / Catamenial
        </div>
        <div className="card-body p-0">
          {treatments.map((t, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fw-bold">{t.drug}</span>
                <span className="badge text-bg-light text-dark border">{t.level}</span>
              </div>
              <div className="row g-2 small">
                <div className="col-md-6">
                  <div className="text-muted"><strong>MOA:</strong> {t.moa}</div>
                  <div className="text-muted"><strong>Dose:</strong> {t.dose}</div>
                  <div className="text-muted"><strong>Efficacy:</strong> {t.efficacy}</div>
                </div>
                <div className="col-md-6">
                  <div className="text-muted"><strong>Safety:</strong> {t.safety}</div>
                  <div className="text-muted"><strong>Monitoring:</strong> {t.monitoring}</div>
                </div>
              </div>
              <div className="small alert alert-primary py-1 px-2 mt-2 mb-0">
                <strong>GABRD Note:</strong> {t.gabrd_note}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm border-danger">
        <div className="card-header fw-semibold text-danger py-2">
          ⛔ Contraindications — GABRD Epilepsy
        </div>
        <div className="card-body p-0">
          {contraindications.map((c, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-bold text-danger">{c.drug}</span>
                <span className={`badge ${c.risk.includes('ABSOLUTE') ? 'bg-danger' : c.risk.includes('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                  {c.risk}
                </span>
              </div>
              <div className="small text-muted">{c.reason}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const references = data.references || [];
  const cis = data.contraindications || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          15 Core Concepts — GABRD Epilepsy
        </div>
        <div className="card-body p-0">
          {concepts.map((c, i) => (
            <div key={i} className="border-bottom px-3 py-2">
              <span className="fw-semibold" style={{ color: ACCENT }}>{c.term}: </span>
              <span className="small text-muted">{c.definition}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
              Clinical Thresholds
            </div>
            <div className="card-body p-0">
              {thresholds.map((t, i) => (
                <div key={i} className="border-bottom px-3 py-2 small">
                  <div className="fw-semibold">{t.label}</div>
                  <div className="text-primary">{t.value} <span className="text-muted">{t.unit}</span></div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100 border-danger">
            <div className="card-header fw-semibold text-danger py-2">⛔ Contraindication Summary</div>
            <div className="card-body p-0">
              {cis.map((c, i) => (
                <div key={i} className="border-bottom px-3 py-2 small d-flex justify-content-between">
                  <span className="text-danger fw-semibold">{c.drug}</span>
                  <span className={`badge ${c.risk.includes('ABSOLUTE') ? 'bg-danger' : c.risk.includes('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                    {c.risk}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          Clinical Standards
        </div>
        <div className="card-body p-0">
          {standards.map((s, i) => (
            <div key={i} className="border-bottom px-3 py-1 small">{s}</div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          Key References
        </div>
        <div className="card-body p-0">
          {references.map((r, i) => (
            <div key={i} className="border-bottom px-3 py-1 small">{r}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function GABRDPage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gabrd/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (activeTab >= 1 && activeTab <= 3 && !breakdown) {
      fetch(`${API}/api/gabrd/breakdown`)
        .then(r => r.json())
        .then(setBreakdown)
        .catch(e => setError(e.message));
    }
    if (activeTab === 4 && !definitions) {
      fetch(`${API}/api/gabrd/definitions`)
        .then(r => r.json())
        .then(setDefinitions)
        .catch(e => setError(e.message));
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="mb-1" style={{ color: ACCENT }}>
          🧬 GABRD Epilepsy — GGE / GEFS+ Spectrum / Catamenial / Tonic Inhibition
        </h4>
        <p className="text-muted small mb-0">
          GABRD (1p36.33) · GABA-A receptor δ (delta) subunit · Extrasynaptic tonic inhibition ·
          Neurosteroid (allopregnanolone) target · Catamenial C1 (perimenstrual ALLO withdrawal) ·
          Ganaxolone precision therapy · AD reduced penetrance (~65%) · 40-patient cohort · ILAE 2022
        </p>
      </div>

      {error && (
        <div className="alert alert-danger small py-2">
          Backend error: {error}
        </div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <SeizuresTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
