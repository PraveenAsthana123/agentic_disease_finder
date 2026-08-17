'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// Teal/green colour theme for SLC13A5 (metabolic/citrate identity)
const C = '#0d7c66';
const CL = '#e0f5f1';

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

function Bar({ label, value, color = C }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
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
      <div className="alert alert-warning py-2 small mb-3" style={{ borderLeft: `4px solid ${C}` }}>
        <strong>SLC13A5 (17p13.1) — NaCT Citrate Transporter:</strong>{' '}
        Biallelic LOF → no BBB citrate import → depleted neuronal citrate → impaired TCA anaplerosis →
        reduced GABA synthesis → <strong>neonatal seizures (Day 1-5)</strong>.{' '}
        <em>PATHOGNOMONIC: <strong>Dental enamel hypoplasia</strong>. Biomarker: <strong>elevated urine citrate</strong>.</em>{' '}
        Precision therapy: <strong>Triheptanoin (C7 anaplerotic — bypasses NaCT, restores TCA flux)</strong>.{' '}
        <span className="text-danger fw-bold">
          CAUTION: Standard 4:1 KD may DEEPEN citrate deficit — use modified KD + Triheptanoin only.
          Tiagabine ABSOLUTE CI. VPA HIGH CAUTION (POLG mandatory). Pyridoxine trial ALL neonatal seizures.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={C} />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Dental Defects" value={`${kpis.dental_defect_pct}%`} color="#fd7e14" />
        <KPI label="Urine Citrate ↑" value={`${kpis.urine_citrate_elevated_pct}%`} color={C} />
        <KPI label="On PB" value={`${kpis.on_pb_pct}%`} color="#6c757d" />
        <KPI label="On LEV" value={`${kpis.on_lev_pct}%`} color="#198754" />
        <KPI label="On Triheptanoin" value={`${kpis.on_triheptanoin_pct}%`} color={C} />
        <KPI label="On KD (modified)" value={`${kpis.on_kd_pct}%`} color="#ffc107" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#6c757d" />
        <KPI label="SUDEP High Risk" value={kpis.sudep_high_risk_n} color="#dc3545" />
        <KPI label="SE History" value={kpis.se_history_n} color="#dc3545" />
        <KPI label="Avg Age (y)" value={kpis.avg_age_years} color="#0dcaf0" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.etiology}</span>
                    <span className="badge" style={{ background: C }}>{e.n} ({e.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.pct / maxEtio * 100)}%`, background: C }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
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
            <div className="card-header fw-semibold py-2" style={{ background: CL }}>
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
            <div className="card-header fw-semibold py-2" style={{ background: CL }}>
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
        <div className="card-header fw-semibold text-danger py-2">
          ⛔ Key Thresholds
        </div>
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
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4 text-danger">{summary.drug_resistant_pct}%</div>
              <div className="small text-muted">Drug Resistant</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: '#fd7e14' }}>{summary.dental_defect_pct}%</div>
              <div className="small text-muted">Dental Enamel Defects</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: C }}>{summary.urine_citrate_elevated_pct}%</div>
              <div className="small text-muted">Urine Citrate Elevated</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: C }}>{summary.on_triheptanoin_n}</div>
              <div className="small text-muted">On Triheptanoin (C7)</div>
            </div>
          </div>
        </div>
      </div>

      {summary.status_epilepticus_hx_n > 0 && (
        <div className="alert alert-danger small mb-3">
          ⚠️ <strong>{summary.status_epilepticus_hx_n} patients with neonatal status epilepticus history</strong> — highest SUDEP risk; nocturnal monitoring mandatory.
        </div>
      )}

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Etiology Catalog
        </div>
        <div className="card-body p-0">
          {etiologies.map((e, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-semibold small">{e.etiology}</span>
                <span className="badge" style={{ background: C }}>{e.n} / {e.pct}%</span>
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
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Patient Sample (n=15 of 40)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0 small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Sex</th><th>Age</th><th>Onset Day</th>
                <th>Category</th><th>PB</th><th>LEV</th><th>VGB</th><th>OXC</th><th>C7</th><th>KD</th>
                <th>POLG</th><th>Dental</th><th>Citrate↑</th><th>DRE</th><th>SE Hx</th><th>SUDEP↑</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age}y</td>
                  <td>D{p.onset_day}</td>
                  <td><span className="badge bg-secondary text-wrap" style={{ fontSize: '0.65rem', maxWidth: 120 }}>{p.category}</span></td>
                  <td>{p.on_pb ? '✅' : '—'}</td>
                  <td>{p.on_lev ? '✅' : '—'}</td>
                  <td>{p.on_vgb ? '✅' : '—'}</td>
                  <td>{p.on_oxc ? '✅' : '—'}</td>
                  <td>{p.on_triheptanoin ? <span style={{ color: C }}>C7</span> : '—'}</td>
                  <td>{p.on_kd ? '🥑' : '—'}</td>
                  <td><span className={p.polg_tested === 'Y' ? 'text-success fw-bold' : 'text-danger fw-bold'}>{p.polg_tested}</span></td>
                  <td>{p.dental_defect ? '🦷' : '—'}</td>
                  <td>{p.urine_citrate_elevated ? '⬆️' : '—'}</td>
                  <td>{p.drug_resistant ? '🔴' : '—'}</td>
                  <td>{p.status_epilepticus_hx ? '⚠️' : '—'}</td>
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
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Seizure Types — SLC13A5 / EIEE25
        </div>
        <div className="card-body p-0">
          {seizures.map((s, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fw-bold">{s.type}</span>
                <span className="badge" style={{ background: C }}>{s.prevalence_pct}%</span>
              </div>
              <div className="progress mb-2" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, background: C }} />
              </div>
              <div className="small text-muted mb-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
              <div className="small alert alert-warning py-1 px-2 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Seizure Triggers — SLC13A5 Specific
        </div>
        <div className="card-body">
          {triggers.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="d-flex justify-content-between small fw-semibold mb-1">
                <span>{t.trigger}</span>
                <span>{t.pct}%</span>
              </div>
              <div className="progress mb-1" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${t.pct}%`, background: i === 0 ? '#dc3545' : C }} />
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
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Treatment Arsenal — SLC13A5 / EIEE25
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
                <strong>SLC13A5 Note:</strong> {t.slc13a5_note}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm border-danger">
        <div className="card-header fw-semibold text-danger py-2">
          ⛔ Contraindications — SLC13A5 / EIEE25
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
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          15 Core Concepts — SLC13A5 / EIEE25
        </div>
        <div className="card-body p-0">
          {concepts.map((c, i) => (
            <div key={i} className="border-bottom px-3 py-2">
              <span className="fw-semibold" style={{ color: C }}>{c.term}: </span>
              <span className="small text-muted">{c.definition}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: CL }}>
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
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Clinical Standards
        </div>
        <div className="card-body p-0">
          {standards.map((s, i) => (
            <div key={i} className="border-bottom px-3 py-1 small">{s}</div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
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

export default function SLC13A5Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/slc13a5/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (activeTab >= 1 && activeTab <= 3 && !breakdown) {
      fetch(`${API}/api/slc13a5/breakdown`)
        .then(r => r.json())
        .then(setBreakdown)
        .catch(e => setError(e.message));
    }
    if (activeTab === 4 && !definitions) {
      fetch(`${API}/api/slc13a5/definitions`)
        .then(r => r.json())
        .then(setDefinitions)
        .catch(e => setError(e.message));
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="mb-1" style={{ color: C }}>
          🦷 SLC13A5 Epilepsy — Citrate Transporter Deficiency / NAFE / EIEE25
        </h4>
        <p className="text-muted small mb-0">
          SLC13A5 (17p13.1) · NaCT sodium-coupled citrate transporter · BBB citrate import → TCA anaplerosis → GABA ·
          Autosomal Recessive (AR) · Neonatal onset Day 1-5 · Dental enamel hypoplasia (pathognomonic) ·
          Triheptanoin (C7) precision therapy · 40-patient cohort · ILAE 2022
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
