'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#7b1fa2';
const LIGHT = '#f3e5f5';

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

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
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
      <div className="alert alert-warning py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
        <strong>GABRA3 (Xq28) — GABA-A α3 Subunit · X-linked:</strong>{' '}
        α3 is the <strong>dominant fetal/neonatal α-isoform</strong> — GABRA3 LOF is most harmful during early brain development.{' '}
        α3 enriched in <strong>Thalamic Reticular Nucleus (TRN)</strong> → TRN LOF → thalamo-cortical hyperexcitability → tonic seizures + absent sleep spindles.{' '}
        <strong>Hyperekplexia (startle disease)</strong> is a hallmark — GOF p.Ile246Val; clonazepam + forward-flexion manoeuvre.{' '}
        <span className="text-danger fw-bold">
          PHB preferred rescue (BZD site preserved); CLB reduced efficacy in LOF (α3 target absent).{' '}
          ABSOLUTE CI: LTG (myoclonic) · TGB (NCSE) · VPA without POLG screen.{' '}
          VGB ≤16 weeks (REMS). POLG mandatory before VPA.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Epileptic Spasms" value={`${kpis.spasms_pct}%`} color="#dc3545" />
        <KPI label="Tonic Seizures" value={`${kpis.tonic_pct}%`} color={COLOR} />
        <KPI label="Hyperekplexia" value={`${kpis.hyperekplexia_pct}%`} color="#fd7e14" />
        <KPI label="Myoclonic Sz" value={`${kpis.myoclonic_pct}%`} color="#6610f2" />
        <KPI label="GTCS" value={`${kpis.gtcs_pct}%`} color="#dc3545" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#198754" />
        <KPI label="On PHB" value={`${kpis.on_phb_pct}%`} color={COLOR} />
        <KPI label="On VPA" value={`${kpis.on_vpa_pct}%`} color="#0d6efd" />
        <KPI label="On CLZ" value={`${kpis.on_clz_pct}%`} color="#0dcaf0" />
        <KPI label="SUDEP High Risk" value={kpis.sudep_high_risk_n} color="#dc3545" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.etiology}</span>
                    <span className="badge" style={{ background: COLOR }}>{e.n} ({e.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.pct / maxEtio * 100)}%`, background: COLOR }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
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
            <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
              Monitoring Schedule (Key Items)
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
            <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
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

      <div className="card shadow-sm border-warning mb-3">
        <div className="card-header fw-semibold py-2" style={{ background: '#fff3cd' }}>
          🧬 GABRA3 X-linked Inheritance — Family Cascade
        </div>
        <div className="card-body small">
          <div className="row g-2">
            <div className="col-md-4 border-end">
              <div className="fw-semibold text-danger mb-1">Hemizygous Males (XY)</div>
              <div className="text-muted">Complete α3 LOF → severe DEE; neonatal onset; hyperekplexia + apnoea; high SUDEP risk; profound ID</div>
            </div>
            <div className="col-md-4 border-end">
              <div className="fw-semibold" style={{ color: COLOR }}>Heterozygous Females (XX)</div>
              <div className="text-muted">Outcome governed by X-inactivation (XCI) skewing: &gt;70% mutant active → severe DEE; random XCI → intermediate; favourable XCI → mild GGE</div>
            </div>
            <div className="col-md-4">
              <div className="fw-semibold text-info mb-1">Cascade Testing</div>
              <div className="text-muted">ALL male first-degree relatives of carrier females must be tested. Xq28 MLPA/SNP array to exclude GABRA3+MECP2 co-deletion.</div>
            </div>
          </div>
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
              <div className="fw-bold fs-4" style={{ color: '#fd7e14' }}>{summary.hyperekplexia_pct}%</div>
              <div className="small text-muted">Hyperekplexia</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: COLOR }}>{summary.acth_trial_pct}%</div>
              <div className="small text-muted">ACTH Trial</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4 text-success">{summary.polg_tested_pct}%</div>
              <div className="small text-muted">POLG Tested</div>
            </div>
          </div>
        </div>
      </div>

      {summary.vpa_without_polg_n > 0 && (
        <div className="alert alert-danger small mb-3">
          ⚠️ <strong>{summary.vpa_without_polg_n} patients on VPA without POLG screen</strong> — Alpers-Huttenlocher risk. Screen immediately.
        </div>
      )}

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
          Etiology Catalog
        </div>
        <div className="card-body p-0">
          {etiologies.map((e, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-semibold small">{e.etiology}</span>
                <span className="badge" style={{ background: COLOR }}>{e.n} / {e.pct}%</span>
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
        <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
          Patient Sample (n=15 of 40)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0 small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Sex</th><th>Age</th><th>Onset</th>
                <th>Category</th><th>PHB</th><th>VPA</th><th>CLZ</th><th>LEV</th><th>KD</th>
                <th>POLG</th><th>Spasms</th><th>Tonic</th><th>Hyperek</th><th>DRE</th><th>SUDEP↑</th>
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
                  <td>{p.on_phb ? '✅' : '—'}</td>
                  <td>{p.on_vpa ? '✅' : '—'}</td>
                  <td>{p.on_clz ? '✅' : '—'}</td>
                  <td>{p.on_lev ? '✅' : '—'}</td>
                  <td>{p.on_kd ? '✅' : '—'}</td>
                  <td><span className={p.polg_tested === 'Y' ? 'text-success fw-bold' : 'text-danger fw-bold'}>{p.polg_tested}</span></td>
                  <td>{p.spasms ? '⚡' : '—'}</td>
                  <td>{p.tonic_seizures ? '⚡' : '—'}</td>
                  <td>{p.hyperekplexia ? '⚠️' : '—'}</td>
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
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          Seizure Types — GABRA3 Epilepsy
        </div>
        <div className="card-body p-0">
          {seizures.map((s, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fw-bold">{s.type}</span>
                <span className="badge" style={{ background: COLOR }}>{s.prevalence_pct}%</span>
              </div>
              <div className="progress mb-2" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, background: COLOR }} />
              </div>
              <div className="small text-muted mb-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
              <div className="small alert alert-warning py-1 px-2 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
          Seizure Triggers — GABRA3 Specific
        </div>
        <div className="card-body">
          {triggers.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="d-flex justify-content-between small fw-semibold mb-1">
                <span>{t.trigger}</span>
                <span>{t.pct}%</span>
              </div>
              <div className="progress mb-1" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${t.pct}%`, background: i === 0 ? '#fd7e14' : COLOR }} />
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
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          Treatment Arsenal — GABRA3 X-linked Epilepsy
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
                <strong>GABRA3 Note:</strong> {t.gabra3_note}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm border-danger">
        <div className="card-header fw-semibold text-danger py-2">
          ⛔ Contraindications — GABRA3 Epilepsy
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
        <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
          15 Core Concepts — GABRA3 Epilepsy
        </div>
        <div className="card-body p-0">
          {concepts.map((c, i) => (
            <div key={i} className="border-bottom px-3 py-2">
              <span className="fw-semibold" style={{ color: COLOR }}>{c.term}: </span>
              <span className="small text-muted">{c.definition}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
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
        <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
          Clinical Standards
        </div>
        <div className="card-body p-0">
          {standards.map((s, i) => (
            <div key={i} className="border-bottom px-3 py-1 small">{s}</div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: LIGHT }}>
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

export default function GABRA3Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gabra3/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (activeTab >= 1 && activeTab <= 3 && !breakdown) {
      fetch(`${API}/api/gabra3/breakdown`)
        .then(r => r.json())
        .then(setBreakdown)
        .catch(e => setError(e.message));
    }
    if (activeTab === 4 && !definitions) {
      fetch(`${API}/api/gabra3/definitions`)
        .then(r => r.json())
        .then(setDefinitions)
        .catch(e => setError(e.message));
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="mb-1" style={{ color: COLOR }}>
          🧬 GABRA3 Epilepsy — X-linked DEE / Hyperekplexia / GABA-A α3 / Xq28
        </h4>
        <p className="text-muted small mb-0">
          GABRA3 (Xq28) · GABA-A receptor α3 subunit · Dominant fetal/neonatal α-isoform ·
          Thalamic Reticular Nucleus (TRN) enrichment · X-linked dominant (de novo/familial) ·
          Hyperekplexia (GOF p.Ile246Val) · PHB preferred rescue · 40-patient cohort · ILAE 2022
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
