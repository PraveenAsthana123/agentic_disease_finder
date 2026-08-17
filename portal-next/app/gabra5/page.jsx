'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];

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

function Bar({ label, value, max, color = '#6f42c1' }) {
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
      <div className="alert alert-warning py-2 small mb-3 border border-warning">
        <strong>&#x26a0;&#xfe0f; GABRA5 (15q12) — GABA-A α5 Subunit — EXTRASYNAPTIC TONIC INHIBITION:</strong>{' '}
        α5-containing receptors are <strong>BZD-INSENSITIVE</strong> (Ile99, not His101).{' '}
        Classical benzodiazepines (diazepam / lorazepam / clonazepam / midazolam){' '}
        <strong>DO NOT effectively act on α5</strong>. Hippocampal seizures in GABRA5-LOF are{' '}
        <strong className="text-danger">BZD-RESISTANT</strong>.{' '}
        Use <strong>PHENOBARBITAL</strong> (barbiturate site — preserved on α5){' '}
        / <strong>CLB</strong> (1,5-BZD, α2/α3 retained){' '}
        / <strong>ACTH</strong> (infantile spasms — ACTH-responsive) for rescue.
        <span className="text-danger fw-bold ms-2">
          ABSOLUTE CI: Tiagabine (NCSE) · VPA without POLG1 screen.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color="#6f42c1" />
        <KPI label="DEE65 Severe" value={kpis.dee65_severe_n} color="#dc3545" />
        <KPI label="Hippocampal TLE" value={kpis.tle_hippocampal_n} color="#fd7e14" />
        <KPI label="West History" value={`${kpis.west_history_pct}%`} color="#6f42c1" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="On KD" value={`${kpis.on_kd_pct}%`} color="#20c997" />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="SUDEP High Risk" value={kpis.sudep_high_risk_n} color="#dc3545" />
        <KPI label="BZD-Insensitive Documented" value={kpis.bzd_insensitivity_documented_n} color="#fd7e14" />
        <KPI label="Avg Age (yr)" value={kpis.avg_age_years} color="#343a40" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#6f42c1' }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.etiology}</span>
                    <span className="badge text-bg-secondary">{e.n} ({e.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.pct / maxEtio * 100)}%`, background: '#6f42c1' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#6f42c1' }}>
              Treatment Arsenal (8 agents)
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
            <div className="card-header fw-semibold py-2" style={{ background: '#e8d5f5' }}>
              Monitoring Schedule (14 items)
            </div>
            <div className="card-body">
              {monitoring.map((m, i) => (
                <div key={i} className="d-flex justify-content-between border-bottom py-1 small">
                  <span>{m.item}</span>
                  <span className="text-muted">{m.frequency}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: '#e8d5f5' }}>
              Patient Lifecycle (6 windows)
            </div>
            <div className="card-body">
              {lifecycle.map((lc, i) => (
                <div key={i} className="mb-2 small">
                  <div className="fw-semibold" style={{ color: '#6f42c1' }}>{lc.window}</div>
                  <div className="text-muted">{lc.focus}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: '#fff3cd' }}>
          Clinical Action Thresholds (12)
        </div>
        <div className="card-body">
          <ul className="mb-0 small">
            {thresholds.map((t, i) => (
              <li key={i} className={
                (t.threshold || '').toLowerCase().includes('absolute') ? 'text-danger fw-bold mb-1'
                : 'mb-1'
              }>{t.threshold}</li>
            ))}
          </ul>
        </div>
      </div>

      {data.note && (
        <div className="alert alert-light border small">{data.note}</div>
      )}
    </div>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const etiologies = data.etiology_distribution || [];
  const patients = data.patient_table || [];

  return (
    <div>
      <h5 className="mb-3" style={{ color: '#6f42c1' }}>Etiology Classes (5)</h5>
      {etiologies.map((e, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header d-flex justify-content-between align-items-center py-2"
            style={{ background: '#e8d5f5' }}>
            <span className="fw-semibold">{e.etiology}</span>
            <span className="badge" style={{ background: '#6f42c1' }}>{e.n} patients ({e.pct}%)</span>
          </div>
          <div className="card-body small">
            <p className="mb-1"><strong>Class:</strong> {e.disease}</p>
            {e.detail && <p className="text-muted mb-0">{e.detail}</p>}
          </div>
        </div>
      ))}

      <h5 className="mt-4 mb-3" style={{ color: '#6f42c1' }}>Patient Cohort (n=40)</h5>
      <div className="table-responsive">
        <table className="table table-sm table-striped table-hover small">
          <thead style={{ background: '#e8d5f5' }}>
            <tr>
              <th>ID</th><th>Age</th><th>Sex</th><th>Disease</th>
              <th>Variant</th><th>Current AEDs</th><th>DRE</th>
              <th>KD</th><th>POLG</th><th>West Hx</th><th>BZD-Ins</th><th>SUDEP Risk</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p, i) => (
              <tr key={i}>
                <td className="fw-semibold">{p.id}</td>
                <td>{p.age_years}y</td>
                <td>{p.sex}</td>
                <td><span className="badge text-bg-light text-dark border">{p.disease}</span></td>
                <td className="text-muted">{p.variant}</td>
                <td>{p.current_aeds}</td>
                <td>{p.drug_resistant ? <span className="badge text-bg-danger">DRE</span>
                  : <span className="badge text-bg-success">Ctrl</span>}</td>
                <td>{p.on_kd ? '✓' : '—'}</td>
                <td>{p.polg_tested}</td>
                <td>{p.west_history ? '✓' : '—'}</td>
                <td>{p.bzd_insensitivity
                  ? <span className="badge text-bg-warning text-dark">⚠ BZD-R</span>
                  : '—'}</td>
                <td>
                  <span className={`badge text-bg-${p.sudep_risk === 'high' ? 'danger'
                    : p.sudep_risk === 'moderate' ? 'warning text-dark' : 'success'}`}>
                    {p.sudep_risk}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizureTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_detail || [];
  const triggers = data.trigger_detail || [];
  const maxSeizure = Math.max(...seizures.map(s => s.prevalence_pct || 0), 1);
  const maxTrigger = Math.max(...triggers.map(t => t.prevalence_pct || 0), 1);

  return (
    <div>
      <div className="row g-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#6f42c1' }}>
              Seizure Types (n=40)
            </div>
            <div className="card-body">
              {seizures.map((s, i) => (
                <div key={i} className="mb-4">
                  <Bar label={s.type} value={s.prevalence_pct} max={maxSeizure} color="#6f42c1" />
                  {s.eeg_correlate && (
                    <div className="small text-muted mb-1">
                      <strong>EEG:</strong> {s.eeg_correlate}
                    </div>
                  )}
                  {s.clinical_tip && (
                    <div className="small text-info">
                      <strong>Tip:</strong> {s.clinical_tip}
                    </div>
                  )}
                  {s.key_drug && (
                    <div className="small mt-1">
                      <span className="badge" style={{ background: '#6f42c1' }}>{s.key_drug}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#fd7e14' }}>
              Seizure Triggers (8)
            </div>
            <div className="card-body">
              {triggers.map((t, i) => (
                <div key={i} className="mb-4">
                  <Bar label={t.trigger} value={t.prevalence_pct} max={maxTrigger} color="#fd7e14" />
                  {t.mechanism && (
                    <div className="small text-muted mb-1">
                      <strong>Mechanism:</strong> {t.mechanism}
                    </div>
                  )}
                  {t.action && (
                    <div className="small text-success">
                      <strong>Action:</strong> {t.action}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatment_catalog || [];
  const contras = data.contraindications || [];

  return (
    <div>
      <h5 className="mb-3" style={{ color: '#6f42c1' }}>Treatment Catalog (8 agents)</h5>
      {treatments.map((t, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header d-flex justify-content-between align-items-center py-2"
            style={{ background: '#e8d5f5' }}>
            <span className="fw-bold">{t.drug}</span>
            <span className="badge" style={{ background: '#6f42c1' }}>{t.level}</span>
          </div>
          <div className="card-body small">
            <div className="row g-2">
              <div className="col-md-6">
                <p className="mb-1"><strong>Dose:</strong> {t.dose}</p>
                <p className="mb-1"><strong>MOA:</strong> {t.moa}</p>
                <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1"><strong>Safety:</strong> {t.safety}</p>
                <p className="mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
              </div>
            </div>
            {t.gabra5_note && (
              <div className="alert alert-warning py-1 px-2 mt-2 mb-0 small">
                <strong>GABRA5 Note:</strong> {t.gabra5_note}
              </div>
            )}
          </div>
        </div>
      ))}

      <h5 className="mt-4 mb-3 text-danger">Contraindications (5)</h5>
      {contras.map((c, i) => (
        <div key={i} className={`card shadow-sm mb-3 border-${
          c.risk && c.risk.toUpperCase().includes('ABSOLUTE') ? 'danger' : 'warning'
        }`}>
          <div className={`card-header d-flex justify-content-between py-2 ${
            c.risk && c.risk.toUpperCase().includes('ABSOLUTE')
              ? 'bg-danger text-white' : 'bg-warning text-dark'
          }`}>
            <span className="fw-bold">{c.drug}</span>
            <span className="badge text-bg-light text-dark">{c.risk}</span>
          </div>
          <div className="card-body small">
            <p className="mb-1"><strong>Mechanism:</strong> {c.mechanism}</p>
            <p className="mb-1"><strong>Action:</strong> {c.action}</p>
            {c.notes && <p className="text-muted mb-0">{c.notes}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.definitions || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const references = data.references || [];

  return (
    <div>
      <h5 className="mb-3" style={{ color: '#6f42c1' }}>Key Concepts (15)</h5>
      <div className="row g-2 mb-4">
        {concepts.map((c, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body py-2 px-3 small">
                <div className="fw-bold mb-1" style={{ color: '#6f42c1' }}>{c.term}</div>
                <div className="text-muted">{c.definition}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row g-3">
        <div className="col-md-4">
          <h6 className="text-warning">Clinical Thresholds (12)</h6>
          <ul className="list-unstyled small">
            {thresholds.map((t, i) => (
              <li key={i} className={`mb-2 ${
                (t.threshold || '').toLowerCase().includes('absolute') ? 'text-danger fw-bold' : ''
              }`}>• {t.threshold}</li>
            ))}
          </ul>
        </div>
        <div className="col-md-4">
          <h6 className="text-info">Clinical Standards (12)</h6>
          {standards.map((s, i) => (
            <div key={i} className="mb-2 small">
              <div className="fw-semibold">{s.standard}</div>
              <div className="text-muted">{s.detail}</div>
            </div>
          ))}
        </div>
        <div className="col-md-4">
          <h6 className="text-secondary">References (6)</h6>
          {references.map((r, i) => (
            <div key={i} className="mb-2 small">
              <div className="fw-semibold" style={{ color: '#6f42c1' }}>{r.ref}</div>
              <div className="text-muted">{r.detail}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function GABRA5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gabra5/overview`).then(r => r.json()),
      fetch(`${API}/api/gabra5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gabra5/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, def]) => {
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(def);
      })
      .catch(e => setError(e.message));
  }, []);

  const renderTab = () => {
    switch (tab) {
      case 0: return <OverviewTab data={overview} />;
      case 1: return <EtiologyTab data={breakdown} />;
      case 2: return <SeizureTab data={breakdown} />;
      case 3: return <TreatmentsTab data={breakdown} />;
      case 4: return <DefinitionsTab data={definitions} />;
      default: return null;
    }
  };

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3">
        <div>
          <h2 className="mb-0 fw-bold" style={{ color: '#6f42c1' }}>
            &#x1f9ec; GABRA5 Epilepsy
          </h2>
          <div className="text-muted small">
            DEE65 · Hippocampal Tonic-Inhibition Deficiency · BZD-Insensitive α5 ·
            West Syndrome (ACTH-Responsive) · 15q12 · 40-patient cohort
          </div>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger">API error: {error}</div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {renderTab()}

      <div className="mt-4 text-muted small border-top pt-3">
        <strong>References:</strong> Maljevic 2019 Front Pharmacol · Möhler 2007 Pharmacol Biochem Behav ·
        Mihalek 1999 PNAS · Pirker 2000 Neuroscience · Bhatt 2023 Epilepsia ·
        UKISS 2004 Lancet Neurol · ILAE 2022 · NICE NG217 · MHRA VPPP 2021 · CPIC-POLG-2023
      </div>
    </div>
  );
}
