'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#7c3aed'; // violet/indigo — TRN-dominant Cav3.3 identity (distinct from CACNA1G teal, CACNA1H amber)

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
  const ciSummary = data.contraindications_summary || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3 border" style={{ borderColor: COLOR }}>
        <strong>⚡ CACNA1I (22q13.1) — Cav3.3 T-type Ca²⁺ Channel — TRN-Dominant — GGE / CAE / JME:</strong>{' '}
        Cav3.3 is the <strong>dominant T-type channel in Thalamic Reticular Nucleus (TRN)</strong> neurons.
        GOF → enhanced TRN bursting → stronger GABA-B IPSPs → deeper TC hyperpolarisation → larger Cav3.1/Cav3.2 LTCS → 3-Hz SWD.{' '}
        <strong>CACNA1I vs siblings:</strong> CACNA1G Cav3.1 (TC-dominant) · CACNA1H Cav3.2 (TC+TRN) · CACNA1I Cav3.3 (TRN-dominant).{' '}
        <strong>ETX Level B</strong> (TRN Cav3.3 less ETX-sensitive than TC Cav3.1/Cav3.2 — ETX+VPA combination often needed).{' '}
        <span className="text-danger fw-bold">
          ABSOLUTE CI: CBZ / OXC / PHT (GGE aggravation → status) · TGB (NCSE). VPA females: VPPP mandatory. POLG1 before VPA.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="On ETX" value={kpis.on_etx_n} color={COLOR} />
        <KPI label="HV-SWD Positive" value={kpis.hv_swd_positive_n} color="#fd7e14" />
        <KPI label="Photo-Sensitive" value={kpis.photosensitive_n} color="#fd7e14" />
        <KPI label="GTCS Present" value={kpis.gtcs_n} color="#dc3545" />
        <KPI label="Myoclonic Jerks" value={kpis.myoclonic_n} color="#dc3545" />
        <KPI label="Catamenial" value={kpis.catamenial_n} color="#6f42c1" />
        <KPI label="Avg Age (yr)" value={kpis.avg_age_years} color="#343a40" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map(e => (
                <Bar key={e.category} label={e.category.replace('GOF-', '')} value={e.pct} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#c0392b' }}>
              Contraindications (Absolute / High Risk)
            </div>
            <div className="card-body">
              {ciSummary.map(ci => (
                <div key={ci.drug} className="mb-2 small border-start border-danger ps-2">
                  <span className="fw-semibold text-danger">{ci.drug.split(' (')[0]}</span>
                  <div className="text-muted">{ci.risk}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#0d6efd' }}>
              Treatments (8 options)
            </div>
            <div className="card-body small">
              {treatments.map(t => (
                <div key={t.drug} className="mb-1">
                  <span className="badge me-2" style={{ background: COLOR }}>{t.level}</span>
                  {t.drug}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#198754' }}>
              Key Monitoring
            </div>
            <div className="card-body small">
              {monitoring.map(m => (
                <div key={m.item} className="mb-1">
                  <span className="fw-semibold">{m.item.split(' (')[0]}</span>
                  <span className="text-muted ms-1">— {m.frequency}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#6f42c1' }}>
              Patient Lifecycle Windows
            </div>
            <div className="card-body">
              <div className="row g-2">
                {lifecycle.map(lc => (
                  <div key={lc.window} className="col-md-4">
                    <div className="border rounded p-2 small h-100">
                      <div className="fw-semibold" style={{ color: COLOR }}>{lc.window}</div>
                      <div className="text-muted">{lc.key}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: '#f8f9fa' }}>
          Key Thresholds
        </div>
        <div className="card-body">
          <div className="row g-2">
            {thresholds.map(t => (
              <div key={t.name} className="col-md-6">
                <div className="small border-start ps-2 mb-1" style={{ borderColor: COLOR }}>
                  <span className="fw-semibold">{t.name}:</span>{' '}
                  <span className="text-muted">{t.value}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function EtiologiesTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const etiologies = data.etiologies || [];
  const patients = data.patients || [];

  return (
    <div>
      <h5 className="mb-3" style={{ color: COLOR }}>5 Etiology Classes — CACNA1I GOF</h5>
      {etiologies.map(e => (
        <div key={e.category} className="card mb-3 shadow-sm">
          <div className="card-header fw-semibold text-white py-2 d-flex justify-content-between align-items-center"
            style={{ background: COLOR }}>
            <span>{e.category}</span>
            <span className="badge bg-light text-dark">{e.pct}%</span>
          </div>
          <div className="card-body small">
            <div className="mb-2"><strong>Etiology:</strong> {e.etiology}</div>
            <div className="mb-2"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="mb-1"><strong>Typical Variants:</strong> {e.typical_variants}</div>
            <div className="mb-1"><strong>Onset Age:</strong> ~{e.onset_age_years}Y</div>
            <div><strong>Outcome:</strong> {e.outcome}</div>
          </div>
        </div>
      ))}

      <h5 className="mt-4 mb-3" style={{ color: COLOR }}>40-Patient Cohort Sample</h5>
      <div className="table-responsive">
        <table className="table table-sm table-striped small">
          <thead className="table-dark">
            <tr>
              <th>Patient</th><th>Etiology</th><th>Syndrome</th><th>Age</th>
              <th>Gender</th><th>Seizure-Free</th><th>ETX</th><th>VPA</th>
            </tr>
          </thead>
          <tbody>
            {patients.slice(0, 20).map(p => (
              <tr key={p.patient_id}>
                <td>{p.patient_id}</td>
                <td>{p.etiology.replace('GOF-', '')}</td>
                <td>{p.syndrome.split(' (')[0]}</td>
                <td>{p.current_age}Y</td>
                <td>{p.gender}</td>
                <td>
                  <span className={`badge ${p.seizure_free ? 'bg-success' : 'bg-warning text-dark'}`}>
                    {p.seizure_free ? 'Yes' : 'No'}
                  </span>
                </td>
                <td>{p.etx_on ? '✓' : '—'}</td>
                <td>{p.vpa_on ? '✓' : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <h5 className="mb-3" style={{ color: COLOR }}>5 Seizure Types — CACNA1I GGE</h5>
      {seizures.map(s => (
        <div key={s.type} className="card mb-3 shadow-sm">
          <div className="card-header fw-semibold text-white py-2 d-flex justify-content-between align-items-center"
            style={{ background: COLOR }}>
            <span>{s.type}</span>
            <span className="badge bg-light text-dark">{s.pct}% of cohort</span>
          </div>
          <div className="card-body small">
            <div className="mb-2"><strong>EEG:</strong> {s.eeg}</div>
            <div className="mb-2"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="alert alert-warning py-1 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
          </div>
        </div>
      ))}

      <h5 className="mt-4 mb-3" style={{ color: COLOR }}>8 Seizure Triggers</h5>
      <div className="row g-3">
        {triggers.map(t => (
          <div key={t.trigger} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body small">
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span className="fw-semibold" style={{ color: COLOR }}>{t.trigger}</span>
                  <span className="badge" style={{ background: COLOR }}>{t.pct}%</span>
                </div>
                <div className="progress mb-2" style={{ height: 6 }}>
                  <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: COLOR }} />
                </div>
                <div className="text-muted">{t.note}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <h5 className="mb-3" style={{ color: COLOR }}>8 Treatments — CACNA1I GGE</h5>
      {treatments.map(t => {
        const isAbs = t.level.includes('ABSOLUTE') || t.level.includes('CAUTION');
        const levelColor = t.level.startsWith('Level A') ? '#198754'
          : t.level.startsWith('Level B') ? '#0d6efd'
          : '#6c757d';
        return (
          <div key={t.drug} className="card mb-3 shadow-sm">
            <div className="card-header fw-semibold text-white py-2 d-flex justify-content-between align-items-center"
              style={{ background: levelColor }}>
              <span>{t.drug}</span>
              <span className="badge bg-light text-dark small">{t.level.split('—')[0].trim()}</span>
            </div>
            <div className="card-body small">
              <div className="row g-3">
                <div className="col-md-6">
                  <div className="mb-2"><strong>Indication:</strong> {t.indication}</div>
                  <div className="mb-2"><strong>Dose:</strong> {t.dose}</div>
                  <div className="mb-2"><strong>MOA:</strong> {t.moa}</div>
                </div>
                <div className="col-md-6">
                  <div className="mb-2"><strong>Efficacy:</strong> {t.efficacy}</div>
                  <div className="mb-2"><strong>Safety:</strong> {t.safety}</div>
                  <div className="mb-2"><strong>Monitoring:</strong> {t.monitoring}</div>
                </div>
              </div>
              {t.cacna1i_note && (
                <div className="alert alert-info py-1 small mb-0 mt-2">
                  <strong>CACNA1I-Specific:</strong> {t.cacna1i_note}
                </div>
              )}
            </div>
          </div>
        );
      })}

      <h5 className="mt-4 mb-3 text-danger">5 Contraindications</h5>
      {contraindications.map(ci => (
        <div key={ci.drug} className="card mb-3 border-danger shadow-sm">
          <div className="card-header fw-semibold text-white py-2" style={{ background: '#dc3545' }}>
            {ci.drug}
          </div>
          <div className="card-body small">
            <div className="mb-2 text-danger fw-semibold">{ci.risk}</div>
            <div className="mb-2"><strong>Mechanism:</strong> {ci.mechanism}</div>
            <div><strong>Action:</strong> {ci.action}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const gene = data.gene_summary || {};
  const defs = data.definitions || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const refs = data.references || [];

  return (
    <div>
      <div className="card mb-4 shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          CACNA1I Gene Summary — Cav3.3 T-type Ca²⁺ Channel (22q13.1, TRN-Dominant)
        </div>
        <div className="card-body small">
          <div className="row g-2">
            {Object.entries(gene).map(([k, v]) => (
              <div key={k} className="col-md-6">
                <div className="border-start ps-2 mb-1" style={{ borderColor: COLOR }}>
                  <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}:</span>{' '}
                  <span className="text-muted">{v}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <h5 className="mb-3" style={{ color: COLOR }}>15 Key Concepts</h5>
      <div className="row g-3 mb-4">
        {defs.map(d => (
          <div key={d.term} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body small">
                <div className="fw-semibold mb-1" style={{ color: COLOR }}>{d.term}</div>
                <div className="text-muted">{d.definition}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h5 className="mb-3" style={{ color: COLOR }}>12 Thresholds</h5>
      <div className="row g-2 mb-4">
        {thresholds.map(t => (
          <div key={t.name} className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-body py-2 small">
                <div className="fw-semibold">{t.name}</div>
                <div className="text-muted">{t.value}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h5 className="mb-3" style={{ color: COLOR }}>12 Standards</h5>
      <div className="row g-2 mb-4">
        {standards.map(s => (
          <div key={s.name} className="col-md-6">
            <div className="small border-start ps-2 mb-1" style={{ borderColor: COLOR }}>
              <span className="fw-semibold">{s.name}:</span>{' '}
              <span className="text-muted">{s.description}</span>
            </div>
          </div>
        ))}
      </div>

      <h5 className="mb-3" style={{ color: COLOR }}>6 References</h5>
      <ul className="small">
        {refs.map((r, i) => (
          <li key={i} className="mb-1 text-muted">{r}</li>
        ))}
      </ul>
    </div>
  );
}

export default function CACNA1IPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/cacna1i/overview`).then(r => r.json()),
          fetch(`${API}/api/cacna1i/breakdown`).then(r => r.json()),
          fetch(`${API}/api/cacna1i/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(br);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: COLOR }}>
          ⚡ CACNA1I Epilepsy Dashboard
        </h2>
        <p className="text-muted mb-0 small">
          GGE / CAE / JME / GEFS+ · Cav3.3 T-type Ca²⁺ Channel · TRN-Dominant · 22q13.1 ·
          ETX Level B · 40 patients · T-type Cav3 Subfamily Complete (CACNA1G · CACNA1H · CACNA1I)
        </p>
      </div>

      {error && (
        <div className="alert alert-danger">Error loading data: {error}</div>
      )}
      {loading && (
        <div className="text-center py-4">
          <div className="spinner-border" style={{ color: COLOR }} role="status" />
          <div className="mt-2 text-muted">Loading CACNA1I dashboard…</div>
        </div>
      )}

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {!loading && (
        <>
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <EtiologiesTab data={breakdown} />}
          {tab === 2 && <SeizuresTab data={breakdown} />}
          {tab === 3 && <TreatmentsTab data={breakdown} />}
          {tab === 4 && <DefinitionsTab data={definitions} />}
        </>
      )}
    </div>
  );
}
