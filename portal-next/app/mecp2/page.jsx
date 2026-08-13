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

function Bar({ label, value, max, color = '#7c3aed' }) {
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
      <div className="alert alert-warning py-2 small mb-3">
        <strong>MECP2 (Xq28) — Dose-Sensitive X-Linked Gene:</strong>{' '}
        <strong>LOF in females → Classic Rett Syndrome</strong> (4-stage regression, hand stereotypies, breathing irregularities, QTc).{' '}
        <strong>DUPLICATION in males → MECP2 Duplication Syndrome</strong> (MDS: progressive encephalopathy, recurrent infections, spasticity).{' '}
        <em>Trofinetide (FDA 2023) = first disease-modifying Rett therapy.</em>{' '}
        <strong>MANDATORY: ECG QTc q12M · Spine X-ray q12M · POLG before VPA.</strong>{' '}
        <span className="text-danger fw-bold">ABSOLUTE CI: abrupt AED withdrawal (autonomic instability + QTc).</span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color="#7c3aed" />
        <KPI label="Rett Syndrome" value={kpis.rett_syndrome_n} color="#dc3545" />
        <KPI label="MDS (males)" value={kpis.mds_n} color="#0d6efd" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#fd7e14" />
        <KPI label="QTc Prolonged" value={`${kpis.qtc_prolonged_pct}%`} color="#dc3545" />
        <KPI label="On Trofinetide" value={`${kpis.trofinetide_rx_pct}%`} color="#198754" />
        <KPI label="Scoliosis" value={`${kpis.scoliosis_pct}%`} color="#6f42c1" />
        <KPI label="Breathing Irr." value={`${kpis.breathing_irregularity_pct}%`} color="#0dcaf0" />
        <KPI label="On KD" value={`${kpis.on_kd_pct}%`} color="#20c997" />
        <KPI label="West (MDS)" value={kpis.west_mds_n} color="#ffc107" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#6c757d" />
        <KPI label="Avg Age (yr)" value={kpis.avg_age_years} color="#343a40" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold bg-purple text-white py-2" style={{ background: '#7c3aed' }}>
              Etiology Distribution (n=41)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.label || e.class_name}</span>
                    <span className="text-muted">{e.n} pts · {e.pct}%</span>
                  </div>
                  <div className="progress" style={{ height: 14 }}>
                    <div className="progress-bar" style={{ width: `${e.pct}%`, background: ['#7c3aed','#dc3545','#0d6efd','#fd7e14','#6c757d'][i % 5] }} />
                  </div>
                  <div className="text-muted" style={{ fontSize: '0.72rem' }}>{e.note}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#0d6efd' }}>
              4-Stage Rett Lifecycle + MDS
            </div>
            <div className="card-body p-2">
              {lifecycle.map((lc, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff', borderLeft: '4px solid #7c3aed' }}>
                  <div className="fw-semibold small">{lc.stage || lc.window}</div>
                  <div className="text-muted" style={{ fontSize: '0.72rem' }}>{lc.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#198754' }}>
              Treatments Summary
            </div>
            <div className="card-body p-2">
              {treatments.map((t, i) => (
                <div key={i} className="d-flex justify-content-between align-items-start mb-2 pb-2 border-bottom">
                  <div>
                    <span className="fw-semibold small">{t.name}</span>
                    <span className={`badge ms-1 ${t.evidence === 'Level A' ? 'bg-success' : t.evidence === 'Level B' ? 'bg-primary' : 'bg-secondary'}`}>{t.evidence}</span>
                    <div className="text-muted" style={{ fontSize: '0.71rem' }}>{t.indication}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#dc3545' }}>
              Monitoring Checklist
            </div>
            <div className="card-body p-2">
              {monitoring.map((m, i) => (
                <div key={i} className="mb-2 pb-1 border-bottom">
                  <div className="fw-semibold small">{m.item}</div>
                  <div className="text-muted" style={{ fontSize: '0.71rem' }}>{m.interval} — {m.rationale}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {thresholds.length > 0 && (
        <div className="card shadow-sm mb-3">
          <div className="card-header fw-semibold text-white py-2" style={{ background: '#6f42c1' }}>
            Clinical Thresholds
          </div>
          <div className="card-body p-0">
            <table className="table table-sm table-striped mb-0">
              <thead><tr><th>Threshold</th><th>Value</th><th>Action</th></tr></thead>
              <tbody>
                {thresholds.map((t, i) => (
                  <tr key={i}>
                    <td className="small fw-semibold">{t.name}</td>
                    <td className="small">{t.value}</td>
                    <td className="small text-muted">{t.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="alert alert-light border py-2 small">
        <strong>Reference:</strong> {data.reference}
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  const [filter, setFilter] = useState('all');
  if (!data) return <div className="text-center py-4 text-muted">Loading patients…</div>;
  const patients = data.patient_table || [];
  const filtered = filter === 'all' ? patients : patients.filter(p =>
    filter === 'catamenial' ? p.disease === 'rett_syndrome' :
    filter === 'mds' ? p.disease === 'mecp2_dup' :
    filter === 'dr' ? p.drug_resistant : true
  );

  return (
    <div>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {[['all','All Patients'],['catamenial','Rett Syndrome'],['mds','MDS (males)'],['dr','Drug-Resistant']].map(([val, label]) => (
          <button key={val} className={`btn btn-sm ${filter === val ? 'btn-primary' : 'btn-outline-secondary'}`}
            onClick={() => setFilter(val)}>{label}</button>
        ))}
        <span className="ms-auto text-muted small align-self-center">{filtered.length} patients</span>
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover table-striped">
          <thead className="table-dark">
            <tr>
              <th>ID</th><th>Age</th><th>Sex</th><th>Diagnosis</th><th>Stage</th>
              <th>Mutation</th><th>DRE</th><th>Treatment</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((p, i) => (
              <tr key={i}>
                <td className="small fw-semibold">{p.patient_id}</td>
                <td className="small">{p.age_years || p.age}</td>
                <td className="small">{p.sex}</td>
                <td className="small">
                  <span className={`badge ${p.disease === 'rett_syndrome' ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>
                    {p.disease === 'rett_syndrome' ? 'Rett' : p.disease === 'mecp2_dup' ? 'MDS' : p.disease}
                  </span>
                </td>
                <td className="small">{p.stage}</td>
                <td className="small">{p.mutation_class}</td>
                <td className="small text-center">
                  {p.drug_resistant ? <span className="badge bg-danger">DRE</span> : <span className="badge bg-success">Controlled</span>}
                </td>
                <td className="small">{p.recommended_treatment}</td>
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
    <div className="row g-3">
      <div className="col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-semibold text-white py-2" style={{ background: '#dc3545' }}>
            Seizure Types in MECP2 Disorders
          </div>
          <div className="card-body">
            {seizures.map((s, i) => (
              <div key={i} className="mb-3 p-2 rounded border-start border-4 ps-3" style={{ borderColor: ['#dc3545','#fd7e14','#ffc107','#0d6efd'][i % 4] }}>
                <div className="fw-semibold small">{s.type}</div>
                <div className="d-flex gap-2 mb-1">
                  <span className="badge bg-secondary">{s.prevalence_pct}%</span>
                  {s.drug_resistant_pct && <span className="badge bg-danger">{s.drug_resistant_pct}% DRE</span>}
                </div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}><strong>EEG:</strong> {s.eeg}</div>
                <div className="text-muted" style={{ fontSize: '0.72rem' }}><strong>Tip:</strong> {s.clinical_tip}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="col-md-6">
        <div className="card shadow-sm h-100">
          <div className="card-header fw-semibold text-white py-2" style={{ background: '#fd7e14' }}>
            Seizure Triggers (% patients affected)
          </div>
          <div className="card-body">
            {triggers.sort((a,b) => (b.prevalence_pct||0)-(a.prevalence_pct||0)).map((t, i) => (
              <div key={i} className="mb-2">
                <div className="d-flex justify-content-between small mb-1">
                  <span>{t.trigger}</span><span className="text-muted">{t.prevalence_pct}%</span>
                </div>
                <div className="progress" style={{ height: 10 }}>
                  <div className="progress-bar bg-warning" style={{ width: `${t.prevalence_pct}%` }} />
                </div>
                {t.mechanism && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{t.mechanism}</div>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatment_catalog || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <div className="alert alert-danger py-2 small mb-3">
        <strong>ABSOLUTE CI: Abrupt AED Withdrawal</strong> — autonomic instability + QTc prolongation in Rett → cardiovascular events + status epilepticus. Always taper ≥6 weeks.{' '}
        <strong>Trofinetide (FDA 2023)</strong> = first disease-modifying therapy; diarrhoea 82% (manage proactively).
      </div>

      <div className="row g-3 mb-3">
        {treatments.map((t, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header d-flex justify-content-between align-items-center py-2">
                <span className="fw-semibold small">{t.name}</span>
                <span className={`badge ${t.evidence === 'Level A' ? 'bg-success' : t.evidence === 'Level B' ? 'bg-primary' : 'bg-secondary'}`}>{t.evidence}</span>
              </div>
              <div className="card-body py-2">
                <div className="row g-1 small">
                  <div className="col-4 text-muted">Dose</div><div className="col-8">{t.dose}</div>
                  <div className="col-4 text-muted">MOA</div><div className="col-8">{t.moa}</div>
                  <div className="col-4 text-muted">Efficacy</div><div className="col-8">{t.efficacy}</div>
                  <div className="col-4 text-muted">Safety</div><div className="col-8">{t.safety}</div>
                  <div className="col-4 text-muted">Monitor</div><div className="col-8">{t.monitoring}</div>
                  {t.mecp2_note && (
                    <><div className="col-4 text-muted">MECP2 note</div><div className="col-8 text-danger fw-semibold">{t.mecp2_note}</div></>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: '#dc3545' }}>
          Contraindications in MECP2 Disorders
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0">
            <thead className="table-dark"><tr><th>Drug / Intervention</th><th>Risk Level</th><th>Mechanism / Reason</th></tr></thead>
            <tbody>
              {contraindications.map((c, i) => (
                <tr key={i}>
                  <td className="small fw-semibold">{c.drug || c.intervention}</td>
                  <td className="small">
                    <span className={`badge ${c.risk === 'ABSOLUTE CI' ? 'bg-danger' : c.risk === 'HIGH' ? 'bg-warning text-dark' : c.risk === 'MODERATE' ? 'bg-info text-dark' : 'bg-secondary'}`}>{c.risk}</span>
                  </td>
                  <td className="small text-muted">{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.concepts || [];
  const standards = data.standards || [];
  const references = data.references || [];
  const thresholds = data.thresholds || [];

  return (
    <div>
      <div className="row g-3 mb-3">
        <div className="col-md-8">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#7c3aed' }}>
              Key Concepts ({concepts.length})
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {concepts.map((c, i) => (
                    <tr key={i}>
                      <td className="small fw-semibold text-nowrap">{c.term}</td>
                      <td className="small text-muted">{c.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#0d6efd' }}>
              Standards & Guidelines ({standards.length})
            </div>
            <div className="card-body p-2">
              {standards.map((s, i) => (
                <div key={i} className="mb-2 pb-1 border-bottom">
                  <div className="small fw-semibold">{s.name || s}</div>
                  {s.note && <div className="text-muted" style={{ fontSize: '0.7rem' }}>{s.note}</div>}
                </div>
              ))}
            </div>
          </div>

          {thresholds.length > 0 && (
            <div className="card shadow-sm">
              <div className="card-header fw-semibold text-white py-2" style={{ background: '#198754' }}>
                Clinical Thresholds
              </div>
              <div className="card-body p-2">
                {thresholds.map((t, i) => (
                  <div key={i} className="mb-1 small">
                    <span className="fw-semibold">{t.name || t}:</span>{' '}
                    <span className="text-muted">{t.value}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: '#6c757d' }}>
          References
        </div>
        <div className="card-body p-2">
          <ol className="mb-0">
            {references.map((r, i) => (
              <li key={i} className="small text-muted mb-1">{r}</li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}

export default function MECP2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mecp2/overview`).then(r => r.json()).then(setOverview).catch(() => setError('Backend unavailable'));
    fetch(`${API}/api/mecp2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/mecp2/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0">
          🧬 MECP2-Related Disorders
          <span className="badge bg-danger ms-2 fs-6">X-Linked · Rett Syndrome · MDS</span>
        </h4>
        <div className="text-muted small">
          MECP2 (Xq28) · Methyl-CpG Binding Protein 2 · LOF → Rett (females) · DUP → MDS (males) · 41-patient cohort
        </div>
        <div className="text-muted small">
          <strong>Trofinetide (FDA 2023)</strong> — first disease-modifying Rett therapy ·{' '}
          <strong>ABSOLUTE CI: abrupt AED withdrawal</strong> (autonomic + QTc)
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={overview} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
