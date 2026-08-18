'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#1b5e20'; // dark forest green — α2-δ-2 cerebellar; distinct from CACNA1 blues
const DANGER = '#b71c1c'; // dark red — CI / gabapentinoid paradox
const SUCCESS = '#1565c0'; // dark blue — seizure freedom
const WARN = '#e65100';   // deep orange — cerebellar atrophy warning

function KPI({ label, value, color = COLOR }) {
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

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold text-white py-2" style={{ background: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const etiologies = data.etiology_distribution || {};
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const ciSummary = data.contraindications_summary || [];
  const seizureSummary = data.seizure_summary || [];
  const etioEntries = Object.entries(etiologies).map(([k, v]) => ({ category: k, count: v }));

  return (
    <div>
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#e8f5e9' }}>
        <strong>🧬 CACNA2D2 (3p21.3) — α2-δ-2 Auxiliary Subunit / EECAT — Epileptic Encephalopathy + Cerebellar Atrophy + Hypotonia:</strong>{' '}
        CACNA2D2 encodes the α2-δ-2 subunit of voltage-gated Ca²⁺ channels — essential for <strong>Cav1/Cav2 trafficking to the plasma membrane</strong>.{' '}
        LOF (biallelic AR): <strong>EECAT</strong> — severe infantile epilepsy + cerebellar ataxia + progressive cerebellar atrophy + hypotonia + global developmental delay.{' '}
        <strong>Ducky mouse model</strong> (Barclay 2001 Nat Neurosci): CACNA2D2 null → SWD + cerebellar ataxia + Purkinje cell loss.{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          ⚠️ GABAPENTINOID PARADOX: Gabapentin/Pregabalin bind α2-δ-2 — already non-functional in CACNA2D2 LOF → unpredictable efficacy + WORSENED CEREBELLAR ATAXIA. HIGH RISK — AVOID.{' '}
          ⚠️ PHT HIGH RISK: Cerebellar toxicity compounds pre-existing Purkinje cell loss. Use only as IV acute SE rescue, NOT maintenance.{' '}
          ABSOLUTE CI: TGB (NCSE) · VPA+POLG1 (Alpers fatal hepatotoxicity).{' '}
          Best DRE: Ketogenic Diet (addresses Cav2 + mTOR + KATP pathways).
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} />
        <KPI label="Seizure-Free" value={`${data.seizure_free_pct}%`} color={SUCCESS} />
        <KPI label="Drug-Resistant" value={`${data.dre_pct}%`} color={DANGER} />
        <KPI label="Infantile Spasms Hx" value={data.infantile_spasms_count} color={WARN} />
        <KPI label="Cerebellar Ataxia" value={data.cerebellar_ataxia_count} color={WARN} />
        <KPI label="Cerebellar Atrophy (MRI)" value={data.cerebellar_atrophy_mri_count} color={DANGER} />
        <KPI label="Hypotonia" value={data.hypotonia_count} color={WARN} />
        <KPI label="KD Use" value={data.kd_use_count} color={COLOR} />
        <KPI label="Myoclonus" value={data.myoclonus_count} color={WARN} />
        <KPI label="Gabapentinoid Exposure" value={data.gabapentinoid_exposure_count} color={DANGER} />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution">
            {etioEntries.map((e, i) => (
              <div key={i} className="d-flex justify-content-between small border-bottom py-1">
                <span>{e.category.replace(/-/g, ' ')}</span>
                <span className="fw-bold">{e.count}</span>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="Seizure Type Frequency">
            {seizureSummary.map((s, i) => (
              <Bar key={i} label={s.type} value={s.frequency_pct} max={100} color={i === 0 ? DANGER : i === 1 ? WARN : COLOR} />
            ))}
          </SectionCard>
        </div>

        <div className="col-md-6">
          <SectionCard title="Treatment Overview">
            {treatments.map((t, i) => (
              <div key={i} className="d-flex justify-content-between small border-bottom py-1">
                <span>{t.drug}</span>
                <span className="badge" style={{ background: COLOR }}>{t.level.replace('Level ', 'L')}</span>
              </div>
            ))}
          </SectionCard>

          <SectionCard title="⚠️ Contraindications Summary" borderColor={DANGER}>
            {ciSummary.map((ci, i) => (
              <div key={i} className="d-flex justify-content-between small border-bottom py-1">
                <span>{ci.drug}</span>
                <span className="badge bg-danger">{ci.risk}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Monitoring Highlights">
            {monitoring.map((m, i) => (
              <div key={i} className="small border-bottom py-1">
                <span className="fw-semibold">{m.item}</span>
                <span className="text-muted ms-2">({m.frequency})</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Lifecycle Stages">
            {lifecycle.map((lc, i) => (
              <div key={i} className="small border-bottom py-1">
                <div className="fw-semibold">{lc.stage}</div>
                <div className="text-muted">{lc.key_action}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Action Thresholds">
        <div className="table-responsive">
          <table className="table table-sm table-striped small mb-0">
            <thead><tr><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { etiologies = [], patients = [] } = data;
  return (
    <div>
      <SectionCard title="Etiology Catalog — 5-Class LOF Spectrum">
        {etiologies.map((e, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between">
              <span className="fw-bold" style={{ color: COLOR }}>{e.category.replace(/-/g, ' ')}</span>
              <span className="badge" style={{ background: COLOR }}>{e.pct}%</span>
            </div>
            <div className="small mt-1 text-muted">{e.mechanism}</div>
            <div className="small mt-1"><strong>EEG:</strong> {e.eeg}</div>
            <div className="small"><strong>Onset:</strong> {e.onset_months} &nbsp;|&nbsp; <strong>Severity:</strong> {e.severity}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Age</th><th>Onset (mo)</th><th>Etiology</th>
                <th>IS Hx</th><th>Ataxia</th><th>Atrophy</th><th>Hypotonia</th>
                <th>Myoclonus</th><th>KD</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td>{p.patient_id}</td>
                  <td>{p.age_years}y</td>
                  <td>{p.onset_months}m</td>
                  <td className="small">{p.etiology.replace(/-/g, ' ').substring(0, 30)}</td>
                  <td>{p.infantile_spasms_hx ? '✓' : ''}</td>
                  <td>{p.cerebellar_ataxia ? '✓' : ''}</td>
                  <td style={{ color: p.cerebellar_atrophy_mri ? DANGER : 'inherit' }}>{p.cerebellar_atrophy_mri ? '✓' : ''}</td>
                  <td>{p.hypotonia ? '✓' : ''}</td>
                  <td>{p.myoclonus ? '✓' : ''}</td>
                  <td>{p.kd_use ? '✓' : ''}</td>
                  <td>
                    <span className={`badge ${p.seizure_free ? 'bg-success' : p.dre ? 'bg-danger' : 'bg-secondary'}`}>
                      {p.outcome}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = data;
  return (
    <div>
      <SectionCard title="Seizure Types — 5 CACNA2D2 Phenotypes">
        {seizure_types.map((s, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between">
              <span className="fw-bold" style={{ color: COLOR }}>{s.type}</span>
              <span className="badge" style={{ background: i < 2 ? DANGER : COLOR }}>{s.frequency_pct}%</span>
            </div>
            <div className="small mt-1"><strong>EEG:</strong> {s.eeg_correlate}</div>
            <div className="small mt-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small mt-1 p-2 rounded" style={{ background: '#e8f5e9' }}>
              <strong>💡 Clinical Tip:</strong> {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (% of patients)">
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <div className="d-flex justify-content-between small mb-1">
              <span className="fw-semibold">{t.trigger}</span>
              <span>{t.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: t.pct >= 75 ? DANGER : t.pct >= 50 ? WARN : COLOR }} />
            </div>
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { treatments = [], contraindications = [] } = data;
  return (
    <div>
      <SectionCard title="Treatments — 8 AEDs/Interventions (No Precision Blocker)">
        {treatments.map((t, i) => (
          <div key={i} className="border rounded p-3 mb-3">
            <div className="d-flex justify-content-between align-items-start">
              <span className="fw-bold" style={{ color: COLOR }}>{t.drug}</span>
              <span className="badge ms-2 flex-shrink-0" style={{ background: COLOR }}>{t.level.split(' (')[0]}</span>
            </div>
            <div className="row mt-2 small g-2">
              <div className="col-md-6"><strong>Dose:</strong> {t.dose}</div>
              <div className="col-md-6"><strong>MOA:</strong> {t.moa}</div>
              <div className="col-md-6"><strong>Efficacy:</strong> {t.efficacy}</div>
              <div className="col-md-6"><strong>Monitoring:</strong> {t.monitoring}</div>
            </div>
            <div className="small mt-2 p-2 rounded" style={{ background: '#e8f5e9', borderLeft: `4px solid ${COLOR}` }}>
              <strong>CACNA2D2 Note:</strong> {t.cacna2d2_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Contraindications (Gabapentinoid Paradox + PHT Risk)" borderColor={DANGER}>
        {contraindications.map((ci, i) => (
          <div key={i} className="border rounded p-3 mb-3" style={{ borderColor: ci.risk === 'ABSOLUTE CI' ? DANGER : WARN }}>
            <div className="d-flex justify-content-between">
              <span className="fw-bold">{ci.drug}</span>
              <span className={`badge ${ci.risk === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'}`}>{ci.risk}</span>
            </div>
            <div className="small mt-1">{ci.mechanism}</div>
            {ci.alternative && <div className="small mt-1 text-success"><strong>Alternative:</strong> {ci.alternative}</div>}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { gene_summary: gs = {}, definitions = [], thresholds = [], standards = [], references = [] } = data;
  return (
    <div>
      <SectionCard title="Gene Summary — CACNA2D2">
        <div className="row small g-2">
          {Object.entries(gs).map(([k, v], i) => (
            <div key={i} className="col-md-6">
              <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key Definitions (15 Concepts)">
        {definitions.map((d, i) => (
          <div key={i} className="border-bottom py-2 small">
            <span className="fw-bold" style={{ color: COLOR }}>{d.term}: </span>
            <span className="text-muted">{d.definition}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Action Thresholds (12)">
        <div className="table-responsive">
          <table className="table table-sm table-striped small mb-0">
            <thead><tr><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Standards (12)">
        <div className="table-responsive">
          <table className="table table-sm table-striped small mb-0">
            <thead><tr><th>Standard</th><th>Applies To</th></tr></thead>
            <tbody>
              {standards.map((s, i) => (
                <tr key={i}><td className="fw-semibold">{s.name}</td><td>{s.applies}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="References (6)">
        {references.map((r, i) => (
          <div key={i} className="small border-bottom py-1">
            <span className="fw-semibold">{r.author} ({r.year}). </span>
            <span className="fst-italic">{r.title}. </span>
            <span className="text-muted">{r.journal}.</span>
            {r.pmid && <span className="ms-2 badge bg-secondary">PMID {r.pmid}</span>}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

export default function CACNA2D2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const base = `${API}/api/cacna2d2`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="pt" data={breakdown} />,
    <SeizuresTab key="sz" data={breakdown} />,
    <TreatmentsTab key="tx" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 CACNA2D2 Epilepsy Dashboard
          </h4>
          <div className="small text-muted">
            EECAT · α2-δ-2 Auxiliary Ca²⁺ Channel Subunit · Gabapentinoid-Binding Protein · AR LOF · 3p21.3 · 40 Patients
          </div>
        </div>
        <span className="badge ms-auto" style={{ background: COLOR }}>CACNA2D2 / α2-δ-2</span>
      </div>

      <div className="alert alert-warning py-2 small mb-3">
        <strong>⚠️ Key Pharmacology:</strong> Gabapentin/Pregabalin bind α2-δ-2 (CACNA2D2) — already non-functional in LOF.{' '}
        <strong>AVOID gabapentinoids</strong> (worsen cerebellar ataxia + unpredictable efficacy). <strong>PHT HIGH RISK</strong> (cerebellar toxicity compounds Purkinje cell loss).{' '}
        No precision blocker available — VPA/LEV + KD is best DRE approach. POLG1 exclusion MANDATORY before VPA.
      </div>

      {loading && <div className="text-center py-4 text-muted">Loading CACNA2D2 data…</div>}
      {error && <div className="alert alert-danger">Error: {error}</div>}

      {!loading && !error && (
        <>
          <ul className="nav nav-tabs mb-3">
            {TABS.map((t, i) => (
              <li key={i} className="nav-item">
                <button
                  className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
                  style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
                  onClick={() => setTab(i)}
                >{t}</button>
              </li>
            ))}
          </ul>
          {tabContent[tab]}
        </>
      )}
    </div>
  );
}
