'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#006064';   // dark cyan — CLN1 / Infantile NCL / Santavuori-Haltia
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep orange — urgent alerts / warnings
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / monitoring

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

function PctBar({ label, pct, color = ACCENT }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#e0f7fa', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

// ── TAB 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        variant="danger"
        text="⛔ ABSOLUTE CI: VGB (Vigabatrin) — ABSOLUTE CI in CLN1: progressive retinal NCL + VGB retinopathy = CATASTROPHIC irreversible blindness acceleration. INFANTILE SPASMS TRAP: VGB is first-line for spasms but ABSOLUTE CI in CLN1 — PPT1 enzyme assay (DBS) BEFORE VGB in any infant with spasms + regression + visual concern. CBZ / Oxcarbazepine / PHT / Fosphenytoin (IV) — ABSOLUTE CI: Na-channel blockers WORSEN myoclonus. PAEDIATRIC ED TRAP: standard SE protocol uses fosphenytoin — CLN1 uses IV LEV 60 mg/kg instead. Tiagabine (TGB) — ABSOLUTE CI: NCSE risk in all NCL/PME."
      />
      <Alert
        variant="warning"
        text="⚠ HIGH RISK: VPA polytherapy in children <3 years (hepatotoxicity highest risk — LFT 3-monthly). POLG1 exclusion BEFORE VPA (POLG1/Alpers mimics CLN1; VPA ABSOLUTE CI in POLG1). GBP/Pregabalin (myoclonus worsening). LTG monotherapy (worsens myoclonus). AED taper: NEVER — CLN1 is progressive fatal NCL; seizures do NOT remit."
      />
      <Alert
        variant="info"
        text="🧠 CLN1 UNIQUE FEATURES: EARLIEST NCL ONSET (6-24 months). GRODs on EM PATHOGNOMONIC (distinct from CLN2 curvilinear, CLN3 fingerprint+curvilinear, CLN4B fingerprint). PPT1 ENZYME ASSAY DBS = diagnosis in 1-3 days (MUST come before gene panel). EEG EXTINCTION (progressive isoelectric by age 2-3y) PATHOGNOMONIC. VEP EXTINGUISHED by 12-18 months DIAGNOSTIC. FATAL 7-12 years. FINNISH HERITAGE enrichment (p.Arg122Trp founder). VPA IS SAFE (lysosomal serine hydrolase, NOT mitochondrial)."
      />
      <Alert
        variant="success"
        text="✅ DIAGNOSIS STEPS: (1) PPT1 enzyme assay DBS (1-3 days — confirm before genetics). (2) Skin biopsy EM → GRODs (pathognomonic). (3) ERG + VEP ophthalmology (VEP extinction by 18m diagnostic). (4) CLN1/PPT1 gene sequencing for family counselling + cascade. (5) BDSRA registry enrolment (trial eligibility). (6) ACP initiated at diagnosis. Sequence: PPT1 enzyme → TPP1 enzyme (if PPT1 normal) → NCL gene panel."
      />

      <div className="row mb-4">
        <KPI label="Cohort" value={`${ov.cohort_size} pts`} color={ACCENT} />
        <KPI label="Regression Onset" value={`${ov.mean_onset_regression_months}m`} color={ACCENT2} />
        <KPI label="Seizure Onset" value={`${ov.mean_onset_seizure_months}m`} color={ACCENT3} />
        <KPI label="Mean Survival" value={`${ov.mean_death_years}y`} color={ACCENT2} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT3} />
        <KPI label="Retinal Degen." value={`${ov.retinal_degeneration_pct}%`} color={ACCENT2} />
        <KPI label="GRODs EM +" value={`${ov.grods_skin_biopsy_pct}%`} color={ACCENT} />
        <KPI label="EEG Suppressed" value={`${ov.eeg_suppression_by_age_3y_pct}%`} color={ACCENT2} />
        <KPI label="VEP Extinct" value={`${ov.vep_extinction_pct}%`} color={ACCENT3} />
        <KPI label="Finnish Heritage" value={`${ov.finnish_heritage_pct}%`} color={ACCENT4} />
        <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="On KD" value={`${ov.on_kd_pct}%`} color={ACCENT4} />
      </div>

      <SectionCard title="Gene & Mechanism">
        <p className="small mb-1"><strong>Gene:</strong> {ov.gene}</p>
        <p className="small mb-1"><strong>Protein:</strong> {ov.protein}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {ov.inheritance}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {ov.omim}</p>
        <p className="small mb-1"><strong>Disease:</strong> {ov.disease}</p>
        <p className="small mb-0"><strong>Mechanism:</strong> {ov.mechanism}</p>
      </SectionCard>

      <SectionCard title="No Disease-Modifying Therapy — CRITICAL Communication" borderColor={ACCENT2}>
        <div className="alert alert-danger py-2" style={{ fontSize: 13 }}>
          <strong>⛔ {ov.no_disease_modifying_therapy}</strong>
        </div>
        <p className="small mb-0">
          No cerliponase equivalent for CLN1 (PPT1 gene therapy under development — AAV-PPT1 preclinical efficacy
          demonstrated; Macauley 2018). BDSRA registry enrolment is the ONLY pathway to trial access.
          Contrast CLN2 (cerliponase alfa ICV — FDA 2017 approved) and CLN3 (no disease-modifying therapy, similar to CLN1).
        </p>
      </SectionCard>

      <SectionCard title="Unique Feature" borderColor={ACCENT}>
        <p className="small mb-0">{ov.unique_feature}</p>
      </SectionCard>

      <SectionCard title="Discovery" borderColor={ACCENT4}>
        <p className="small mb-0">{ov.discovery}</p>
      </SectionCard>

      <SectionCard title="Absolute & High-Risk Contraindications" borderColor={ACCENT2}>
        {ov.absolute_ci?.map((ci, i) => (
          <div key={i} className="alert alert-danger py-1 mb-2" style={{ fontSize: 13 }}>
            <strong>⛔ ABSOLUTE CI:</strong> {ci}
          </div>
        ))}
        {ov.high_risk_ci?.map((ci, i) => (
          <div key={i} className="alert alert-warning py-1 mb-2" style={{ fontSize: 13 }}>
            <strong>⚠ HIGH RISK:</strong> {ci}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Treatment Overview" borderColor={ACCENT}>
        <div className="row small">
          <div className="col-md-6">
            <PctBar label="On VPA" pct={ov.on_vpa_pct} />
            <PctBar label="On LEV" pct={ov.on_lev_pct} />
            <PctBar label="On Ketogenic Diet" pct={ov.on_kd_pct} />
            <PctBar label="GRODs Skin Biopsy +" pct={ov.grods_skin_biopsy_pct} />
          </div>
          <div className="col-md-6">
            <PctBar label="Retinal Degeneration" pct={ov.retinal_degeneration_pct} color={ACCENT2} />
            <PctBar label="EEG Suppressed by Age 3y" pct={ov.eeg_suppression_by_age_3y_pct} color={ACCENT2} />
            <PctBar label="VEP Extinction" pct={ov.vep_extinction_pct} color={ACCENT3} />
            <PctBar label="Drug-Resistant" pct={ov.drug_resistant_pct} color={ACCENT3} />
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── TAB 2: Patients & Etiology ────────────────────────────────────────────────
function PatientsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="6-Class Etiology Distribution" borderColor={ACCENT}>
        {bd.etiologies?.map((e, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{e.class}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.pct}% ({e.count} pts)</span>
            </div>
            <p className="small text-muted mb-1">{e.description}</p>
            <p className="small mb-1"><em>Mechanism:</em> {e.gene_mechanism}</p>
            <div className="d-flex flex-wrap gap-1">
              {e.key_variants?.map((v, j) => (
                <span key={j} className="badge bg-secondary" style={{ fontSize: 11 }}>{v}</span>
              ))}
            </div>
            <div className="progress mt-2" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 3: Seizures & Triggers ────────────────────────────────────────────────
function SeizuresTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        variant="info"
        text="🧠 CLN1 KEY: INFANTILE SPASMS TRAP — VGB is first-line for spasms but ABSOLUTE CI in CLN1 (retinal NCL + VGB retinopathy = catastrophic). PPT1 enzyme assay BEFORE VGB in any infant with spasms + regression. EEG EXTINCTION is pathognomonic (normal → high amplitude → progressive suppression → isoelectric by 2-3y). VEP extinguished by 12-18 months diagnostic. FEVER is most potent seizure trigger (88%). IV LEV 60 mg/kg for SE (NOT fosphenytoin)."
      />
      <SectionCard title="Seizure Types" borderColor={ACCENT}>
        {bd.seizure_types?.map((s, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{s.type}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{s.pct}%</span>
            </div>
            <p className="small text-muted mb-1">{s.description}</p>
            <p className="small mb-1"><em>EEG:</em> {s.eeg}</p>
            <p className="small mb-1"><em>Semiology:</em> {s.semiology}</p>
            <div className="alert alert-info py-1 mb-0" style={{ fontSize: 12 }}>
              💡 {s.clinical_tip}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers" borderColor={ACCENT3}>
        {bd.triggers?.map((t, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{t.trigger}</strong>
              <span className="badge" style={{ backgroundColor: t.pct === 100 ? ACCENT2 : ACCENT3 }}>{t.pct}%</span>
            </div>
            <p className="small text-muted mb-1">{t.description}</p>
            <p className="small mb-0"><em>Management:</em> {t.management}</p>
            <div className="progress mt-2" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: t.pct === 100 ? ACCENT2 : ACCENT3 }} />
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 4: Treatments ──────────────────────────────────────────────────────────
function TreatmentsTab({ bd }) {
  if (!bd) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Alert
        variant="success"
        text="✅ CLN1 PHARMACOLOGY: VPA IS SAFE (lysosomal serine hydrolase, NOT mitochondrial — no POLG CI; but monitor LFT 3-monthly in <3y on polytherapy). IV LEV 60 mg/kg = SE rescue (replace fosphenytoin in all ED protocols). Ketogenic diet: Level C adjunct after ≥3 AED failures. POLG1 exclusion before VPA in infants <2 years. VGB ABSOLUTE CI. BDSRA enrolment mandatory. Palliative care from diagnosis."
      />
      <SectionCard title="Treatments" borderColor={ACCENT}>
        {bd.treatments?.map((t, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{t.drug}</strong>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{t.level}</span>
            </div>
            <p className="small mb-1"><em>Dose:</em> {t.dose}</p>
            <p className="small mb-1"><em>MOA:</em> {t.moa}</p>
            <p className="small mb-1"><em>Efficacy:</em> {t.efficacy}</p>
            <p className="small mb-1"><em>Monitoring:</em> {t.monitoring}</p>
            <div className="alert alert-success py-1 mb-0" style={{ fontSize: 12 }}>
              🧠 CLN1 Note: {t.cln1_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications" borderColor={ACCENT2}>
        {bd.contraindications?.map((ci, i) => (
          <div
            key={i}
            className="mb-3 p-2 border rounded"
            style={{ borderLeft: `4px solid ${ci.severity === 'ABSOLUTE' ? '#b71c1c' : '#e65100'}` }}
          >
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{ci.drug}</strong>
              <span
                className="badge"
                style={{ backgroundColor: ci.severity === 'ABSOLUTE' ? '#b71c1c' : ci.severity === 'AVOID' ? '#6a1b9a' : '#e65100' }}
              >
                {ci.severity}
              </span>
            </div>
            <p className="small text-muted mb-1">{ci.reason}</p>
            <p className="small mb-0"><em>Alternative:</em> {ci.alternative}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        <ul className="list-unstyled small mb-0">
          {bd.monitoring?.map((m, i) => (
            <li key={i} className="mb-2 pb-2 border-bottom">
              <strong>{m.item}:</strong> {m.rationale}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Disease Lifecycle" borderColor="#4a148c">
        {bd.lifecycle?.map((l, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong className="small">{l.stage}</strong>
              <span className="badge bg-secondary">{l.age}</span>
            </div>
            <p className="small mb-0 text-muted">{l.description}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── TAB 5: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ defs }) {
  if (!defs) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        {defs.concepts?.map((c, i) => (
          <div key={i} className="mb-3 p-2 border rounded">
            <strong className="small d-block mb-1" style={{ color: ACCENT }}>{c.concept}</strong>
            <p className="small text-muted mb-1">{c.definition}</p>
            <span className="badge bg-secondary" style={{ fontSize: 11 }}>Standard: {c.standard}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds" borderColor={ACCENT3}>
        <ul className="list-unstyled small mb-0">
          {defs.thresholds?.map((t, i) => (
            <li key={i} className="mb-2 pb-2 border-bottom">
              <strong>{t.threshold}</strong>
              <span className="text-muted ms-2">({t.standard})</span>
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Standards & Guidelines" borderColor={ACCENT4}>
        {defs.standards?.map((s, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom small">
            <strong>{s.standard}:</strong> {s.detail}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="References" borderColor="#546e7a">
        {defs.references?.map((r, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom small">
            <strong>{r.ref}:</strong> {r.citation}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function CLN1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cln1/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/cln1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/cln1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <div
          className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 48, height: 48, backgroundColor: ACCENT, fontSize: 18 }}
        >
          C1
        </div>
        <div>
          <h4 className="mb-0" style={{ color: ACCENT }}>
            CLN1 Epilepsy — Infantile Batten Disease (Santavuori-Haltia)
          </h4>
          <div className="text-muted small">
            CLN1/PPT1 · 1p34.2 · AR · GRODs EM PATHOGNOMONIC · PPT1 Enzyme Assay DBS · EEG Extinction Age 2-3y · VEP Extinction 12-18m · VGB ABSOLUTE CI · Fatal 7-12y
          </div>
        </div>
      </div>

      <div className="alert alert-warning py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>CLN1 vs CLN2 vs CLN3 vs CLN4B CRITICAL DISTINCTIONS:</strong>{' '}
        CLN1 → INFANTILE onset (6-24m); GRODs EM; PPT1 enzyme DBS; EEG isoelectric by 2-3y; VEP extinct 12-18m; VGB ABSOLUTE CI; NO disease-modifying Rx; fatal 7-12y.{' '}
        CLN2 → LATE-INFANTILE (2-4y); curvilinear bodies + fingerprint EM; TPP1 enzyme DBS; giant SSPS 1-3 Hz EEG; cerliponase alfa ICV (FDA 2017 — ONLY NCL ERT); VGB ABSOLUTE CI.{' '}
        CLN3 → JUVENILE (4-10y); combined fingerprint+curvilinear EM; vacuolated lymphocytes PATHOGNOMONIC; visual failure FIRST; no disease-modifying Rx.{' '}
        CLN4B → ADULT (25-45y); fingerprint profiles EM; BEHAVIORAL onset FIRST; NO visual failure; AD inheritance; no disease-modifying Rx.{' '}
        VPA SAFE in ALL (lysosomal/presynaptic — not mitochondrial). IV LEV 60 mg/kg SE rescue in ALL.
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={overview} />}
      {tab === 1 && <PatientsTab bd={breakdown} />}
      {tab === 2 && <SeizuresTab bd={breakdown} />}
      {tab === 3 && <TreatmentsTab bd={breakdown} />}
      {tab === 4 && <DefinitionsTab defs={definitions} />}
    </div>
  );
}
