'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#2e7d32';   // forest-green — CLN3 / JNCL / Juvenile Batten Disease
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / danger / fatal disease
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
      <div className="card-header fw-bold" style={{ backgroundColor: '#e8f5e9', color: borderColor }}>
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
        text="⛔ ABSOLUTE CI: Vigabatrin / VGB (irreversible retinal toxicity — CLN3 has progressive retinal degeneration leading to complete blindness; VGB superimposes vigabatrin-associated retinopathy → catastrophic combined visual loss). IV Fosphenytoin/Phenytoin: HIGH RISK (worsens myoclonus — use IV LEV 60 mg/kg for SE)."
      />
      <Alert
        variant="warning"
        text="⚠ HIGH RISK: CBZ / OXC / PHT (Na-channel blockers worsen myoclonic component — misdiagnosis as JME/GGE common). GBP / Pregabalin (myoclonus worsening; Crespel 1999). D2-blocking antipsychotics / Haloperidol (worsen late-disease Parkinsonism). AED taper: NEVER taper for remission in progressive CLN3 epilepsy."
      />
      <Alert
        variant="info"
        text="🔵 CLN3 UNIQUE FEATURES: VISUAL FAILURE IS FIRST (before seizures by 2-5 years) — ophthalmologist-led diagnosis. Vacuolated lymphocytes on blood film PATHOGNOMONIC. No disease-modifying therapy (contrast CLN2 cerliponase alfa). Parkinsonism in late disease → L-DOPA. SSRI for anxiety/OCD (cardinal disease features). VPA IS SAFE (lysosomal disorder, NOT mitochondrial)."
      />
      <Alert
        variant="success"
        text="✅ DIAGNOSIS STEPS: (1) Blood film vacuolated lymphocytes — rapid. (2) CLN3 1.02 kb deletion PCR (73% alleles) — same day. (3) CLN3 full sequencing — second allele. (4) Skin biopsy EM — fingerprint profiles + curvilinear bodies confirms NCL. No CLN3 enzyme assay (Battenin is membrane protein, not soluble enzyme — contrast CLN2/TPP1 enzyme assay)."
      />

      <div className="row mb-4">
        <KPI label="Cohort" value={`${ov.cohort_size} pts`} color={ACCENT} />
        <KPI label="Visual Onset" value={`${ov.mean_onset_visual_years}y`} color={ACCENT4} />
        <KPI label="Seizure Onset" value={`${ov.mean_onset_seizure_years}y`} color={ACCENT3} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT3} />
        <KPI label="Visual Failure 1st" value={`${ov.visual_failure_first_pct}%`} color={ACCENT4} />
        <KPI label="Vacuolated Lymphs" value={`${ov.vacuolated_lymphocytes_pct}%`} color={ACCENT} />
        <KPI label="Parkinsonism" value={`${ov.parkinsonism_late_disease_pct}%`} color={ACCENT3} />
        <KPI label="Psych/Behav" value={`${ov.behavioural_psychiatric_pct}%`} color={ACCENT4} />
        <KPI label="Photosens." value={`${ov.photosensitivity_pct}%`} color="#6a1b9a" />
        <KPI label="On SSRI" value={`${ov.on_ssri_pct}%`} color={ACCENT} />
        <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="Delay Dx" value={`${ov.mean_delay_diagnosis_visual_years}y`} color={ACCENT2} />
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
          Contrast with CLN2 (cerliponase alfa ICV enzyme replacement, FDA 2017 — the ONLY approved NCL therapy).
          CLN3 Battenin is a membrane protein, not a soluble enzyme — enzyme replacement is not applicable.
          Refer all CLN3 patients to NCL clinical trial networks (NCL Resource, BDSRA registry, Hamburg Battenin team)
          for AAV-CLN3 gene therapy trial eligibility.
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
            <PctBar label="On LTG" pct={ov.on_ltg_pct} />
            <PctBar label="On SSRI" pct={ov.on_ssri_pct} />
          </div>
          <div className="col-md-6">
            <PctBar label="On L-DOPA (late disease)" pct={ov.on_levodopa_late_pct} />
            <PctBar label="Exon 7+8 Deletion" pct={ov.homozygous_exon78_deletion_pct} />
            <PctBar label="Cognitive Decline 100%" pct={ov.cognitive_decline_pct} />
            <PctBar label="Vacuolated Lymphocytes" pct={ov.vacuolated_lymphocytes_pct} />
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
        text="🔵 CLN3 KEY: GTCS emerges AFTER visual failure by 2-5 years (onset 10-13 years). This 'visual first, seizures later' sequence is PATHOGNOMONIC for CLN3. Any adolescent with GTCS + prior visual failure = CLN3 until proven. CLN3 does NOT show giant SSPS at 1-3 Hz (that is CLN2-specific). Standard IPS protocol 3-50 Hz is appropriate."
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
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.pct}%</span>
            </div>
            <p className="small text-muted mb-1">{t.description}</p>
            <p className="small mb-0"><em>Management:</em> {t.management}</p>
            <div className="progress mt-2" style={{ height: 6 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: ACCENT3 }} />
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
        text="✅ CLN3 PHARMACOLOGY: VPA IS SAFE (lysosomal, not mitochondrial). SSRI (sertraline/fluoxetine) IS a CORE TREATMENT for anxiety/OCD/agitation — not optional. L-DOPA/carbidopa for late-disease Parkinsonism — unique among NCLs. IV LEV 60 mg/kg = SE rescue. No disease-modifying therapy — refer to gene therapy trials."
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
              🌿 CLN3 Note: {t.cln3_note}
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
                style={{ backgroundColor: ci.severity === 'ABSOLUTE' ? '#b71c1c' : '#e65100' }}
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
export default function CLN3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cln3/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/cln3/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/cln3/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <div
          className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 48, height: 48, backgroundColor: ACCENT, fontSize: 18 }}
        >
          C3
        </div>
        <div>
          <h4 className="mb-0" style={{ color: ACCENT }}>
            CLN3 Epilepsy — Juvenile Batten Disease (JNCL)
          </h4>
          <div className="text-muted small">
            CLN3 / Battenin · 16p12.1 · AR · Most Common NCL Worldwide · Visual Failure FIRST · No Disease-Modifying Therapy
          </div>
        </div>
      </div>

      <div className="alert alert-success py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>CLN3 vs CLN2 CRITICAL DISTINCTION:</strong> CLN3 → Visual failure FIRST (4-10y), THEN seizures (10-13y);
        Vacuolated lymphocytes PATHOGNOMONIC; NO approved disease-modifying therapy; Parkinsonism in late disease (L-DOPA).
        CLN2 → Seizures FIRST (2-4y); Giant SSPS 1-3 Hz EEG PATHOGNOMONIC; Cerliponase alfa ICV (FDA 2017 — ONLY NCL ERT);
        NO vacuolated lymphocytes. VGB ABSOLUTE CI in BOTH.
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
