'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#5d4037';   // dark sienna — CLN4B / Adult Batten / behavioral-dominant adult-onset NCL
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
      <div className="card-header fw-bold" style={{ backgroundColor: '#efebe9', color: borderColor }}>
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
        text="⛔ ABSOLUTE CI: CBZ / Oxcarbazepine / PHT (Na-channel blockers WORSEN myoclonus — PRIMARY CLN4B MISDIAGNOSIS TRAP: behavioral onset + GTCS → GP prescribes CBZ for 'focal epilepsy' → ACUTE MYOCLONIC DETERIORATION). Tiagabine (TGB): ABSOLUTE CI — NCSE risk in PME. Stop CBZ/OXC IMMEDIATELY if patient already receiving at CLN4B diagnosis."
      />
      <Alert
        variant="warning"
        text="⚠ HIGH RISK: GBP / Pregabalin (myoclonus worsening — adult pain prescribing trap; check all specialist letters). LTG monotherapy (worsens CLN4B myoclonus). VGB (AVOID — visual field constriction; less strict than CLN2/CLN3 but avoid). AED taper: NEVER taper for remission in progressive CLN4B NCL."
      />
      <Alert
        variant="info"
        text="🧠 CLN4B UNIQUE FEATURES: BEHAVIORAL/PSYCHIATRIC ONSET FIRST (before seizures by 3-5 years — 92% cohort). NO visual failure (retina SPARED — distinct from CLN2/CLN3). NO vacuolated lymphocytes (blood film normal — distinct from CLN3). AUTOSOMAL DOMINANT (50% offspring risk — unique among NCLs). FINGERPRINT PROFILES on EM pathognomonic. Mean Dx delay 5.8 years. VPA IS SAFE (presynaptic, NOT mitochondrial)."
      />
      <Alert
        variant="success"
        text="✅ DIAGNOSIS STEPS: (1) Skin biopsy EM → fingerprint profiles (adult NCL confirmed, 2-4 weeks). (2) DNAJC5 sequencing — p.Leu115Arg / c.344_347del targeted first. (3) Family cascade testing (AD — all 1st degree relatives ≥18y). (4) Functional palmitoylation assay for VUS. No CLN4B enzyme assay (CSPα is a presynaptic chaperone, not a soluble lysosomal enzyme)."
      />

      <div className="row mb-4">
        <KPI label="Cohort" value={`${ov.cohort_size} pts`} color={ACCENT} />
        <KPI label="Behavioral Onset" value={`${ov.mean_onset_behavioral_years}y`} color={ACCENT4} />
        <KPI label="Seizure Onset" value={`${ov.mean_onset_seizure_years}y`} color={ACCENT3} />
        <KPI label="Drug-Resistant" value={`${ov.drug_resistant_pct}%`} color={ACCENT3} />
        <KPI label="Behavioral 1st" value={`${ov.behavioural_psychiatric_onset_pct}%`} color={ACCENT} />
        <KPI label="No Visual Fail" value="✓ Retina SPARED" color={ACCENT4} />
        <KPI label="Vacuolated Lymphs" value={`${ov.vacuolated_lymphocytes_pct}% (Normal film)`} color={ACCENT} />
        <KPI label="Skin Biopsy +" value={`${ov.skin_biopsy_fingerprint_pct}%`} color={ACCENT4} />
        <KPI label="Photosens." value={`${ov.photosensitivity_pct}%`} color="#6a1b9a" />
        <KPI label="Mean Dx Delay" value={`${ov.mean_delay_diagnosis_behavioral_years}y`} color={ACCENT2} />
        <KPI label="On VPA" value={`${ov.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="AD Confirmed" value={`${ov.ad_inheritance_confirmed_pct}%`} color={ACCENT} />
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
          No cerliponase equivalent for CLN4B (CSPα is a presynaptic chaperone, not a soluble lysosomal enzyme — ERT not applicable).
          Investigational: AAV-DNAJC5 gene therapy (early-phase); CSPα stabiliser compounds.
          Register ALL CLN4B patients with BDSRA registry and NCL Network Europe for trial access.
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
            <PctBar label="Skin Biopsy Fingerprint +" pct={ov.skin_biopsy_fingerprint_pct} />
            <PctBar label="AD Inheritance Confirmed" pct={ov.ad_inheritance_confirmed_pct} />
          </div>
          <div className="col-md-6">
            <PctBar label="Behavioral Onset First" pct={ov.behavioural_psychiatric_onset_pct} />
            <PctBar label="Cognitive Decline 100%" pct={ov.cognitive_decline_pct} />
            <PctBar label="Retinal Spared" pct={ov.retinal_spared_pct} />
            <PctBar label="Drug-Resistant" pct={ov.drug_resistant_pct} />
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
        text="🧠 CLN4B KEY: BEHAVIORAL/PSYCHIATRIC symptoms PRECEDE seizures by 3-5 years (mean 5.8-year diagnostic delay). Action myoclonus (80%) is the primary early functional disability. Focal seizures + behavioral onset → DO NOT start CBZ/OXC (ABSOLUTE CI → myoclonic deterioration). Standard IPS protocol 3-50 Hz — CLN4B is NOT CLN2 (no 1-3 Hz SSPS). VPA + LEV + piracetam for action myoclonus."
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
        text="✅ CLN4B PHARMACOLOGY: VPA IS SAFE (presynaptic, NOT mitochondrial — no POLG CI). LEV has DIRECT SV2A-CSPα presynaptic pathway complementarity. Piracetam FIRST-LINE for action myoclonus. SSRI (sertraline) is CORE treatment — behavioral features are primary disease (not secondary). Neuropsychiatry is MDT CO-LEAD. Genetic counselling IMMEDIATE (AD 50% offspring risk). IV LEV 60 mg/kg = SE rescue."
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
              🧠 CLN4B Note: {t.cln4b_note}
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
export default function CLN4BPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cln4b/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    fetch(`${API}/api/cln4b/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/cln4b/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <div
          className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 48, height: 48, backgroundColor: ACCENT, fontSize: 18 }}
        >
          C4B
        </div>
        <div>
          <h4 className="mb-0" style={{ color: ACCENT }}>
            CLN4B Epilepsy — Adult Batten Disease (Kufs Type B)
          </h4>
          <div className="text-muted small">
            DNAJC5 / CSPα · 19q13.11 · AD · Behavioral Onset FIRST · NO Visual Failure · NO Vacuolated Lymphocytes · 50% Offspring Risk
          </div>
        </div>
      </div>

      <div className="alert alert-warning py-2 mb-3" style={{ fontSize: 13 }}>
        <strong>CLN4B vs CLN3 vs CLN2 CRITICAL DISTINCTIONS:</strong>{' '}
        CLN4B → BEHAVIORAL onset FIRST (25-40y); NO visual failure (retina SPARED); NO vacuolated lymphocytes; AUTOSOMAL DOMINANT (50% risk); fingerprint profiles EM; CBZ ABSOLUTE CI.{' '}
        CLN3 → Visual failure FIRST (4-10y); vacuolated lymphocytes PATHOGNOMONIC; AR; VGB ABSOLUTE CI (progressive retinal disease).{' '}
        CLN2 → GTCS FIRST (2-4y); giant SSPS 1-3 Hz PATHOGNOMONIC; cerliponase alfa ICV (FDA 2017 — ONLY NCL ERT); VGB ABSOLUTE CI (retinal NCL).{' '}
        VPA SAFE in ALL three. IV LEV 60 mg/kg SE rescue in ALL three.
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
