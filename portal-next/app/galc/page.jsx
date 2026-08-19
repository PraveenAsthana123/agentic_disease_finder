'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // dark-indigo — GALC / lysosomal leukodystrophy
const ACCENT2 = '#b71c1c';   // dark-red — HIGH RISK / ABSOLUTE / danger
const ACCENT3 = '#e65100';   // deep-orange — RELATIVE CI / PATHOGNOMONIC
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / HSCT / VPA
const ACCENT5 = '#4a148c';   // deep-purple — molecular / psychosine
const ACCENT6 = '#006064';   // dark-cyan — biomarkers / NBS / psychosine DBS

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
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

// ── Overview tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert
        text="⚠ UNEXPLAINED FEVER (82%) + EXTREME HYPERIRRITABILITY + PERIPHERAL NEUROPATHY (NCS absent SNAPs) in infant <6 months = Krabbe Disease UNTIL PROVEN OTHERWISE. DBS galactocerebrosidase + psychosine IMMEDIATELY. This is the HSCT window — delay is catastrophic."
        variant="danger"
      />
      <Alert
        text="⚠ HSCT ONLY effective PRE-SYMPTOMATICALLY (NBS-detected) or early-juvenile. After EI Stage 2 symptoms: NO neurological benefit from HSCT (oligodendrocytes already apoptosed). NO ERT, NO SRT available for Krabbe."
        variant="danger"
      />
      <Alert
        text="⚠ VGB (Vigabatrin) HIGH RISK — IRREVERSIBLE visual field constriction COMPOUNDS progressive optic atrophy + cortical blindness in Krabbe. Choose ACTH (Level A) over VGB for infantile spasms. CRITICAL distinction from other IS etiologies."
        variant="warning"
      />
      <Alert
        text="ℹ CBZ/OXC/PHT: RELATIVE CI (worsen spasticity/rigidity + peripheral neuropathy) — NOT ABSOLUTE CI. KEY DISTINCTION: Krabbe lacks PME mechanism (unlike GBA/HEXA/HEXB where CBZ is ABSOLUTE CI for myoclonus). Prefer LEV + VPA. Fosphenytoin RELATIVE CI — IV LEV replaces in SE."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${d.infantile_spasms_pct}%`} color={ACCENT3} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="PNS Neuropathy" value={`${d.peripheral_neuropathy_pct}%`} color={ACCENT3} />
        <KPI label="Unexplained Fever" value={`${d.unexplained_fever_pct}%`} color={ACCENT3} />
        <KPI label="On HSCT %" value={`${d.on_hsct_pct}%`} color={ACCENT4} />
        <KPI label="Globoid Cells %" value={`${d.globoid_cells_biopsy_pct}%`} color={ACCENT5} />
        <KPI label="On VPA %" value={`${d.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="On LEV %" value={`${d.on_lev_pct}%`} color={ACCENT4} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (GALC — 14q31.3)" borderColor={ACCENT5}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>Locus:</strong> {d.locus}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Psychosine Cytotoxicity → Globoid Cells → Demyelination" borderColor={ACCENT5}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚠ Unexplained Fever Episodes — PATHOGNOMONIC Stage 1 EI (82%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.unexplained_fever_note}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚠ Globoid Cells — PATHOGNOMONIC on Brain Biopsy (87%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.globoid_cells_note}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="⚠ Peripheral Neuropathy — 100% EI/LI — CNS + PNS Combined Demyelination (UNIQUE)" borderColor={ACCENT2}>
        <p className="small mb-0">{d.peripheral_neuropathy_note}</p>
      </SectionCard>

      <SectionCard title="🧬 HSCT — Narrow Therapeutic Window (Pre-Symptomatic ONLY for EI)" borderColor={ACCENT4}>
        <p className="small mb-0">{d.hsct_window_note}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical & Seizure Profile" borderColor={ACCENT4}>
            <PctBar label="Unexplained Fever (Vasomotor — Stage 1 PATHOGNOMONIC)" pct={d.unexplained_fever_pct} color={ACCENT3} />
            <PctBar label="Peripheral Neuropathy (100% EI/LI)" pct={d.peripheral_neuropathy_pct} color={ACCENT3} />
            <PctBar label="Seizures (any type)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Infantile Spasms" pct={d.infantile_spasms_pct} color={ACCENT2} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="On HSCT (Pre-Symptomatic / Early-JV)" pct={d.on_hsct_pct} color={ACCENT4} />
            <PctBar label="On VPA (Level B)" pct={d.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV (Level B)" pct={d.on_lev_pct} color={ACCENT4} />
            <PctBar label="On ACTH (Level A — IS)" pct={d.on_acth_pct} color={ACCENT4} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT5}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique GALC Features — Psychosine + HSCT Window + No ERT" borderColor={ACCENT}>
        <p className="small mb-0">{d.unique_feature}</p>
      </SectionCard>

      {d.key_pharmacological_distinctions && (
        <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
          {Object.entries(d.key_pharmacological_distinctions).map(([k, v]) => (
            <div key={k} className="mb-2 pb-2 border-bottom">
              <div className="small fw-semibold" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</div>
              <div className="small text-muted">{v}</div>
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── Patients & Etiology tab ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies } = data;
  return (
    <div>
      <Alert
        text="ℹ MLPA/CNV analysis MANDATORY in all GALC workup — 30 kb deletion (c.1700+10kbdel) accounts for 50-70% of European EI Krabbe compound het cases; NOT detectable by WES alone. GALC pseudodeficiency alleles (1-2% of population) → low enzyme but NORMAL psychosine + benign genotype → NO HSCT."
        variant="info"
      />
      <Alert
        text="⚠ Saposin A (PSAP gene, 10q22.2) — Krabbe phenocopy: normal GALC enzyme activity but psychosine elevated. If GALC enzyme normal but clinical + DBS psychosine elevated → check PSAP. Very rare (~20 cases worldwide) but critical to recognize."
        variant="warning"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Krabbe Etiology Classes — 6 Classes (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT }}>{e.class}</h6>
              <span className="badge" style={{ backgroundColor: ACCENT, color: '#fff', fontSize: 13 }}>{e.pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            <p className="small mb-1">{e.description}</p>
            <div className="row small text-muted">
              <div className="col-md-6"><strong>Typical onset:</strong> {e.typical_onset}</div>
              <div className="col-md-6"><strong>Genotype notes:</strong> {e.genotype_notes}</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Seizures & Triggers tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers } = data;
  return (
    <div>
      <Alert
        text="⚠ VGB HIGH RISK for infantile spasms in Krabbe — irreversible visual field defects ADDITIVE to progressive optic atrophy + cortical blindness. Use ACTH (Level A) as first-line. VGB only if ACTH fails AND patient already cortically blind (clinical judgment)."
        variant="danger"
      />
      <Alert
        text="ℹ Krabbe seizures are NOT progressive myoclonic epilepsy (PME). Seizure types: infantile spasms, generalized tonic, focal-autonomic, GTCS. Secondary myoclonus (encephalopathic) ≠ PME cortical reflex myoclonus. No giant SEPs. KEY DISTINCTION from GBA/HEXA/HEXB (PME mechanism)."
        variant="info"
      />

      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Seizure Types</h6>
      {seizure_types?.map((s, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT2 }}>{s.type}</h6>
              <span className="badge bg-danger">{s.prevalence_pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 6 }}>
              <div className="progress-bar bg-danger" style={{ width: `${s.prevalence_pct}%` }} />
            </div>
            <div className="small mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small text-muted"><strong>Tips:</strong> {s.clinical_tips}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT3 }}>Seizure Triggers</h6>
      {triggers?.map((t, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT3 }}>{t.trigger}</span>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct}%</span>
            </div>
            <div className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
            <div className="small text-muted"><strong>Management:</strong> {t.management}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Treatments tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications, lifecycle_stages } = data;
  return (
    <div>
      <Alert
        text="⚠ CBZ/OXC/PHT: RELATIVE CI (worsen spasticity + PNS neuropathy) — NOT ABSOLUTE CI in Krabbe (no PME). Fosphenytoin: RELATIVE CI (PNS neuropathy exacerbation) — IV LEV preferred in SE. VGB: HIGH RISK (irreversible visual field → optic atrophy). TGB: HIGH RISK (myoclonus + spasticity)."
        variant="danger"
      />
      <Alert
        text="🧬 HSCT ONLY for NBS-detected pre-symptomatic EI or early-juvenile Krabbe. NO ERT, NO SRT available. VPA SAFE (lysosomal). ACTH Level A for IS (preferred over VGB). Baclofen essential for spasticity — NEVER abruptly stop (withdrawal = hyperthermia + SE)."
        variant="info"
      />

      <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Treatments (8)</h6>
      {treatments?.map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT4 }}>{t.drug}</h6>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small text-muted"><strong>Monitoring:</strong> {t.monitoring}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT2 }}>Contraindications (7)</h6>
      {contraindications?.map((c, i) => (
        <div key={i} className="card mb-2 shadow-sm"
          style={{ borderLeft: `4px solid ${c.severity === 'RELATIVE CI' ? ACCENT3 : ACCENT2}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold"
                style={{ color: c.severity === 'RELATIVE CI' ? ACCENT3 : ACCENT2 }}>{c.drug}</span>
              <span className={`badge ${c.severity === 'RELATIVE CI' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                {c.severity}
              </span>
            </div>
            <div className="small mb-1"><strong>Reason:</strong> {c.reason}</div>
            <div className="small text-muted"><strong>Alternative:</strong> {c.alternative}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Lifecycle Stages (6)</h6>
      {lifecycle_stages?.map((s, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
          <div className="card-body py-2">
            <div className="small fw-bold" style={{ color: ACCENT }}>{s.stage}</div>
            <div className="small text-muted">{s.description}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Definitions tab ────────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { concepts, thresholds, standards, references } = data;
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Key Concepts (16)</h6>
      {concepts?.map((c, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
          <div className="card-body py-2">
            <div className="small fw-bold" style={{ color: ACCENT5 }}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT3 }}>Clinical Thresholds (12)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered">
          <thead className="table-light">
            <tr>
              <th>Parameter</th>
              <th>Value / Threshold</th>
              <th>Clinical Action</th>
            </tr>
          </thead>
          <tbody>
            {thresholds?.map((t, i) => (
              <tr key={i}>
                <td className="small">{t.parameter}</td>
                <td className="small fw-bold" style={{ color: ACCENT3 }}>{t.value}</td>
                <td className="small">{t.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT6 }}>Standards & Guidelines (12)</h6>
      {standards?.map((s, i) => (
        <div key={i} className="d-flex mb-1">
          <span className="badge me-2 flex-shrink-0" style={{ backgroundColor: ACCENT6, fontSize: 11 }}>
            {s.ref}
          </span>
          <span className="small text-muted">{s.summary}</span>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Key References (6)</h6>
      {references?.map((r, i) => (
        <div key={i} className="mb-1">
          <span className="small fw-semibold" style={{ color: ACCENT }}>[{r.ref}] </span>
          <span className="small text-muted">{r.detail}</span>
        </div>
      ))}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function GALCPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/galc/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/galc/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    fetch(`${API}/api/galc/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <div className="me-3" style={{
          width: 48, height: 48, borderRadius: '50%',
          backgroundColor: ACCENT, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontSize: 22
        }}>🧬</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            GALC Epilepsy — Krabbe Disease (Globoid Cell Leukodystrophy)
          </h4>
          <div className="text-muted small">
            GALC (14q31.3) · Galactocerebrosidase Deficiency · Psychosine Cytotoxin ·
            Unexplained Fever PATHOGNOMONIC (82%) · Globoid Cells PATHOGNOMONIC ·
            HSCT Pre-Symptomatic ONLY · No ERT / No SRT · NBS NY 2006 Hunter's Hope ·
            CBZ/OXC/PHT RELATIVE CI (NOT Absolute — No PME) · AR Biallelic LOF · 40-Patient Cohort
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
