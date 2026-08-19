'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep-purple — ARSA / MLD / leukodystrophy
const ACCENT2 = '#b71c1c';   // dark-red — HIGH RISK / danger
const ACCENT3 = '#e65100';   // deep-orange — CAUTION / PATHOGNOMONIC
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / gene therapy / VPA
const ACCENT5 = '#880e4f';   // dark-pink — molecular / sulfatide / lysosulfatide
const ACCENT6 = '#01579b';   // dark-blue — biomarkers / NBS / urine sulfatides

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
        text="⚠ ARSA PSEUDODEFICIENCY PITFALL: ~1-2% of Europeans carry c.1049A>G (p.Asn350Ser) + c.542T>C (p.Ile181Thr) in CIS → leukocyte ARSA very low BUT no disease. Urine sulfatide quantitation MANDATORY — normal in pseudodeficiency. ARSA enzyme alone is insufficient for NBS or diagnosis."
        variant="danger"
      />
      <Alert
        text="⚠ CBZ/OXC/PHT HIGH RISK in MLD — worsen PERIPHERAL NEUROPATHY (demyelinating NCS already impaired in late-infantile MLD) + spasticity. IV LEV preferred over Fosphenytoin in SE. VGB HIGH RISK — visual field defects compound MLD optic/visual cortex lesions."
        variant="danger"
      />
      <Alert
        text="⚠ ADULT MLD PSYCHIATRIC MISDIAGNOSIS (35%): schizophrenia-like psychosis often precedes motor signs by years. Typical antipsychotics HIGH RISK — worsen spasticity + NMS risk. Any young adult with psychiatric symptoms + white matter changes on MRI = exclude MLD immediately."
        variant="warning"
      />
      <Alert
        text="🧬 Arsa-cel (Lenmeldy): EMA approved July 2020 / FDA accelerated approval March 2024 — first ex vivo HSC gene therapy for MLD. Effective in pre-symptomatic late-infantile and early juvenile MLD only. HSCT limited benefit (inferior to gene therapy); no ERT approved."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${d.infantile_spasms_pct}%`} color={ACCENT3} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="PNS Neuropathy" value={`${d.peripheral_neuropathy_pct}%`} color={ACCENT3} />
        <KPI label="Tigroid MRI" value={`${d.tigroid_mri_pct}%`} color={ACCENT5} />
        <KPI label="Urine Sulfatides +" value={`${d.elevated_urine_sulfatides_pct}%`} color={ACCENT6} />
        <KPI label="On Gene Therapy" value={`${d.on_gene_therapy_pct}%`} color={ACCENT4} />
        <KPI label="On HSCT %" value={`${d.on_hsct_pct}%`} color={ACCENT4} />
        <KPI label="Psychiatric Dx" value={`${d.psychiatric_misdiagnosis_pct}%`} color={ACCENT2} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (ARSA — 22q13.33)" borderColor={ACCENT5}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>Locus:</strong> {d.locus}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Sulfatide/Lysosulfatide Accumulation → Metachromatic Granules → Demyelination" borderColor={ACCENT5}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚠ Tigroid (Leopard Skin) MRI Pattern — PATHOGNOMONIC (85%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.tigroid_note}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚠ ARSA Pseudodeficiency — NBS Pitfall (1-2% Europeans)" borderColor={ACCENT2}>
            <p className="small mb-0">{d.pseudodeficiency_note}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="⚠ Saposin B / PSAP Phenocopy — Normal ARSA Activity, Same MLD Phenotype" borderColor={ACCENT3}>
        <p className="small mb-0">{d.saposin_b_note}</p>
      </SectionCard>

      <SectionCard title="🧬 Arsa-cel (Lenmeldy) — EMA 2020 / FDA 2024 — Ex Vivo HSC Gene Therapy" borderColor={ACCENT4}>
        <p className="small mb-0">{d.arsa_cel_note}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical & Seizure Profile" borderColor={ACCENT4}>
            <PctBar label="Tigroid MRI (Periventricular PATHOGNOMONIC)" pct={d.tigroid_mri_pct} color={ACCENT5} />
            <PctBar label="Peripheral Neuropathy (LI/EJ form)" pct={d.peripheral_neuropathy_pct} color={ACCENT3} />
            <PctBar label="Seizures (any type)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Infantile Spasms (early LI)" pct={d.infantile_spasms_pct} color={ACCENT2} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
            <PctBar label="Adult Psychiatric Misdiagnosis" pct={d.psychiatric_misdiagnosis_pct} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="On Gene Therapy (Arsa-cel / Lenmeldy)" pct={d.on_gene_therapy_pct} color={ACCENT4} />
            <PctBar label="On HSCT (Early JV/Adult pre-symptomatic)" pct={d.on_hsct_pct} color={ACCENT4} />
            <PctBar label="On VPA (Level B)" pct={d.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV (Level B)" pct={d.on_lev_pct} color={ACCENT4} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT5}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique ARSA/MLD Features — Tigroid MRI + Pseudodeficiency Trap + Lenmeldy Gene Therapy" borderColor={ACCENT}>
        <p className="small mb-0">{d.unique_feature}</p>
      </SectionCard>
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
        text="ℹ ARSA PSEUDODEFICIENCY (1-2% Europeans): c.1049A>G (p.Asn350Ser) + c.542T>C (p.Ile181Thr) in CIS → very low leukocyte ARSA but normal urine sulfatides and NO disease. Urine sulfatide quantitation MANDATORY before any ARSA diagnosis. ARSA gene sequencing separates pseudodeficiency from true disease."
        variant="info"
      />
      <Alert
        text="⚠ Saposin B / PSAP phenocopy (10q22.1): ARSA enzyme activity NORMAL, urine sulfatides ELEVATED, PSAP gene abnormal. Clinically indistinguishable from ARSA MLD. Check PSAP if ARSA enzyme normal but MLD clinical phenotype + elevated sulfatides."
        variant="warning"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>MLD Etiology Classes — 6 Classes (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT }}>{e.class || e.name}</h6>
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
        text="⚠ VGB HIGH RISK in MLD — visual field defects compound progressive optic tract + visual cortex lesions (especially adult MLD). Use ACTH (Level A) as first-line for infantile spasms. CBZ/OXC/PHT worsen peripheral neuropathy — choose LEV + VPA."
        variant="danger"
      />
      <Alert
        text="ℹ MLD seizures are NOT primary progressive myoclonic epilepsy. Seizure types: focal-onset, GTCS, myoclonic (later stages), infantile spasms (very early LI). Late myoclonus is encephalopathic rather than PME-cortical-reflex. KEY DISTINCTION from HEXA/HEXB/GBA (true PME)."
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
        text="⚠ CBZ/OXC HIGH RISK — worsen demyelinating peripheral neuropathy (prominent in LI MLD) + spasticity. PHT/Fosphenytoin HIGH RISK — neuropathy worsening; IV LEV preferred in SE. TGB HIGH RISK — worsen myoclonus + spasticity. VGB HIGH RISK — visual field defects additive to MLD visual lesions."
        variant="danger"
      />
      <Alert
        text="🧬 Arsa-cel (Lenmeldy) Level A — EMA 2020 / FDA 2024 — ONLY effective pre-symptomatic LI or early-stage LI/EJ. VPA SAFE (lysosomal NOT mitochondrial) — POLG1 exclusion mandatory. HSCT: pre-symptomatic juvenile/adult only (inferior to gene therapy). No approved ERT for MLD."
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
          style={{ borderLeft: `4px solid ${c.severity === 'CAUTION' ? ACCENT3 : ACCENT2}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold"
                style={{ color: c.severity === 'CAUTION' ? ACCENT3 : ACCENT2 }}>{c.drug}</span>
              <span className={`badge ${c.severity === 'CAUTION' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                {c.severity}
              </span>
            </div>
            <div className="small mb-1"><strong>Reason:</strong> {c.reason}</div>
            <div className="small text-muted"><strong>Alternative:</strong> {c.alternative}</div>
          </div>
        </div>
      ))}

      {lifecycle_stages && lifecycle_stages.length > 0 && (
        <>
          <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>MLD Clinical Stages</h6>
          {lifecycle_stages.map((s, i) => (
            <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
              <div className="card-body py-2">
                <div className="small fw-bold" style={{ color: ACCENT }}>{s.stage}</div>
                <div className="small text-muted">{s.description}</div>
              </div>
            </div>
          ))}
        </>
      )}
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

      {references && references.length > 0 && (
        <>
          <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Key References</h6>
          {references.map((r, i) => (
            <div key={i} className="mb-1">
              <span className="small fw-semibold" style={{ color: ACCENT }}>[{r.ref}] </span>
              <span className="small text-muted">{r.detail}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function ARSAPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/arsa/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/arsa/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    fetch(`${API}/api/arsa/definitions`)
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
            ARSA Epilepsy — Metachromatic Leukodystrophy (MLD)
          </h4>
          <div className="text-muted small">
            ARSA (22q13.33) · Arylsulfatase A Deficiency · Sulfatide/Lysosulfatide Accumulation ·
            Tigroid (Leopard Skin) MRI PATHOGNOMONIC (85%) · ARSA Pseudodeficiency NBS Pitfall ·
            Saposin B/PSAP Phenocopy · Arsa-cel/Lenmeldy EMA 2020 / FDA 2024 ·
            CBZ/OXC/PHT HIGH RISK Peripheral Neuropathy · Adult Psychiatric Misdiagnosis 35% ·
            AR Biallelic LOF · 40-Patient Cohort
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
