'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep-indigo — SMPD1 / sphingomyelin / lysosomal
const ACCENT2 = '#b71c1c';   // dark-red — HIGH RISK / danger
const ACCENT3 = '#e65100';   // deep-orange — CAUTION / PATHOGNOMONIC
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / ERT / olipudase alfa
const ACCENT5 = '#4a148c';   // dark-purple — molecular / sphingomyelin / ceramide
const ACCENT6 = '#01579b';   // dark-blue — biomarkers / LysoSM / NBS

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
        text="⚠ CHERRY RED SPOT (75-85% NPA) + VGB HIGH RISK: VGB causes irreversible visual field constriction ADDITIVE to cherry red spot / retinal ganglion cell sphingomyelin storage in NPA. ACTH preferred over VGB for infantile spasms in NPA. Do NOT use VGB in any Niemann-Pick Type A patient."
        variant="danger"
      />
      <Alert
        text="⚠ CBZ/OXC RELATIVE CI in NPA — worsen myoclonus (generalised myoclonic component in NPA); no demyelinating neuropathy mechanism (unlike MLD/Krabbe). Fosphenytoin/PHT RELATIVE CI in SE — IV LEV preferred. Typical antipsychotics HIGH RISK — worsen myoclonus + NMS risk in encephalopathic NPA."
        variant="danger"
      />
      <Alert
        text="⚠ VPA SAFE (lysosomal NOT mitochondrial) BUT enhanced hepatic monitoring MANDATORY — SMPD1 causes hepatomegaly with Kupffer cell sphingomyelin storage; VPA hepatotoxicity risk is additive. POLG1 exclusion mandatory. LFTs every 6-8 weeks (not 3-monthly as in MLD) in NPA hepatic disease."
        variant="warning"
      />
      <Alert
        text="🧬 Olipudase alfa (Xenpozyme): FDA Aug 2022 / EMA Sep 2022 — FIRST approved treatment for ASMD (NPA-B/NPB). Recombinant ASM ERT (IV q2 weeks); reduces liver/spleen/lung sphingomyelin burden. DOES NOT CROSS BBB — no CNS benefit in NPA neurodegeneration. Approved for non-neuronopathic NPB and NPA-B intermediate."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${d.infantile_spasms_pct}%`} color={ACCENT3} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Cherry Red Spot" value={`${d.cherry_red_pct}%`} color={ACCENT3} />
        <KPI label="Hepatosplenomegaly" value={`${d.hepatosplenomegaly_pct}%`} color={ACCENT5} />
        <KPI label="Pulmonary Disease" value={`${d.pulmonary_disease_pct}%`} color={ACCENT6} />
        <KPI label="On ERT (Xenpozyme)" value={`${d.on_ert_pct}%`} color={ACCENT4} />
        <KPI label="On HSCT %" value={`${d.on_hsct_pct}%`} color={ACCENT4} />
        <KPI label="On VPA" value={`${d.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (SMPD1 — 11p15.4)" borderColor={ACCENT5}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>Locus:</strong> {d.locus}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Sphingomyelin Accumulation → Foam Cells → NPA Neurodegeneration / NPB Visceral Disease" borderColor={ACCENT5}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚠ Cherry Red Macular Spot — PATHOGNOMONIC NPA (75-85%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.cherry_red_note}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚠ Sea-Blue Histiocytes in Bone Marrow — PATHOGNOMONIC NPB (90%)" borderColor={ACCENT2}>
            <p className="small mb-0">{d.sea_blue_histiocyte_note}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🧬 Olipudase Alfa (Xenpozyme) — FDA Aug 2022 / EMA Sep 2022 — First ERT for ASMD" borderColor={ACCENT4}>
        <p className="small mb-0">{d.olipudase_alfa_note}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical & Seizure Profile" borderColor={ACCENT4}>
            <PctBar label="Cherry Red Macular Spot (NPA PATHOGNOMONIC)" pct={d.cherry_red_pct} color={ACCENT3} />
            <PctBar label="Hepatosplenomegaly (NPA + NPB)" pct={d.hepatosplenomegaly_pct} color={ACCENT5} />
            <PctBar label="Pulmonary Interstitial Disease (NPB 100%)" pct={d.pulmonary_disease_pct} color={ACCENT6} />
            <PctBar label="Seizures (any type, mainly NPA)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Infantile Spasms (NPA early onset)" pct={d.infantile_spasms_pct} color={ACCENT2} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="On ERT — Olipudase Alfa / Xenpozyme (visceral)" pct={d.on_ert_pct} color={ACCENT4} />
            <PctBar label="On HSCT (experimental, NPA-B)" pct={d.on_hsct_pct} color={ACCENT4} />
            <PctBar label="On VPA (Level B)" pct={d.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV (Level B)" pct={d.on_lev_pct} color={ACCENT4} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT5}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique SMPD1/ASMD Features — Cherry Red + Sea-Blue + Xenpozyme + AJ Founders" borderColor={ACCENT}>
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
        text="ℹ SMPD1 REVISED CLASSIFICATION (Wasserstein 2022): NPA (acid sphingomyelinase deficiency, neuronopathic severe) / NPB (acid sphingomyelinase deficiency, non-neuronopathic) / NPA-B intermediate (ASMD subtype C). Residual ASM activity determines phenotype: ~0% → NPA fatal infantile; ~5-15% → NPB visceral survival."
        variant="info"
      />
      <Alert
        text="⚠ AJ FOUNDER MUTATIONS: p.Arg496Leu + p.Leu302Pro + fsP330 (c.996delC) account for ~97% of Ashkenazi Jewish NPA alleles. p.Arg496Leu = NPA (neuronopathic); p.Leu302Pro = NPA; fsP330 = NPA. Nova Scotia (French-Canadian Acadian) deltaR608 = NPB distinct phenotype."
        variant="warning"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>ASMD Etiology Classes — 6 Classes (40 Patients)</h6>
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
            <p className="small mb-1">{e.description || e.key_feature}</p>
            <div className="row small text-muted">
              <div className="col-md-6"><strong>Seizure %:</strong> {e.seizure_pct}%</div>
              <div className="col-md-6"><strong>Key trigger:</strong> {e.trigger}</div>
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
        text="⚠ VGB ABSOLUTE CONTRAINDICATION in NPA — visual field defects additive to cherry red spot / retinal ganglion cell sphingomyelin accumulation. ACTH Level A preferred for infantile spasms. CBZ/OXC RELATIVE CI — worsen myoclonus (NPA). Fosphenytoin RELATIVE CI in SE — IV LEV preferred."
        variant="danger"
      />
      <Alert
        text="ℹ NPA SEIZURE SPECTRUM: Infantile spasms predominate in early NPA (2-6 months onset). Myoclonic + focal seizures emerge as neurodegeneration progresses. NPB rarely has seizures (non-neuronopathic). Hypoxia-triggered seizures in NPA-B intermediate via pulmonary interstitial disease."
        variant="info"
      />

      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Seizure Types</h6>
      {seizure_types?.map((s, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT2 }}>{s.type}</h6>
              <span className="badge bg-danger">{s.prevalence_pct || s.pct}%</span>
            </div>
            <div className="progress mb-2" style={{ height: 6 }}>
              <div className="progress-bar bg-danger" style={{ width: `${s.prevalence_pct || s.pct}%` }} />
            </div>
            {s.eeg_pattern && <div className="small mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>}
            {s.semiology && <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>}
            <div className="small text-muted">{s.clinical_tips || s.notes}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT3 }}>Seizure Triggers</h6>
      {triggers?.map((t, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT3 }}>{t.trigger}</span>
              <span className="badge" style={{ backgroundColor: ACCENT3 }}>{t.prevalence_pct || t.pct}%</span>
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
        text="⚠ VGB HIGH RISK — visual field defects additive to cherry red/retinal ganglion sphingomyelin (NPA). CBZ/OXC RELATIVE CI — worsen myoclonus. Fosphenytoin RELATIVE CI in SE — IV LEV preferred. TGB HIGH RISK — worsen myoclonus. Typical antipsychotics HIGH RISK — myoclonus worsening + NMS in encephalopathic NPA."
        variant="danger"
      />
      <Alert
        text="🧬 Olipudase alfa (Xenpozyme) Level A — FDA Aug 2022 / EMA Sep 2022 — ERT for NPB/NPA-B intermediate: reduces liver/spleen/lung sphingomyelin. DOES NOT CROSS BBB — no CNS benefit in NPA. VPA SAFE (lysosomal NOT mitochondrial) — enhanced hepatic monitoring (LFTs q6-8w in NPA). POLG1 exclusion mandatory."
        variant="info"
      />

      <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Treatments (8)</h6>
      {treatments?.map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <h6 className="fw-bold mb-0" style={{ color: ACCENT4 }}>{t.drug || t.name}</h6>
              <span className="badge" style={{ backgroundColor: ACCENT4 }}>{t.level}</span>
            </div>
            {t.dose && <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>}
            {t.moa && <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>}
            {t.efficacy && <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>}
            <div className="small text-muted">{t.monitoring || t.notes}</div>
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
                style={{ color: c.severity === 'CAUTION' ? ACCENT3 : ACCENT2 }}>{c.drug || c.name}</span>
              <span className={`badge ${c.severity === 'CAUTION' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                {c.severity}
              </span>
            </div>
            <div className="small mb-1"><strong>Reason:</strong> {c.reason}</div>
            {c.alternative && <div className="small text-muted"><strong>Alternative:</strong> {c.alternative}</div>}
          </div>
        </div>
      ))}

      {lifecycle_stages && lifecycle_stages.length > 0 && (
        <>
          <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>ASMD Clinical Stages</h6>
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
            <div className="small fw-bold" style={{ color: ACCENT5 }}>{c.term || c.name}</div>
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
            {Array.isArray(thresholds)
              ? thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="small">{t.parameter}</td>
                  <td className="small fw-bold" style={{ color: ACCENT3 }}>{t.value}</td>
                  <td className="small">{t.action}</td>
                </tr>
              ))
              : thresholds && Object.entries(thresholds).map(([k, v], i) => (
                <tr key={i}>
                  <td className="small">{k.replace(/_/g, ' ')}</td>
                  <td className="small fw-bold" style={{ color: ACCENT3 }} colSpan={2}>{v}</td>
                </tr>
              ))
            }
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT6 }}>Standards & Guidelines (12)</h6>
      {standards?.map((s, i) => (
        <div key={i} className="d-flex mb-1">
          <span className="badge me-2 flex-shrink-0" style={{ backgroundColor: ACCENT6, fontSize: 11 }}>
            {s.ref || s.author_year}
          </span>
          <span className="small text-muted">{s.summary || s.relevance || s.title}</span>
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
export default function SMPD1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/smpd1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/smpd1/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    fetch(`${API}/api/smpd1/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <div className="me-3" style={{
          width: 48, height: 48, borderRadius: '50%',
          backgroundColor: ACCENT, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontSize: 22
        }}>🫁</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            SMPD1 Epilepsy — Niemann-Pick Disease Types A &amp; B (ASMD)
          </h4>
          <div className="text-muted small">
            SMPD1 (11p15.4) · Acid Sphingomyelinase Deficiency · Sphingomyelin Accumulation ·
            Cherry Red Macular Spot PATHOGNOMONIC NPA (75-85%) · Sea-Blue Histiocytes PATHOGNOMONIC NPB (90%) ·
            Olipudase Alfa/Xenpozyme FDA 2022 / EMA 2022 — First ERT ·
            VGB HIGH RISK Cherry Red Retinal · CBZ/OXC RELATIVE CI Myoclonus ·
            AJ Founders p.Arg496Leu/p.Leu302Pro/fsP330 · AR Biallelic LOF · 40-Patient Cohort
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
