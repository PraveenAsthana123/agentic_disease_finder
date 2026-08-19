'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep-indigo — NPC1 / cholesterol transport / lysosomal
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — high-risk warnings / PATHOGNOMONIC
const ACCENT4 = '#2e7d32';   // deep-green — safe treatments / disease-modifying
const ACCENT5 = '#4a148c';   // deep-purple — molecular biology / sterol-sensing
const ACCENT6 = '#006064';   // dark-cyan — gene therapy / research / HPβCD

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
        text="⚠ VSGP (95%) + Gelastic Cataplexy (72%) — BOTH PATHOGNOMONIC NPC. VSGP + ataxia + young onset = NPC workup MANDATORY."
        variant="danger"
      />
      <Alert
        text="⚠ CBZ / OXC / PHT — ABSOLUTE CI (worsens NPC epilepsy + cognition + cataplexy). Fosphenytoin — ABSOLUTE CI; IV LEV replaces in SE."
        variant="danger"
      />
      <Alert
        text="⚠ Typical antipsychotics — HIGH RISK (worsen gelastic cataplexy + EPS in NPC). Atypical antipsychotics ONLY if psychiatric features."
        variant="warning"
      />
      <Alert
        text="ℹ Miglustat (Zavesca) — ONLY EMA-approved disease-modifying NPC therapy (Level A, 2009). Start at diagnosis. VPA SAFE (lysosomal, not mitochondrial). POLG1/MERRF exclusion mandatory."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="VSGP %" value={`${d.vsgp_pct}%`} color={ACCENT3} />
        <KPI label="Cataplexy %" value={`${d.gelastic_cataplexy_pct}%`} color={ACCENT3} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Ataxia %" value={`${d.ataxia_pct}%`} color={ACCENT} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
        <KPI label="On Miglustat %" value={`${d.on_miglustat_pct}%`} color={ACCENT4} />
        <KPI label="On VPA %" value={`${d.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="On LEV %" value={`${d.on_lev_pct}%`} color={ACCENT4} />
        <KPI label="Psychiatric %" value={`${d.psychiatric_features_pct}%`} color={ACCENT5} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (NPC1 — 18q11.2)" borderColor={ACCENT5}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Lysosomal Cholesterol Accumulation" borderColor={ACCENT5}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚠ VSGP — Vertical Supranuclear Gaze Palsy — PATHOGNOMONIC (95%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.vsgp_note}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚠ Gelastic Cataplexy — PATHOGNOMONIC (72%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.cataplexy_note}</p>
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical & Seizure Profile" borderColor={ACCENT4}>
            <PctBar label="VSGP (PATHOGNOMONIC)" pct={d.vsgp_pct} color={ACCENT3} />
            <PctBar label="Gelastic Cataplexy (PATHOGNOMONIC)" pct={d.gelastic_cataplexy_pct} color={ACCENT3} />
            <PctBar label="Seizures" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
            <PctBar label="Cerebellar Ataxia" pct={d.ataxia_pct} color={ACCENT} />
            <PctBar label="Psychiatric Features" pct={d.psychiatric_features_pct} color={ACCENT5} />
            <PctBar label="Dysphagia" pct={d.dysphagia_pct} color={ACCENT} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="On Miglustat (Disease-Modifying)" pct={d.on_miglustat_pct} color={ACCENT4} />
            <PctBar label="On VPA" pct={d.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV" pct={d.on_lev_pct} color={ACCENT4} />
            <PctBar label="Late Infantile Type (40%)" pct={d.late_infantile_pct} color={ACCENT} />
            <PctBar label="Juvenile Type (35%)" pct={d.juvenile_pct} color={ACCENT5} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT6}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique NPC1 Features" borderColor={ACCENT}>
        <p className="small mb-0">{d.unique_feature}</p>
      </SectionCard>

      {d.key_pharmacological_distinctions && (
        <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
          {Object.entries(d.key_pharmacological_distinctions).map(([k, v]) => (
            <div key={k} className="mb-2 pb-2 border-bottom">
              <div className="small fw-semibold" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</div>
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
        text="ℹ NPC1 (18q11.2) accounts for 95% of NPC; NPC2 (14q24.3) 5% — SAME phenotype, different gene. NPC1 WES + CNV/MLPA mandatory. No major founder effect globally."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>NPC1 Etiology Classes — 6 Classes (40 Patients)</h6>
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
        text="⚠ CBZ / OXC / PHT ABSOLUTE CI — focal seizures + GTCS misidentified as GGE → CBZ prescribed → worsening NPC epilepsy + cognition."
        variant="danger"
      />
      <Alert
        text="⚠ Gelastic cataplexy (72%) — EEG NORMAL during event (NOT epileptic). Do NOT treat with AEDs — treat underlying NPC with Miglustat."
        variant="warning"
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
        text="⚠ CBZ / OXC / PHT / Fosphenytoin — ABSOLUTE CI. Typical antipsychotics HIGH RISK. GBP/PGB HIGH RISK (worsen ataxia)."
        variant="danger"
      />
      <Alert
        text="ℹ Miglustat (Zavesca 100mg TID) — ONLY EMA-approved disease-modifying NPC therapy (Level A). VPA SAFE (lysosomal, not mitochondrial). POLG1/MERRF exclusion mandatory before VPA."
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
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.npc1_note && (
              <div className="small p-2 rounded mt-1" style={{ backgroundColor: '#e8eaf6', color: '#1a237e' }}>
                <strong>NPC1 Note:</strong> {t.npc1_note}
              </div>
            )}
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT2 }}>Contraindications</h6>
      {contraindications?.map((c, i) => (
        <div key={i} className="card mb-2 shadow-sm border-danger">
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT2 }}>{c.drug}</span>
              <span className={`badge ${c.severity === 'ABSOLUTE CI' ? 'bg-danger' : c.severity.includes('HIGH RISK') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                {c.severity}
              </span>
            </div>
            <div className="small mb-1">{c.reason}</div>
            <div className="small text-muted"><strong>Note:</strong> {c.note}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT6 }}>Disease Lifecycle Stages</h6>
      {lifecycle_stages?.map((s, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold" style={{ color: ACCENT6 }}>{s.stage}</span>
              <span className="badge" style={{ backgroundColor: ACCENT6 }}>{s.age_range}</span>
            </div>
            <p className="small mb-1">{s.description}</p>
            <ul className="mb-0 ps-3">
              {s.priorities?.map((p, j) => <li key={j} className="small">{p}</li>)}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Definitions tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <SectionCard title="Disease & Gene" borderColor={ACCENT}>
        <div className="small mb-1"><strong>Disease:</strong> {data.disease_name}</div>
        <div className="small mb-1"><strong>Gene (full):</strong> {data.gene_full}</div>
        <div className="small mb-1"><strong>OMIM Gene:</strong> {data.omim_gene}</div>
        <div className="small mb-1"><strong>OMIM Disease:</strong> {data.omim_disease}</div>
        <div className="small mb-1"><strong>Inheritance:</strong> {data.inheritance_mode}</div>
        <div className="small mb-0"><strong>Onset age:</strong> {data.onset_age}</div>
      </SectionCard>

      <SectionCard title="Protein Structure (NPC1)" borderColor={ACCENT5}>
        <p className="small mb-0">{data.protein_full}</p>
      </SectionCard>

      <SectionCard title="NPC1 + NPC2 Two-Protein Cholesterol Hand-Off System" borderColor={ACCENT5}>
        <p className="small mb-0">{data.npc1_npc2_cholesterol_handoff}</p>
      </SectionCard>

      <SectionCard title="VSGP Anatomy in NPC — Brainstem Storage Pathway" borderColor={ACCENT3}>
        <p className="small mb-0">{data.vsgp_anatomy}</p>
      </SectionCard>

      <SectionCard title="Gelastic Cataplexy — Mechanism (NOT Epileptic)" borderColor={ACCENT3}>
        <p className="small mb-0">{data.gelastic_cataplexy_mechanism}</p>
      </SectionCard>

      <SectionCard title="NPC Biomarker Diagnostic Hierarchy" borderColor={ACCENT6}>
        <p className="small mb-0">{data.biomarker_hierarchy}</p>
      </SectionCard>

      <SectionCard title="Miglustat Mechanism — Substrate Reduction Therapy" borderColor={ACCENT4}>
        <p className="small mb-0">{data.miglustat_mechanism_detail}</p>
      </SectionCard>

      <SectionCard title="NPC2 Distinction — Same Phenotype, Different Gene" borderColor={ACCENT}>
        <p className="small mb-0">{data.npc2_distinction}</p>
      </SectionCard>

      {data.sandhoff_vs_npc_key_differences && (
        <SectionCard title="Sandhoff (HEXB) vs NPC1 — Key Differences" borderColor={ACCENT2}>
          <div className="table-responsive">
            <table className="table table-sm small mb-0">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th style={{ color: '#3e2723' }}>Sandhoff (HEXB)</th>
                  <th style={{ color: ACCENT }}>NPC1</th>
                </tr>
              </thead>
              <tbody>
                {data.sandhoff_vs_npc_key_differences.map((r, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{r.feature}</td>
                    <td>{r.sandhoff}</td>
                    <td>{r.npc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      <h6 className="fw-bold mb-3 mt-3" style={{ color: ACCENT }}>Key Concepts (15)</h6>
      {data.concepts?.map((c, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `3px solid ${ACCENT}` }}>
          <div className="card-body py-2">
            <div className="small fw-bold mb-1" style={{ color: ACCENT }}>{c.name}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        </div>
      ))}

      <SectionCard title="Thresholds" borderColor={ACCENT3}>
        {data.thresholds?.map((t, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <div className="small fw-semibold" style={{ color: ACCENT3 }}>{t.parameter}</div>
            <div className="small"><strong>Value:</strong> {t.value}</div>
            <div className="small text-muted"><strong>Action:</strong> {t.action}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Standards" borderColor={ACCENT6}>
        <ol className="mb-0 ps-3">
          {data.standards?.map((s, i) => <li key={i} className="small mb-1">{s}</li>)}
        </ol>
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT}>
        <ol className="mb-0 ps-3">
          {data.references?.map((r, i) => <li key={i} className="small mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function NPC1Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/npc1/overview`).then(r => r.json()),
      fetch(`${API}/api/npc1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/npc1/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="container py-4"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, #283593 100%)` }}>
        <h4 className="mb-1 fw-bold">&#x1f52e; NPC1 Epilepsy — Niemann-Pick Disease Type C</h4>
        <div className="small opacity-90">
          NPC Intracellular Cholesterol Transporter 1 Deficiency · NPC1 (18q11.2) · 13-TM Lysosomal Membrane Protein ·
          VSGP Downward-Gaze-First PATHOGNOMONIC (95%) · Gelastic-Cataplexy-Emotion-Triggered-Atonia PATHOGNOMONIC (72%) ·
          Plasma-Oxysterols + Lyso-SM-509 Non-Invasive Screen · Filipin-Staining Gold-Standard ·
          Miglustat EMA-Approved 2009 Level-A Disease-Modifying · HPβCD Phase-3 ACEND Trial ·
          CBZ/OXC/PHT ABSOLUTE CI · VPA SAFE (POLG1 Exclusion Mandatory) · IV-LEV Replaces Fosphenytoin ·
          Typical-Antipsychotics HIGH-RISK · GBP/PGB HIGH-RISK-Ataxia · 95% NPC1 / 5% NPC2 Same-Phenotype ·
          AR Biallelic LOF · OMIM #257220
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setActiveTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      <div>
        {activeTab === 0 && <OverviewTab data={overview} />}
        {activeTab === 1 && <PatientsTab data={breakdown} />}
        {activeTab === 2 && <SeizuresTab data={breakdown} />}
        {activeTab === 3 && <TreatmentsTab data={breakdown} />}
        {activeTab === 4 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
