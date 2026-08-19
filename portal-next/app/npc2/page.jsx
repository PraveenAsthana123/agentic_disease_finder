'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#004d40';   // dark-teal — NPC2 / lysosomal soluble protein / pulmonary
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — high-risk / PATHOGNOMONIC
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / miglustat
const ACCENT5 = '#006064';   // dark-cyan — PAP / pulmonary / WLL
const ACCENT6 = '#4a148c';   // deep-purple — molecular biology / ML domain

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
        text="⚠ VSGP (93%) + Gelastic Cataplexy (70%) — BOTH PATHOGNOMONIC NPC2. NPC2 child + dyspnea + ground-glass CT = NPC2 PAP + workup MANDATORY."
        variant="danger"
      />
      <Alert
        text="⚠ CBZ / OXC / PHT — ABSOLUTE CI (worsens NPC2 epilepsy + cognition). Fosphenytoin — ABSOLUTE CI; IV LEV replaces in SE."
        variant="danger"
      />
      <Alert
        text="⚠ PAP (Pulmonary Alveolar Proteinosis) — NPC2-SPECIFIC (40-60%). Miglustat does NOT treat PAP — WLL required. Screen ALL NPC2 patients for lung disease."
        variant="warning"
      />
      <Alert
        text="ℹ Miglustat (Zavesca) — Level A neurological NPC2. WLL for PAP. VPA SAFE (lysosomal, not mitochondrial). POLG1/MERRF exclusion mandatory. NPC2 = 5% of all NPC."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="VSGP %" value={`${d.vsgp_pct}%`} color={ACCENT3} />
        <KPI label="Cataplexy %" value={`${d.gelastic_cataplexy_pct}%`} color={ACCENT3} />
        <KPI label="PAP %" value={`${d.pap_pct}%`} color={ACCENT5} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="Ataxia %" value={`${d.ataxia_pct}%`} color={ACCENT} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
        <KPI label="On Miglustat %" value={`${d.on_miglustat_pct}%`} color={ACCENT4} />
        <KPI label="On VPA %" value={`${d.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="Early Infantile %" value={`${d.early_infantile_pct}%`} color={ACCENT5} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (NPC2 — 14q24.3)" borderColor={ACCENT6}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Lysosomal Cholesterol + Pulmonary Surfactant Accumulation" borderColor={ACCENT6}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚠ VSGP — Vertical Supranuclear Gaze Palsy — PATHOGNOMONIC (93%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.vsgp_note}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚠ Gelastic Cataplexy — PATHOGNOMONIC (70%) — NOT Epileptic" borderColor={ACCENT3}>
            <p className="small mb-0">{d.cataplexy_note}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🫁 PAP — Pulmonary Alveolar Proteinosis — NPC2-SPECIFIC (40-60%)" borderColor={ACCENT5}>
        <p className="small mb-0">{d.pap_note}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical & Seizure Profile" borderColor={ACCENT4}>
            <PctBar label="VSGP (PATHOGNOMONIC)" pct={d.vsgp_pct} color={ACCENT3} />
            <PctBar label="Gelastic Cataplexy (PATHOGNOMONIC)" pct={d.gelastic_cataplexy_pct} color={ACCENT3} />
            <PctBar label="PAP (Pulmonary Alveolar Proteinosis)" pct={d.pap_pct} color={ACCENT5} />
            <PctBar label="Seizures" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
            <PctBar label="Cerebellar Ataxia" pct={d.ataxia_pct} color={ACCENT} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment & Type Profile" borderColor={ACCENT4}>
            <PctBar label="On Miglustat (Neurological Disease-Modifying)" pct={d.on_miglustat_pct} color={ACCENT4} />
            <PctBar label="On VPA" pct={d.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV" pct={d.on_lev_pct} color={ACCENT4} />
            <PctBar label="Early Infantile Form (30%)" pct={d.early_infantile_pct} color={ACCENT5} />
            <PctBar label="Neonatal/Perinatal Form (10%)" pct={d.neonatal_pct} color={ACCENT2} />
            <PctBar label="Hepatic Failure in Infancy (18%)" pct={d.hepatic_failure_infancy_pct} color={ACCENT3} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT6}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique NPC2 Features" borderColor={ACCENT}>
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
        text="ℹ NPC2 (14q24.3) accounts for 5% of NPC; NPC1 (18q11.2) 95% — SAME neurological phenotype, different gene. NPC2 has MORE pulmonary (PAP) and hepatic disease. NPC1+NPC2 WES + CNV/MLPA panel mandatory."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>NPC2 Etiology Classes — 6 Classes (40 Patients)</h6>
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
        text="⚠ CBZ / OXC / PHT ABSOLUTE CI — focal seizures + GTCS in NPC2 misidentified as GGE → CBZ prescribed → catastrophic worsening."
        variant="danger"
      />
      <Alert
        text="⚠ Gelastic cataplexy (70%) — EEG NORMAL during event (NOT epileptic). Do NOT treat with AEDs. Miglustat reduces cataplexy."
        variant="warning"
      />
      <Alert
        text="⚠ Respiratory exacerbation (PAP) lowers seizure threshold — hypoxia-triggered seizures. Monitor SpO2 during WLL recovery."
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
        text="🫁 Miglustat (Zavesca 100mg TID) — Level A for NEUROLOGICAL NPC2 ONLY. Does NOT treat PAP. WLL (whole-lung lavage) required for PAP — separate treatment axis."
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
          style={{ borderLeft: `4px solid ${c.severity === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}` }}>
          <div className="card-body py-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold"
                style={{ color: c.severity === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3 }}>{c.drug}</span>
              <span className={`badge ${c.severity === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'}`}>
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
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
          <div className="card-body py-2">
            <div className="small fw-bold" style={{ color: ACCENT6 }}>{c.term}</div>
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

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT5 }}>Standards & Guidelines (12)</h6>
      {standards?.map((s, i) => (
        <div key={i} className="d-flex mb-1">
          <span className="badge me-2 flex-shrink-0" style={{ backgroundColor: ACCENT5, fontSize: 11 }}>
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
export default function NPC2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/npc2/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/npc2/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    fetch(`${API}/api/npc2/definitions`)
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
            NPC2 Epilepsy — Niemann-Pick Disease Type C, Type 2
          </h4>
          <div className="text-muted small">
            NPC2 (14q24.3) · Soluble Lysosomal Cholesterol Carrier · PAP-Associated ·
            VSGP + Gelastic Cataplexy PATHOGNOMONIC · AR Biallelic LOF · 5% of NPC ·
            40-Patient Cohort
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
