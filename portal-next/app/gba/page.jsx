'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // dark-indigo — GBA1 / lysosomal glucocerebrosidase
const ACCENT2 = '#b71c1c';   // dark-red — ABSOLUTE CI / danger
const ACCENT3 = '#e65100';   // deep-orange — high-risk / PATHOGNOMONIC
const ACCENT4 = '#1b5e20';   // dark-green — safe treatments / ERT / ambroxol
const ACCENT5 = '#4a148c';   // deep-purple — molecular / Parkinson risk
const ACCENT6 = '#006064';   // dark-cyan — biomarkers / lyso-Gb1

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
        text="⚠ HORIZONTAL Saccade Palsy (88%) + Action Myoclonus (75%) — BOTH PATHOGNOMONIC Gaucher Type 3. NOT NPC (NPC = VERTICAL gaze). GBA1 enzyme + lyso-Gb1 + WES mandatory."
        variant="danger"
      />
      <Alert
        text="⚠ CBZ / OXC / PHT — ABSOLUTE CI (worsen myoclonic epilepsy/PME). Fosphenytoin — ABSOLUTE CI; IV LEV replaces in SE. TGB — ABSOLUTE CI."
        variant="danger"
      />
      <Alert
        text="⚠ ERT (imiglucerase/velaglucerase) DOES NOT CROSS THE BBB — treats visceral only. Add ambroxol (Level B) for neurological. NEVER stop ERT (visceral rebound)."
        variant="warning"
      />
      <Alert
        text="ℹ GBA1 het carriers → 5-10× Parkinson disease risk. ALL Type 3 families MUST be counselled. Ambroxol (Level B neurological) chaperone + TFEB. VPA SAFE (lysosomal, not mitochondrial) — POLG1 exclusion mandatory."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Mean Onset (y)" value={d.mean_onset_years} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Action Myoclonus" value={`${d.action_myoclonus_pct}%`} color={ACCENT3} />
        <KPI label="H-Saccade Palsy" value={`${d.horizontal_saccade_palsy_pct}%`} color={ACCENT3} />
        <KPI label="Hepatospleno %" value={`${d.hepatosplenomegaly_pct}%`} color={ACCENT} />
        <KPI label="Drug Resistant" value={`${d.drug_resistant_pct}%`} color={ACCENT2} />
        <KPI label="On ERT %" value={`${d.on_ert_pct}%`} color={ACCENT4} />
        <KPI label="Dx Delay (y)" value={d.mean_diagnosis_delay_years} color={ACCENT3} />
        <KPI label="On VPA %" value={`${d.on_vpa_pct}%`} color={ACCENT4} />
        <KPI label="On Ambroxol %" value={`${d.on_ambroxol_pct}%`} color={ACCENT4} />
        <KPI label="On LEV %" value={`${d.on_lev_pct}%`} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary" borderColor={ACCENT}>
        <p className="small mb-0">{d.disease}</p>
      </SectionCard>

      <SectionCard title="Gene & Protein (GBA1 — 1q22)" borderColor={ACCENT5}>
        <p className="small mb-1"><strong>Gene:</strong> {d.gene}</p>
        <p className="small mb-1"><strong>OMIM:</strong> {d.omim}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0"><strong>Protein:</strong> {d.protein}</p>
      </SectionCard>

      <SectionCard title="Pathomechanism — Lysosomal Glucosylceramide + Lyso-Gb1 Neuronal Accumulation" borderColor={ACCENT5}>
        <p className="small mb-0">{d.mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚠ Horizontal Supranuclear Saccade Palsy — PATHOGNOMONIC (88%)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.horizontal_saccade_palsy_note}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚠ Action Myoclonus — Cortical Reflex Myoclonus — PATHOGNOMONIC (75% Type 3a)" borderColor={ACCENT3}>
            <p className="small mb-0">{d.action_myoclonus_note}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🚫 ERT Blood-Brain Barrier Limitation — CRITICAL" borderColor={ACCENT2}>
        <p className="small mb-0">{d.ert_bbb_note}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical & Seizure Profile" borderColor={ACCENT4}>
            <PctBar label="Horizontal Saccade Palsy (PATHOGNOMONIC)" pct={d.horizontal_saccade_palsy_pct} color={ACCENT3} />
            <PctBar label="Action Myoclonus (PATHOGNOMONIC Type 3a)" pct={d.action_myoclonus_pct} color={ACCENT3} />
            <PctBar label="Seizures (any type)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug Resistant" pct={d.drug_resistant_pct} color={ACCENT2} />
            <PctBar label="Hepatosplenomegaly" pct={d.hepatosplenomegaly_pct} color={ACCENT} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="On ERT (Visceral — Level A)" pct={d.on_ert_pct} color={ACCENT4} />
            <PctBar label="On Ambroxol (Neurological — Level B)" pct={d.on_ambroxol_pct} color={ACCENT4} />
            <PctBar label="On VPA (AED — Level B)" pct={d.on_vpa_pct} color={ACCENT4} />
            <PctBar label="On LEV (AED — Level B)" pct={d.on_lev_pct} color={ACCENT4} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Discovery History" borderColor={ACCENT5}>
        <p className="small mb-0">{d.discovery}</p>
      </SectionCard>

      <SectionCard title="Unique GBA1 Features — Dual Role: Lysosomal Disease + PD Risk Gene" borderColor={ACCENT}>
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
        text="ℹ GBA1 (1q22) adjacent to pseudogene GBAP1 (93% homology). WES alone may MISS complex rearrangements (RecNciI). CNV/MLPA MANDATORY in all GBA1 diagnosis. Enzyme activity + lyso-Gb1 confirm diagnosis."
        variant="info"
      />
      <Alert
        text="⚠ N370S (AJ founder) — NEVER neuronopathic when homozygous. L444P — neuronopathic (Type 3). D409H homozygous — Type 3c (cardiac + oculomotor). Genotype-phenotype correlation is imperfect."
        variant="warning"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>GBA Etiology Classes — 6 Classes (40 Patients)</h6>
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
        text="⚠ CBZ / OXC / PHT ABSOLUTE CI — horizontal saccade palsy can be mistaken for focal seizure → CBZ prescribed → myoclonic status epilepticus. Video-EEG mandatory before AED prescription."
        variant="danger"
      />
      <Alert
        text="⚠ Action myoclonus (cortical reflex) — DO NOT treat with CBZ/typical antipsychotics. Best: clonazepam + piracetam + VPA combination. C-reflex distinguishes from tremor."
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
        text="⚠ CBZ / OXC / PHT / Fosphenytoin / TGB — ABSOLUTE CI. Typical antipsychotics HIGH RISK. GBP/PGB HIGH RISK (worsen ataxia). VGB HIGH RISK (worsen myoclonus + irreversible visual fields)."
        variant="danger"
      />
      <Alert
        text="🧬 ERT (imiglucerase/velaglucerase): Level A visceral — DOES NOT CROSS BBB. ADD ambroxol (Level B, pharmacological chaperone) for neurological. NEVER stop ERT."
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
export default function GBAPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/gba/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/gba/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    fetch(`${API}/api/gba/definitions`)
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
            GBA Epilepsy — Gaucher Disease Type 3 (Neuronopathic)
          </h4>
          <div className="text-muted small">
            GBA1 (1q22) · Glucocerebrosidase Deficiency · Lyso-Gb1 Neurotoxic ·
            Horizontal Saccade Palsy PATHOGNOMONIC (88%) · Action Myoclonus PATHOGNOMONIC (75%) ·
            ERT Does NOT Cross BBB · AR Biallelic LOF · 40-Patient Cohort
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
