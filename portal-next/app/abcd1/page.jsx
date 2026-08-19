'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — ABCD1 / peroxisomal / X-ALD core
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / adrenal crisis / danger
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / EXTREME HAZARD / enzyme induction
const ACCENT4 = '#2e7d32';   // deep green — safe treatments / HSCT / gene therapy
const ACCENT5 = '#4a148c';   // deep purple — molecular biology / peroxisomal mechanism
const ACCENT6 = '#00695c';   // teal/dark-cyan — gene therapy / Lorenzo's oil / NBS

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
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
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

function Badge({ text, color }) {
  return (
    <span className="badge me-1" style={{ backgroundColor: color, color: '#fff', fontSize: 11 }}>{text}</span>
  );
}

function CICard({ drug, level, reason, alternative }) {
  const color = level?.includes('ABSOLUTE') ? ACCENT2 : level?.includes('EXTREME') ? ACCENT2 : level?.includes('HIGH') ? ACCENT3 : ACCENT3;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level} color={color} />
        </div>
        <p className="small text-danger mb-1">{reason}</p>
        {alternative && <p className="small text-muted mb-0"><strong>Alternative:</strong> {alternative}</p>}
      </div>
    </div>
  );
}

function TreatmentCard({ drug, class: cls, evidence, dose, moa, monitoring, ci }) {
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={cls?.split('—')[0]?.trim()} color={ACCENT4} />
        </div>
        <p className="small mb-1"><strong>Evidence:</strong> {evidence}</p>
        {dose && <p className="small mb-1"><strong>Dose:</strong> {dose}</p>}
        {moa && <p className="small mb-1"><strong>MOA:</strong> {moa}</p>}
        {monitoring && <p className="small mb-1 text-muted"><strong>Monitor:</strong> {monitoring}</p>}
        {ci && <p className="small mb-0 text-warning"><strong>CI:</strong> {ci}</p>}
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert
        text="🚨 PHT/Phenytoin — ABSOLUTE CI: CYP3A4 induction → cortisol drop → adrenal crisis (ALL X-ALD males with AI). IV Fosphenytoin ALSO ABSOLUTE CI in SE — use IV LEV instead."
        variant="danger"
      />
      <Alert
        text="🚨 Anaesthesia — EXTREME HAZARD: 100mg IV hydrocortisone at induction + 50mg q6h × 24h mandatory. Inform ALL surgical/anaesthesia teams of X-ALD adrenal insufficiency BEFORE any procedure."
        variant="danger"
      />
      <Alert
        text="⚠ CBZ / OXC / PB — RELATIVE CI: CYP3A4 induction reduces cortisol levels → adrenal crisis risk. Monitor cortisol q2W × 3M after initiation. LEV is first-line AED (no enzyme induction)."
        variant="warning"
      />
      <Alert
        text="ℹ HSCT Level A (CCALD Loes ≤9 + NRS ≤1 + Gd+) or Skysona gene therapy (FDA Aug 2022, CCALD ≤17yr). HSCT/GT WINDOW = gadolinium enhancement — once passed, benefit lost. Adrenal hormone replacement MANDATORY ALL males."
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="CCALD %" value={`${d.ccald_pct}%`} color={ACCENT3} />
        <KPI label="AMN %" value={`${d.amn_pct}%`} color={ACCENT} />
        <KPI label="Adrenal Insuff %" value={`${d.adrenal_insufficiency_pct}%`} color={ACCENT3} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="On HSCT %" value={`${d.on_hsct_pct}%`} color={ACCENT4} />
        <KPI label="On Gene Therapy %" value={`${d.on_gt_pct}%`} color={ACCENT6} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT5} />
        <KPI label="OMIM Disease" value={`#${d.omim_disease}`} color={ACCENT5} />
        <KPI label="Locus" value={d.locus} color={ACCENT} />
        <KPI label="NBS Rate" value={d.nbs_positive_rate?.split(' ')[0]} color={ACCENT6} />
      </div>

      <SectionCard title="Disease Summary — X-linked Adrenoleukodystrophy (X-ALD)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="CCALD (cerebral forms)" pct={d.ccald_pct} color={ACCENT3} />
            <PctBar label="AMN (adrenomyeloneuropathy)" pct={d.amn_pct} color={ACCENT} />
            <PctBar label="Adrenal Insufficiency" pct={d.adrenal_insufficiency_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="On HSCT (disease-modifying)" pct={d.on_hsct_pct} color={ACCENT4} />
            <PctBar label="On Gene Therapy (Skysona)" pct={d.on_gt_pct} color={ACCENT6} />
            <div className="mt-2 small text-muted">
              <strong>VLCFA Diagnostic Sensitivity:</strong> {d.vlcfa_diagnostic_c26_sensitivity_pct}% (plasma C26:0 in males)
            </div>
            <div className="small text-muted mt-1">
              <strong>NBS Rate:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="PHT/Phenytoin — ABSOLUTE CI — CYP3A4 induction → cortisol catabolism → adrenal crisis in 71% of X-ALD males with AI. NEVER use in X-ALD." variant="danger" />
        <Alert text="CBZ/OXC/PB — RELATIVE CI — enzyme induction → cortisol falls 30-40%. Cortisol levels q2W × 3M after starting any of these. LEV replaces as first-line." variant="warning" />
        <Alert text="HSCT (Level A) or Skysona gene therapy (FDA 2022) — ONLY disease-modifying options. Loes ≤9 + NRS ≤1 + Gd+ = IMMEDIATE REFERRAL. Adrenal hormone replacement MANDATORY all males." variant="info" />
      </SectionCard>

      <SectionCard title="Key Concepts" borderColor={ACCENT5}>
        <ul className="list-unstyled mb-0">
          {d.key_concepts?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT5 }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Reference Standards" borderColor={ACCENT4}>
        <ul className="mb-0">
          {d.standards?.map((s, i) => <li key={i} className="small">{s}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Patients & Etiology Tab ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { etiologies, patients } = data;
  const ETIOL_COLORS = {
    'CCALD': ACCENT2,
    'Adolescent-Cerebral': ACCENT3,
    'Adult-Cerebral': '#c62828',
    'AMN': ACCENT,
    'Addison-only': ACCENT5,
    'Female-het-AMN': ACCENT6,
  };
  return (
    <div>
      <Alert
        text="ℹ GENOTYPE DOES NOT PREDICT PHENOTYPE — same ABCD1 mutation → CCALD in one brother + AMN in another + Addison-only in a third. Modifier genes suspected. Serial MRI mandatory all males age 3-12yr."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>ABCD1 Phenotypic Classes — 6 Classes (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ETIOL_COLORS[e.name?.split(' ')[0]] || ACCENT}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: ETIOL_COLORS[e.name?.split(' ')[0]] || ACCENT }}>{e.name}</h6>
              <div>
                <Badge text={`${e.pct}%`} color={ETIOL_COLORS[e.name?.split(' ')[0]] || ACCENT} />
                <Badge text={`n=${e.n}`} color="#555" />
                <Badge text={e.sex} color="#888" />
              </div>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ETIOL_COLORS[e.name?.split(' ')[0]] || ACCENT }} />
            </div>
            <div className="row small">
              <div className="col-md-6">
                <p className="mb-1"><strong>Onset:</strong> {e.onset_age}</p>
                <p className="mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
                <p className="mb-1"><strong>Loes range:</strong> {e.loes_range}</p>
                <div className="d-flex gap-2 flex-wrap mt-1">
                  {e.hsct_eligible && <Badge text="HSCT eligible" color={ACCENT4} />}
                  {e.gt_eligible && <Badge text="Gene Therapy eligible" color={ACCENT6} />}
                  {!e.ert_available && <Badge text="No ERT available" color="#888" />}
                </div>
              </div>
            </div>
            <p className="small text-muted mb-0 mt-2">{e.variant_detail}</p>
          </div>
        </div>
      ))}

      <SectionCard title="Patient Table (40 patients)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Sex</th><th>Loes</th><th>AI</th>
                <th>Seizures</th><th>AED</th><th>Response</th><th>HSCT</th><th>GT</th>
              </tr>
            </thead>
            <tbody>
              {patients?.map((p, i) => (
                <tr key={i}>
                  <td><code>{p.patient_id}</code></td>
                  <td><Badge text={p.phenotype} color={ETIOL_COLORS[p.phenotype] || ACCENT} /></td>
                  <td>{p.sex}</td>
                  <td>{p.loes_score}</td>
                  <td>{p.has_ai ? <span className="text-warning fw-bold">AI+</span> : <span className="text-muted">—</span>}</td>
                  <td>{p.has_seizures ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">No</span>}</td>
                  <td className="small">{p.primary_aed || '—'}</td>
                  <td>
                    {p.drug_response ? (
                      <Badge text={p.drug_response}
                        color={p.drug_response === 'Drug-resistant' ? ACCENT2 :
                               p.drug_response === 'Partially controlled' ? ACCENT3 : ACCENT4} />
                    ) : '—'}
                  </td>
                  <td>{p.on_hsct ? <Badge text="HSCT" color={ACCENT4} /> : '—'}</td>
                  <td>{p.on_gt ? <Badge text="GT" color={ACCENT6} /> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [], monitoring = [], thresholds = [], lifecycle = [] } = data;
  return (
    <>
      <SectionCard title="Seizure Types (X-ALD — Posterior-Predominant)" borderColor={ACCENT2}>
        {seizure_types.map((st, i) => (
          <div key={i} className="mb-3">
            <PctBar label={st.type} pct={st.pct} color={ACCENT2} />
            <div className="small text-muted">{st.eeg}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (Adrenal Crisis — DOMINANT TRIGGER)" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <PctBar label={t.trigger} pct={t.pct} color={ACCENT3} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        <ul className="list-unstyled mb-0">
          {monitoring.map((m, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT4 }}>▸</span>{m}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="Clinical Thresholds & Action Points" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{t.parameter}</td>
                  <td><Badge text={t.threshold} color={ACCENT3} /></td>
                  <td className="small text-muted">{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle (6 Stages)" borderColor={ACCENT}>
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-3 pb-3 border-bottom">
            <div className="fw-bold small" style={{ color: ACCENT }}>{l.stage}</div>
            <div className="small text-muted">{l.features}</div>
            <div className="small mt-1"><strong>Action:</strong> {l.action}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <Alert
        text="ℹ HSCT (Level A, CCALD Loes ≤9 + NRS ≤1) or Skysona gene therapy (FDA Aug 2022, CCALD ≤17yr, no HLA match) are the ONLY disease-modifying options. Adrenal hormone replacement MANDATORY all males. LEV = first-line AED (no enzyme induction)."
        variant="info"
      />
      <SectionCard title="Treatments (Disease-Modifying + AEDs + Adrenal)" borderColor={ACCENT4}>
        {data.treatments?.map((t, i) => (
          <TreatmentCard key={i} {...t} />
        ))}
      </SectionCard>
      <SectionCard title="Contraindications & Hazards" borderColor={ACCENT2}>
        {data.contraindications?.map((c, i) => (
          <CICard key={i} {...c} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading definitions…</div>;
  return (
    <>
      <SectionCard title="Key Concepts (16)" borderColor={ACCENT5}>
        <ul className="list-unstyled mb-0">
          {data.key_concepts?.map((c, i) => (
            <li key={i} className="mb-2 small">
              <span className="me-1" style={{ color: ACCENT5 }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="10-Step Diagnostic Algorithm" borderColor={ACCENT}>
        <ol className="mb-0">
          {data.diagnostic_algorithm?.map((step, i) => (
            <li key={i} className="small mb-2">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Pharmacological Distinctions (12 Points)" borderColor={ACCENT2}>
        <ol className="mb-0">
          {data.pharmacological_distinctions?.map((p, i) => (
            <li key={i} className="small mb-2">{p}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Differential Diagnosis" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Condition</th><th>Distinguishing Features</th></tr></thead>
            <tbody>
              {data.differential_diagnosis?.map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold small">{d.condition}</td>
                  <td className="small text-muted">{d.distinction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Reference Standards" borderColor={ACCENT4}>
        <ul className="mb-0">
          {data.standards?.map((s, i) => <li key={i} className="small mb-1">{s}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function ABCD1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/abcd1/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/abcd1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/abcd1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          ABCD1 Epilepsy — X-linked Adrenoleukodystrophy (X-ALD)
        </h2>
        <p className="text-muted small mb-2">
          ABCD1 (Xq28) · Adrenoleukodystrophy Protein (ALDP) · Peroxisomal VLCFA beta-oxidation failure ·
          X-linked (males hemizygous) · CCALD seizures 90%+ · Adrenal Insufficiency 71% ·
          HSCT Level A (Loes ≤9) · Skysona gene therapy FDA 2022 ·
          PHT ABSOLUTE CI · Anaesthesia EXTREME HAZARD · 40 patients
        </p>
        {err && <div className="alert alert-danger small">{err}</div>}
      </div>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
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
