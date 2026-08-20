'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep green — ZSD / peroxisomal biogenesis / DHA
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK CI / VPA / VGB / ZS severe
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / fasting / NALD
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / DHA / ACTH / LEV
const ACCENT5 = '#4a148c';   // deep purple — molecular biology / PEX3 membrane receptor
const ACCENT6 = '#00695c';   // teal — DHA / NBS / PEX19 docking

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
  const color = level?.includes('ABSOLUTE') || level?.includes('HIGH RISK') || level?.includes('EXTREME')
    ? ACCENT2 : ACCENT3;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('(')[0]?.trim().split(' ').slice(0, 3).join(' ')} color={color} />
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
        text="🚨 VPA — HIGH RISK (near-Absolute CI in ZS/NALD): THREE mechanisms — (1) hepatotoxicity (baseline cholestatic liver in ZS/NALD) + (2) peroxisomal beta-oxidation inhibition (worsens VLCFA accumulation) + (3) carnitine depletion. POLG1 exclusion MANDATORY (CPIC Grade A) before any VPA."
        variant="danger"
      />
      <Alert
        text="🚨 VGB (Vigabatrin) — HIGH RISK: ZSD retinopathy (universal ZS/NALD; 15–20% IRD) + VGB irreversible VF constriction = additive irreversible blindness. ACTH (Level A) preferred for infantile spasms. AVOID VGB in all ZSD forms."
        variant="danger"
      />
      <Alert
        text="⚠ Fasting / Anaesthesia / NPO — EXTREME HAZARD: starvation → adipose phytanic acid release → acute neurotoxicity + seizures. IV dextrose MANDATORY during any pre-surgical fast. Pre-op: PT/INR + LFT + platelet count + IV Vitamin K (cholestatic coagulopathy in ZS/NALD)."
        variant="warning"
      />
      <Alert
        text={`ℹ PEX3 = DOCKING RECEPTOR FOR PEX19 on peroxisomal membrane (373 aa, 6q24.2, type I integral PMP). PEX3 N-terminal TMD anchors it; C-terminal cytoplasmic domain binds PEX19 (cytosolic PMP chaperone). PEX3 LOF → PEX19 cannot dock → ALL peroxisomal membrane proteins (PMPs) fail to insert → peroxisomes absent or 'ghost' → ALL matrix import fails. PEX3 acts UPSTREAM of PEX26, PEX1/PEX6, and PEX2/PEX10/PEX12 — it establishes the peroxisomal membrane that all downstream peroxins require. ~1–2% of all PBD-ZSD; ~10–20 cases worldwide 2026. COMPLETES the PEX3-PEX16-PEX19 early membrane biogenesis axis.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="ZS (Severe) %" value={`${d.zs_pct}%`} color={ACCENT2} />
        <KPI label="NALD %" value={`${d.nald_pct}%`} color={ACCENT3} />
        <KPI label="IRD (Attenuated) %" value={`${d.ird_pct}%`} color={ACCENT} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="On DHA %" value={`${d.on_dha_pct}%`} color={ACCENT6} />
        <KPI label="Liver Disease %" value={`${d.liver_disease_pct}%`} color={ACCENT3} />
        <KPI label="Retinopathy %" value={`${d.retinopathy_pct}%`} color={ACCENT2} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT5} />
        <KPI label="Locus" value={d.locus} color={ACCENT} />
        <KPI label="NBS Rate" value={d.nbs_positive_rate?.split(' ')[0]} color={ACCENT6} />
      </div>

      <SectionCard title="Disease Summary — PEX3 / Zellweger Spectrum Disorder (ZSD)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Allele Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="ZS (severe — neonatal, null/null)" pct={d.zs_pct} color={ACCENT2} />
            <PctBar label="NALD (intermediate — null/L61Pro)" pct={d.nald_pct} color={ACCENT3} />
            <PctBar label="IRD (attenuated — L61Pro/mild)" pct={d.ird_pct} color={ACCENT} />
            <PctBar label="Liver disease (ZS + NALD)" pct={d.liver_disease_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical & Treatment Profile" borderColor={ACCENT4}>
            <PctBar label="VLCFA elevated (C26:0)" pct={d.vlcfa_elevated_pct} color={ACCENT2} />
            <PctBar label="Plasmalogens LOW (ZSD marker)" pct={d.plasmalogen_low_pct} color={ACCENT5} />
            <PctBar label="DHA deficiency" pct={d.dha_low_pct} color={ACCENT3} />
            <PctBar label="Retinopathy (ZS/NALD universal)" pct={d.retinopathy_pct} color={ACCENT2} />
            <PctBar label="On DHA supplementation" pct={d.on_dha_pct} color={ACCENT6} />
            <div className="mt-2 small text-muted">
              <strong>NBS:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="PEX3 Docking Receptor Mechanism — PEX3-PEX16-PEX19 Early Membrane Biogenesis Axis" borderColor={ACCENT5}>
        <div className="row small">
          <div className="col-md-4">
            <p className="fw-bold mb-1" style={{ color: ACCENT5 }}>PEX3 — Docking Receptor (MEMBRANE)</p>
            <ul className="mb-0">
              <li>Type I integral PMP: N-terminal TMD in membrane</li>
              <li>Large C-terminal cytoplasmic domain = PEX19 receptor</li>
              <li>PEX3 LOF → PEX19 cannot stably engage membrane</li>
              <li>ALL PMPs fail to insert → ghost peroxisomes</li>
              <li>MOST UPSTREAM ZSD defect in this series</li>
            </ul>
          </div>
          <div className="col-md-4">
            <p className="fw-bold mb-1" style={{ color: ACCENT6 }}>PEX19 — Cytosolic PMP Chaperone (CARGO CARRIER)</p>
            <ul className="mb-0">
              <li>Farnesylated cytosolic/peripheral membrane protein</li>
              <li>C-terminal domain: binds newly synthesised PMPs</li>
              <li>Docks onto PEX3 cytoplasmic domain</li>
              <li>Releases PMPs into peroxisomal membrane (class I)</li>
              <li>PEX19 requires PEX3 to deliver ALL PMPs</li>
            </ul>
          </div>
          <div className="col-md-4">
            <p className="fw-bold mb-1" style={{ color: ACCENT4 }}>Downstream Consequences of PEX3 LOF</p>
            <ul className="mb-0">
              <li>PEX26 (tail-anchored PMP) cannot be inserted</li>
              <li>→ PEX1-PEX6 motor cannot dock</li>
              <li>PEX14 translocon cannot form</li>
              <li>PEX2/PEX10/PEX12 RING complex absent from membrane</li>
              <li>ALL downstream peroxins fail simultaneously</li>
            </ul>
          </div>
        </div>
        <div className="mt-2 small">
          <Badge text="p.Leu61Pro (L61Pro)" color={ACCENT5} /> Near C-terminal end of TMD → reduces PEX3 membrane insertion efficiency (~20–40% residual) → partial PEX19 docking → enables NALD/IRD (15% IRD in PEX3-ZSD).
          <Badge text="p.Gln85*" color={ACCENT2} className="ms-2" /> Most common null truncating variant → ZS and NALD (no functional PEX3 → ghost peroxisomes).
        </div>
      </SectionCard>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="VPA — HIGH RISK: THREE independent mechanisms — hepatotoxicity (cholestatic liver ZS/NALD) + peroxisomal BO inhibition (worsens VLCFA) + carnitine depletion. POLG1 MANDATORY (CPIC Grade A)." variant="danger" />
        <Alert text="VGB — HIGH RISK: retinopathy (universal ZS/NALD) + VGB irreversible VF constriction = additive blindness. ACTH Level A for IS. AVOID in all ZSD forms." variant="danger" />
        <Alert text="PHT/CBZ/OXC — RELATIVE CI (CYP3A4 → DHA depletion + hepatic). NOT adrenal crisis (unlike ABCD1 ABSOLUTE CI). IV LEV replaces in ZSD status epilepticus." variant="warning" />
        <Alert text="DHA Level B (NALD/IRD). ACTH Level A for IS. LEV first-line all ZSD forms. Phytol-restricted diet Level B (IRD). No ERT / No HSCT / Lorenzo's Oil INEFFECTIVE (membrane absent)." variant="info" />
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
    'ZS (Zellweger-Severe)': ACCENT2,
    'NALD (Neonatal-ALD-Intermediate)': ACCENT3,
    'IRD (Infantile-Refsum-Attenuated)': ACCENT,
    'Atypical / Late-Onset ZSD': ACCENT6,
  };
  const getColor = (name) => {
    for (const [key, color] of Object.entries(ETIOL_COLORS)) {
      if (name?.includes(key.split(' ')[0])) return color;
    }
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ ZSD SPECTRUM (PEX3): null/null → ZS (35% — more ZS than PEX26 25%, reflecting no founder hypomorphic) · null/p.Leu61Pro → NALD (45%) · p.Leu61Pro/mild → IRD (15% — less IRD than PEX6 45%, reflecting less-studied partial alleles) · Atypical (5%). PEX3 has p.Leu61Pro partial-function allele (near TMD; IRD/NALD-enabling). Biochemically identical to all other PBD-ZSD; only NGS distinguishes."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>PEX3-ZSD Phenotypic Classes — 4 Classes (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${getColor(e.name)}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: getColor(e.name) }}>{e.name}</h6>
              <div>
                <Badge text={`${e.pct}%`} color={getColor(e.name)} />
                <Badge text={`n=${e.n}`} color="#555" />
                <Badge text={e.sex} color="#888" />
              </div>
            </div>
            <div className="progress mb-2" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: getColor(e.name) }} />
            </div>
            <div className="row small">
              <div className="col-md-6">
                <p className="mb-1"><strong>Onset:</strong> {e.onset_age}</p>
                <p className="mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
                <div className="d-flex gap-2 flex-wrap mt-1">
                  {e.dha_supplement && <Badge text="DHA supplementation" color={ACCENT6} />}
                  {!e.hsct_eligible && <Badge text="HSCT not indicated" color="#888" />}
                  {!e.ert_available && <Badge text="No ERT (type I integral membrane protein)" color="#888" />}
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
                <th>ID</th><th>Phenotype</th><th>Sex</th><th>Genotype</th><th>Liver</th>
                <th>Seizures</th><th>AED</th><th>Response</th><th>DHA</th>
              </tr>
            </thead>
            <tbody>
              {patients?.map((p, i) => (
                <tr key={i}>
                  <td><code>{p.patient_id}</code></td>
                  <td><Badge text={p.phenotype?.split(' ')[0] || p.phenotype} color={getColor(p.phenotype)} /></td>
                  <td>{p.sex}</td>
                  <td className="small">{p.genotype}</td>
                  <td>{p.liver_disease ? <span className="text-warning fw-bold">Yes</span> : <span className="text-muted">—</span>}</td>
                  <td>{p.has_seizures ? <span className="text-danger fw-bold">Yes</span> : <span className="text-muted">No</span>}</td>
                  <td className="small">{p.primary_aed || '—'}</td>
                  <td>
                    {p.drug_response ? (
                      <Badge text={p.drug_response}
                        color={p.drug_response === 'Drug-resistant' ? ACCENT2 :
                               p.drug_response === 'Partially controlled' ? ACCENT3 : ACCENT4} />
                    ) : '—'}
                  </td>
                  <td>{p.on_dha ? <Badge text="DHA" color={ACCENT6} /> : '—'}</td>
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
      <SectionCard title="Seizure Types (ZSD — Neonatal to Focal, ZS → NALD → IRD)" borderColor={ACCENT2}>
        {seizure_types.map((st, i) => (
          <div key={i} className="mb-3">
            <PctBar label={st.type} pct={st.pct} color={ACCENT2} />
            <div className="small text-muted">{st.eeg}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (Febrile + Phytanic Acid + Fasting)" borderColor={ACCENT3}>
        {triggers.map((t, i) => (
          <div key={i} className="mb-3">
            <PctBar label={t.trigger} pct={t.pct} color={ACCENT3} />
            <div className="small text-muted">{t.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Monitoring Protocol (14 Parameters)" borderColor={ACCENT4}>
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
        text="ℹ LEV = first-line AED (all ZSD forms — ZS, NALD, IRD). DHA Level B (NALD/IRD). ACTH Level A for infantile spasms (preferred over VGB due to ZSD retinopathy). Phytol-restricted diet Level B (IRD). Cholic acid/UDCA Level C (cholestasis). No ERT (type I integral membrane protein) / No HSCT (not inflammatory) / Lorenzo's Oil INEFFECTIVE (peroxisomal membrane absent — PEX3 LOF prevents all PMP insertion)."
        variant="info"
      />
      <SectionCard title="Treatments (AEDs + Disease-Modifying + Supportive)" borderColor={ACCENT4}>
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

      <SectionCard title="13-Step Diagnostic Algorithm" borderColor={ACCENT}>
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

      <SectionCard title="Differential Diagnosis (vs PEX1/PEX6/PEX26/RING-triad/ABCD1/PHYH/RCDP/Mito)" borderColor={ACCENT3}>
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
export default function PEX3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/pex3/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/pex3/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/pex3/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          PEX3 Epilepsy — Zellweger Spectrum Disorder (ZSD)
        </h2>
        <p className="text-muted small mb-2">
          PEX3 (6q24.2) · Type I integral peroxisomal membrane protein — DOCKING RECEPTOR for PEX19 (373 aa) ·
          PEX3 C-terminal cytoplasmic domain binds PEX19 (cytosolic PMP chaperone) ·
          PEX3 LOF → PEX19 cannot dock → ALL PMPs fail to insert → ghost peroxisomes → ALL matrix import fails ·
          MOST UPSTREAM ZSD defect — PEX3 establishes the membrane before PEX26, PEX1/PEX6, PEX2/PEX10/PEX12 ·
          AR biallelic LOF · p.Gln85* (null, truncating) · p.Leu61Pro (partial membrane insertion, NALD/IRD-enabling) ·
          ~1–2% of all PBD-ZSD · ~10–20 cases worldwide 2026 ·
          ZS 35% · NALD 45% · IRD 15% · Atypical 5% ·
          Biochemically identical to PEX1/PEX6/PEX2/PEX10/PEX12/PEX26-ZSD — only NGS panel distinguishes ·
          Plasmalogens (RBC) LOW · DHA low → migration defects ·
          VPA HIGH RISK · VGB HIGH RISK · LEV first-line · DHA Level B · ACTH Level A IS ·
          HSCT NOT indicated · Lorenzo's Oil INEFFECTIVE · No ERT · 40 patients ·
          COMPLETES the PEX3-PEX16-PEX19 early membrane biogenesis axis
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
