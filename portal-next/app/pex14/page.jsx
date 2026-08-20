'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1b5e20';   // deep green — ZSD / peroxisomal biogenesis / importomer
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK CI / VPA / VGB / ZS severe
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / fasting / NALD
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / DHA / ACTH / LEV
const ACCENT5 = '#4a148c';   // deep purple — molecular biology / importomer scaffold / PEX5 docking
const ACCENT6 = '#00695c';   // teal — DHA / NBS / PEX14 IF absence clue

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
        text="🚨 VGB (Vigabatrin) — HIGH RISK: ZSD retinopathy (universal ZS/NALD) + VGB irreversible VF constriction = additive irreversible blindness. ACTH (Level A) preferred for infantile spasms. AVOID VGB in all ZSD forms."
        variant="danger"
      />
      <Alert
        text="⚠ Fasting / Anaesthesia / NPO — EXTREME HAZARD: starvation → adipose phytanic acid release → acute neurotoxicity + seizures. IV dextrose MANDATORY during any pre-surgical fast. Pre-op: PT/INR + LFT + platelet count + IV Vitamin K (cholestatic coagulopathy in ZS/NALD)."
        variant="warning"
      />
      <Alert
        text={`ℹ PEX14 = IMPORTOMER CENTRAL SCAFFOLD (376 aa, 1p36.22) — type I integral PMP; N-terminal cytoplasmic domain (aa 1–115) = PRIMARY HIGH-AFFINITY PEX5 docking site. PEX5 N-terminal pentapeptide repeats (WxxxF/WxxxY) bind PEX14. PEX14 LOF → importomer scaffold absent → PEX5 has NO primary docking site → ALL matrix import fails → ALL peroxisomal pathways fail. IF TRIPLET (key to diagnosis): PMP70+ (membrane present) + PEX14− (scaffold absent) + catalase cytoplasmic (import failed). Distinguishes from PEX13 LOF (where PEX14 IS present). ~<1% all PBD-ZSD; ~5–8 cases worldwide 2026 — rarest importomer defect. ZS 40% · NALD 40% · IRD 15% · Atypical 5%.`}
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

      <SectionCard title="Disease Summary — PEX14 / Zellweger Spectrum Disorder (ZSD)" borderColor={ACCENT}>
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
            <PctBar label="NALD (intermediate — null/p.Leu78His)" pct={d.nald_pct} color={ACCENT3} />
            <PctBar label="IRD (attenuated — p.Leu78His/mild)" pct={d.ird_pct} color={ACCENT} />
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

      <SectionCard title="PEX14 Importomer Scaffold — IF Triplet (PMP70 / PEX14 / Catalase) — KEY DIAGNOSTIC DISTINCTION" borderColor={ACCENT5}>
        <div className="row small">
          <div className="col-md-4">
            <p className="fw-bold mb-1" style={{ color: ACCENT6 }}>PEX3-PEX16-PEX19 Axis — Membrane Biogenesis (UPSTREAM)</p>
            <ul className="mb-0">
              <li>PEX16: ER→peroxisome vesicles, recruits PEX3</li>
              <li>PEX3: type I PMP, docking receptor for PEX19</li>
              <li>PEX19: farnesylated chaperone, inserts Class I PMPs (including PEX14)</li>
              <li>LOF → membrane absent (ghost peroxisomes)</li>
              <li>IF: PMP70 ABSENT; PEX14 ABSENT; catalase cytoplasmic</li>
            </ul>
          </div>
          <div className="col-md-4">
            <p className="fw-bold mb-1" style={{ color: ACCENT5 }}>PEX14 — Importomer Central Scaffold (THIS GENE)</p>
            <ul className="mb-0">
              <li>376 aa; N-terminal domain = PRIMARY PEX5 docking site</li>
              <li>PEX13·PEX14·PEX17 = importomer complex</li>
              <li>PEX5 WxxxF pentapeptide repeats bind PEX14 N-domain</li>
              <li>LOF → PEX14 absent from membrane; PEX5 has no primary docking site</li>
              <li><strong>IF TRIPLET: PMP70+ / PEX14− / catalase cytoplasmic</strong></li>
              <li>PEX19 LOF: PMP70− + PEX14− (vs PEX14 LOF: PMP70+ + PEX14−)</li>
            </ul>
          </div>
          <div className="col-md-4">
            <p className="fw-bold mb-1" style={{ color: ACCENT3 }}>PEX13 — Importomer SH3 Brace (adjacent gene)</p>
            <ul className="mb-0">
              <li>403 aa; SH3 domain binds PEX5 (SECONDARY) + PEX14</li>
              <li>PEX14 IS present in PEX13 LOF (key IF distinction)</li>
              <li>PEX13 LOF: PMP70+ / PEX14+ / catalase cytoplasmic</li>
              <li>PEX14 LOF: PMP70+ / PEX14− / catalase cytoplasmic</li>
              <li>Anti-PEX14 IF separates PEX13 from PEX14 definitively</li>
            </ul>
          </div>
        </div>
        <div className="mt-2 small">
          <Badge text="p.Leu78His (N-terminal PEX5-binding domain)" color={ACCENT5} /> Reduces PEX5 pentapeptide-repeat binding → partial scaffold function → NALD/IRD as compound het.
          <Badge text="IF triplet diagnostic" color={ACCENT6} className="ms-2" /> PMP70+(membrane intact) + PEX14−(scaffold absent) + catalase cytoplasmic(import failed) = unambiguous PEX14-ZSD fingerprint. PEX19 LOF: PMP70− + PEX14−. PEX13 LOF: PMP70+ + PEX14+.
        </div>
      </SectionCard>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="VPA — HIGH RISK: THREE independent mechanisms — hepatotoxicity (cholestatic liver ZS/NALD) + peroxisomal BO inhibition (worsens VLCFA) + carnitine depletion. POLG1 MANDATORY (CPIC Grade A)." variant="danger" />
        <Alert text="VGB — HIGH RISK: retinopathy (universal ZS/NALD) + VGB irreversible VF constriction = additive blindness. ACTH Level A for IS. AVOID in all ZSD forms." variant="danger" />
        <Alert text="PHT/CBZ/OXC — RELATIVE CI (CYP3A4 → DHA depletion + hepatic). NOT absolute CI (unlike ABCD1 where adrenal crisis risk is ABSOLUTE). IV LEV replaces in ZSD status epilepticus." variant="warning" />
        <Alert text="DHA Level B (NALD/IRD). ACTH Level A for IS. LEV first-line all ZSD forms. Phytol-restricted diet Level B (IRD). No ERT (type I integral PMP — not a soluble lysosomal enzyme) / No HSCT (not inflammatory) / Lorenzo's Oil INEFFECTIVE (importomer scaffold absent — PEX14 LOF removes primary PEX5 docking site)." variant="info" />
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
        text="ℹ ZSD SPECTRUM (PEX14): null/null → ZS (40%) · null/p.Leu78His → NALD (40%) · p.Leu78His/mild → IRD (15%) · Atypical (5%). ZS fraction 40% — higher than PEX13 (35%) reflecting absence of dominant partial-function allele equivalent. Biochemically identical to all other PBD-ZSD; only NGS distinguishes. IF TRIPLET: PMP70+ / PEX14− / catalase cytoplasmic — unambiguous PEX14-ZSD fingerprint. PEX14 absent (unlike PEX13 LOF where PEX14 IS present; unlike PEX19 LOF where PMP70 also absent)."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>PEX14-ZSD Phenotypic Classes — 4 Classes (40 Patients)</h6>
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
                  {!e.ert_available && <Badge text="No ERT (type I integral PMP)" color="#888" />}
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

      <SectionCard title="Seizure Triggers (Febrile + Phytanic Acid + Fasting EXTREME HAZARD)" borderColor={ACCENT3}>
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
        text="ℹ LEV = first-line AED (all ZSD forms — ZS, NALD, IRD). DHA Level B (NALD/IRD). ACTH Level A for infantile spasms (preferred over VGB due to ZSD retinopathy). Phytol-restricted diet Level B (IRD). Cholic acid/UDCA Level C (cholestasis). No ERT (type I integral PMP — not a soluble lysosomal enzyme) / No HSCT (not inflammatory) / Lorenzo's Oil INEFFECTIVE (importomer scaffold absent — PEX5 has no primary docking site even though peroxisomal membrane is present)."
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
      <SectionCard title="Key Concepts (17)" borderColor={ACCENT5}>
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

      <SectionCard title="Differential Diagnosis (vs PEX13/PEX1/PEX19/PEX3/PEX6/ABCD1/PHYH/RCDP)" borderColor={ACCENT3}>
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
export default function PEX14Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/pex14/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/pex14/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/pex14/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          PEX14 Epilepsy — Zellweger Spectrum Disorder (ZSD)
        </h2>
        <p className="text-muted small mb-2">
          PEX14 (1p36.22) · Importomer Central Scaffold — PEX13·PEX14·PEX17 complex (376 aa) ·
          N-terminal cytoplasmic domain (aa 1–115) = PRIMARY HIGH-AFFINITY PEX5 docking site ·
          PEX5 WxxxF pentapeptide repeats bind PEX14 N-domain (primary contact; PEX13 SH3 = secondary) ·
          PEX14 LOF → importomer scaffold absent → PEX5 has NO primary docking site → ALL matrix import fails ·
          IF TRIPLET: PMP70+(membrane present) + PEX14−(scaffold absent) + catalase cytoplasmic ·
          Distinguishes from PEX13 LOF (PEX14 IS present) and PEX19 LOF (PMP70 also absent) ·
          AR biallelic LOF · p.Leu78His (N-terminal PEX5-binding domain, partial-function, NALD/IRD-enabling) · p.Trp47* (null) ·
          &lt;1% of all PBD-ZSD · ~5–8 cases worldwide 2026 — rarest importomer defect ·
          ZS 40% · NALD 40% · IRD 15% · Atypical 5% ·
          Biochemically identical to PEX1/PEX6/PEX3/PEX16/PEX19/PEX26/PEX2/PEX10/PEX12/PEX13-ZSD — only NGS panel distinguishes ·
          VPA HIGH RISK · VGB HIGH RISK · LEV first-line · DHA Level B · ACTH Level A IS ·
          HSCT NOT indicated · Lorenzo's Oil INEFFECTIVE · No ERT (type I integral PMP) · 40 patients
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
