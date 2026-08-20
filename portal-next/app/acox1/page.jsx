'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — ACOX1 peroxisomal beta-oxidation / rare severe
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / VGB retinal / VLCFA accumulation
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / caution / thresholds
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / first-line / CAN USE
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (plasmalogen/phytanic) / safe
const ACCENT6 = '#4a148c';   // deep purple — experimental / DHA / mechanism notes

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
  const color = level?.includes('HIGH RISK') || level?.includes('HAZARD') || level?.includes('NOT RECOMMENDED')
    ? ACCENT2 : level?.includes('RELATIVE') ? ACCENT3
    : level?.includes('NOT APPLICABLE') ? '#616161' : ACCENT4;
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

// ── Overview Tab ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading overview…</div>;
  const d = data;
  return (
    <div>
      <Alert
        text="🚨 CRITICAL DISTINCTION FROM ZSD (PEX1/PEX6): VLCFA is ELEVATED in BOTH ACOX1 and ZSD — plasma VLCFA panel CANNOT distinguish them alone. KEY TEST: RBC plasmalogens are NORMAL in ACOX1 (peroxisome biogenesis INTACT) vs SEVERELY LOW in ZSD. Always order BOTH VLCFA + plasmalogens simultaneously."
        variant="danger"
      />
      <Alert
        text="🚨 PHT/CBZ/OXC — CAN BE USED in ACOX1 (no adrenal insufficiency). CRITICAL CONTRAST: In ABCD1/X-ALD, enzyme inducers = ABSOLUTE CI (accelerate cortisol degradation → adrenal crisis). ACOX1 does NOT cause adrenal insufficiency requiring this precaution."
        variant="danger"
      />
      <Alert
        text="⚠ VGB — RELATIVE CI: Retinal degeneration in 85% (ERG abnormal) + VGB irreversible peripheral VF constriction = additive visual impairment. NOT absolute CI (unlike ZSD). Prefer ACTH for IS. Monthly VEP/VF mandatory if VGB used."
        variant="warning"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: Hepatotoxicity (3 mechanisms: carnitine depletion, peroxisomal beta-oxidation interference, mitochondrial toxicity). POLG1 MANDATORY (CPIC Grade A). Lorenzo oil NOT recommended (mechanism mismatch — reduces synthesis, cannot restore oxidation). Fasting = HAZARD (IV dextrose mandatory)."
        variant="warning"
      />
      <Alert
        text={`ℹ ACOX1 = Acyl-CoA Oxidase 1 (660 aa PTS1-SRL homotrimer). First rate-limiting step of peroxisomal straight-chain VLCFA beta-oxidation. LOF → C26:0/C24:0 accumulate. Phytanic NORMAL (PHYH intact). Pristanic NORMAL (ACOX2 not ACOX1). Plasmalogens NORMAL. Leukodystrophy + retinal degeneration + SNHL. ~50 cases worldwide 2026. DHA Level C. No ERT. No HSCT (classic form).`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Classic Severe %" value={`${d.classic_severe_pct}%`} color={ACCENT2} />
        <KPI label="Intermediate %" value={`${d.intermediate_pct}%`} color={ACCENT3} />
        <KPI label="Attenuated %" value={`${d.attenuated_pct}%`} color={ACCENT5} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Retinal Degen." value={`${d.retinal_degeneration_pct}%`} color={ACCENT3} />
        <KPI label="SNHL %" value={`${d.snhl_pct}%`} color={ACCENT3} />
        <KPI label="Plasmalogens" value="NORMAL" color={ACCENT5} />
        <KPI label="Phytanic" value="NORMAL" color={ACCENT5} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — ACOX1 / Pseudo-Neonatal Adrenoleukodystrophy (Pseudo-NALD)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Classic Severe (null/null)" pct={d.classic_severe_pct} color={ACCENT2} />
            <PctBar label="Intermediate (null/hypomorphic)" pct={d.intermediate_pct} color={ACCENT3} />
            <PctBar label="Attenuated (hypomorphic/hypomorphic)" pct={d.attenuated_pct} color={ACCENT5} />
            <PctBar label="Retinal degeneration (ERG abnormal)" pct={d.retinal_degeneration_pct} color={ACCENT3} />
            <PctBar label="SNHL (sensorineural hearing loss)" pct={d.snhl_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="VLCFA C26:0 ELEVATED (same as ZSD on panel)" pct={100} color={ACCENT2} />
            <PctBar label="Plasmalogens (RBC) NORMAL — KEY DISTINCTION from ZSD" pct={100} color={ACCENT5} />
            <PctBar label="Phytanic acid NORMAL (PHYH/alpha-ox intact)" pct={100} color={ACCENT5} />
            <PctBar label="Pristanic acid NORMAL (ACOX2 not ACOX1)" pct={100} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>NBS:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="PHT/CBZ/OXC — CAN BE USED (no adrenal mechanism in ACOX1). CRITICAL: In ABCD1/X-ALD, enzyme inducers = ABSOLUTE CI (cortisol degradation → adrenal crisis). ACOX1 does NOT carry this risk." variant="secondary" />
        <Alert text="VGB — RELATIVE CI (ERG abnormal 85% + VGB VF constriction = additive retinal risk). NOT absolute CI as in ZSD. Monthly VEP/VF if used. Prefer ACTH for infantile spasms." variant="warning" />
        <Alert text="VPA — RELATIVE CI (3 mechanisms: carnitine depletion + peroxisomal beta-ox interference + POLG1/mitochondrial). POLG1 MANDATORY CPIC A. LFT q3 months." variant="warning" />
        <Alert text="Lorenzo oil NOT RECOMMENDED in ACOX1 — reduces VLCFA synthesis (elongase inhibition) but cannot restore beta-oxidation. Mechanism mismatch: oxidation failure ≠ import failure (ABCD1). Fasting HAZARD: IV dextrose perioperatively mandatory." variant="danger" />
        <Alert text="LEV first-line all forms. Phenobarbital neonatal. ACTH Level B for IS. OXC/CBZ focal (safe — no adrenal). DHA Level C. No ERT. No HSCT for classic severe form." variant="info" />
      </SectionCard>

      <SectionCard title="Key Concepts" borderColor={ACCENT}>
        <ul className="list-unstyled mb-0">
          {d.key_concepts?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
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
    'Classic Severe': ACCENT2,
    'Intermediate': ACCENT3,
    'Attenuated': ACCENT5,
  };
  const getColor = (name) => {
    if (name?.includes('Classic')) return ACCENT2;
    if (name?.includes('Intermediate')) return ACCENT3;
    if (name?.includes('Attenuated')) return ACCENT5;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ ACOX1 SPECTRUM: Classic Severe (null/null → biallelic null → neonatal onset, seizures 90%, 55%) · Intermediate (null/hypomorphic → infantile onset, seizures 60-70%, 30%) · Attenuated (hypomorphic/hypomorphic → childhood onset, seizures 25-40%, 15%). No founder mutation — all private variants. Plasmalogens NORMAL in ALL forms."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>ACOX1-Pseudo-NALD Phenotypic Classes — 3 Forms (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${getColor(e.name)}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: getColor(e.name) }}>{e.name}</h6>
              <span className="badge" style={{ backgroundColor: getColor(e.name) }}>{e.pct}% · n={e.n}</span>
            </div>
            <div className="row small">
              <div className="col-md-6">
                <p className="mb-1"><strong>Sex:</strong> {e.sex}</p>
                <p className="mb-1"><strong>Onset:</strong> {e.onset_age}</p>
                <p className="mb-1"><strong>Seizure risk:</strong> {e.seizure_risk}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg}</p>
                <p className="mb-1"><strong>MRI:</strong> {e.mri}</p>
              </div>
              <div className="col-md-6">
                <p className="mb-1 text-muted">{e.variant_detail}</p>
                <div className="mt-1">
                  <Badge text={e.dha_supplement ? 'DHA: Yes' : 'DHA: No'} color={ACCENT5} />
                  <Badge text={e.hsct_eligible ? 'HSCT: Yes' : 'HSCT: No'} color={e.hsct_eligible ? ACCENT4 : '#9e9e9e'} />
                  <Badge text={e.ert_available ? 'ERT: Available' : 'ERT: None'} color={e.ert_available ? ACCENT4 : '#9e9e9e'} />
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — ACOX1-01 to ACOX1-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Phenotype</th><th>Sex</th><th>Onset (mo)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resistant</th>
              <th>Retinal Degen.</th><th>SNHL</th><th>VLCFA C26 (µmol/L)</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                <td className="small">{p.phenotype}</td>
                <td className="small">{p.sex}</td>
                <td className="small">{p.onset_age_months}</td>
                <td className="small">
                  <Badge text={p.has_seizures ? 'Yes' : 'No'} color={p.has_seizures ? ACCENT2 : ACCENT5} />
                </td>
                <td className="small">{p.seizure_type || '—'}</td>
                <td className="small">
                  {p.drug_resistant && <Badge text="DRE" color={ACCENT2} />}
                  {!p.drug_resistant && p.has_seizures && <Badge text="Controlled" color={ACCENT5} />}
                  {!p.has_seizures && <Badge text="Sz-free" color={ACCENT5} />}
                </td>
                <td className="small">
                  <Badge text={p.retinal_degeneration ? 'Yes' : 'No'} color={p.retinal_degeneration ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.snhl ? 'Yes' : 'No'} color={p.snhl ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small text-muted">{typeof p.vlcfa_c26_umol_l === 'number' ? p.vlcfa_c26_umol_l.toFixed(2) : p.vlcfa_c26_umol_l}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Seizures & Triggers Tab ───────────────────────────────────────────────────
function SeizuresTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { seizure_types, triggers, monitoring, lifecycle } = data;
  return (
    <div>
      <Alert
        text="⚠ SEIZURE PROFILE ACOX1: Neonatal multifocal clonic 38% (LEV+PB). Infantile spasms 25% (ACTH Level B — avoid VGB: retinal degeneration ERG 85% + VGB VF = additive). Focal 20% (LEV/OXC — PHT/CBZ SAFE, no adrenal). Myoclonic 12%. SE 5%. Drug resistance 35%. VLCFA-mediated membrane toxicity."
        variant="warning"
      />

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types (5 Types)" borderColor={ACCENT2}>
            {seizure_types?.map((s, i) => (
              <div key={i} className="mb-3">
                <PctBar label={s.type} pct={s.pct} color={ACCENT2} />
                <div className="small text-muted">
                  <strong>Preferred Rx:</strong> {s.preferred_tx}
                </div>
                {s.notes && <div className="small text-muted mt-1">{s.notes}</div>}
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Triggers (7)" borderColor={ACCENT3}>
            {triggers?.map((t, i) => (
              <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT3} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Monitoring Parameters (7)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead><tr><th>Parameter</th><th>Threshold / Target</th><th>Frequency</th></tr></thead>
            <tbody>
              {monitoring?.map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.parameter}</td>
                  <td className="small text-muted">{m.threshold}</td>
                  <td className="small">{m.frequency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle (6 Stages)" borderColor={ACCENT}>
        {lifecycle?.map((l, i) => (
          <div key={i} className="mb-2">
            <div className="fw-bold small" style={{ color: ACCENT }}>{l.stage}</div>
            <div className="small text-muted">{l.features}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments, contraindications } = data;
  return (
    <div>
      <Alert
        text="ℹ TREATMENT PRINCIPLES ACOX1: LEV first-line. Phenobarbital neonatal. ACTH Level B for IS (preferred over VGB — avoid adding to retinal degeneration). OXC/CBZ focal seizures — SAFE (no adrenal, contrast ABCD1). DHA Level C. POLG1 MANDATORY before VPA. Lorenzo oil NOT recommended (mechanism mismatch). Fasting HAZARD. No ERT. No HSCT (classic)."
        variant="info"
      />

      <SectionCard title="Approved & Supportive Treatments (6)" borderColor={ACCENT4}>
        {treatments?.map((t, i) => (
          <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-body py-2 px-3">
              <div className="d-flex justify-content-between align-items-start mb-1">
                <span className="fw-bold small">{t.drug}</span>
                <div>
                  <Badge text={t.class} color={ACCENT4} />
                  <Badge text={t.level} color={ACCENT} />
                </div>
              </div>
              <p className="small text-muted mb-1"><strong>Dose:</strong> {t.dose}</p>
              <p className="small text-muted mb-1">{t.notes}</p>
              {t.ci && t.ci !== 'None' && t.ci !== 'None specific' && t.ci !== 'None specific to ACOX1' && (
                <p className="small text-danger mb-0"><strong>CI:</strong> {t.ci}</p>
              )}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications & High-Risk Drugs (5)" borderColor={ACCENT2}>
        {contraindications?.map((c, i) => (
          <CICard key={i} drug={c.drug} level={c.level} reason={c.reason} alternative={c.alternative} />
        ))}
      </SectionCard>

      <SectionCard title="Lorenzo Oil & Fasting Hazard Notes" borderColor={ACCENT6}>
        <Alert text="❌ LORENZO OIL — NOT RECOMMENDED in ACOX1: Lorenzo oil inhibits VLCFA elongase (reduces C24:0/C26:0 synthesis from C22:0) — may partially reduce plasma VLCFA but does NOT restore peroxisomal beta-oxidation capacity. Mechanism mismatch: ACOX1 deficiency is an oxidation failure, not an import failure (ABCD1). No clinical benefit proven in ACOX1." variant="light" />
        <Alert text="⚠ FASTING — HAZARD: Peroxisomal beta-oxidation is integral to fasting fatty acid metabolism. Prolonged fasting → VLCFA surge and metabolic decompensation. IV glucose (dextrose 10%) mandatory perioperatively and during intercurrent illness. Never fast > 4-6 hours in young children." variant="warning" />
        <Alert text="🧪 DHA (Docosahexaenoic acid) supplementation: Reduces secondary DHA deficit from impaired peroxisomal beta-oxidation of precursors. 200 mg/day infants, 500 mg/day children. Safe, no drug interactions. Level C." variant="secondary" />
        <Alert text="❌ NO ERT (2026) — ACOX1 is peroxisomal matrix enzyme (PTS1-SRL); systemic enzyme replacement cannot cross peroxisomal membrane as functional enzyme. No secreted form." variant="light" />
        <Alert text="❌ NO HSCT for Classic Severe — VLCFA-mediated neurodegeneration is not inflammatory. HSCT is effective for inflammatory demyelination (Krabbe GALC, ABCD1-CCALD). Attenuated ACOX1 form (neuroinflammatory component) — HSCT being explored experimentally." variant="light" />
      </SectionCard>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading definitions…</div>;
  return (
    <>
      <SectionCard title="Key Concepts (15)" borderColor={ACCENT}>
        <ul className="list-unstyled mb-0">
          {data.key_concepts?.map((c, i) => (
            <li key={i} className="mb-2 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="12-Step Diagnostic Algorithm" borderColor={ACCENT4}>
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

      <SectionCard title="Differential Diagnosis (7 Conditions)" borderColor={ACCENT3}>
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
export default function ACOX1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/acox1/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/acox1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/acox1/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          ACOX1 Epilepsy — Pseudo-Neonatal Adrenoleukodystrophy (Pseudo-NALD)
        </h2>
        <p className="text-muted small mb-2">
          ACOX1 / Acyl-CoA Oxidase 1 (17q25.1) · 660 aa peroxisomal matrix enzyme (PTS1-SRL homotrimer) ·
          First rate-limiting step VLCFA beta-oxidation · AR biallelic LOF · ~50 cases worldwide 2026 ·
          VLCFA C26:0 ELEVATED (same as ZSD on plasma panel) ·
          Plasmalogens NORMAL — KEY DISTINCTION from ZSD/PEX1/PEX6 ·
          Phytanic NORMAL · Pristanic NORMAL ·
          No adrenal insufficiency (PHT/CBZ SAFE — contrast ABCD1 ABSOLUTE CI) ·
          VGB RELATIVE CI (retinal degeneration ERG 85%) ·
          VPA RELATIVE CI (POLG1 MANDATORY CPIC A) · Lorenzo oil NOT recommended ·
          Fasting HAZARD · LEV first-line · ACTH Level B IS · DHA Level C · No ERT · No HSCT · 40 patients
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
