'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — HSD17B4 / DBP / peroxisomal beta-oxidation steps 2+3
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / pristanic elevated / VGB risk / VLCFA
const ACCENT3 = '#e65100';   // deep orange — RELATIVE CI / THCA elevated / caution
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / CAN USE / first-line
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (plasmalogens) / safe
const ACCENT6 = '#4a148c';   // deep purple — DHA / bile acid / experimental / mechanism notes

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
        text="🚨 KEY DISTINCTION FROM ACOX1: Both HSD17B4 and ACOX1 have VLCFA ELEVATED — plasma VLCFA panel CANNOT distinguish them alone. PRISTANIC ACID SEVERELY ELEVATED in HSD17B4 (NORMAL in ACOX1) — pristanic measurement is MANDATORY when VLCFA is elevated. THCA/DHCA also elevated in HSD17B4 (normal in ACOX1)."
        variant="danger"
      />
      <Alert
        text="🚨 PHT/CBZ/OXC — CAN BE USED in HSD17B4 (no adrenal insufficiency). CRITICAL CONTRAST: In ABCD1/X-ALD, enzyme inducers = ABSOLUTE CI (accelerate cortisol degradation → adrenal crisis). HSD17B4 does NOT cause adrenal insufficiency — same safe profile as ACOX1 and RCDP series."
        variant="danger"
      />
      <Alert
        text="⚠ VGB — HIGH RISK (higher than ACOX1): Retinal dystrophy 75% + OPTIC ATROPHY 60% + VGB irreversible peripheral VF constriction = HIGH additive visual risk. Risk is GREATER than ACOX1 (which has retinal degeneration only, no optic atrophy). Strongly prefer ACTH for IS. Monthly VEP/VF mandatory if VGB used."
        variant="warning"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: Hepatotoxicity (3 mechanisms: carnitine depletion, peroxisomal beta-oxidation interference — WORSENS already deficient HSD17B4 pathway, mitochondrial toxicity). ADDITIONALLY bile acid accumulation (THCA/DHCA) increases hepatic burden beyond ACOX1. POLG1 MANDATORY (CPIC Grade A). Lorenzo oil NOT recommended."
        variant="warning"
      />
      <Alert
        text={`ℹ HSD17B4 = D-bifunctional protein / MFP-2 (736 aa PTS1-SRL). Steps 2+3 peroxisomal beta-oxidation for ALL substrates: VLCFA + pristanic (branched-chain) + THCA/DHCA (bile acid). LOF → ALL accumulate. Plasmalogens NORMAL (contrast ZSD). No adrenal insufficiency (contrast ABCD1). Types I/II neonatal severe (~80%); Type III milder (~20%). SNHL 85%. Retinal dystrophy 75%. Optic atrophy 60%. ~100 cases worldwide 2026. DHA Level C. No ERT. No HSCT. LEV first-line. ACTH Level B IS.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Type I %" value={`${d.type_i_pct}%`} color={ACCENT2} />
        <KPI label="Type II %" value={`${d.type_ii_pct}%`} color={ACCENT3} />
        <KPI label="Type III %" value={`${d.type_iii_pct}%`} color={ACCENT5} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Retinal Dystrophy" value={`${d.retinal_dystrophy_pct}%`} color={ACCENT3} />
        <KPI label="Optic Atrophy" value={`${d.optic_atrophy_pct}%`} color={ACCENT3} />
        <KPI label="SNHL %" value={`${d.snhl_pct}%`} color={ACCENT3} />
        <KPI label="Plasmalogens" value="NORMAL" color={ACCENT5} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — HSD17B4 / D-Bifunctional Protein Deficiency (DBP / MFP-2)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Seizures (overall cohort)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Type I (D1+D2 deficient, most severe)" pct={d.type_i_pct} color={ACCENT2} />
            <PctBar label="Type II (D1 hydratase only, severe)" pct={d.type_ii_pct} color={ACCENT3} />
            <PctBar label="Type III (D2 3-HSD only, milder)" pct={d.type_iii_pct} color={ACCENT5} />
            <PctBar label="Retinal dystrophy" pct={d.retinal_dystrophy_pct} color={ACCENT3} />
            <PctBar label="Optic atrophy (adds to VGB risk)" pct={d.optic_atrophy_pct} color={ACCENT3} />
            <PctBar label="SNHL (highest in peroxisomal series)" pct={d.snhl_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="VLCFA C26:0 ELEVATED (same as ACOX1/ZSD on panel)" pct={100} color={ACCENT2} />
            <PctBar label="PRISTANIC SEVERELY ELEVATED — KEY DISTINCTION from ACOX1" pct={100} color={ACCENT2} />
            <PctBar label="THCA/DHCA ELEVATED (bile acid intermediates)" pct={d.thca_elevated_pct} color={ACCENT3} />
            <PctBar label="Plasmalogens NORMAL — distinct from ZSD/RCDP" pct={d.plasmalogen_normal_pct} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>NBS:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="PHT/CBZ/OXC — CAN BE USED (no adrenal mechanism in HSD17B4). CRITICAL: In ABCD1/X-ALD, enzyme inducers = ABSOLUTE CI (cortisol degradation → adrenal crisis). HSD17B4 does NOT carry this risk." variant="secondary" />
        <Alert text="VGB — HIGH RISK (retinal dystrophy 75% + OPTIC ATROPHY 60% + VGB VF constriction = HIGH additive visual risk). HIGHER than ACOX1 (no optic atrophy in ACOX1). Monthly VEP/VF if used. Strongly prefer ACTH for IS." variant="warning" />
        <Alert text="VPA — RELATIVE CI (3 mechanisms: carnitine depletion + peroxisomal beta-ox interference worsening deficient HSD17B4 pathway + POLG1/mitochondrial). ADDITIONALLY bile acid accumulation (THCA/DHCA) raises hepatic risk above ACOX1. POLG1 MANDATORY CPIC A." variant="warning" />
        <Alert text="Lorenzo oil NOT RECOMMENDED in HSD17B4 — does not restore steps 2+3 of beta-oxidation; pristanic and THCA/DHCA unaffected. Fasting HAZARD (IV dextrose perioperatively mandatory). DHA Level C." variant="danger" />
        <Alert text="LEV first-line all forms. Phenobarbital neonatal. ACTH Level B for IS (strongly preferred over VGB — combined retinal + optic atrophy risk). OXC/CBZ focal (CAN USE). No ERT. No HSCT." variant="info" />
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
  const getColor = (name) => {
    if (name?.includes('Type I')) return ACCENT2;
    if (name?.includes('Type II')) return ACCENT3;
    if (name?.includes('Type III')) return ACCENT5;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ HSD17B4 / DBP SPECTRUM: Type I (D1+D2 both deficient → all substrates affected → neonatal severe, 50%) · Type II (D1 hydratase only → same severity as I since hydratase required for all substrates, 30%) · Type III (D2 3-HSD only → milder peroxisomal disease + gonadal phenotype in males, 20%). No founder mutation — all private variants. Plasmalogens NORMAL in ALL types. Pristanic SEVERELY ELEVATED in ALL types."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>HSD17B4-DBP Phenotypic Classes — 3 Types (40 Patients)</h6>
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

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — HSD17B4-01 to HSD17B4-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Type</th><th>Sex</th><th>Onset (mo)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resistant</th>
              <th>Retinal</th><th>Optic Atrophy</th><th>SNHL</th>
              <th>VLCFA C26 (µmol/L)</th><th>Pristanic (µmol/L)</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                <td className="small">{p.phenotype?.replace('DBP ', '')}</td>
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
                  <Badge text={p.retinal_dystrophy ? 'Yes' : 'No'} color={p.retinal_dystrophy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.optic_atrophy ? 'Yes' : 'No'} color={p.optic_atrophy ? ACCENT2 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.snhl ? 'Yes' : 'No'} color={p.snhl ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small text-muted">{typeof p.vlcfa_c26_umol_l === 'number' ? p.vlcfa_c26_umol_l.toFixed(2) : p.vlcfa_c26_umol_l}</td>
                <td className="small" style={{ color: ACCENT2 }}>{typeof p.pristanic_umol_l === 'number' ? p.pristanic_umol_l.toFixed(2) : p.pristanic_umol_l}</td>
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
        text="⚠ SEIZURE PROFILE HSD17B4: Neonatal multifocal clonic 35% (LEV+PB). Infantile spasms 27% (ACTH Level B — VGB HIGH RISK: retinal dystrophy 75% + optic atrophy 60% = additive; HIGHER risk than ACOX1). Focal 20% (LEV/OXC — PHT/CBZ SAFE, no adrenal). Myoclonic 13%. SE 5%. Drug resistance 38%. Pristanic + VLCFA neurotoxicity."
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

      <SectionCard title="Monitoring Parameters (8)" borderColor={ACCENT4}>
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
        text="ℹ TREATMENT PRINCIPLES HSD17B4: LEV first-line. Phenobarbital neonatal. ACTH Level B for IS (STRONGLY preferred over VGB — combined retinal dystrophy + optic atrophy = HIGHER risk than ACOX1). OXC/CBZ focal seizures — SAFE (no adrenal, contrast ABCD1). DHA Level C. POLG1 MANDATORY before VPA (bile acid burden adds hepatic risk above ACOX1). Lorenzo oil NOT recommended. Fasting HAZARD. No ERT. No HSCT."
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
              {t.ci && t.ci !== 'None' && t.ci !== 'None specific' && t.ci !== 'None specific to HSD17B4' && (
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

      <SectionCard title="Bile Acid, Lorenzo Oil & Fasting Notes" borderColor={ACCENT6}>
        <Alert text="⚠ BILE ACID ACCUMULATION (THCA/DHCA): HSD17B4 required for peroxisomal chain shortening of THCA/DHCA to CDCA/CA. Accumulation causes additional hepatic metabolic stress. Bile acid monitoring (THCA/DHCA) every 6 months. VPA hepatic risk HIGHER than ACOX1 due to combined beta-oxidation interference + bile acid burden." variant="warning" />
        <Alert text="❌ LORENZO OIL — NOT RECOMMENDED in HSD17B4: Lorenzo oil inhibits VLCFA elongase (may reduce C26:0) but does NOT restore HSD17B4 steps 2+3; pristanic acid and THCA/DHCA are UNAFFECTED (different metabolic pathway). No clinical benefit." variant="light" />
        <Alert text="⚠ FASTING — HAZARD: Peroxisomal beta-oxidation (steps 2+3) essential for fasting fatty acid metabolism. Fasting → VLCFA + pristanic surge. Additional bile acid synthesis impairment under metabolic stress. IV glucose (dextrose 10%) mandatory perioperatively. Avoid fasting >4-6 hours in young children." variant="warning" />
        <Alert text="🧪 DHA (Docosahexaenoic acid) supplementation Level C: Retinal photoreceptors depend on DHA — DHA supplementation may partially support retinal function given retinal dystrophy 75%. 200 mg/day infants, 500 mg/day children. Safe, no drug interactions." variant="secondary" />
        <Alert text="❌ NO ERT (2026) — HSD17B4 is peroxisomal matrix enzyme (PTS1-SRL); systemic ERT cannot reach peroxisomal matrix. No secreted isoform available." variant="light" />
        <Alert text="❌ NO HSCT — DBP deficiency = substrate accumulation neurotoxicity (VLCFA + pristanic + THCA). NOT inflammatory demyelination. HSCT effective only for Krabbe (GALC) and ABCD1-CCALD (inflammatory). HSCT not indicated for HSD17B4." variant="light" />
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
export default function HSD17B4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hsd17b4/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/hsd17b4/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/hsd17b4/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          HSD17B4 Epilepsy — D-Bifunctional Protein Deficiency (DBP / MFP-2)
        </h2>
        <p className="text-muted small mb-2">
          HSD17B4 / D-Bifunctional Protein (5q23.1) · 736 aa peroxisomal matrix enzyme (PTS1-SRL) · 3 domains: enoyl-CoA hydratase (D1) + 3-hydroxyacyl-CoA dehydrogenase (D2) + SCP2-like (D3) ·
          Steps 2+3 ALL peroxisomal beta-oxidation substrates (VLCFA + pristanic + THCA/DHCA) ·
          AR biallelic LOF · ~100 cases worldwide 2026 ·
          VLCFA C26:0 ELEVATED · PRISTANIC SEVERELY ELEVATED — KEY DISTINCTION from ACOX1 (pristanic NORMAL in ACOX1) ·
          THCA/DHCA ELEVATED · Plasmalogens NORMAL ·
          No adrenal insufficiency (PHT/CBZ SAFE — contrast ABCD1 ABSOLUTE CI) ·
          VGB HIGH RISK (retinal dystrophy 75% + optic atrophy 60% — higher risk than ACOX1) ·
          VPA RELATIVE CI (POLG1 MANDATORY + bile acid burden) · Types I/II neonatal severe · Type III milder ·
          LEV first-line · ACTH Level B IS · DHA Level C · No ERT · No HSCT · 40 patients
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
