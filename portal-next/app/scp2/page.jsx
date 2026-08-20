'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — SCP2 / SCPx / peroxisomal thiolase step 4
const ACCENT2 = '#b71c1c';   // dark red — movement disorder / RELATIVE CI / pristanic elevated
const ACCENT3 = '#e65100';   // deep orange — THCA elevated / caution / bile acid
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / CAN USE / normal markers
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (VLCFA normal) / safe
const ACCENT6 = '#4a148c';   // deep purple — azoospermia / adult-onset / phytol diet / experimental

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
  const color = level?.includes('NOT APPLICABLE') || level?.includes('NOT RECOMMENDED')
    ? '#616161'
    : level?.includes('RELATIVE CI') ? ACCENT3
    : level?.includes('CAUTION') ? ACCENT3
    : level?.includes('CAN USE') ? ACCENT4
    : ACCENT2;
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
        text="🚨 KEY DISTINCTION: VLCFA NORMAL IN SCP2 — unlike HSD17B4 (VLCFA significantly elevated), ACOX1 (VLCFA elevated), and ZSD (VLCFA elevated). If VLCFA is NORMAL but pristanic + THCA elevated → think SCP2 or AMACR. SCP2 (SCPx) blocks STEP 4 (thiolytic cleavage) of branched-chain peroxisomal beta-oxidation. Straight-chain VLCFA uses separate thiolase (ACAA1) → VLCFA unaffected."
        variant="danger"
      />
      <Alert
        text="🚨 ADULT-ONSET (20s–50s) — COMPLETELY different from neonatal/infantile peroxisomal disorders (HSD17B4, ACOX1, PEX1/6). Core triad: MOVEMENT DISORDER (dystonia/chorea/myoclonus, 90%) + POLYNEUROPATHY (75%) + AZOOSPERMIA in males (~95%). Seizures in ~40% (less severe than neonatal forms). ~20 cases worldwide 2026."
        variant="danger"
      />
      <Alert
        text="⚠ PHT/CBZ/OXC — CAN BE USED in SCP2 (no adrenal insufficiency). CRITICAL CONTRAST: In ABCD1/X-ALD, PHT/CBZ = ABSOLUTE CI (enzyme inducers → cortisol degradation → adrenal crisis). SCP2 has NO adrenal mechanism — same safe profile as HSD17B4, ACOX1, RCDP series."
        variant="warning"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: 3 hepatotoxicity mechanisms + THCA/DHCA bile acid accumulation increases hepatic burden. POLG1 MANDATORY (CPIC Grade A). Use LEV + clonazepam for myoclonus instead. VGB — RELATIVE CI: irreversible peripheral VF constriction in adult with polyneuropathy/movement disorder (QoL impact)."
        variant="warning"
      />
      <Alert
        text={`ℹ SCP2/SCPx (547 aa, PTS1-SKL, 1p32.3): Step 4 thiolytic cleavage of branched-chain peroxisomal beta-oxidation. Pristanic ELEVATED + THCA/DHCA ELEVATED; VLCFA NORMAL (straight-chain ACAA1 intact). Adult-onset movement disorder + polyneuropathy + seizures (40%) + azoospermia (males ~95%). No ERT. No HSCT. LEV first-line. Clonazepam for myoclonus. Phytol-restricted diet Level C. ~20 cases worldwide 2026. AR biallelic LOF.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Movement Disorder %" value={`${d.movement_disorder_pct}%`} color={ACCENT2} />
        <KPI label="Polyneuropathy %" value={`${d.polyneuropathy_pct}%`} color={ACCENT3} />
        <KPI label="Azoospermia (males)" value={`${d.azoospermia_pct_males}%`} color={ACCENT6} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Leukoencephalopathy" value={`${d.leuko_pct}%`} color={ACCENT3} />
        <KPI label="SNHL %" value={`${d.snhl_pct}%`} color={ACCENT3} />
        <KPI label="VLCFA" value="NORMAL" color={ACCENT5} />
        <KPI label="Plasmalogens" value="NORMAL" color={ACCENT5} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — SCP2 / SCPx (Sterol Carrier Protein X / 3-Oxoacyl-CoA Thiolase Deficiency)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Onset:</strong> {d.onset_age}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Movement disorder (dystonia/chorea/myoclonus)" pct={d.movement_disorder_pct} color={ACCENT2} />
            <PctBar label="Polyneuropathy (axonal, length-dependent)" pct={d.polyneuropathy_pct} color={ACCENT3} />
            <PctBar label="Seizures (myoclonic + focal)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Azoospermia (affected males — most specific marker)" pct={d.azoospermia_pct_males} color={ACCENT6} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Leukoencephalopathy on MRI" pct={d.leuko_pct} color={ACCENT3} />
            <PctBar label="SNHL" pct={d.snhl_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="VLCFA C26:0 — NORMAL (KEY DISTINCTION from HSD17B4/ACOX1/ZSD)" pct={d.vlcfa_normal_pct} color={ACCENT5} />
            <PctBar label="Pristanic acid — SEVERELY ELEVATED (step 4 block)" pct={d.pristanic_elevated_pct} color={ACCENT2} />
            <PctBar label="THCA/DHCA — ELEVATED (bile acid intermediates)" pct={d.thca_elevated_pct} color={ACCENT3} />
            <PctBar label="Plasmalogens — NORMAL (PTS2 intact)" pct={d.plasmalogen_normal_pct} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>Diagnosis note:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="PHT/CBZ/OXC — CAN BE USED (no adrenal mechanism in SCP2). CRITICAL: In ABCD1/X-ALD, enzyme inducers = ABSOLUTE CI (CYP3A4 cortisol degradation → adrenal crisis). SCP2 does NOT carry this risk. Standard hepatic monitoring." variant="secondary" />
        <Alert text="VPA — RELATIVE CI: 3 hepatotoxicity mechanisms (carnitine depletion + peroxisomal beta-ox interference at step 4 block + POLG1/mitochondrial) + bile acid burden (THCA/DHCA accumulation). POLG1 MANDATORY CPIC A. Use LEV + clonazepam for myoclonus." variant="warning" />
        <Alert text="VGB — RELATIVE CI: Irreversible peripheral VF constriction; in adult with polyneuropathy + movement disorder, additional visual impairment = significant QoL impact. Monthly VEP/VF if used. NO absolute CI (different mechanism from neonatal retinal dystrophy in HSD17B4/ACOX1)." variant="warning" />
        <Alert text="VLCFA NORMAL in SCP2 — if VLCFA elevated, reconsider HSD17B4 (steps 2+3 block, VLCFA elevated, neonatal) or ACOX1 (step 1 block, VLCFA elevated, infantile). Lorenzo oil NOT applicable. Phytol-restricted diet Level C." variant="info" />
        <Alert text="LEV first-line. Clonazepam for myoclonus. OXC focal (CAN USE). Phytol-restricted diet Level C. DHA Level C. No ERT. No HSCT. POLG1 MANDATORY before VPA." variant="info" />
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
    if (name?.includes('Classic')) return ACCENT;
    if (name?.includes('Neuropathy')) return ACCENT4;
    if (name?.includes('Severe')) return ACCENT2;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ SCP2 SPECTRUM: Classic adult-onset (movement disorder + polyneuropathy + azoospermia, 60%) · Neuropathy-predominant variant (25%) · Severe early-adult onset with rapid leukoencephalopathy (15%). VLCFA NORMAL in ALL classes — PRIMARY DISTINCTION from HSD17B4/ACOX1. Pristanic ELEVATED in ALL classes. Azoospermia in ~95% of affected males — most specific clinical marker. AR biallelic LOF. No founder mutation."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>SCP2/SCPx Phenotypic Classes — 3 Classes (40 Patients)</h6>
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
                  <Badge text={e.diet_therapy ? 'Phytol-restricted diet: Yes' : 'Diet: Standard'} color={ACCENT6} />
                  <Badge text={e.hsct_eligible ? 'HSCT: Yes' : 'HSCT: No'} color={e.hsct_eligible ? ACCENT4 : '#9e9e9e'} />
                  <Badge text={e.ert_available ? 'ERT: Available' : 'ERT: None'} color={e.ert_available ? ACCENT4 : '#9e9e9e'} />
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — SCP2-01 to SCP2-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Class</th><th>Sex</th><th>Onset (yr)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resistant</th>
              <th>Movement D/O</th><th>Polyneuropathy</th><th>Azoospermia</th>
              <th>Pristanic (µmol/L)</th><th>VLCFA C26 (µmol/L)</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                <td className="small">{p.phenotype}</td>
                <td className="small">{p.sex}</td>
                <td className="small">{p.onset_age_years}</td>
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
                  <Badge text={p.movement_disorder ? 'Yes' : 'No'} color={p.movement_disorder ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.polyneuropathy ? 'Yes' : 'No'} color={p.polyneuropathy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  {p.sex === 'M'
                    ? <Badge text={p.azoospermia ? 'Yes' : 'No'} color={p.azoospermia ? ACCENT6 : ACCENT5} />
                    : <span className="text-muted">N/A</span>}
                </td>
                <td className="small" style={{ color: ACCENT2 }}>{typeof p.pristanic_umol_l === 'number' ? p.pristanic_umol_l.toFixed(2) : p.pristanic_umol_l}</td>
                <td className="small" style={{ color: ACCENT5 }}>{typeof p.vlcfa_c26_umol_l === 'number' ? p.vlcfa_c26_umol_l.toFixed(3) : p.vlcfa_c26_umol_l}</td>
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
        text="⚠ SEIZURE PROFILE SCP2: Myoclonic (42%) — LEV + clonazepam; Focal temporal (35%) — LEV / OXC (CAN USE, no adrenal); Generalised secondary (15%); Myoclonic status (8%). Overall seizure rate 40% (LESS severe than neonatal peroxisomal). Drug resistance ~20%. VPA RELATIVE CI. VGB RELATIVE CI (adult QoL — polyneuropathy + VF constriction)."
        variant="warning"
      />

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types (4 Types)" borderColor={ACCENT2}>
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

      <SectionCard title="Monitoring Parameters (9)" borderColor={ACCENT4}>
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

      <SectionCard title="Disease Lifecycle (5 Stages)" borderColor={ACCENT}>
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
        text="ℹ TREATMENT PRINCIPLES SCP2: LEV first-line (all seizure types). Clonazepam for myoclonus (preferred over VPA which is RELATIVE CI). OXC/CBZ focal seizures — SAFE (no adrenal, contrast ABCD1). Phytol-restricted diet Level C (reduces pristanic substrate). DHA Level C. POLG1 MANDATORY before VPA (bile acid burden adds hepatic risk). No ERT. No HSCT. Lorenzo oil NOT applicable (VLCFA normal — no elongase block)."
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
              {t.ci && t.ci !== 'None' && t.ci !== 'None specific' && t.ci !== 'None specific to SCP2' && (
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

      <SectionCard title="Diet, Lorenzo Oil, Fasting & Reproductive Notes" borderColor={ACCENT6}>
        <Alert text="🥗 PHYTOL-RESTRICTED DIET Level C: Phytol (from chlorophyll, dairy fat, ruminant fats) → phytanic acid (via gut/microbiome) → pristanic acid (via PHYH alpha-oxidation) → enters peroxisomal beta-oxidation. Restricting phytol reduces pristanic substrate load for blocked SCPx step 4. Metabolic dietitian supervision mandatory. Limited evidence (very rare disease). Similar rationale to Refsum disease diet." variant="secondary" />
        <Alert text="❌ LORENZO OIL — NOT APPLICABLE in SCP2: VLCFA is NORMAL in SCP2 (no elongase problem). Lorenzo oil inhibits VLCFA elongase — no effect on SCPx thiolase deficiency or pristanic/THCA accumulation (completely different metabolic step). No rationale for use in SCP2." variant="light" />
        <Alert text="⚠ FASTING — CAUTION (not extreme hazard): Fasting mobilises branched-chain fatty acids → increased pristanic/THCA flux through blocked step 4 → substrate surge. Less acute than PHYH/Refsum (where phytanic surge causes acute neuropathy worsening). Regular meals advisable; IV glucose if acutely ill with anorexia." variant="warning" />
        <Alert text="👨 AZOOSPERMIA: Virtually all affected males (~95%); SCPx thiolase + SCP2 lipid-transfer domain both required for testicular germ cell maturation and spermatogenesis. Fertility counselling mandatory at diagnosis. Sperm banking rarely possible (diagnostic delay common). Testosterone usually intact (unlike HSD17B4 Type III)." variant="secondary" />
        <Alert text="❌ NO ERT (2026) — SCPx is peroxisomal matrix enzyme (PTS1-SKL); no secreted isoform; systemic ERT cannot reach peroxisomal matrix as functional enzyme." variant="light" />
        <Alert text="❌ NO HSCT — SCP2 deficiency is substrate accumulation toxicity (pristanic + THCA). NOT inflammatory demyelination. HSCT targets only inflammatory demyelinators (Krabbe/GALC, ABCD1-CCALD). HSCT not indicated for SCP2." variant="light" />
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
export default function SCP2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/scp2/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3) {
      fetch(`${API}/api/scp2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/scp2/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      <div className="mb-4">
        <h2 className="fw-bold mb-1" style={{ color: ACCENT }}>
          SCP2 Epilepsy — SCPx / Sterol Carrier Protein X Deficiency (3-Oxoacyl-CoA Thiolase)
        </h2>
        <p className="text-muted small mb-2">
          SCP2/SCPx (1p32.3) · 547 aa peroxisomal matrix enzyme (PTS1: SKL) · Step 4 (thiolytic cleavage) branched-chain peroxisomal beta-oxidation ·
          PRISTANIC ACID SEVERELY ELEVATED · THCA/DHCA ELEVATED · VLCFA NORMAL (KEY DISTINCTION from HSD17B4/ACOX1/ZSD) ·
          Adult-onset (20s–50s) — COMPLETELY different from neonatal/infantile peroxisomal disorders ·
          Core triad: movement disorder 90% + polyneuropathy 75% + azoospermia males ~95% ·
          Seizures ~40% (myoclonic + focal; less severe than neonatal) · Drug resistance ~20% ·
          AR biallelic LOF · ~20 cases worldwide 2026 · No adrenal insufficiency (PHT/CBZ SAFE — contrast ABCD1 ABSOLUTE CI) ·
          VPA RELATIVE CI (POLG1 MANDATORY + bile acid burden) · VGB RELATIVE CI (adult QoL — VF constriction + polyneuropathy) ·
          LEV first-line · Clonazepam myoclonus · Phytol-restricted diet Level C · DHA Level C · No ERT · No HSCT · 40 patients
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
