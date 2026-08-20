'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4527a0';   // deep indigo — ACOX2 / branched-chain step 1 / peroxisomal matrix
const ACCENT2 = '#b71c1c';   // dark red — absolute CI / ELEVATED markers / liver disease / crisis
const ACCENT3 = '#e65100';   // deep orange — relative CI / caution / hepatic burden
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / CAN USE / NORMAL markers
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (VLCFA normal / plasmalogens normal)
const ACCENT6 = '#37474f';   // dark blue-grey — hepatic / bile acid / azoospermia / SNHL

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

function CICard({ drug, level, reason }) {
  const color = level?.includes('NOT APPLICABLE') || level?.includes('NO ERT') || level?.includes('NO HSCT')
    ? '#616161'
    : level?.includes('ABSOLUTE CI') ? ACCENT2
    : level?.includes('RELATIVE CI') || level?.includes('ELEVATED RISK') ? ACCENT3
    : level?.includes('MANDATORY') ? ACCENT2
    : level?.includes('CAN USE') || level?.includes('FIRST-LINE') || level?.includes('Second-line') || level?.includes('Level')
    ? ACCENT4
    : level?.includes('GOLD STANDARD') ? ACCENT5
    : ACCENT3;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('(')[0]?.trim().split(' ').slice(0, 5).join(' ')} color={color} />
        </div>
        <p className="small text-muted mb-0">{reason}</p>
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
        text="⚠ VPA RELATIVE CI — ELEVATED RISK IN ACOX2 (highest of adult branched-chain peroxisomal diseases): Cholestatic liver disease ~70% baseline + bile acid burden + VPA hepatotoxicity = significant additive hepatic risk. POLG1 MANDATORY CPIC A before any VPA. LEV first-line strongly preferred. Avoid VPA unless LEV/LTG/CLZ have failed and liver function permits."
        variant="danger"
      />
      <Alert
        text="⚠ VGB RELATIVE CI — Retinopathy ~30%. VGB causes irreversible peripheral VF constriction additive to retinal damage. ERG + formal VF MANDATORY at baseline. If retinopathy on ERG: contraindicated. Less severe than PHYH (ABSOLUTE CI RP ~95%), higher than SCP2 (~20%)."
        variant="warning"
      />
      <Alert
        text="ℹ BIOCHEMICAL SIGNATURE — BRANCHED-CHAIN STEP 1 BLOCK: Pristanic acid ELEVATED (step 1 blocked, cannot progress to HSD17B4/SCP2). THCA + DHCA ELEVATED (bile acid CoA esters blocked → cholestatic liver disease). VLCFA NORMAL (ACOX1 pathway intact — KEY: excludes ACOX1/ZSD/HSD17B4). Phytanic NORMAL. Plasmalogens NORMAL. Biochemical profile identical to SCP2/AMACR → GENE SEQUENCING MANDATORY."
        variant="warning"
      />
      <Alert
        text={`ℹ ACOX2 (~672 aa, PTS1-SRL, 3p25.1, OMIM Gene *601641): Step 1 branched-chain peroxisomal beta-oxidation (after AMACR racemization). FAD-dependent oxidase, homotrimer. Deficiency: pristanic ↑ + THCA/DHCA ↑ + VLCFA NORMAL. Cholestatic liver disease ~70% (DOMINANT distinguishing feature vs SCP2/AMACR). Epilepsy ~35% (myoclonic, focal). Adult onset 20s-50s. Under 30 cases worldwide 2026. AR biallelic LOF. LEV first-line. PHT/CBZ CAN USE. No ERT. No HSCT. No Lorenzo's Oil.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Hepatic Cholestasis" value={`${d.hepatic_pct}%`} color={ACCENT2} />
        <KPI label="Polyneuropathy" value={`${d.polyneuropathy_pct}%`} color={ACCENT3} />
        <KPI label="Cerebellar Ataxia" value={`${d.cerebellar_ataxia_pct}%`} color={ACCENT3} />
        <KPI label="Epilepsy" value={`${d.seizure_pct}%`} color={ACCENT3} />
        <KPI label="Retinopathy" value={`${d.retinopathy_pct}%`} color={ACCENT3} />
        <KPI label="Azoospermia (M)" value={`${d.azoospermia_males_pct}%`} color={ACCENT6} />
        <KPI label="SNHL" value={`${d.snhl_pct}%`} color={ACCENT6} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="VLCFA" value="NORMAL" color={ACCENT5} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — ACOX2 (Acyl-CoA Oxidase 2) Deficiency / Branched-Chain Peroxisomal Beta-Oxidation Step 1" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Onset:</strong> {d.onset_age}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Cholestatic liver disease (THCA/DHCA → impaired bile acid synthesis)" pct={d.hepatic_pct} color={ACCENT2} />
            <PctBar label="Peripheral polyneuropathy (axonal or mixed)" pct={d.polyneuropathy_pct} color={ACCENT3} />
            <PctBar label="Cerebellar ataxia (gait instability, dysmetria)" pct={d.cerebellar_ataxia_pct} color={ACCENT3} />
            <PctBar label="Epilepsy (myoclonic + focal; less than AMACR ~60%)" pct={d.seizure_pct} color={ACCENT3} />
            <PctBar label="Pigmentary retinopathy (less than AMACR ~45%)" pct={d.retinopathy_pct} color={ACCENT3} />
            <PctBar label="Cognitive decline / white matter changes" pct={d.cognitive_decline_pct} color={ACCENT6} />
            <PctBar label="Azoospermia (males — bile acid disrupts spermatogenesis)" pct={d.azoospermia_males_pct} color={ACCENT6} />
            <PctBar label="Sensorineural hearing loss (SNHL)" pct={d.snhl_pct} color={ACCENT6} />
            <PctBar label="Drug-resistant seizures (of those with epilepsy)" pct={d.drug_resistance_pct} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="Pristanic acid — SEVERELY ELEVATED (step 1 of branched-chain beta-ox blocked)" pct={100} color={ACCENT2} />
            <PctBar label="THCA + DHCA — ELEVATED (bile acid CoA esters blocked → cholestasis)" pct={100} color={ACCENT2} />
            <PctBar label="VLCFA C26:0 — NORMAL (KEY: excludes ACOX1 / ZSD / HSD17B4)" pct={d.vlcfa_normal_pct} color={ACCENT5} />
            <PctBar label="Phytanic acid — NORMAL (excludes PHYH/Refsum; alpha-ox intact)" pct={d.phytanic_normal_pct} color={ACCENT5} />
            <PctBar label="Plasmalogens — NORMAL (excludes ZSD / ALL RCDP types; PTS2 intact)" pct={d.plasmalogen_normal_pct} color={ACCENT5} />
            <PctBar label="Pipecolic acid — NORMAL (excludes ZSD — PEX1/PEX6)" pct={d.pipecolic_normal_pct} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>Sequencing note:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="🚨 VPA — RELATIVE CI (HIGHEST HEPATIC RISK OF ADULT BRANCHED-CHAIN PEROXISOMAL GROUP): Cholestatic liver disease ~70% + bile acid burden = elevated baseline hepatotoxicity risk. VPA adds to this. POLG1 MANDATORY CPIC A. Avoid as first-line. LEV strongly preferred." variant="danger" />
        <Alert text="⚠ VGB — RELATIVE CI: Retinopathy ~30%. VGB + retinopathy = additive irreversible VF loss. ERG + VF MANDATORY at baseline. Less severe than PHYH (ABSOLUTE CI RP ~95%)." variant="warning" />
        <Alert text="✅ PHT/CBZ — CAN USE: NO adrenal insufficiency in ACOX2 (contrast ABCD1: ABSOLUTE CI). CYP450 induction → monitor LFTs in cholestatic patients. PHT: avoid if severe ataxia present." variant="secondary" />
        <Alert text="⚠ GENE SEQUENCING MANDATORY to distinguish ACOX2 from SCP2 and AMACR: Biochemical profile near-identical (pristanic ↑ + THCA ↑ + VLCFA NORMAL). Order panel: ACOX2 + SCP2 + AMACR simultaneously." variant="warning" />
        <Alert text="🔬 HEPATIC MONITORING: ALT/AST/GGT/ALP every 3 months. UDCA Level C (empirical). Phytol-restricted diet Level C. Avoid hepatotoxic drugs. LEV first-line. CLZ adjunct (myoclonus). No ERT. No HSCT. No Lorenzo's Oil (VLCFA normal)." variant="info" />
      </SectionCard>

      <SectionCard title="Critical Distinctions" borderColor={ACCENT}>
        <ul className="list-unstyled mb-0">
          {d.critical_distinctions?.map((c, i) => (
            <li key={i} className="mb-1 small">
              <span className="me-1" style={{ color: ACCENT }}>▸</span>{c}
            </li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Patients & Etiology Tab ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { phenotypic_classes, patients } = data;
  const getColor = (cls) => {
    if (cls?.includes('Hepato')) return ACCENT2;
    if (cls?.includes('Neurological-Dominant')) return ACCENT4;
    if (cls?.includes('Mixed')) return ACCENT3;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ ACOX2 SPECTRUM (40 synthetic patients): Hepato-Neurological (50%) — liver + neuropathy dominant · Neurological-Dominant (35%) — ataxia + neuropathy, mild hepatic · Mixed Multisystem (15%) — liver + neurology + retina. Pristanic + THCA ELEVATED in ALL classes. VLCFA NORMAL in ALL. AR biallelic LOF. Under 30 cases worldwide 2026. Adult onset 20s-50s."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>ACOX2 Phenotypic Classes — 3 Classes (40 Patients)</h6>
      {phenotypic_classes?.map((e, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${getColor(e.class)}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-start mb-2">
              <h6 className="fw-bold mb-0" style={{ color: getColor(e.class) }}>{e.class}</h6>
              <span className="badge" style={{ backgroundColor: getColor(e.class) }}>{e.pct}% · n={e.count}</span>
            </div>
            <p className="small mb-1">{e.description}</p>
            <p className="small text-muted mb-0"><strong>Seizure control:</strong> {e.seizure_control}</p>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — ACOX2-01 to ACOX2-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Class</th><th>Sex</th><th>Onset (yr)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resist.</th>
              <th>Hepatic</th><th>Neuropathy</th><th>Ataxia</th>
              <th>Retinopathy</th><th>SNHL</th>
              <th>Pristanic (µmol/L)</th><th>THCA (µmol/L)</th><th>VLCFA C26 (µmol/L)</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>ACOX2-{String(p.id).padStart(2,'0')}</td>
                <td className="small">{p.phenotypic_class}</td>
                <td className="small">{p.sex}</td>
                <td className="small">{p.onset_age}</td>
                <td className="small">
                  <Badge
                    text={p.primary_seizure !== 'None' ? 'Yes' : 'No'}
                    color={p.primary_seizure !== 'None' ? ACCENT2 : ACCENT5}
                  />
                </td>
                <td className="small">{p.primary_seizure !== 'None' ? p.primary_seizure : '—'}</td>
                <td className="small">
                  {p.drug_resistant
                    ? <Badge text="DRE" color={ACCENT2} />
                    : p.primary_seizure !== 'None'
                    ? <Badge text="Controlled" color={ACCENT5} />
                    : <Badge text="Sz-free" color={ACCENT5} />}
                </td>
                <td className="small">
                  <Badge text={p.has_hepatic ? 'Yes' : 'No'} color={p.has_hepatic ? ACCENT2 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_neuropathy ? 'Yes' : 'No'} color={p.has_neuropathy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_ataxia ? 'Yes' : 'No'} color={p.has_ataxia ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_retinopathy ? 'Yes' : 'No'} color={p.has_retinopathy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_snhl ? 'Yes' : 'No'} color={p.has_snhl ? ACCENT6 : ACCENT5} />
                </td>
                <td className="small" style={{ color: ACCENT2 }}>{p.pristanic_umol_L?.toFixed(1)}</td>
                <td className="small" style={{ color: ACCENT2 }}>{p.thca_umol_L?.toFixed(1)}</td>
                <td className="small" style={{ color: ACCENT5 }}>{p.vlcfa_c26_umol_L?.toFixed(2)}</td>
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
        text="⚠ SEIZURE PROFILE ACOX2: Epilepsy ~35% (less prominent than AMACR ~60% or SCP2 ~40%). Myoclonic ~45% of seizure patients (correlates with ataxia). Focal temporal ~35%. Drug resistance ~20%. VPA RELATIVE CI (elevated hepatic risk). VGB RELATIVE CI (retinopathy ~30%). LEV first-line. Hepatic decompensation is a unique seizure trigger in ACOX2."
        variant="warning"
      />
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types (5 Types — in those with epilepsy)" borderColor={ACCENT2}>
            {seizure_types?.map((s, i) => (
              <div key={i} className="mb-3">
                <PctBar label={s.type} pct={s.pct} color={ACCENT2} />
                <div className="small text-muted">{s.notes}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Triggers (7 Triggers)" borderColor={ACCENT3}>
            {triggers?.map((t, i) => (
              <div key={i} className="mb-3">
                <PctBar label={t.trigger} pct={t.pct} color={ACCENT3} />
                <div className="small text-muted">{t.mechanism}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Monitoring Parameters (11 Parameters)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm">
            <thead className="table-light">
              <tr><th>Parameter</th><th>Frequency</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {monitoring?.map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold" style={{ color: ACCENT }}>{m.parameter}</td>
                  <td className="small">{m.frequency}</td>
                  <td className="small text-muted">{m.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle (5 Stages)" borderColor={ACCENT6}>
        {lifecycle?.map((s, i) => (
          <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
            <div className="card-body py-2 px-3">
              <div className="fw-bold small mb-1" style={{ color: ACCENT6 }}>{s.stage}</div>
              <div className="small text-muted">{s.description}</div>
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatments } = data;
  const getColor = (status) => {
    if (!status) return ACCENT;
    if (status.includes('ABSOLUTE CI')) return ACCENT2;
    if (status.includes('ELEVATED RISK')) return ACCENT2;
    if (status.includes('NOT APPLICABLE') || status.includes('NO ERT') || status.includes('NO HSCT')) return '#616161';
    if (status.includes('MANDATORY')) return ACCENT2;
    if (status.includes('RELATIVE CI')) return ACCENT3;
    if (status.includes('FIRST-LINE') || status.includes('Level A') || status.includes('As indicated') || status.includes('case-by-case')) return ACCENT4;
    if (status.includes('Level B') || status.includes('Level C') || status.includes('Second-line') || status.includes('CAN USE')) return ACCENT4;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="🚨 VPA — RELATIVE CI WITH ELEVATED RISK: Cholestatic liver disease ~70% + bile acid THCA/DHCA burden = highest hepatotoxicity risk in adult branched-chain peroxisomal group. POLG1 MANDATORY CPIC A. Avoid first-line. LEV strongly preferred."
        variant="danger"
      />
      <Alert
        text="⚠ VGB — RELATIVE CI: Retinopathy ~30%. ERG + formal VF MANDATORY at baseline. VGB + retinopathy = additive irreversible VF loss. If retinopathy present: contraindicated."
        variant="warning"
      />
      <Alert
        text="🔬 HEPATIC MONITORING ESSENTIAL: ALT/AST/GGT/ALP every 3 months (cholestatic liver ~70%). UDCA Level C (empirical hepatoprotection). Phytol-restricted diet Level C. Avoid hepatotoxic drugs. LFTs before and during any AED associated with liver burden."
        variant="warning"
      />
      <Alert
        text="✅ LEV first-line. PHT/CBZ CAN USE (no adrenal). CLZ adjunct (myoclonus/ataxia). DHA Level C. POLG1 MANDATORY before VPA. No ERT. No HSCT. No Lorenzo's Oil (VLCFA normal). Sequencing panel ACOX2+SCP2+AMACR mandatory."
        variant="info"
      />
      <h6 className="fw-bold mb-3 mt-2" style={{ color: ACCENT }}>Treatment Options (14 Options)</h6>
      {treatments?.map((t, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${getColor(t.status)}` }}>
          <div className="card-body py-2 px-3">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <div>
                <span className="fw-bold small">{t.name}</span>
                <span className="ms-2 text-muted small">[{t.type}]</span>
              </div>
              <Badge text={t.status?.split('(')[0]?.trim().split(' ').slice(0, 5).join(' ')} color={getColor(t.status)} />
            </div>
            <p className="small mb-1"><strong>Mechanism:</strong> {t.mechanism}</p>
            <p className="small text-muted mb-0"><strong>Notes:</strong> {t.contra}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { key_concepts, diagnostic_algorithm, pharmacological_distinctions, differential_diagnoses } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (15 Definitions)" borderColor={ACCENT}>
        {key_concepts?.map((c, i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
            {i < key_concepts.length - 1 && <hr className="my-2" />}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Diagnostic Algorithm (13 Steps)" borderColor={ACCENT4}>
        <ol className="mb-0">
          {diagnostic_algorithm?.map((step, i) => (
            <li key={i} className="small mb-1">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Pharmacological Distinctions (14 Options)" borderColor={ACCENT2}>
        {pharmacological_distinctions?.map((p, i) => (
          <CICard key={i} drug={p.drug} level={p.status} reason={p.reason} />
        ))}
      </SectionCard>

      <SectionCard title="Differential Diagnoses (7 Conditions)" borderColor={ACCENT3}>
        {differential_diagnoses?.map((d, i) => (
          <div key={i} className="card mb-2 shadow-sm">
            <div className="card-body py-2 px-3">
              <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{d.disease}</div>
              <div className="small mb-1"><strong>Key distinction:</strong> {d.key_distinction}</div>
              <div className="small text-muted"><strong>Shared features:</strong> {d.shared_features}</div>
            </div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function ACOX2Page() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/acox2/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Patients & Etiology' || tab === 'Seizures & Triggers' || tab === 'Treatments') {
      if (!breakdown) {
        fetch(`${API}/api/acox2/breakdown`)
          .then(r => r.json()).then(setBreakdown)
          .catch(e => setError(e.message));
      }
    }
    if (tab === 'Definitions') {
      if (!definitions) {
        fetch(`${API}/api/acox2/definitions`)
          .then(r => r.json()).then(setDefinitions)
          .catch(e => setError(e.message));
      }
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 ACOX2 Epilepsy — Branched-Chain Peroxisomal Beta-Oxidation Step 1
        </h4>
        <span className="badge" style={{ backgroundColor: ACCENT }}>3p25.1 · AR · *601641</span>
        <span className="badge bg-secondary">Step 1 Branched-Chain PeroxBO</span>
        <span className="badge" style={{ backgroundColor: ACCENT5 }}>VLCFA NORMAL</span>
        <span className="badge" style={{ backgroundColor: ACCENT2 }}>PRISTANIC ↑↑</span>
        <span className="badge" style={{ backgroundColor: ACCENT2 }}>THCA ↑↑</span>
        <span className="badge" style={{ backgroundColor: ACCENT3 }}>Liver ~70%</span>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Etiology' && <PatientsTab data={breakdown} />}
      {tab === 'Seizures & Triggers' && <SeizuresTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
