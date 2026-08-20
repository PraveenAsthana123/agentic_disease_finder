'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a0072';   // deep violet — PHYH / alpha-oxidation / phytanic
const ACCENT2 = '#b71c1c';   // dark red — absolute CI / phytanic ELEVATED / crisis
const ACCENT3 = '#e65100';   // deep orange — relative CI / caution / cardiac / fasting
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / CAN USE / NORMAL markers
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (VLCFA normal / plasmalogens normal)
const ACCENT6 = '#37474f';   // dark blue-grey — cardiac / anosmia / SNHL / lifestyle

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
    : level?.includes('RELATIVE CI') ? ACCENT3
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
        text="🚨 VGB ABSOLUTE CI IN ARD — STRONGEST VGB CI OF ALL PEROXISOMAL EPILEPSY DISORDERS: Retinitis pigmentosa ~95% in PHYH deficiency. VGB causes irreversible peripheral visual field constriction — combined with RP = CATASTROPHIC, irreversible additive visual loss. VGB ABSOLUTELY CONTRAINDICATED in Adult Refsum Disease. NEVER USE VGB in PHYH. ERG + VF MANDATORY at baseline."
        variant="danger"
      />
      <Alert
        text="🚨 FASTING HAZARD: Phytanic acid stored in adipose. ANY catabolic state (illness, surgery, fasting, anaesthesia) triggers adipose lipolysis → phytanic surge → acute crisis (polyneuropathy + cardiac arrhythmia + seizures). IV 10% DEXTROSE MANDATORY during any illness or fasting. Pre-operative phytanic <200 µmol/L required; plasmapheresis if needed."
        variant="danger"
      />
      <Alert
        text="⚠ BIOCHEMICAL PROFILE — ISOLATED PHYTANIC ELEVATION: Phytanic acid SEVERELY ELEVATED (>200 µmol/L; normal <10). VLCFA NORMAL (excludes ZSD/ABCD1/HSD17B4/ACOX1). Plasmalogens NORMAL (excludes ZSD/RCDP). Pristanic NORMAL (excludes AMACR/SCP2 — if pristanic elevated, different disease). THCA NORMAL. ONLY phytanic elevated — isolated alpha-oxidation defect."
        variant="warning"
      />
      <Alert
        text={`ℹ PHYH (338 aa, PTS2, 10p13, OMIM *602026): Phytanoyl-CoA 2-hydroxylase — step 1 of alpha-oxidation. Deficiency = Adult Refsum Disease (ARD, OMIM #266500). Phytanic acid accumulates → integrates into membranes → retinal/myelin/cardiac toxicity. RP ~95%, cerebellar ataxia ~90%, polyneuropathy ~90%, anosmia ~70%, SNHL ~50%, cardiac arrhythmia ~40%, epilepsy ~20% (LESS than AMACR/SCP2). GOLD STANDARD: phytol-restricted diet (Level A). ~200 cases worldwide 2026. AR biallelic LOF. LEV first-line. No ERT. No HSCT.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="RP (Retinitis Pigmentosa)" value={`${d.retinitis_pigmentosa_pct}%`} color={ACCENT2} />
        <KPI label="Cerebellar Ataxia" value={`${d.cerebellar_ataxia_pct}%`} color={ACCENT3} />
        <KPI label="Polyneuropathy" value={`${d.polyneuropathy_pct}%`} color={ACCENT3} />
        <KPI label="Anosmia" value={`${d.anosmia_pct}%`} color={ACCENT6} />
        <KPI label="Cardiac Arrhythmia" value={`${d.cardiac_pct}%`} color={ACCENT2} />
        <KPI label="Epilepsy" value={`${d.seizure_pct}%`} color={ACCENT3} />
        <KPI label="SNHL" value={`${d.snhl_pct}%`} color={ACCENT6} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="VLCFA" value="NORMAL" color={ACCENT5} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — PHYH (Phytanoyl-CoA Hydroxylase) Deficiency / Adult Refsum Disease" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Onset:</strong> {d.onset_age}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Retinitis pigmentosa (RP) — rod-cone dystrophy" pct={d.retinitis_pigmentosa_pct} color={ACCENT2} />
            <PctBar label="Cerebellar ataxia — gait ataxia, dysmetria" pct={d.cerebellar_ataxia_pct} color={ACCENT3} />
            <PctBar label="Peripheral polyneuropathy — hypertrophic demyelinating" pct={d.polyneuropathy_pct} color={ACCENT3} />
            <PctBar label="Anosmia / hyposmia — pathognomonic clue" pct={d.anosmia_pct} color={ACCENT6} />
            <PctBar label="SNHL (sensorineural hearing loss)" pct={d.snhl_pct} color={ACCENT6} />
            <PctBar label="Cardiac conduction defects / arrhythmia" pct={d.cardiac_pct} color={ACCENT2} />
            <PctBar label="Epilepsy (less prominent than AMACR/SCP2)" pct={d.seizure_pct} color={ACCENT3} />
            <PctBar label="Ichthyosis" pct={d.ichthyosis_pct} color={ACCENT6} />
            <PctBar label="Epiphyseal dysplasia" pct={d.epiphyseal_dysplasia_pct} color={ACCENT6} />
            <PctBar label="Drug-resistant seizures (of those with epilepsy)" pct={d.drug_resistance_pct} color={ACCENT2} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="Phytanic acid — SEVERELY ELEVATED (KEY: only elevated metabolite)" pct={100} color={ACCENT2} />
            <PctBar label="VLCFA C26:0 — NORMAL (KEY: excludes ZSD/ABCD1/HSD17B4/ACOX1)" pct={d.vlcfa_normal_pct} color={ACCENT5} />
            <PctBar label="Plasmalogens — NORMAL (KEY: excludes ZSD/ALL RCDP types)" pct={d.plasmalogen_normal_pct} color={ACCENT5} />
            <PctBar label="Pristanic acid — NORMAL (KEY: excludes AMACR/SCP2/HSD17B4)" pct={d.pristanic_normal_pct} color={ACCENT5} />
            <PctBar label="Pipecolic acid — NORMAL (excludes ZSD — PEX1/PEX6)" pct={d.pipecolic_normal_pct} color={ACCENT5} />
            <PctBar label="THCA/DHCA — NORMAL (excludes HSD17B4/SCP2/AMACR)" pct={95} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>Diagnosis note:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="🚫 VGB — ABSOLUTE CI (STRONGEST VGB CI IN ALL PEROXISOMAL EPILEPSY): RP ~95% in ARD. VGB + RP = CATASTROPHIC irreversible additive visual loss. NEVER USE VGB in Adult Refsum Disease. ERG + formal VF testing MANDATORY at baseline before any AED." variant="danger" />
        <Alert text="⚠ VPA — RELATIVE CI: Hepatotoxicity + hepatic phytanic acid processing burden + POLG1 MANDATORY CPIC A. Avoid first-line. LEV preferred." variant="warning" />
        <Alert text="✅ PHT/CBZ — CAN USE: NO adrenal insufficiency in PHYH — distinct from ABCD1 (X-ALD) where PHT/CBZ = ABSOLUTE CI. PHT: avoid if severe ataxia present (cerebellar toxicity). CBZ: monitor sodium." variant="secondary" />
        <Alert text="🥗 PHYTOL-RESTRICTED DIET = GOLD STANDARD (Level A): The ONLY disease-modifying therapy that reduces phytanic acid. Restrict dairy fat, ruminant fat, oily fish, chlorophyll-rich foods. Target phytanic <200 µmol/L. Lifelong. Dietitian essential." variant="success" />
        <Alert text="🚨 FASTING PROTOCOL: IV 10% dextrose MANDATORY during illness, surgery, prolonged fasting (prevents adipose lipolysis → phytanic surge → crisis). Adequate caloric intake always. Plasmapheresis for acute crisis or if phytanic >400 µmol/L." variant="danger" />
        <Alert text="LEV first-line for seizures. CLZ adjunct for myoclonus/ataxia. Cardiac monitoring essential (Holter + echo annually — AV block life-threatening). No ERT. No HSCT. No Lorenzo's Oil (VLCFA normal)." variant="info" />
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
    if (cls?.includes('Classic')) return ACCENT2;
    if (cls?.includes('Attenuated')) return ACCENT4;
    if (cls?.includes('Cardiac')) return ACCENT3;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ PHYH/ARD SPECTRUM (40 synthetic patients): Classic Tetrad (55%) — RP + ataxia + polyneuropathy + elevated CSF protein · Attenuated RP-Dominant (30%) — often misdiagnosed as isolated hereditary RP · Cardiac-Prominent (15%) — arrhythmia/AV block as life-threatening feature. Phytanic acid ELEVATED in ALL classes. VLCFA NORMAL in ALL. AR biallelic LOF. ~200 cases worldwide 2026."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>PHYH Phenotypic Classes — 3 Classes (40 Patients)</h6>
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

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — PHYH-01 to PHYH-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Class</th><th>Sex</th><th>Onset (yr)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resist.</th>
              <th>RP</th><th>Ataxia</th><th>Neuropathy</th>
              <th>Anosmia</th><th>Cardiac</th>
              <th>Phytanic (µmol/L)</th><th>VLCFA C26 (µmol/L)</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>PHYH-{String(p.id).padStart(2,'0')}</td>
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
                  <Badge text={p.has_rp ? 'Yes' : 'No'} color={p.has_rp ? ACCENT2 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_ataxia ? 'Yes' : 'No'} color={p.has_ataxia ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_neuropathy ? 'Yes' : 'No'} color={p.has_neuropathy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_anosmia ? 'Yes' : 'No'} color={p.has_anosmia ? ACCENT6 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_cardiac ? 'Yes' : 'No'} color={p.has_cardiac ? ACCENT2 : ACCENT5} />
                </td>
                <td className="small" style={{ color: ACCENT2 }}>{p.phytanic_umol_L?.toFixed(1)}</td>
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
        text="⚠ SEIZURE PROFILE PHYH/ARD: Epilepsy LESS PROMINENT (~20%) than AMACR (~60%) or SCP2 (~40%) — RP + ataxia + neuropathy dominate. Multifocal myoclonic ~40% (of those with seizures). Focal occipital ~30% (RP-related cortical irritation). Drug resistance ~30% of epilepsy subset. VGB ABSOLUTE CI. FASTING = strongest seizure trigger (phytanic surge). VPA RELATIVE CI. LEV first-line."
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

      <SectionCard title="Monitoring Parameters (10 Parameters)" borderColor={ACCENT4}>
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
    if (status.includes('GOLD STANDARD')) return ACCENT5;
    if (status.includes('MANDATORY')) return ACCENT2;
    if (status.includes('RELATIVE CI')) return ACCENT3;
    if (status.includes('NOT APPLICABLE') || status.includes('NO ERT') || status.includes('NO HSCT')) return '#616161';
    if (status.includes('FIRST-LINE') || status.includes('Level A') || status.includes('As indicated')) return ACCENT4;
    if (status.includes('Level B') || status.includes('Level C') || status.includes('Second-line')) return ACCENT4;
    if (status.includes('CAN USE')) return ACCENT4;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="🚫 VGB ABSOLUTE CI — DO NOT USE IN ARD: Retinitis pigmentosa ~95%. VGB causes irreversible VF constriction — catastrophic additive damage to already-impaired retina. STRONGEST VGB contraindication in all peroxisomal epilepsy. ERG + VF MANDATORY before ANY AED initiation."
        variant="danger"
      />
      <Alert
        text="🥗 PHYTOL-RESTRICTED DIET = GOLD STANDARD (Level A): Primary and most effective treatment. Restrict dairy fat, ruminant fat, oily fish, large chlorophyll portions. Target phytanic <200 µmol/L (ideally <50). 50-80% phytanic reduction with strict compliance. Lifelong. Dietitian mandatory."
        variant="success"
      />
      <Alert
        text="🚨 FASTING HAZARD — MANDATORY PROTOCOL: IV 10% dextrose during ANY illness, surgery, or fasting. Adequate caloric intake at ALL times. Pre-op: measure phytanic; plasmapheresis if >200 µmol/L. Do NOT allow voluntary fasting or caloric restriction."
        variant="danger"
      />
      <Alert
        text="⚠ POLG1 MANDATORY before VPA (CPIC Grade A). Cardiac Holter + echo annually (AV block ~40% — life-threatening). Ophthalmology every 6 months (RP monitoring). Audiogram annually (SNHL ~50%). LEV first-line for seizures. CLZ adjunct for myoclonus/ataxia."
        variant="warning"
      />
      <h6 className="fw-bold mb-3 mt-2" style={{ color: ACCENT }}>Treatment Options (13 Options)</h6>
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
export default function PHYHPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/phyh/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Patients & Etiology' || tab === 'Seizures & Triggers' || tab === 'Treatments') {
      if (!breakdown) {
        fetch(`${API}/api/phyh/breakdown`)
          .then(r => r.json()).then(setBreakdown)
          .catch(e => setError(e.message));
      }
    }
    if (tab === 'Definitions') {
      if (!definitions) {
        fetch(`${API}/api/phyh/definitions`)
          .then(r => r.json()).then(setDefinitions)
          .catch(e => setError(e.message));
      }
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          🧬 PHYH Epilepsy — Adult Refsum Disease
        </h4>
        <span className="ms-3 badge" style={{ backgroundColor: ACCENT }}>10p13 · AR · #266500</span>
        <span className="ms-2 badge bg-secondary">Alpha-Oxidation Step 1</span>
        <span className="ms-2 badge" style={{ backgroundColor: ACCENT5 }}>VLCFA NORMAL</span>
        <span className="ms-2 badge" style={{ backgroundColor: ACCENT2 }}>PHYTANIC ↑↑↑</span>
        <span className="ms-2 badge" style={{ backgroundColor: ACCENT2 }}>VGB ABSOLUTE CI</span>
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
