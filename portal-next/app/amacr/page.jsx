'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4a148c';   // deep purple — AMACR / racemase / prerequisite step
const ACCENT2 = '#b71c1c';   // dark red — epilepsy prominent / RELATIVE CI / pristanic elevated
const ACCENT3 = '#e65100';   // deep orange — THCA elevated / caution / bile acid / retinopathy
const ACCENT4 = '#1565c0';   // deep blue — safe treatments / LEV / CAN USE / normal markers
const ACCENT5 = '#1b5e20';   // dark green — NORMAL markers (VLCFA normal / phytanic normal)
const ACCENT6 = '#37474f';   // dark blue-grey — azoospermia / adult-onset / phytol diet

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
    : level?.includes('RELATIVE CI') ? ACCENT3
    : level?.includes('CAUTION') ? ACCENT3
    : level?.includes('CAN USE') || level?.includes('FIRST-LINE') || level?.includes('Second-line') || level?.includes('Adjunct') || level?.includes('Level C')
    ? ACCENT4
    : ACCENT2;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('(')[0]?.trim().split(' ').slice(0, 4).join(' ')} color={color} />
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
        text="🚨 AMACR vs SCP2 — BIOCHEMICALLY NEAR-IDENTICAL: Both have pristanic ELEVATED + THCA/DHCA ELEVATED + VLCFA NORMAL. GENE SEQUENCING (AMACR vs SCP2) is the ONLY way to distinguish. KEY CLINICAL DIFFERENCE: AMACR epilepsy ~60% (FOCAL TEMPORAL — more prominent); SCP2 epilepsy ~40% (movement disorder ~90% — more prominent). Retinopathy ~45% in AMACR (vs ~20% in SCP2) → VGB carries HIGHER visual risk in AMACR."
        variant="danger"
      />
      <Alert
        text="🚨 VLCFA NORMAL IN AMACR — unlike HSD17B4 (VLCFA significantly elevated, steps 2+3 block), ACOX1 (VLCFA elevated, step 1 block), ZSD (all pathways elevated). AMACR blocks the PREREQUISITE racemization step before ACOX2 can act. Phytanic acid NORMAL (PHYH intact) — excludes Adult Refsum Disease. Plasmalogens NORMAL — excludes ZSD/RCDP."
        variant="danger"
      />
      <Alert
        text="⚠ VGB — RELATIVE CI (HIGH VISUAL RISK in AMACR): Pigmentary retinopathy ~45% in AMACR patients. VGB causes irreversible peripheral VF constriction. Additive visual loss risk is HIGHER than SCP2 (retinopathy ~20%) or ACOX1. ERG + VF baseline MANDATORY before VGB. Avoid if alternatives exist."
        variant="warning"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: Standard hepatotoxicity + THCA/DHCA bile acid accumulation adds hepatic burden. POLG1 MANDATORY CPIC A before prescribing. PHT/CBZ/OXC — CAN USE (no adrenal insufficiency in AMACR — contrast ABCD1 where ABSOLUTE CI). Prostate cancer allele p.Ser113Leu ≠ neurological pathogenic variant."
        variant="warning"
      />
      <Alert
        text={`ℹ AMACR (382 aa, PTS1: AKL, 5p13.2): Prerequisite racemization of (R)-pristanoyl-CoA → (S)-form before branched-chain peroxisomal beta-oxidation (ACOX2 → HSD17B4 → SCP2). Without racemization, entire branched-chain pathway stalls. Pristanic ELEVATED + THCA/DHCA ELEVATED; VLCFA NORMAL; Phytanic NORMAL. Adult-onset focal temporal epilepsy (~60%) + polyneuropathy + pigmentary retinopathy (~45%). ~25-30 cases worldwide 2026. AR biallelic LOF. LEV first-line. Phytol-restricted diet Level C. No ERT. No HSCT.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Seizure %" value={`${d.seizure_pct}%`} color={ACCENT2} />
        <KPI label="Polyneuropathy %" value={`${d.polyneuropathy_pct}%`} color={ACCENT3} />
        <KPI label="Retinopathy %" value={`${d.retinopathy_pct}%`} color={ACCENT3} />
        <KPI label="Tremor/Ataxia %" value={`${d.tremor_ataxia_pct}%`} color={ACCENT3} />
        <KPI label="Azoospermia (males)" value={`${d.azoospermia_pct_males}%`} color={ACCENT6} />
        <KPI label="Drug Resistant %" value={`${d.drug_resistance_pct}%`} color={ACCENT2} />
        <KPI label="Cognitive Decline %" value={`${d.cognitive_decline_pct}%`} color={ACCENT3} />
        <KPI label="VLCFA" value="NORMAL" color={ACCENT5} />
        <KPI label="Phytanic" value="NORMAL" color={ACCENT5} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT} />
        <KPI label="Locus" value={d.locus} color={ACCENT4} />
      </div>

      <SectionCard title="Disease Summary — AMACR (Alpha-methylacyl-CoA Racemase) Deficiency" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Onset:</strong> {d.onset_age}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-1"><strong>Variant Spectrum:</strong> {d.common_variant}</p>
        <p className="small mb-0">{d.disease_mechanism}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile" borderColor={ACCENT4}>
            <PctBar label="Epilepsy — focal temporal (most prominent feature)" pct={d.seizure_pct} color={ACCENT2} />
            <PctBar label="Polyneuropathy (axonal sensorimotor)" pct={d.polyneuropathy_pct} color={ACCENT3} />
            <PctBar label="Pigmentary retinopathy" pct={d.retinopathy_pct} color={ACCENT3} />
            <PctBar label="Cognitive decline" pct={d.cognitive_decline_pct} color={ACCENT3} />
            <PctBar label="Tremor / cerebellar ataxia" pct={d.tremor_ataxia_pct} color={ACCENT3} />
            <PctBar label="Drug-resistant seizures" pct={d.drug_resistance_pct} color={ACCENT2} />
            <PctBar label="Azoospermia (affected males)" pct={d.azoospermia_pct_males} color={ACCENT6} />
            <PctBar label="SNHL" pct={d.snhl_pct} color={ACCENT3} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Biochemical Profile — KEY DISTINCTIONS" borderColor={ACCENT4}>
            <PctBar label="VLCFA C26:0 — NORMAL (KEY: excludes ZSD/HSD17B4/ACOX1)" pct={d.vlcfa_normal_pct} color={ACCENT5} />
            <PctBar label="Phytanic acid — NORMAL (KEY: excludes Refsum/PHYH)" pct={d.phytanic_normal_pct} color={ACCENT5} />
            <PctBar label="Pristanic acid — SEVERELY ELEVATED (racemase block)" pct={d.pristanic_elevated_pct} color={ACCENT2} />
            <PctBar label="THCA/DHCA — ELEVATED (bile acid intermediates)" pct={d.thca_elevated_pct} color={ACCENT3} />
            <PctBar label="Plasmalogens — NORMAL (PTS2 / RCDP pathway intact)" pct={d.plasmalogen_normal_pct} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>Diagnosis note:</strong> {d.nbs_positive_rate}
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key Pharmacological Distinctions" borderColor={ACCENT2}>
        <Alert text="PHT/CBZ/OXC — CAN BE USED (no adrenal mechanism in AMACR). CRITICAL CONTRAST: ABCD1/X-ALD PHT/CBZ = ABSOLUTE CI (enzyme induction → cortisol degradation → adrenal crisis). AMACR has NO adrenal insufficiency — same safe profile as SCP2, HSD17B4, ACOX1." variant="secondary" />
        <Alert text="VGB — RELATIVE CI (HIGHEST visual risk in peroxisomal group): Pigmentary retinopathy ~45% in AMACR. VGB causes irreversible peripheral VF constriction. Additive risk markedly increases total visual loss. ERG + VF MANDATORY at baseline before VGB. Avoid unless no alternatives." variant="danger" />
        <Alert text="VPA — RELATIVE CI: Hepatotoxicity (3 mechanisms) + THCA/DHCA bile acid burden adds hepatic metabolic load. POLG1 MANDATORY CPIC A before prescribing. LEV first-line for focal seizures." variant="warning" />
        <Alert text="PROSTATE CANCER ALLELE WARNING: p.Ser113Leu (rs10794086) is a COMMON allele (~5-15% pop). NOT pathogenic for neurological AMACR deficiency. Leukocyte enzyme assay + plasma biochemistry MANDATORY before genetic attribution." variant="warning" />
        <Alert text="LEV first-line. OXC/CBZ for focal (CAN USE — no adrenal). CLZ adjunct for myoclonus. Phytol-restricted diet Level C. DHA Level C. No ERT (matrix enzyme, not secreted). No HSCT (non-inflammatory). Distinguish AMACR from SCP2 by gene sequencing." variant="info" />
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
    if (cls?.includes('Epilepsy')) return ACCENT2;
    if (cls?.includes('Neuropathy')) return ACCENT4;
    if (cls?.includes('Mixed')) return ACCENT3;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ AMACR SPECTRUM (40 synthetic patients): Epilepsy-Predominant (40%) — focal temporal lobe presenting feature · Neuropathy-Predominant (35%) — axonal sensorimotor polyneuropathy first; seizures secondary · Mixed/Multisystem (25%) — simultaneous epilepsy + neuropathy + retinopathy + cognitive decline. VLCFA NORMAL in ALL classes. Pristanic ELEVATED in ALL. AR biallelic LOF. No founder mutation. ~25-30 cases worldwide 2026."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>AMACR Phenotypic Classes — 3 Classes (40 Patients)</h6>
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

      <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT }}>Individual Patients (40 Synthetic — AMACR-01 to AMACR-40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Class</th><th>Sex</th><th>Onset (yr)</th>
              <th>Seizures</th><th>Sz Type</th><th>Drug Resist.</th>
              <th>Retinopathy</th><th>Neuropathy</th><th>Azoospermia</th>
              <th>Pristanic (µmol/L)</th><th>VLCFA C26 (µmol/L)</th>
            </tr>
          </thead>
          <tbody>
            {patients?.map((p, i) => (
              <tr key={i}>
                <td className="small fw-bold" style={{ color: ACCENT }}>AMACR-{String(p.id).padStart(2,'0')}</td>
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
                  <Badge text={p.has_retinopathy ? 'Yes' : 'No'} color={p.has_retinopathy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  <Badge text={p.has_polyneuropathy ? 'Yes' : 'No'} color={p.has_polyneuropathy ? ACCENT3 : ACCENT5} />
                </td>
                <td className="small">
                  {p.sex === 'Male'
                    ? <Badge text={p.azoospermia ? 'Yes' : 'No'} color={p.azoospermia ? ACCENT6 : ACCENT5} />
                    : <span className="text-muted">N/A</span>}
                </td>
                <td className="small" style={{ color: ACCENT2 }}>{p.pristanic_umol_L?.toFixed(1)}</td>
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
        text="⚠ SEIZURE PROFILE AMACR: Focal temporal (aware) 55% — MOST CHARACTERISTIC; often epigastric aura. Focal-BTCS 40%. Myoclonic 25%. Focal parietal-occipital 15% (correlates with retinopathy). Status epilepticus 5% (metabolic). Drug resistance ~25%. EPILEPSY MORE PROMINENT THAN SCP2 (~60% vs 40%). VPA RELATIVE CI. VGB RELATIVE CI (pigmentary retinopathy — HIGH visual risk)."
        variant="warning"
      />
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types (5 Types)" borderColor={ACCENT2}>
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

      <SectionCard title="Monitoring Parameters (9 Parameters)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead className="table-light">
              <tr><th>Parameter</th><th>Frequency</th><th>Target</th></tr>
            </thead>
            <tbody>
              {monitoring?.map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.parameter}</td>
                  <td className="small">{m.frequency}</td>
                  <td className="small text-muted">{m.target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Disease Lifecycle (6 Stages)" borderColor={ACCENT}>
        {lifecycle?.map((l, i) => (
          <div key={i} className="mb-2 small">
            <div className="d-flex align-items-start">
              <Badge text={`Stage ${i + 1}`} color={ACCENT} />
              <div className="ms-2">
                <strong>{l.stage}</strong> <span className="text-muted">({l.age_range})</span>
                <div className="text-muted">{l.features}</div>
              </div>
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
  return (
    <div>
      <Alert
        text="🚨 VGB — RELATIVE CI (HIGHEST visual risk in peroxisomal series): Pigmentary retinopathy ~45% in AMACR. VGB causes irreversible peripheral VF constriction. Combined additive visual loss risk is higher than in SCP2 (~20% retinopathy) or ACOX1. ERG + VF MANDATORY at baseline. Avoid if alternatives exist."
        variant="danger"
      />
      <Alert
        text="⚠ VPA — RELATIVE CI: Standard hepatotoxicity + THCA/DHCA bile acid accumulation (hepatic burden). POLG1 MANDATORY CPIC A. PHT/CBZ/OXC — CAN USE (no adrenal CI — unlike ABCD1 where ABSOLUTE CI). Lorenzo's Oil NOT APPLICABLE (VLCFA normal, no mechanism for pristanic/racemase block)."
        variant="warning"
      />
      <SectionCard title="Treatment Catalog — AMACR Deficiency (11 Entries)" borderColor={ACCENT}>
        {treatments?.map((t, i) => (
          <CICard key={i} drug={t.drug} level={t.level} reason={t.ci || t.indication} />
        ))}
      </SectionCard>
      <SectionCard title="Treatment Summary" borderColor={ACCENT4}>
        <div className="row small">
          <div className="col-md-6">
            <p className="mb-1"><strong style={{ color: ACCENT4 }}>First-line:</strong> LEV (focal + generalised)</p>
            <p className="mb-1"><strong style={{ color: ACCENT4 }}>Focal (CAN USE):</strong> OXC, CBZ, PHT (no adrenal CI)</p>
            <p className="mb-1"><strong style={{ color: ACCENT4 }}>Myoclonus adjunct:</strong> Clonazepam (Level C)</p>
            <p className="mb-1"><strong style={{ color: ACCENT4 }}>Diet:</strong> Phytol-restricted (Level C) — reduces pristanic load</p>
            <p className="mb-1"><strong style={{ color: ACCENT4 }}>DHA:</strong> Level C — theoretical neuroprotection</p>
          </div>
          <div className="col-md-6">
            <p className="mb-1"><strong style={{ color: ACCENT2 }}>VGB:</strong> RELATIVE CI — pigmentary retinopathy ~45% (HIGHEST visual risk in peroxisomal group)</p>
            <p className="mb-1"><strong style={{ color: ACCENT2 }}>VPA:</strong> RELATIVE CI — hepatotoxicity + bile acid burden; POLG1 mandatory</p>
            <p className="mb-1"><strong style={{ color: '#616161' }}>Lorenzo's Oil:</strong> NOT APPLICABLE — VLCFA normal</p>
            <p className="mb-1"><strong style={{ color: '#616161' }}>ERT:</strong> None available (matrix enzyme)</p>
            <p className="mb-1"><strong style={{ color: '#616161' }}>HSCT:</strong> Not indicated (non-inflammatory)</p>
          </div>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { key_concepts, diagnostic_algorithm, pharmacological_distinctions, differential_diagnoses } = data;
  return (
    <div>
      <SectionCard title="Key Concepts (15 Concepts)" borderColor={ACCENT}>
        {key_concepts?.map((c, i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="12-Step Diagnostic Algorithm" borderColor={ACCENT4}>
        <ol className="mb-0">
          {diagnostic_algorithm?.map((step, i) => (
            <li key={i} className="small mb-1">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Pharmacological Distinctions (13 Drugs)" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover">
            <thead className="table-light">
              <tr><th>Drug</th><th>Status</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {pharmacological_distinctions?.map((d, i) => {
                const color = d.status?.includes('RELATIVE CI') ? ACCENT3
                  : d.status?.includes('NOT APPLICABLE') || d.status?.includes('NO ERT') || d.status?.includes('NO HSCT') ? '#616161'
                  : d.status?.includes('FIRST-LINE') || d.status?.includes('CAN USE') || d.status?.includes('Second-line') || d.status?.includes('Adjunct') || d.status?.includes('Level C') ? ACCENT4
                  : ACCENT;
                return (
                  <tr key={i}>
                    <td className="small fw-bold">{d.drug}</td>
                    <td className="small"><Badge text={d.status} color={color} /></td>
                    <td className="small text-muted">{d.reason}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Differential Diagnoses (7 Conditions)" borderColor={ACCENT3}>
        {differential_diagnoses?.map((dd, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#fafafa', borderLeft: `3px solid ${ACCENT3}` }}>
            <div className="fw-bold small" style={{ color: ACCENT3 }}>{dd.disease}</div>
            <div className="small"><strong>Key distinction:</strong> {dd.key_distinction}</div>
            <div className="small text-muted"><strong>Shared features:</strong> {dd.shared_features}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function AMACRPage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    const base = `${API}/api/amacr`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); })
      .catch(e => setError(e.message));
  }, []);

  const renderTab = () => {
    if (activeTab === 0) return <OverviewTab data={overview} />;
    if (activeTab === 1) return <PatientsTab data={breakdown} />;
    if (activeTab === 2) return <SeizuresTab data={breakdown} />;
    if (activeTab === 3) return <TreatmentsTab data={breakdown} />;
    if (activeTab === 4) return <DefinitionsTab data={defs} />;
    return null;
  };

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-start mb-3">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
            🧬 AMACR Epilepsy
          </h4>
          <div className="text-muted small">
            Alpha-methylacyl-CoA Racemase Deficiency · Prerequisite Racemization Step ·
            (R)→(S) Pristanoyl-CoA + THCA-CoA · PRISTANIC-SEVERELY-ELEVATED ·
            VLCFA-NORMAL · PHYTANIC-NORMAL · Adult-Onset · Focal-Temporal-Epilepsy-60pct ·
            AMACR-vs-SCP2-Gene-Sequencing-Only · 5p13.2 · ~25-30-Cases-2026 ·
            VGB-RELATIVE-CI-Retinopathy-45pct-HIGHEST-Visual-Risk · VPA-RELATIVE-CI-POLG1-Mandatory ·
            PHT-CBZ-OXC-CAN-USE-No-Adrenal · LEV-FIRST-LINE · Phytol-Restricted-Diet-Level-C ·
            No-ERT · No-HSCT · AR-Biallelic-LOF
          </div>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger">API error: {error}</div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((tab, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active fw-bold' : ''}`}
              style={activeTab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setActiveTab(i)}
            >
              {tab}
            </button>
          </li>
        ))}
      </ul>

      {renderTab()}
    </div>
  );
}
