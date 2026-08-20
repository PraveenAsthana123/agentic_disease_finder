'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#283593';   // deep indigo — GCS/glycine/NMDA
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / VPA CI / classic neonatal
const ACCENT3 = '#e65100';   // deep orange — CAUTION / attenuated / subtherapeutic
const ACCENT4 = '#1565c0';   // deep blue — treatments / sodium benzoate / LEV
const ACCENT5 = '#4a148c';   // deep purple — GCS mechanism / NMDAr / DXM
const ACCENT6 = '#00695c';   // teal — DXM / NMDA antagonists / attenuated phenotype

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
  const color = level?.includes('HIGH RISK') || level?.includes('ABSOLUTE')
    ? ACCENT2
    : level?.includes('CAUTION')
    ? ACCENT3 : '#546e7a';
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('—')[0]?.trim().split(' ').slice(0, 4).join(' ')} color={color} />
        </div>
        <p className="small text-danger mb-1">{reason}</p>
        {alternative && <p className="small text-muted mb-0"><strong>Alternative:</strong> {alternative}</p>}
      </div>
    </div>
  );
}

function TreatmentCard({ drug, level, dose, moa, efficacy, monitoring, nkh_note }) {
  const color = level?.includes('HIGH RISK') ? ACCENT2
    : level?.includes('Level A') ? ACCENT
    : level?.includes('Level B') ? ACCENT4
    : ACCENT6;
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('—')[0]?.trim()} color={color} />
        </div>
        {dose && <p className="small mb-1"><strong>Dose:</strong> {dose}</p>}
        {moa && <p className="small mb-1"><strong>MOA:</strong> {moa}</p>}
        {efficacy && <p className="small mb-1"><strong>Efficacy:</strong> {efficacy}</p>}
        {monitoring && <p className="small mb-1 text-muted"><strong>Monitor:</strong> {monitoring}</p>}
        {nkh_note && <p className="small mb-0" style={{ color: ACCENT5 }}><strong>NKH Note:</strong> {nkh_note}</p>}
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
        text="⚠ VPA — HIGH RISK IN NKH (DISEASE-SPECIFIC): VPA directly inhibits GLDC (residual GCS P-protein activity) AND impairs N-methylglycine pathway → RAISES glycine in plasma + CSF → worsens NMDAr excitotoxicity → paradoxical seizure worsening or encephalopathy. DISTINCT from carnitine/POLG CI (those ALSO apply). AVOID VPA. Use LEV + CLB + DXM + sodium benzoate."
        variant="danger"
      />
      <Alert
        text="⚠ VGB — HIGH RISK for IS in NKH: VGB (GABA-T inhibitor) → GABA↑ → GABA-glycine co-transport interaction → may raise CSF glycine → worsens NMDAr burden. Also: VGB retinal toxicity applies independently. PREFER ACTH (Level A) for IS in NKH. ACTH + DXM + sodium benzoate is the NKH-IS triple combination."
        variant="warning"
      />
      <Alert
        text="🚨 HICCUPS = PATHOGNOMONIC NKH CLUE: Persistent neonatal hiccups + hypotonia + apnea = NKH until proven otherwise. Glycine excess → GlyR activation at phrenic nerve nucleus (C3–C5) → rhythmic diaphragm contractions. STAT: CSF:plasma glycine ratio (simultaneous draw). Normal ratio <0.02; NKH ≥0.08."
        variant="info"
      />
      <Alert
        text={`🔬 GLDC = P-protein of GCS. GLDC LOF (75–80% of NKH) → GCS inoperable → glycine accumulates → DUAL mechanism: (1) GlyR brainstem activation → neonatal hypotonia/apnea/hiccups; (2) NMDAr GluN1 co-agonism → cortical excitotoxicity → burst-suppression → IS → DRE. OMIM *${d.omim_gene}/#${d.omim_disease}. Locus: ${d.locus}.`}
        variant="info"
      />

      <div className="row mb-3">
        <KPI label="Cohort Size" value={d.cohort_size} color={ACCENT} />
        <KPI label="Classic Neonatal %" value={`${d.classic_pct}%`} color={ACCENT2} />
        <KPI label="Burst-Suppression %" value={`${d.burst_suppression_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms %" value={`${d.is_pct}%`} color={ACCENT3} />
        <KPI label="Drug-Resistant %" value={`${d.dre_pct}%`} color={ACCENT2} />
        <KPI label="Attenuated %" value={`${d.attenuated_pct}%`} color={ACCENT6} />
        <KPI label="On Sodium Benzoate %" value={`${d.benzoate_pct}%`} color={ACCENT4} />
        <KPI label="On DXM %" value={`${d.dxm_pct}%`} color={ACCENT5} />
        <KPI label="Avg Plasma Gly (µmol/L)" value={d.avg_plasma_glycine} color={ACCENT3} />
        <KPI label="Avg CSF:Plasma Ratio" value={d.avg_csf_plasma_ratio} color={ACCENT2} />
        <KPI label="OMIM Gene" value={`*${d.omim_gene}`} color={ACCENT5} />
        <KPI label="Locus" value={d.locus} color={ACCENT} />
      </div>

      <SectionCard title="Disease Summary — GLDC / Non-Ketotic Hyperglycinemia (NKH / Glycine Encephalopathy)" borderColor={ACCENT}>
        <p className="small mb-1"><strong>Inheritance:</strong> {d.inheritance}</p>
        <p className="small mb-0">{d.gene}</p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Profile (40-Patient Cohort)" borderColor={ACCENT4}>
            <PctBar label="Epilepsy (overall)" pct={d.epilepsy_pct} color={ACCENT2} />
            <PctBar label="Classic neonatal phenotype" pct={d.classic_pct} color={ACCENT2} />
            <PctBar label="Burst-suppression EEG (neonatal)" pct={d.burst_suppression_pct} color={ACCENT2} />
            <PctBar label="Infantile Spasms / West Syndrome" pct={d.is_pct} color={ACCENT3} />
            <PctBar label="Drug-resistant epilepsy (DRE)" pct={d.dre_pct} color={ACCENT2} />
            <PctBar label="Attenuated phenotype" pct={d.attenuated_pct} color={ACCENT6} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Utilisation" borderColor={ACCENT4}>
            <PctBar label="Sodium benzoate (glycine-lowering)" pct={d.benzoate_pct} color={ACCENT4} />
            <PctBar label="Dextromethorphan (NMDAr antagonist)" pct={d.dxm_pct} color={ACCENT5} />
            <div className="mt-2 small text-muted">
              <strong>Avg plasma glycine:</strong> {d.avg_plasma_glycine} µmol/L (target &lt;500 µmol/L)<br />
              <strong>Avg CSF:plasma ratio:</strong> {d.avg_csf_plasma_ratio} (diagnostic ≥0.08)
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="GCS Pathway — 4-Protein Complex (GLDC Position)" borderColor={ACCENT5}>
        <div className="row small">
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT2 }}>P-protein (GLDC — THIS GENE)</p>
            <ul className="mb-0">
              <li>9p24.1; 1020 aa; PLP-cofactor</li>
              <li>Decarboxylates glycine → CO₂</li>
              <li>Transfers aminomethyl group to H-protein</li>
              <li>LOF → GCS non-functional</li>
              <li><strong>75–80% of NKH</strong></li>
              <li className="text-danger fw-bold">Glycine accumulates → NMDAr excitotoxicity</li>
            </ul>
          </div>
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT3 }}>H-protein (GCSH)</p>
            <ul className="mb-0">
              <li>16q23.2; 125 aa; lipoic acid carrier</li>
              <li>Shuttles aminomethyl group P→T protein</li>
              <li>LOF → ~1% of NKH</li>
              <li>Biochemically identical to GLDC-NKH</li>
              <li>Gene sequencing distinguishes</li>
            </ul>
          </div>
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT4 }}>T-protein (AMT)</p>
            <ul className="mb-0">
              <li>3p21.31; 403 aa; aminomethyltransferase</li>
              <li>Transfers NH₄⁺ to THF → 5,10-methyleneTHF</li>
              <li>LOF → ~15% of NKH</li>
              <li>Same glycine accumulation as GLDC</li>
              <li>5,10-methyleneTHF not produced → folate impact</li>
            </ul>
          </div>
          <div className="col-md-3">
            <p className="fw-bold mb-1" style={{ color: ACCENT6 }}>GCS Net Reaction</p>
            <ul className="mb-0">
              <li>Glycine + THF + NAD⁺</li>
              <li>→ 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH</li>
              <li>1-carbon metabolism (folate cycle input)</li>
              <li>NKH: reaction fails completely</li>
              <li className="fw-bold">Folate + carnitine supplementation needed</li>
            </ul>
          </div>
        </div>
        <div className="mt-2 small">
          <Badge text="HICCUPS PATHOGNOMONIC" color={ACCENT2} /> Phrenic GlyR → rhythmic diaphragm → hiccups.
          <Badge text="Burst-Suppression → Hypsarrhythmia" color={ACCENT2} className="ms-2" /> NMDAr excitotoxic seizure evolution.
          <Badge text="CSF:Plasma Ratio ≥0.08" color={ACCENT5} className="ms-2" /> Simultaneous draw; highly specific diagnostic threshold.
        </div>
      </SectionCard>

      <SectionCard title="NMDAr Co-Agonism Mechanism — Why Glycine Causes Seizures" borderColor={ACCENT5}>
        <div className="row small">
          <div className="col-md-6">
            <p className="fw-bold mb-1">Cortex / Brain: Excitatory (NMDAr Co-Agonism)</p>
            <ul>
              <li>GluN1 subunit of NMDAr: OBLIGATE glycine-binding site (Km ~0.5–5 µM)</li>
              <li>Normal CSF glycine: 3–10 µM → partial GluN1 occupancy</li>
              <li>NKH CSF glycine: 100–2000 µM → GluN1 SATURATED</li>
              <li>Result: NMDAr in maximum activation state when glutamate present</li>
              <li>→ Excitotoxic cascade → burst-suppression → seizures</li>
              <li className="fw-bold text-danger">DXM/felbamate block NMDAr channel → beneficial</li>
            </ul>
          </div>
          <div className="col-md-6">
            <p className="fw-bold mb-1">Brainstem / Spinal Cord: Inhibitory (GlyR Agonism)</p>
            <ul>
              <li>Glycine receptor (GlyR, GLRA1/GLRB): Cl⁻ channel — INHIBITORY</li>
              <li>Predominant in brainstem reticular formation, phrenic nucleus</li>
              <li>Excess glycine → GlyR over-activation → brainstem inhibition</li>
              <li>→ Profound hypotonia + apnea (ventilator need)</li>
              <li>→ Phrenic nucleus: HICCUPS (rhythmic diaphragm)</li>
              <li className="fw-bold text-warning">PB/propofol potentiate GlyR → additive respiratory depression risk</li>
            </ul>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Key Pharmacological Distinctions — GLDC vs Other Metabolic Epilepsies" borderColor={ACCENT2}>
        <Alert text="VPA HIGH RISK — NKH-SPECIFIC MECHANISM: VPA inhibits GLDC directly → raises glycine → worsens NMDAr excitotoxicity. NOT the same CI as carnitine depletion or POLG (those ALSO apply — triple risk)." variant="danger" />
        <Alert text="VGB HIGH RISK for IS: GABA↑ from GABA-T inhibition → glycine-GABA co-transport → CSF glycine may rise. Plus: VGB retinal toxicity. Use ACTH Level A for IS instead." variant="warning" />
        <Alert text="Na-channel blockers (CBZ/PHT/OXC/LTG): RELATIVE CI — worsen myoclonic seizures in NKH. Replace with LEV + CLB + DXM." variant="warning" />
        <Alert text="Ketamine (IV): POTENTIALLY BENEFICIAL — NMDAr antagonist at channel site (same as DXM mechanism). Consider ketamine for refractory SE in NKH instead of IV VPA." variant="info" />
        <Alert text="Phenobarbital CAUTION: potentiates GlyR (brainstem inhibitory) — additive with excess glycine causing respiratory depression in neonatal NKH. Use IV LEV for SE; PB only if LEV fails." variant="warning" />
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

// ── Patients & Phenotype Tab ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { phenotype_class_distribution = [], etiologies = [], per_patient_profiles = [] } = data;
  const getColor = (cls) => {
    if (cls?.includes('Classic')) return ACCENT2;
    if (cls?.includes('Attenuated')) return ACCENT6;
    if (cls?.includes('Transient')) return ACCENT4;
    return ACCENT;
  };
  return (
    <div>
      <Alert
        text="ℹ NKH SPECTRUM: Classic Neonatal (null/null or null/severe-missense) → neonatal crisis, burst-suppression, IS, DRE, severe ID. Attenuated (p.Gly761Arg partial-function allele) → later onset, milder ID, seizures manageable. Transient → glycine normalises by 8 weeks, benign but monitor. GENOTYPE-PHENOTYPE: null alleles = severe; p.Gly761Arg homozygous = mildest. CSF:plasma ratio ≥0.08 is diagnostic for all."
        variant="info"
      />
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>NKH Phenotypic Classes (40 Patients)</h6>
      {phenotype_class_distribution?.map((cls, i) => (
        <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${cls.colour || ACCENT}` }}>
          <div className="card-body">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <h6 className="fw-bold mb-0" style={{ color: cls.colour }}>{cls.class}</h6>
              <Badge text={`${cls.n} patients (${cls.pct}%)`} color={cls.colour} />
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Genotype Distribution (40 Patients)</h6>
      {etiologies?.map((e, i) => (
        <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT}` }}>
          <div className="card-body py-2 px-3">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="small fw-bold">{e.etiology}</span>
              <Badge text={`${e.pct}% (n=${e.n})`} color={ACCENT} />
            </div>
            <div className="row small text-muted">
              <div className="col-sm-6"><strong>CSF:Plasma ratio:</strong> {e.csf_plasma_ratio}</div>
              <div className="col-sm-6"><strong>Outcome:</strong> {e.outcome}</div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-3 mb-2" style={{ color: ACCENT }}>Per-Patient Profiles (top 20 by CSF:Plasma ratio)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead><tr>
            <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (days)</th>
            <th>CSF:Plasma</th><th>Plasma Gly</th><th>IS</th><th>DRE</th>
            <th>Benzoate</th><th>DXM</th>
          </tr></thead>
          <tbody>
            {per_patient_profiles?.slice(0, 20).map((p, i) => (
              <tr key={i}>
                <td>{p.patient_id}</td>
                <td>{p.sex}</td>
                <td><span style={{ color: getColor(p.phenotype_class) }}>{p.phenotype_class?.substring(0, 12)}</span></td>
                <td>{p.onset_age_days}</td>
                <td style={{ color: p.csf_plasma_ratio >= 0.08 ? ACCENT2 : ACCENT6 }}>{p.csf_plasma_ratio?.toFixed(3)}</td>
                <td style={{ color: p.plasma_glycine_umol > 500 ? ACCENT2 : ACCENT6 }}>{Math.round(p.plasma_glycine_umol)}</td>
                <td>{p.has_infantile_spasms ? '✓' : '–'}</td>
                <td>{p.drug_resistant ? '✓' : '–'}</td>
                <td>{p.on_benzoate ? '✓' : '–'}</td>
                <td>{p.on_dxm ? '✓' : '–'}</td>
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
  const { seizure_type_distribution = [], trigger_distribution = [],
          csf_plasma_ratio_histogram = [], plasma_glycine_histogram = [],
          monitoring = [], lifecycle = [] } = data;
  return (
    <div>
      <Alert
        text="⚠ NKH NEONATAL SE: Electrical SE (EEG-confirmed without motor correlation) is VERY COMMON in neonatal NKH due to profound hypotonia masking clinical manifestations. Continuous video-EEG MANDATORY in first weeks. IV LEV 60 mg/kg loading for SE. IV sodium benzoate simultaneously — metabolic + AED dual approach."
        variant="warning"
      />
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Seizure Types — NKH" borderColor={ACCENT2}>
            {seizure_type_distribution?.map((s, i) => (
              <div key={i} className="mb-3">
                <PctBar label={s.type} pct={s.pct} color={ACCENT2} />
                <div className="small text-muted ms-2">
                  <strong>EEG:</strong> {s.eeg}<br />
                  <strong>Clinical:</strong> {s.semiology}<br />
                  <strong>Pearls:</strong> {s.tips}
                </div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Triggers" borderColor={ACCENT3}>
            {trigger_distribution?.map((t, i) => (
              <div key={i} className="mb-2">
                <PctBar label={t.trigger} pct={t.pct} color={ACCENT3} />
                <div className="small text-muted ms-2">{t.mechanism}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="CSF:Plasma Glycine Ratio Distribution" borderColor={ACCENT5}>
            {csf_plasma_ratio_histogram?.map((h, i) => (
              <div key={i} className="mb-1 small">
                <div className="d-flex justify-content-between">
                  <span>{h.range}</span>
                  <span>{h.n} patients ({h.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${h.pct}%`, backgroundColor: ACCENT5 }} />
                </div>
              </div>
            ))}
            <p className="small text-muted mt-2">Diagnostic threshold: ≥0.08. Simultaneous CSF + plasma draw mandatory.</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Plasma Glycine Distribution (on treatment)" borderColor={ACCENT4}>
            {plasma_glycine_histogram?.map((h, i) => (
              <div key={i} className="mb-1 small">
                <div className="d-flex justify-content-between">
                  <span>{h.range}</span>
                  <span>{h.n} patients ({h.pct}%)</span>
                </div>
                <div className="progress" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${h.pct}%`, backgroundColor: ACCENT4 }} />
                </div>
              </div>
            ))}
            <p className="small text-muted mt-2">Target: &lt;500 µmol/L (ideally &lt;300 µmol/L) on sodium benzoate.</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Monitoring Protocol" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Parameter</th><th>Frequency</th><th>Target</th></tr></thead>
            <tbody>
              {monitoring?.map((m, i) => (
                <tr key={i}>
                  <td>{m.parameter}</td>
                  <td>{m.frequency}</td>
                  <td className="text-muted">{m.target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle — NKH Across Developmental Stages" borderColor={ACCENT6}>
        {lifecycle?.map((l, i) => (
          <div key={i} className="mb-2 small">
            <strong style={{ color: ACCENT5 }}>{l.stage}</strong>
            <span className="text-muted ms-2">({l.age})</span>
            <p className="mb-0 mt-1">{l.description}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Treatments Tab ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { treatment_counts = {}, contraindications = [] } = data;
  const allTreatments = data.treatments_raw || [];
  return (
    <div>
      <Alert
        text="⚠ NKH TREATMENT HIERARCHY: (1) Sodium Benzoate (Level A) — glycine-lowering backbone; PLUS carnitine supplement. (2) DXM (Level B) — NMDAr antagonist adjunct. (3) LEV (Level B) — AED, safe, no glycine interaction. (4) ACTH (Level A for IS). AVOID: VPA (inhibits GLDC directly), VGB for IS (raises glycine), CBZ/PHT/OXC (worsen myoclonus)."
        variant="warning"
      />
      <Alert
        text="ℹ CARNITINE CO-SUPPLEMENTATION: Sodium benzoate conjugates glycine → hippuric acid, depleting carnitine. L-carnitine 50–100 mg/kg/day MANDATORY with benzoate therapy. Monitor free carnitine target >20 µmol/L."
        variant="info"
      />
      <SectionCard title="Treatment Counts (40-Patient Cohort)" borderColor={ACCENT4}>
        <div className="row">
          {Object.entries(treatment_counts).map(([drug, n], i) => (
            <div key={i} className="col-6 col-md-4 col-lg-2 mb-2">
              <div className="card text-center shadow-sm">
                <div className="card-body py-2 px-1">
                  <div className="fw-bold fs-5" style={{ color: ACCENT4 }}>{n}</div>
                  <div className="text-muted small">{drug}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Contraindications — NKH-Specific" borderColor={ACCENT2}>
        {contraindications?.map((c, i) => (
          <CICard key={i} drug={c.drug} level={c.level} reason={c.reason} alternative={c.alternative} />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Definitions Tab ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { gene_card = {}, pathway = {}, biomarkers = [], key_concepts = [], thresholds = [],
          treatments = [], references = [], differential_diagnosis = [] } = data;
  return (
    <div>
      <SectionCard title="Gene Card — GLDC (P-protein)" borderColor={ACCENT}>
        <div className="row small">
          {Object.entries(gene_card).map(([k, v], i) => (
            <div key={i} className="col-md-6 mb-1">
              <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong>{' '}
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="GCS Pathway — 4-Protein Complex" borderColor={ACCENT5}>
        <p className="small mb-2"><strong>Net reaction:</strong> {pathway.net_reaction}</p>
        <p className="small mb-2 text-danger"><strong>Clinical consequence:</strong> {pathway.clinical_consequence}</p>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Step</th><th>Enzyme</th><th>Gene</th><th>Cofactor</th><th>Reaction</th><th>Clinical</th></tr></thead>
            <tbody>
              {pathway.steps?.map((s, i) => (
                <tr key={i}>
                  <td>{s.step}</td>
                  <td className="fw-bold">{s.enzyme}</td>
                  <td style={{ color: ACCENT5 }}>{s.gene}</td>
                  <td>{s.cofactor}</td>
                  <td>{s.reaction}</td>
                  <td className="text-muted">{s.clinical}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Diagnostic Biomarkers" borderColor={ACCENT4}>
        {biomarkers?.map((b, i) => (
          <div key={i} className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
            <div className="card-body py-2 px-3">
              <div className="fw-bold small mb-1">{b.marker}</div>
              <div className="row small">
                <div className="col-md-4"><strong>Method:</strong> {b.method}</div>
                <div className="col-md-3"><strong>Normal:</strong> {b.reference_range}</div>
                <div className="col-md-2" style={{ color: ACCENT2 }}><strong>NKH:</strong> {b.nkh_range}</div>
                <div className="col-md-3 text-muted">{b.notes}</div>
              </div>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key Clinical Thresholds" borderColor={ACCENT3}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Parameter</th><th>Value</th><th>Clinical Meaning</th></tr></thead>
            <tbody>
              {thresholds?.map((t, i) => (
                <tr key={i}>
                  <td>{t.parameter}</td>
                  <td className="fw-bold" style={{ color: ACCENT }}>{t.value}</td>
                  <td className="text-muted">{t.clinical}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Treatment Pharmacology (NKH-Specific Annotations)" borderColor={ACCENT4}>
        {treatments?.map((t, i) => (
          <TreatmentCard key={i} {...t} />
        ))}
      </SectionCard>

      <SectionCard title="Key Concepts" borderColor={ACCENT5}>
        {key_concepts?.map((c, i) => (
          <div key={i} className="mb-2 small">
            <span className="fw-bold" style={{ color: ACCENT5 }}>{c.term}:</span>{' '}
            <span className="text-muted">{c.definition}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Differential Diagnosis" borderColor={ACCENT3}>
        {differential_diagnosis?.map((d, i) => (
          <div key={i} className="mb-2 small">
            <span className="fw-bold">{d.condition}:</span>{' '}
            <span className="text-muted">{d.distinction}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="References" borderColor={ACCENT6}>
        <ul className="mb-0">
          {references?.map((r, i) => <li key={i} className="small">{r}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function GLDCPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/gldc/overview`).then(r => r.json()),
      fetch(`${API}/api/gldc/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gldc/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => {
      setError(e.message);
      setLoading(false);
    });
  }, []);

  const renderTab = () => {
    if (loading) return <div className="text-center py-5"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
    if (error) return <div className="alert alert-danger">Error: {error}</div>;
    switch (tab) {
      case 0: return <OverviewTab data={overview} />;
      case 1: return <PatientsTab data={breakdown} />;
      case 2: return <SeizuresTab data={breakdown} />;
      case 3: return <TreatmentsTab data={breakdown} />;
      case 4: return <DefinitionsTab data={definitions} />;
      default: return null;
    }
  };

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <div style={{ width: 10, height: 40, backgroundColor: ACCENT, borderRadius: 4, flexShrink: 0 }} />
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            GLDC Epilepsy — Non-Ketotic Hyperglycinemia (NKH / Glycine Encephalopathy)
          </h4>
          <div className="text-muted small">
            Glycine Decarboxylase Deficiency · P-protein of GCS · 9p24.1 · AR · OMIM *238300/#605899 ·
            CSF:Plasma Glycine Ratio ≥0.08 · Hiccups Pathognomonic · Sodium Benzoate + DXM · VPA HIGH RISK
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3 flex-wrap">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: ACCENT, borderColor: ACCENT, borderBottomColor: '#fff' } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {renderTab()}
    </div>
  );
}
