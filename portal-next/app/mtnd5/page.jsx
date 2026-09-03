'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Biochemistry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — mtDNA CI / maternal inheritance
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#b71c1c';   // red — severe / Leigh
const COLOR4 = '#e65100';   // deep orange — MELAS/stroke-like highlight
const COLOR5 = '#2e7d32';   // green — treatments / normal findings

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, color = COLOR }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const s = data.cohort_statistics || {};
  const feats = data.cohort_summary_features || [];
  const pheno_dist = data.phenotype_distribution || [];
  const onset = data.onset_distribution || {};

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MT-ND5 — Largest mtDNA-Encoded Complex I Subunit / Distal Antiporter Module
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Genome:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Alias:</strong> {data.alias}
        </p>
        <p className="mb-1 small">
          <strong>Inheritance:</strong> {data.inheritance} &nbsp;|&nbsp;
          <strong>Cohort:</strong> {data.cohort_n} patients (seed {data.seed})
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🔵 MT-ND5: LARGEST of 7 mtDNA-encoded CI subunits (603 aa / 16 TM helices) — distal antiporter module (with ND4/ND2).
          <span style={{ color: COLOR4 }}> MATERNAL inheritance + HETEROPLASMY</span> — key DDx from ALL nuclear CI defects (AR biallelic).
          <span style={{ color: COLOR4 }}> Leigh/MELAS overlap (stroke-like {s.stroke_like_pct}%)</span> — isolated CI deficiency ({s.avg_ci_activity_pct}% avg residual).
          WES MISSES MT-ND5 — dedicated mtDNA sequencing required.
          Metformin / VPA / Linezolid / Propofol / IV-tPA (stroke-like): ABSOLUTE CONTRAINDICATIONS.
        </p>
      </div>

      {/* Phenotype distribution */}
      <SectionCard title="📊 Phenotype Spectrum (n=40 cohort)" borderColor={COLOR}>
        <div className="row">
          {pheno_dist.map(p => (
            <div key={p.phenotype} className="col-md-6 mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span className="text-truncate" style={{ maxWidth: '75%' }}>{p.phenotype}</span>
                <span className="text-muted">{p.pct}%</span>
              </div>
              <div className="progress" style={{ height: 10 }}>
                <div className="progress-bar" style={{
                  width: `${p.pct}%`,
                  backgroundColor: p.phenotype.includes('Neonatal') ? COLOR3 :
                                   p.phenotype.includes('Overlap') ? COLOR4 :
                                   p.phenotype.includes('CPEO') ? COLOR5 : COLOR,
                }} />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Avg CI Activity %" value={`${s.avg_ci_activity_pct}%`} color={COLOR3} />
        <KPI label="Leigh MRI" value={`${s.leigh_mri_pct}%`} color={COLOR} />
        <KPI label="Stroke-like Episodes" value={`${s.stroke_like_pct}%`} color={COLOR4} />
        <KPI label="Avg Lactate (mM)" value={s.avg_lactic_acid_mmolL} color={COLOR3} />
        <KPI label="Avg Heteroplasmy (blood)" value={`${s.avg_heteroplasmy_blood_pct}%`} color={COLOR2} />
        <KPI label="Mortality" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>
      <div className="row mb-4">
        <KPI label="Lactic Acidosis" value={`${s.lactic_acidosis_pct}%`} color={COLOR} />
        <KPI label="Hypotonia" value={`${s.hypotonia_pct}%`} color={COLOR} />
        <KPI label="Developmental Delay" value={`${s.developmental_delay_pct}%`} color={COLOR} />
        <KPI label="Seizures" value={`${s.seizures_pct}%`} color={COLOR2} />
        <KPI label="SNHL (hearing loss)" value={`${s.hearing_loss_pct}%`} color={COLOR2} />
        <KPI label="CPEO (ophthalmoplegia)" value={`${s.cpeo_pct}%`} color={COLOR5} />
      </div>

      {/* Feature bars */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Features (% of 40 patients)" borderColor={COLOR}>
            {feats.slice(0, Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={f.feature.includes('stroke') || f.feature.includes('MELAS') ? COLOR4 :
                       f.feature.includes('Ragged') || f.feature.includes('CPEO') ? COLOR5 :
                       f.pct > 80 ? COLOR3 : COLOR} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Clinical Features (continued)" borderColor={COLOR}>
            {feats.slice(Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={f.feature.includes('stroke') || f.feature.includes('MELAS') ? COLOR4 :
                       f.feature.includes('CPEO') || f.feature.includes('Ragged') ? COLOR5 :
                       f.pct > 80 ? COLOR3 : COLOR} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Onset distribution */}
      <SectionCard title="Onset Distribution" borderColor={COLOR2}>
        <div className="row">
          {Object.entries(onset).map(([k, v]) => (
            <div key={k} className="col-6 col-md-3 mb-2 text-center">
              <div className="fw-bold" style={{ color: COLOR }}>{v}%</div>
              <div className="text-muted small">{k.replace(/_pct$/, '').replace(/_/g, ' ')}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Heteroplasmy thresholds */}
      {data.heteroplasmy_thresholds && (
        <SectionCard title="⚡ Heteroplasmy Threshold Paradigm (blood)" borderColor={COLOR4}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0">
              <thead style={{ background: COLOR4, color: '#fff' }}>
                <tr><th>Heteroplasmy (blood)</th><th>Clinical Phenotype</th></tr>
              </thead>
              <tbody>
                {Object.entries(data.heteroplasmy_thresholds).map(([k, v]) => (
                  <tr key={k}>
                    <td className="fw-bold">{k}</td>
                    <td className="small">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Key alerts */}
      <SectionCard title="⚠️ Clinical Alerts & Contraindications" borderColor={COLOR3}>
        {(data.key_clinical_alerts || []).map((a, i) => (
          <div key={i} className="mb-1 small p-2 rounded"
            style={{
              background: a.startsWith('🚫') ? '#ffebee' : a.startsWith('⚠️') ? '#fff8e1' : '#e8f5e9',
              borderLeft: `3px solid ${a.startsWith('🚫') ? COLOR3 : a.startsWith('⚠️') ? COLOR4 : COLOR5}`,
            }}>
            {a}
          </div>
        ))}
      </SectionCard>

      {/* Sample patients */}
      <SectionCard title="Sample Patient Records (first 10)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead style={{ background: LIGHT }}>
              <tr>
                <th>ID</th><th>Phenotype</th><th>Variant</th>
                <th>Heteroplasmy %</th><th>CI %</th><th>Lactate</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td className="small">{p.patient_id}</td>
                  <td className="small">{p.phenotype}</td>
                  <td className="small fw-bold" style={{ color: COLOR }}>{p.variant}</td>
                  <td className="small">{p.heteroplasmy_blood_pct}%</td>
                  <td className="small" style={{ color: p.ci_activity_pct < 10 ? COLOR3 : 'inherit' }}>
                    {p.ci_activity_pct}%
                  </td>
                  <td className="small">{p.lactic_acid_mmolL}</td>
                  <td className="small">{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Biochemistry ──────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.all_variants || [];
  const vdist = data.variant_distribution || [];
  const biochem = data.biochemistry_distribution || {};
  const bnPage = data.bn_page_pattern || {};
  const immuno = data.immunoblot_pattern || {};

  return (
    <div>
      <SectionCard title="🧬 Pathogenic Variants in MT-ND5 (mtDNA rCRS)" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>mtDNA HGVS</th><th>Protein Change</th><th>Domain</th>
                <th>Type</th><th>Phenotype</th><th>Severity</th><th>Penetrance</th>
              </tr>
            </thead>
            <tbody>
              {variants.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold small" style={{ color: COLOR }}>{v.hgvs_mtdna}</td>
                  <td className="small">{v.protein}</td>
                  <td className="small">{v.domain}</td>
                  <td className="small">{v.type}</td>
                  <td className="small">{v.phenotype}</td>
                  <td className="small" style={{
                    color: v.severity === 'Severe' ? COLOR3 :
                           v.severity === 'Moderate–Severe' ? COLOR4 : COLOR5
                  }}>{v.severity}</td>
                  <td className="small">{v.penetrance_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          {variants.map((v, i) => (
            <div key={i} className="small mb-2 p-2 rounded" style={{ background: LIGHT }}>
              <strong style={{ color: COLOR }}>{v.hgvs_mtdna}</strong>: {v.notes}
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Variant Frequency Distribution (cohort)" borderColor={COLOR}>
            {vdist.map(v => (
              <Bar key={v.variant} label={v.variant} value={v.freq_pct} color={COLOR} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="CI Activity Distribution" borderColor={COLOR3}>
            <Bar label="CI <10% (severe)" value={biochem.ci_below_10_pct} color={COLOR3} />
            <Bar label="CI 10–20% (moderate)" value={biochem.ci_10_to_20_pct} color={COLOR4} />
            <Bar label="CI 20–30% (mild-moderate)" value={biochem.ci_20_to_30_pct} color={COLOR2} />
            <Bar label="CI >30% (mild/CPEO)" value={biochem.ci_above_30_pct} color={COLOR5} />
            <hr />
            <Bar label="Lactate >10 mM (severe)" value={biochem.lactic_above_10_pct} color={COLOR3} />
            <Bar label="Lactate 5–10 mM (moderate)" value={biochem.lactic_5_to_10_pct} color={COLOR4} />
            <Bar label="Lactate <5 mM (mild)" value={biochem.lactic_below_5_pct} color={COLOR5} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Heteroplasmy Distribution (blood)" borderColor={COLOR4}>
        <div className="row">
          {[
            ['<60% (subclinical)', 'het_below_60_pct', COLOR5],
            ['60–80% (Leigh/MELAS overlap)', 'het_60_to_80_pct', COLOR4],
            ['80–95% (severe Leigh)', 'het_80_to_95_pct', COLOR3],
            ['>95% (neonatal lethal)', 'het_above_95_pct', COLOR3],
          ].map(([label, key, color]) => (
            <div key={key} className="col-md-6 mb-2">
              <Bar label={label} value={biochem[key] || 0} color={color} />
            </div>
          ))}
        </div>
        <p className="small text-muted mt-2">
          ⚠️ Blood heteroplasmy may UNDERESTIMATE muscle heteroplasmy by 15–25 percentage points.
          Muscle biopsy preferred for diagnostic accuracy. WES does NOT detect mtDNA variants.
        </p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="BN-PAGE Pattern" borderColor={COLOR2}>
            <p className="small fw-bold mb-1">{bnPage.finding}</p>
            <p className="small mb-1">{bnPage.interpretation}</p>
            <p className="small text-muted">{bnPage.ddx_value}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Immunoblot Pattern" borderColor={COLOR2}>
            <table className="table table-sm mb-0">
              <tbody>
                {Object.entries(immuno).map(([subunit, finding]) => (
                  <tr key={subunit}>
                    <td className="small fw-bold" style={{ color: COLOR }}>{subunit}</td>
                    <td className="small">{finding}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>
        </div>
      </div>

      {/* Outcome distribution */}
      <SectionCard title="Outcome Distribution" borderColor={COLOR3}>
        <div className="row">
          {(data.outcome_distribution || []).map(o => (
            <div key={o.outcome} className="col-md-6 mb-2">
              <Bar label={o.outcome} value={o.pct}
                color={o.outcome.includes('Deceased') ? COLOR3 :
                       o.outcome.includes('severe') ? COLOR4 :
                       o.outcome.includes('CPEO') ? COLOR5 : COLOR} />
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Treatment uptake */}
      {data.treatment_uptake && (
        <SectionCard title="Treatment Uptake (cohort)" borderColor={COLOR5}>
          <div className="table-responsive">
            <table className="table table-sm mb-0">
              <tbody>
                {Object.entries(data.treatment_uptake).map(([drug, n]) => (
                  <tr key={drug}>
                    <td className="small">{drug}</td>
                    <td className="small fw-bold" style={{ color: COLOR5 }}>{n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DdxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.ddx_table || [];

  return (
    <div>
      <SectionCard title="🔍 Differential Diagnosis — MT-ND5 vs Key Mimics" borderColor={COLOR3}>
        {ddx.map((d, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR3}` }}>
            <h6 className="fw-bold mb-1" style={{ color: COLOR3 }}>vs {d.condition}</h6>
            <p className="small mb-1"><strong style={{ color: COLOR }}>Key distinction:</strong> {d.key_distinction}</p>
            <p className="small mb-0 text-muted"><strong>Shared features:</strong> {d.shared}</p>
          </div>
        ))}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="🚫 Absolute Contraindications" borderColor={COLOR3}>
            {(data.cohort_statistics ? [] : []).map((c, i) => (
              <div key={i} className="small mb-2">{c}</div>
            ))}
            {[
              { drug: 'Metformin', reason: 'Complex I inhibitor — additive CI failure → fatal lactic acidosis' },
              { drug: 'Valproic acid (VPA)', reason: 'CoA sequestration + POLG inhibition + OXPHOS impairment + hepatotoxicity' },
              { drug: 'Linezolid', reason: 'Inhibits mt-23S rRNA → reduces MT-ND5 synthesis directly' },
              { drug: 'Chloramphenicol', reason: 'mt-ribosome inhibitor — reduces ALL 7 mt-encoded CI subunits' },
              { drug: 'Propofol (PRIS)', reason: 'Directly inhibits CI — compounding MT-ND5 CI deficiency → fatal' },
              { drug: 'Tobacco (all forms)', reason: 'Cyanide is a CI inhibitor — doubles CI inhibition in MT-ND5' },
              { drug: 'IV tPA (stroke-like episodes)', reason: 'Stroke-like episodes are METABOLIC not thrombotic — tPA causes harm' },
            ].map((c, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#ffebee', borderLeft: `3px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>🚫 {c.drug}</div>
                <div className="small">{c.reason}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="✅ Evidence-Based Treatments" borderColor={COLOR5}>
            {[
              { drug: 'Thiamine (B1) IV/oral', level: 'Mandatory empiric', note: '10–20 mg/kg IV in acute decompensation; PDH cofactor + CI-associated' },
              { drug: 'GIR 6–8 mg/kg/min dextrose', level: 'Mandatory', note: 'NEVER fast; continuous dextrose prevents CI crisis; increase during illness' },
              { drug: 'CoQ10 / Ubiquinol', level: 'Level C', note: '10–20 mg/kg/day; CI-to-CIII electron carrier; ubiquinol (reduced form) preferred' },
              { drug: 'Riboflavin (B2)', level: 'Level C', note: 'FAD-dependent CI assembly co-factors; empiric' },
              { drug: 'NaHCO₃ IV (acute)', level: 'Acute rescue', note: 'Target pH >7.2; 1–2 mEq/kg IV; lactic acidosis crisis' },
              { drug: 'LEV (Levetiracetam)', level: 'Preferred AED', note: 'Renal clearance; no mitochondrial toxicity; avoid VPA/PHT/CBZ' },
              { drug: 'Idebenone (investigational)', level: 'Level C', note: 'Short-chain CoQ analog; bypasses CI to CII→CIII→CIV; EU-approved LHON; investigational in MT-ND5' },
              { drug: 'Succinate (CI-bypass)', level: 'Investigational', note: 'CII substrate bypasses CI block; case reports of CI crisis rescue' },
              { drug: 'Sevoflurane (anaesthesia)', level: 'Preferred', note: 'Over Propofol; avoid halothane (CI inhibitor); NIV/BiPAP for respiratory support' },
            ].map((t, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#e8f5e9', borderLeft: `3px solid ${COLOR5}` }}>
                <div className="fw-bold small" style={{ color: COLOR5 }}>✅ {t.drug} <span className="badge ms-1" style={{ background: COLOR5, fontSize: '0.65em' }}>{t.level}</span></div>
                <div className="small">{t.note}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🏥 Diagnostic Workup — MT-ND5 CI Deficiency" borderColor={COLOR2}>
        <ol className="small mb-0">
          {[
            'Plasma lactate + pyruvate + L:P ratio (>20 = mitochondrial aetiology)',
            'Plasma amino acids (elevated alanine — PDH inhibition by high NADH)',
            'Urine organic acids (3-methylglutaconic acid non-specific; elevated in mito disease)',
            'Brain MRI: bilateral symmetric T2 hyperintensity putamen/brainstem (Leigh); cortical changes if MELAS overlap (posterior > anterior)',
            'Dedicated mtDNA sequencing (NOT WES): blood first; MUSCLE biopsy preferred for accurate heteroplasmy (blood may underestimate by 15–25%)',
            'Large deletion screening: long-read sequencing or Southern blot for Pearson/KSS phenotypes',
            'Muscle biopsy: RRF (Gomori trichrome), COX/SDH dual histochemistry (COX-positive RRF in MT-ND5), EM for mitochondrial proliferation',
            'OXPHOS enzyme activities (muscle or fibroblasts): CI severely reduced; CII/CIII/CIV normal → isolated CI deficiency fingerprint',
            'BN-PAGE + 2D-BN: CI severely reduced; P-intermediate (ND5-lacking) sub-complex accumulates — diagnostically characteristic',
            'Immunoblot: ND5 reduced/absent; secondary ND4/NDUFB11 reduction',
            'Maternal family heteroplasmy cascade testing (siblings, mother, maternal aunts)',
          ].map((step, i) => (
            <li key={i} className="mb-1">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="🧠 Acute Leigh Crisis / Metabolic Decompensation Management" borderColor={COLOR3}>
        <div className="row">
          {[
            { step: '1. IV thiamine 10–20 mg/kg IMMEDIATELY', color: COLOR3 },
            { step: '2. GIR 6–8 mg/kg/min (10% dextrose) — NEVER fast', color: COLOR3 },
            { step: '3. NaHCO₃ 1–2 mEq/kg if pH <7.2', color: COLOR4 },
            { step: '4. NIV/BiPAP for respiratory support — avoid Propofol', color: COLOR4 },
            { step: '5. Seizures → LEV IV (NOT VPA); benzodiazepines for acute status', color: COLOR },
            { step: '6. For stroke-like: NO tPA; IV thiamine + hydration + avoid fasting', color: COLOR4 },
            { step: '7. Strict infection control — infection triggers decompensation', color: COLOR },
            { step: '8. Metabolic genetics + neurology consult URGENTLY', color: COLOR5 },
          ].map((s, i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="small p-2 rounded fw-semibold"
                style={{ background: '#fff3e0', borderLeft: `3px solid ${s.color}`, color: s.color }}>
                {s.step}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <SectionCard title="📖 Gene & Disease Identity" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm mb-0">
            <tbody>
              {[
                ['Gene', data.gene],
                ['Full name', data.full_name],
                ['Alias', data.alias],
                ['OMIM Gene', `*${data.omim_gene}`],
                ['Disease', data.disease_name],
                ['Chromosome', data.chromosome],
                ['Protein', `${data.protein_aa} aa, ${data.protein_kDa} kDa, ${data.tm_helices} TM helices`],
                ['Inheritance', data.inheritance],
                ['Cohort', `${data.n_patients} patients, seed ${data.cohort_seed}`],
              ].map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-bold small" style={{ color: COLOR, width: '35%' }}>{k}</td>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔑 Key Concepts" borderColor={COLOR2}>
        {Object.entries(data.key_concepts || {}).map(([k, v]) => (
          <div key={k} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR2 }}>{k}</div>
            <div className="small">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead style={{ background: COLOR3, color: '#fff' }}>
              <tr><th>Drug</th><th>Severity</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {(data.contraindications || []).map((c, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{c.drug}</td>
                  <td className="small" style={{ color: COLOR3 }}>{c.severity}</td>
                  <td className="small">{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="✅ Treatments" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead style={{ background: COLOR5, color: '#fff' }}>
              <tr><th>Drug/Intervention</th><th>Level</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(data.treatments || []).map((t, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{t.drug}</td>
                  <td className="small" style={{ color: COLOR5 }}>{t.level}</td>
                  <td className="small">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔬 Diagnostic Workup Steps" borderColor={COLOR2}>
        <ol className="small mb-0">
          {(data.diagnostic_workup || []).map((step, i) => (
            <li key={i} className="mb-1">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="📚 Key References" borderColor={COLOR}>
        {(data.references || []).map((r, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <div className="small fw-bold" style={{ color: COLOR }}>{r.citation}</div>
            <div className="small">{r.relevance}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────────
export default function MtND5Page() {
  const [tab, setTab] = useState(0);
  const [overviewData, setOverviewData]   = useState(null);
  const [variantData,  setVariantData]    = useState(null);
  const [defsData,     setDefsData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/mtnd5/overview`).then(r => r.json()),
      fetch(`${API}/api/mtnd5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtnd5/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, br, df]) => {
        setOverviewData(ov);
        setVariantData(br);
        setDefsData(df);
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1200 }}>
      {/* Page header */}
      <div className="d-flex align-items-center gap-3 mb-4">
        <div className="rounded-circle d-flex align-items-center justify-content-center fw-bold text-white"
          style={{ width: 56, height: 56, background: COLOR, fontSize: 22 }}>
          5
        </div>
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
            MT-ND5 — Complex I Deficiency / Leigh–MELAS Overlap
          </h4>
          <p className="text-muted mb-0 small">
            Largest mtDNA-Encoded CI Subunit · Distal Antiporter Module · Maternal Inheritance · 40-patient cohort (seed 737)
          </p>
        </div>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}
      {loading && <div className="text-center py-4"><div className="spinner-border" style={{ color: COLOR }} /></div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {!loading && (
        <>
          {tab === 0 && <OverviewTab data={overviewData} />}
          {tab === 1 && <VariantsTab data={variantData} />}
          {tab === 2 && <DdxTab data={variantData} />}
          {tab === 3 && <DefinitionsTab data={defsData} />}
        </>
      )}
    </div>
  );
}
