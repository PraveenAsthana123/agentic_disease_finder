'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'HCM vs Leigh vs KSS', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#bf360c';   // deep burnt orange — Complex V / ATP synthase (adjacent to ATP6)
const LIGHT  = '#fbe9e7';
const COLOR2 = '#e64a19';   // deep orange — HCM / cardiac
const COLOR3 = '#b71c1c';   // dark red — Leigh / severe
const COLOR4 = '#1565c0';   // deep blue — HCM / cardiomyopathy
const COLOR5 = '#4a148c';   // deep purple — overlap mutation / combined CI+CV

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
  const mol_feats = data.key_molecular_features || [];
  const alerts = data.clinical_alerts || [];
  const overlap = data.overlap_region || {};

  return (
    <div>
      {/* Banner */}
      <div className="alert fw-bold mb-4" style={{ backgroundColor: LIGHT, borderLeft: `5px solid ${COLOR}`, color: COLOR }}>
        🟤 MT-ATP8: COMPLEX V F0 PERIPHERAL STALK ANCHOR (68 aa, 2 TM) — <strong>HCM dominant phenotype</strong> (annual echo+Holter mandatory).
        Overlap mutation m.8528T&gt;C: dual ATP8+ATP6 disruption → <strong>combined CI+CV deficiency.</strong>
        <span className="text-danger"> NO Retinitis Pigmentosa</span> (distinguishes from MT-ATP6/NARP).
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Avg CV Activity" value={`${s.avg_cv_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg Lactate" value={`${s.avg_lactic_acid_mmolL} mmol/L`} color={COLOR2} />
        <KPI label="HCM" value={`${s.hypertrophic_cardiomyopathy_pct}%`} color={COLOR4} />
        <KPI label="LVOT Obstruction" value={`${s.lvot_obstruction_pct}%`} color={COLOR4} />
        <KPI label="Arrhythmia" value={`${s.arrhythmia_pct}%`} color={COLOR4} />
        <KPI label="Leigh MRI" value={`${s.leigh_mri_pct}%`} color={COLOR3} />
        <KPI label="Exercise Intolerance" value={`${s.exercise_intolerance_pct}%`} color={COLOR2} />
        <KPI label="Avg Heteroplasmy" value={`${s.avg_heteroplasmy_blood_pct}%`} color={COLOR} />
        <KPI label="TM Helices" value={data.tm_helices} color={COLOR} />
        <KPI label="aa Length" value={`${data.aa_length} aa`} color={COLOR} />
        <KPI label="MW (kDa)" value={`${data.molecular_weight_kda} kDa`} color={COLOR} />
      </div>

      {/* Gene Info */}
      <SectionCard title="Gene & Protein" borderColor={COLOR}>
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Gene:</strong> {data.gene} ({data.omim_gene})</div>
          <div className="col-md-6"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-md-6"><strong>Module/Position:</strong> {data.module}</div>
          <div className="col-md-6"><strong>rCRS positions:</strong> {data.rcrs_positions} ({data.strand})</div>
          <div className="col-md-6"><strong>Inheritance:</strong> {data.inheritance}</div>
          <div className="col-md-12"><strong>Primary Disease:</strong> {data.primary_disease}</div>
        </div>
      </SectionCard>

      {/* Overlap region */}
      {overlap.rcrs && (
        <SectionCard title="ATP8/ATP6 Overlap Region — rCRS 8527-8572 (46 bp)" borderColor={COLOR5}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small mb-0">
              <tbody>
                <tr><th style={{ width: 200 }}>Overlap rCRS</th><td>{overlap.rcrs} ({overlap.length_bp} bp)</td></tr>
                <tr><th>Description</th><td>{overlap.description}</td></tr>
                <tr><th>Key Mutation</th><td className="fw-semibold" style={{ color: COLOR5 }}>{overlap.key_mutation}</td></tr>
                <tr><th>Consequence</th><td className="text-danger fw-semibold">{overlap.consequence}</td></tr>
              </tbody>
            </table>
          </div>
          <p className="small text-muted mt-2 mb-0">
            ⚠️ All variants in rCRS 8527-8572 must be reported for BOTH MT-ATP8 AND MT-ATP6 simultaneously.
            Non-overlap ATP8 mutations (rCRS 8366-8526) cause isolated CV deficiency only.
          </p>
        </SectionCard>
      )}

      {/* Phenotype distribution */}
      {pheno_dist.length > 0 && (
        <SectionCard title="Phenotype Distribution" borderColor={COLOR4}>
          {pheno_dist.map((p, i) => (
            <div key={i} className="mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span>{p.phenotype}</span><span className="text-muted">{p.count} pts</span>
              </div>
              <div className="progress" style={{ height: 10 }}>
                <div className="progress-bar" style={{
                  width: `${(p.count / data.n_patients) * 100}%`,
                  backgroundColor: p.phenotype.includes('Leigh') ? COLOR3 :
                                   p.phenotype.includes('HCM') ? COLOR4 :
                                   p.phenotype.includes('Exercise') ? COLOR2 : COLOR
                }} />
              </div>
            </div>
          ))}
        </SectionCard>
      )}

      {/* Clinical feature bars */}
      <SectionCard title="Clinical Features (% of 40-patient cohort)" borderColor={COLOR2}>
        <div className="row">
          <div className="col-md-6">
            {feats.slice(0, Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR4 : f.pct > 30 ? COLOR2 : COLOR3} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR4 : f.pct > 30 ? COLOR2 : COLOR3} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Molecular features */}
      {mol_feats.length > 0 && (
        <SectionCard title="Key Molecular Features — MT-ATP8 Complex V F0 Peripheral Stalk Anchor" borderColor={COLOR}>
          {mol_feats.map((f, i) => (
            <div key={i} className="mb-3 pb-3 border-bottom">
              <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{f.feature}</div>
              <div className="small text-muted">{f.significance}</div>
            </div>
          ))}
        </SectionCard>
      )}

      {/* Clinical alerts */}
      {alerts.length > 0 && (
        <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
          {alerts.map((a, i) => (
            <div key={i} className="alert alert-light border-start border-3 mb-2 py-2 small"
              style={{ borderColor: a.startsWith('🔴') ? COLOR3 : a.startsWith('⚠️') ? COLOR2 : COLOR }}>
              {a}
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab: HCM vs Leigh vs KSS ──────────────────────────────────────────────────
function HCMvsTab({ data, overview }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const variants = data.variant_breakdown || [];
  const cv_bands = data.cv_activity_bands || {};
  const het_bands = data.heteroplasmy_bands || {};
  const outcomes = data.outcome_distribution || [];
  const pts = data.patient_table || [];
  const hcm_table = data.hcm_severity_table || [];
  const s = overview?.cohort_statistics || {};

  return (
    <div>
      {/* HCM vs Leigh features */}
      <SectionCard title="Cardiac Features vs Leigh/Encephalomyopathy Features" borderColor={COLOR4}>
        <div className="row">
          <div className="col-md-6">
            <p className="small fw-semibold mb-2" style={{ color: COLOR4 }}>Cardiac Features (HCM-predominant)</p>
            {[
              { label: 'Hypertrophic Cardiomyopathy (HCM)', key: 'hypertrophic_cardiomyopathy_pct' },
              { label: 'Exercise intolerance', key: 'exercise_intolerance_pct' },
              { label: 'Myopathy', key: 'myopathy_pct' },
              { label: 'LVOT obstruction (dynamic)', key: 'lvot_obstruction_pct' },
              { label: 'Arrhythmia (AF / VT)', key: 'arrhythmia_pct' },
              { label: 'Cardiac conduction defect', key: 'cardiac_conduction_pct' },
            ].map((f, i) => (
              <Bar key={i} label={f.label} value={s[f.key] ?? 0} color={COLOR4} />
            ))}
          </div>
          <div className="col-md-6">
            <p className="small fw-semibold mb-2" style={{ color: COLOR3 }}>Leigh/Encephalomyopathy Features</p>
            {[
              { label: 'Leigh MRI (bilateral BG/brainstem)', key: 'leigh_mri_pct' },
              { label: 'Lactic acidosis', key: 'lactic_acidosis_pct' },
              { label: 'Encephalopathy', key: 'encephalopathy_pct' },
              { label: 'Hypotonia', key: 'hypotonia_pct' },
              { label: 'Seizures', key: 'seizures_pct' },
              { label: 'Respiratory failure', key: 'respiratory_failure_pct' },
            ].map((f, i) => (
              <Bar key={i} label={f.label} value={s[f.key] ?? 0} color={COLOR3} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* HCM severity vs CV activity table */}
      {hcm_table.length > 0 && (
        <SectionCard title="HCM Severity vs Complex V Activity — Phenotype Thresholds" borderColor={COLOR4}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small mb-0">
              <thead><tr style={{ backgroundColor: LIGHT }}>
                <th>CV Activity</th><th>Phenotype</th><th>HCM Prevalence</th><th>LVOT Obstruction</th>
              </tr></thead>
              <tbody>
                {hcm_table.map((row, i) => (
                  <tr key={i} style={{
                    backgroundColor: row.cv_activity.startsWith('4') ? '#ffebee' :
                                     row.cv_activity.startsWith('5-22') ? '#fff8e1' :
                                     row.cv_activity.startsWith('>50') ? '#f1f8e9' : undefined
                  }}>
                    <td className="fw-semibold" style={{ color: row.cv_activity.startsWith('4') ? COLOR3 : COLOR4 }}>
                      {row.cv_activity}
                    </td>
                    <td>{row.phenotype}</td>
                    <td style={{ color: COLOR4 }}>{row.hcm_prevalence}</td>
                    <td>{row.lvot}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Variant breakdown table */}
      <SectionCard title="Variant Breakdown — MT-ATP8 Pathogenic Variants" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Variant</th><th>Protein</th><th>Domain</th><th>n</th>
              <th>Avg CV%</th><th>Avg Heter%</th><th>HCM%</th>
              <th>Leigh MRI%</th><th>CI+CV%</th>
            </tr></thead>
            <tbody>
              {variants.map((v, i) => (
                <tr key={i} style={{ backgroundColor: v.variant === 'm.8528T>C' ? '#fce4ec' : undefined }}>
                  <td className="fw-semibold" style={{ color: v.variant === 'm.8528T>C' ? COLOR5 : undefined }}>
                    {v.variant}
                    {v.variant === 'm.8528T>C' && <span className="badge ms-1" style={{ backgroundColor: COLOR5, fontSize: '0.65rem' }}>OVERLAP</span>}
                  </td>
                  <td style={{ fontSize: '0.8rem' }}>{v.protein}</td>
                  <td className="text-muted" style={{ maxWidth: 180, fontSize: '0.78rem' }}>{v.domain}</td>
                  <td>{v.n_patients}</td>
                  <td style={{ color: v.avg_cv_activity_pct < 20 ? COLOR3 : v.avg_cv_activity_pct < 35 ? COLOR2 : COLOR4 }}>
                    {v.avg_cv_activity_pct}%
                  </td>
                  <td>{v.avg_heteroplasmy_pct}%</td>
                  <td style={{ color: COLOR4 }}>{v.hcm_pct}%</td>
                  <td>{v.leigh_mri_pct}%</td>
                  <td style={{ color: v.combined_ci_cv_pct > 0 ? COLOR5 : undefined }}>
                    {v.combined_ci_cv_pct}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small text-muted mt-2 mb-0">
          🟣 m.8528T>C (OVERLAP) row highlighted — simultaneous ATP8 C-terminal + ATP6 Met1 disruption → combined CI+CV deficiency.
        </p>
      </SectionCard>

      {/* CV activity & heteroplasmy bands */}
      <div className="row mb-4">
        <div className="col-md-6">
          <SectionCard title="CV Activity Distribution (Complex V ATP synthesis)" borderColor={COLOR3}>
            {Object.entries(cv_bands).map(([band, count]) => (
              <Bar key={band} label={`CV ${band}`} value={Math.round((count / 40) * 100)}
                color={band === '<15%' ? COLOR3 : band === '15-30%' ? COLOR2 : COLOR4} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Heteroplasmy Distribution (blood)" borderColor={COLOR}>
            {Object.entries(het_bands).map(([band, count]) => (
              <Bar key={band} label={`${band} heteroplasmy`} value={Math.round((count / 40) * 100)}
                color={band === '>90%' ? COLOR3 : band === '70-90%' ? COLOR2 : COLOR} />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Outcomes */}
      <SectionCard title="Outcome Distribution (40-patient cohort)" borderColor={COLOR2}>
        {outcomes.map((o, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{o.outcome}</span><span className="text-muted">{o.count} pts</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{
                width: `${(o.count / 40) * 100}%`,
                backgroundColor: o.outcome.includes('Deceased') ? '#616161' :
                                 o.outcome.includes('ICD') ? COLOR4 : COLOR2
              }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient table (first 15) */}
      <SectionCard title="Patient Cohort Sample (seed-779)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>ID</th><th>Phenotype</th><th>Variant</th><th>Heter%</th>
              <th>CV%</th><th>Lactate</th><th>HCM</th><th>Arrhythmia</th><th>Leigh MRI</th><th>CI+CV</th><th>Outcome</th>
            </tr></thead>
            <tbody>
              {pts.slice(0, 15).map((p, i) => (
                <tr key={i} style={{ backgroundColor: p.variant === 'm.8528T>C' ? '#fce4ec' : undefined }}>
                  <td className="fw-semibold">{p.id}</td>
                  <td style={{ maxWidth: 180, fontSize: '0.75rem' }}>{p.phenotype}</td>
                  <td style={{ fontSize: '0.8rem' }}>{p.variant}</td>
                  <td>{p.heteroplasmy_pct}%</td>
                  <td style={{ color: p.cv_pct < 20 ? COLOR3 : p.cv_pct < 35 ? COLOR2 : COLOR4 }}>
                    {p.cv_pct}%
                  </td>
                  <td>{p.lactate}</td>
                  <td style={{ color: COLOR4 }}>{p.hcm ? '💙' : '—'}</td>
                  <td>{p.arrhythmia ? '⚡' : '—'}</td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td style={{ color: COLOR5 }}>{p.combined_ci_cv ? '🟣' : '—'}</td>
                  <td style={{ fontSize: '0.72rem' }}>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted small mt-2">
          Showing 15 of {pts.length} patients (seed-779, synthetic cohort).
          🟣 = combined CI+CV deficiency (overlap mutation m.8528T&gt;C).
        </p>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ overview, breakdown }) {
  const ddx = breakdown?.differential_diagnosis || [];
  const contraindications = overview?.absolute_contraindications || [];
  const mandatory = overview?.mandatory_empiric_treatments || [];
  const levelc = overview?.level_c_treatments || [];
  const cardiac = overview?.cardiac_monitoring_protocol;

  return (
    <div>
      {/* Cardiac monitoring — highlighted */}
      {cardiac && (
        <div className="alert fw-semibold mb-4" style={{ backgroundColor: '#e3f2fd', borderLeft: `5px solid ${COLOR4}`, color: COLOR4 }}>
          💙 CARDIAC MONITORING PROTOCOL: {cardiac}
        </div>
      )}

      {/* Differential Diagnosis */}
      <SectionCard title="Differential Diagnosis — MT-ATP8 HCM/Leigh vs Key Mimics" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Entity</th><th>Distinguishing Feature</th><th>Key Test</th>
            </tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ minWidth: 160 }}>{d.entity}</td>
                  <td>{d.distinguishing_feature}</td>
                  <td className="text-muted">{d.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Absolute contraindications */}
      <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
        <ul className="mb-0 small">
          {contraindications.map((c, i) => (
            <li key={i} className="mb-1"><span className="text-danger fw-semibold">🚫 </span>{c}</li>
          ))}
        </ul>
      </SectionCard>

      {/* Mandatory treatments */}
      <SectionCard title="Mandatory Empiric Treatments" borderColor={COLOR4}>
        <ul className="mb-0 small">
          {mandatory.map((t, i) => (
            <li key={i} className="mb-1"><span style={{ color: COLOR4 }}>✅ </span>{t}</li>
          ))}
        </ul>
      </SectionCard>

      {/* Level C treatments */}
      <SectionCard title="Level C Treatments" borderColor={COLOR}>
        <ul className="mb-0 small">
          {levelc.map((t, i) => (
            <li key={i} className="mb-1"><span style={{ color: COLOR }}>🔵 </span>{t}</li>
          ))}
        </ul>
        {overview?.preferred_aed && (
          <div className="mt-2 small"><strong>Preferred AED:</strong> {overview.preferred_aed}</div>
        )}
      </SectionCard>

      {/* Non-overlap vs overlap biochemistry */}
      <SectionCard title="Non-overlap vs Overlap Mutation Biochemistry" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Mutation Type</th><th>rCRS Region</th><th>Genes Affected</th><th>Biochemistry</th><th>Phenotype</th>
            </tr></thead>
            <tbody>
              <tr>
                <td className="fw-semibold">Non-overlap (m.8411/8423/8438)</td>
                <td>8366–8526 (ATP8 only)</td>
                <td>MT-ATP8 only</td>
                <td>Isolated CV deficiency (CI/CII/CIII/CIV normal)</td>
                <td>HCM / Leigh (isolated)</td>
              </tr>
              <tr style={{ backgroundColor: '#fce4ec' }}>
                <td className="fw-semibold" style={{ color: COLOR5 }}>Overlap (m.8528T&gt;C)</td>
                <td style={{ color: COLOR5 }}>8527–8572 (shared ATP8+ATP6)</td>
                <td className="fw-semibold text-danger">MT-ATP8 + MT-ATP6</td>
                <td className="fw-semibold text-danger">Combined CI+CV deficiency</td>
                <td className="fw-semibold text-danger">HCM + Leigh-like (worst prognosis)</td>
              </tr>
            </tbody>
          </table>
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
      <SectionCard title="Gene & Protein" borderColor={COLOR}>
        <div className="row g-2 small">
          {[
            ['Gene', data.gene], ['OMIM Gene', data.omim_gene],
            ['Full Name', data.full_name], ['Protein', data.protein_name],
            ['aa Length', `${data.aa_length} aa`], ['MW', `${data.molecular_weight_kda} kDa`],
            ['TM Helices', data.tm_helices], ['rCRS', data.rcrs_positions],
            ['Strand', data.strand], ['Module', data.module],
          ].map(([k, v], i) => (
            <div key={i} className="col-md-6"><strong>{k}:</strong> {v}</div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Complex V Structure" borderColor={COLOR}>
        <p className="small mb-0">{data.complex_v_structure}</p>
      </SectionCard>

      <SectionCard title="Overlap Region (rCRS 8527-8572)" borderColor={COLOR5}>
        <p className="small mb-0 fw-semibold" style={{ color: COLOR5 }}>{data.overlap_definition}</p>
      </SectionCard>

      <SectionCard title="HCM Definition" borderColor={COLOR4}>
        <p className="small mb-0 fw-semibold" style={{ color: COLOR4 }}>{data.hcm_definition}</p>
      </SectionCard>

      <SectionCard title="Leigh Syndrome Definition" borderColor={COLOR3}>
        <p className="small mb-0 fw-semibold text-danger">{data.leigh_definition}</p>
      </SectionCard>

      <SectionCard title="ATP8/ATP6 Assembly Mechanism" borderColor={COLOR}>
        <p className="small mb-0">{data.assembly_mechanism_definition}</p>
      </SectionCard>

      <SectionCard title="Combined CI+CV Deficiency (Overlap Mutations)" borderColor={COLOR5}>
        <p className="small mb-0" style={{ color: COLOR5 }}>{data.combined_ci_cv_definition}</p>
      </SectionCard>

      <SectionCard title="Key Variants" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}><th>Variant</th><th>Details</th></tr></thead>
            <tbody>
              {Object.entries(data.key_variants || {}).map(([variant, detail], i) => (
                <tr key={i} style={{ backgroundColor: variant === 'm.8528T>C' ? '#fce4ec' : undefined }}>
                  <td className="fw-semibold" style={{ color: variant === 'm.8528T>C' ? COLOR5 : undefined }}>
                    {variant}
                    {variant === 'm.8528T>C' && <span className="badge ms-1" style={{ backgroundColor: COLOR5, fontSize: '0.65rem' }}>OVERLAP</span>}
                  </td>
                  <td>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="m.8411A>G Definition" borderColor={COLOR4}>
        <p className="small mb-0">{data.m8411ag_definition}</p>
      </SectionCard>

      <SectionCard title="m.8423T>C Definition" borderColor={COLOR3}>
        <p className="small mb-0">{data.m8423tc_definition}</p>
      </SectionCard>

      <SectionCard title="m.8528T>C (Overlap Mutation) Definition" borderColor={COLOR5}>
        <p className="small mb-0 fw-semibold" style={{ color: COLOR5 }}>{data.m8528tc_definition}</p>
      </SectionCard>

      <SectionCard title="Maternal Inheritance" borderColor={COLOR}>
        <p className="small mb-0">{data.maternal_inheritance_definition}</p>
      </SectionCard>

      <SectionCard title="BTBGD Mandatory Exclusion" borderColor={COLOR3}>
        <p className="small mb-0 fw-semibold text-danger">{data.btbgd_mandatory_exclusion}</p>
      </SectionCard>

      <SectionCard title="Biochemical Fingerprint" borderColor={COLOR2}>
        {Object.entries(data.biochemical_fingerprint || {}).map(([k, v], i) => (
          <div key={i} className="mb-2 small">
            <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Recommended Treatments" borderColor={COLOR4}>
        {Object.entries(data.recommended_treatments || {}).map(([k, v], i) => (
          <div key={i} className="mb-1 small">
            <strong>{k.replace(/_/g, ' ').toUpperCase()}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor={COLOR3}>
        {Object.entries(data.absolute_contraindications || {}).map(([k, v], i) => (
          <div key={i} className="mb-1 small text-danger">
            <strong>🚫 {k}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Specialist Monitoring" borderColor={COLOR}>
        {Object.entries(data.specialist_monitoring || {}).map(([k, v], i) => (
          <div key={i} className="mb-1 small">
            <strong>{k}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="KSS Annual Holter Policy" borderColor={COLOR3}>
        <p className="small mb-0">{data.kss_holter_policy}</p>
      </SectionCard>

      <SectionCard title="WES Coverage Warning" borderColor={COLOR3}>
        <p className="small mb-0 fw-semibold text-danger">{data.wes_coverage}</p>
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR2}>
        <ol className="small mb-0">
          {(data.key_references || []).map((r, i) => (
            <li key={i} className="mb-1">{r}</li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MTATP8Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mtatp8/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 && !breakdown)
      fetch(`${API}/api/mtatp8/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setErr(e.message));
    if (tab === 3 && !defs)
      fetch(`${API}/api/mtatp8/definitions`)
        .then(r => r.json()).then(setDefs).catch(e => setErr(e.message));
  }, [tab]);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  return (
    <div className="container-fluid py-4">
      <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
        MT-ATP8 — Complex V F0 Peripheral Stalk Anchor — HCM / Leigh / Encephalomyopathy
      </h4>
      <p className="text-muted small mb-3">
        68 aa · 7.6 kDa · 2 TM helices · H-strand rCRS 8366-8572 · OMIM *516070 ·
        HCM (annual echo+Holter mandatory) · Overlap mutation m.8528T&gt;C → combined CI+CV ·
        NO Retinitis Pigmentosa (distinguishes from MT-ATP6/NARP) · 40-patient cohort (seed-779)
      </p>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <HCMvsTab data={breakdown} overview={overview} />}
      {tab === 2 && <DDxTab overview={overview} breakdown={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
