'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'NARP vs Leigh vs KSS', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#e65100';   // deep orange — CV/ATP synthase energy metabolism
const LIGHT  = '#fff3e0';
const COLOR2 = '#bf360c';   // dark orange — severe NARP/Leigh
const COLOR3 = '#b71c1c';   // dark red — Leigh MRI / severe
const COLOR4 = '#1b5e20';   // dark green — NARP triad
const COLOR5 = '#4527a0';   // deep purple — retinitis pigmentosa

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

  return (
    <div>
      {/* Banner */}
      <div className="alert fw-bold mb-4" style={{ backgroundColor: LIGHT, borderLeft: `5px solid ${COLOR}`, color: COLOR2 }}>
        🟠 MT-ATP6: COMPLEX V F0 PROTON CHANNEL (226 aa, 8 TM) — NARP (m.8993T&gt;G/C, 70-90% heteroplasmy) vs LEIGH/MILS (≥90%).
        Retinitis Pigmentosa: hallmark of NARP — only mtDNA gene causing RP as primary feature.
        CV ATP synthesis: <strong>isolated deficiency — CI/II/III/IV normal.</strong>
      </div>

      {/* KPIs */}
      <div className="row mb-4">
        <KPI label="Patients" value={data.n_patients} color={COLOR} />
        <KPI label="Avg CV Activity" value={`${s.avg_cv_activity_pct}%`} color={COLOR3} />
        <KPI label="Avg Lactate" value={`${s.avg_lactic_acid_mmolL} mmol/L`} color={COLOR2} />
        <KPI label="Leigh MRI" value={`${s.leigh_mri_pct}%`} color={COLOR3} />
        <KPI label="Lactic Acidosis" value={`${s.lactic_acidosis_pct}%`} color={COLOR2} />
        <KPI label="NARP Triad" value={`${s.narp_triad_pct}%`} color={COLOR4} />
        <KPI label="Retinitis Pigmentosa" value={`${s.retinitis_pigmentosa_pct}%`} color={COLOR5} />
        <KPI label="Cerebellar Ataxia" value={`${s.cerebellar_ataxia_pct}%`} color={COLOR4} />
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

      {/* Heteroplasmy threshold summary */}
      <SectionCard title="Heteroplasmy Thresholds — m.8993T&gt;G/C NARP vs Leigh/MILS" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Heteroplasmy (blood)</th><th>Phenotype (m.8993T&gt;G)</th><th>CV Activity</th><th>Retinitis Pigmentosa</th>
            </tr></thead>
            <tbody>
              <tr><td className="text-muted">&lt;70%</td><td>Subclinical / carrier</td><td>&gt;60%</td><td>Absent</td></tr>
              <tr style={{ backgroundColor: '#fff8e1' }}>
                <td className="fw-semibold" style={{ color: COLOR4 }}>70–90%</td>
                <td className="fw-semibold">NARP — weakness + ataxia + RP</td>
                <td>30–60%</td>
                <td style={{ color: COLOR5 }}>Present (85%)</td>
              </tr>
              <tr style={{ backgroundColor: '#ffebee' }}>
                <td className="fw-semibold" style={{ color: COLOR3 }}>≥90%</td>
                <td className="fw-semibold text-danger">Leigh/MILS — bilateral BG/brainstem</td>
                <td style={{ color: COLOR3 }}>5–30%</td>
                <td className="text-muted">Rare (8%)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="small text-muted mt-2 mb-0">m.8993T&gt;C (milder): predominantly NARP at 65-88%; Leigh rare even at &gt;90%. m.9176T&gt;C: Leigh-predominant from ≥85%.</p>
      </SectionCard>

      {/* Phenotype distribution */}
      {pheno_dist.length > 0 && (
        <SectionCard title="Phenotype Distribution" borderColor={COLOR3}>
          {pheno_dist.map((p, i) => (
            <div key={i} className="mb-2">
              <div className="d-flex justify-content-between small mb-1">
                <span>{p.phenotype}</span><span className="text-muted">{p.count} pts</span>
              </div>
              <div className="progress" style={{ height: 10 }}>
                <div className="progress-bar" style={{
                  width: `${(p.count / data.n_patients) * 100}%`,
                  backgroundColor: p.phenotype.includes('Leigh') ? COLOR3 : p.phenotype.includes('NARP') ? COLOR4 : COLOR2
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
                color={f.pct > 60 ? COLOR3 : f.pct > 30 ? COLOR2 : COLOR4} />
            ))}
          </div>
          <div className="col-md-6">
            {feats.slice(Math.ceil(feats.length / 2)).map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct > 60 ? COLOR3 : f.pct > 30 ? COLOR2 : COLOR4} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Molecular features */}
      {mol_feats.length > 0 && (
        <SectionCard title="Key Molecular Features — MT-ATP6 Complex V F0 Proton Channel" borderColor={COLOR}>
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

// ── Tab: NARP vs Leigh vs KSS ─────────────────────────────────────────────────
function NARPvsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const variants = data.variant_breakdown || [];
  const cv_bands = data.cv_activity_bands || {};
  const het_bands = data.heteroplasmy_bands || {};
  const outcomes = data.outcome_distribution || [];
  const pts = data.patient_table || [];
  const narp_thresh = data.narp_heteroplasmy_threshold || [];

  return (
    <div>
      {/* NARP Triad feature bars */}
      <SectionCard title="NARP Triad Features vs Leigh Features" borderColor={COLOR4}>
        <div className="row">
          <div className="col-md-6">
            <p className="small fw-semibold mb-2" style={{ color: COLOR4 }}>NARP Features (moderate heteroplasmy)</p>
            {[
              { label: 'Cerebellar ataxia', key: 'cerebellar_ataxia_pct' },
              { label: 'Retinitis Pigmentosa (RP)', key: 'retinitis_pigmentosa_pct' },
              { label: 'NARP Triad (all 3)', key: 'narp_triad_pct' },
              { label: 'Peripheral neuropathy', key: 'peripheral_neuropathy_pct' },
              { label: 'Sensorineural hearing loss', key: 'sensorineural_hearing_loss_pct' },
            ].map((f, i) => (
              <Bar key={i} label={f.label} value={data.cohort_statistics?.[f.key] ?? 0}
                color={f.key === 'retinitis_pigmentosa_pct' ? COLOR5 : COLOR4} />
            ))}
          </div>
          <div className="col-md-6">
            <p className="small fw-semibold mb-2" style={{ color: COLOR3 }}>Leigh/MILS Features (high heteroplasmy)</p>
            {[
              { label: 'Leigh MRI (bilateral BG/brainstem)', key: 'leigh_mri_pct' },
              { label: 'Lactic acidosis', key: 'lactic_acidosis_pct' },
              { label: 'Encephalopathy', key: 'encephalopathy_pct' },
              { label: 'Hypotonia', key: 'hypotonia_pct' },
              { label: 'Seizures', key: 'seizures_pct' },
            ].map((f, i) => (
              <Bar key={i} label={f.label} value={data.cohort_statistics?.[f.key] ?? 0}
                color={COLOR3} />
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Heteroplasmy threshold diagram */}
      {narp_thresh.length > 0 && (
        <SectionCard title="Heteroplasmy Threshold Diagram — m.8993T>G NARP / Leigh / Subclinical" borderColor={COLOR}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small mb-0">
              <thead><tr style={{ backgroundColor: LIGHT }}>
                <th>Heteroplasmy Range</th><th>Phenotype</th><th>CV Activity</th><th>Retinitis Pigmentosa</th>
              </tr></thead>
              <tbody>
                {narp_thresh.map((row, i) => (
                  <tr key={i} style={{
                    backgroundColor: row.heteroplasmy_range === '≥90%' ? '#ffebee' :
                                     row.heteroplasmy_range === '70-90%' ? '#fff8e1' : '#f1f8e9'
                  }}>
                    <td className="fw-semibold">{row.heteroplasmy_range}</td>
                    <td>{row.phenotype}</td>
                    <td style={{ color: row.heteroplasmy_range === '≥90%' ? COLOR3 : COLOR4 }}>{row.cv_activity}</td>
                    <td style={{ color: COLOR5 }}>{row.retinitis_pigmentosa}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Variant breakdown table */}
      <SectionCard title="Variant Breakdown — MT-ATP6 Pathogenic Variants" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>Variant</th><th>Protein</th><th>Domain</th><th>n</th>
              <th>Avg CV%</th><th>Avg Heter%</th><th>Leigh MRI%</th>
              <th>NARP Triad%</th><th>RP%</th>
            </tr></thead>
            <tbody>
              {variants.map((v, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{v.variant}</td>
                  <td>{v.protein}</td>
                  <td className="text-muted" style={{ maxWidth: 180 }}>{v.domain}</td>
                  <td>{v.n_patients}</td>
                  <td style={{ color: v.avg_cv_activity_pct < 20 ? COLOR3 : v.avg_cv_activity_pct < 40 ? COLOR2 : COLOR4 }}>
                    {v.avg_cv_activity_pct}%
                  </td>
                  <td>{v.avg_heteroplasmy_pct}%</td>
                  <td>{v.leigh_mri_pct}%</td>
                  <td style={{ color: COLOR4 }}>{v.narp_triad_pct}%</td>
                  <td style={{ color: COLOR5 }}>{v.retinitis_pigmentosa_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* CV activity & heteroplasmy bands side by side */}
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
          <SectionCard title="Heteroplasmy Distribution (blood) — NARP/Leigh Thresholds" borderColor={COLOR}>
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
                backgroundColor: o.outcome.includes('Deceased') ? '#616161' : COLOR2
              }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Patient table (first 15) */}
      <SectionCard title="Patient Cohort Sample (seed-775)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>ID</th><th>Phenotype</th><th>Variant</th><th>Heter%</th>
              <th>CV%</th><th>Lactate</th><th>NARP Triad</th><th>RP</th><th>Leigh MRI</th><th>Outcome</th>
            </tr></thead>
            <tbody>
              {pts.slice(0, 15).map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.id}</td>
                  <td style={{ maxWidth: 180, fontSize: '0.75rem' }}>{p.phenotype}</td>
                  <td>{p.variant}</td>
                  <td>{p.heteroplasmy_pct}%</td>
                  <td style={{ color: p.cv_pct < 20 ? COLOR3 : p.cv_pct < 40 ? COLOR2 : COLOR4 }}>
                    {p.cv_pct}%
                  </td>
                  <td>{p.lactate}</td>
                  <td>{p.narp_triad ? '✅' : '—'}</td>
                  <td style={{ color: COLOR5 }}>{p.retinitis_pigmentosa ? '🟣' : '—'}</td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td style={{ fontSize: '0.72rem' }}>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted small mt-2">Showing 15 of {pts.length} patients (seed-775, synthetic cohort)</p>
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

  return (
    <div>
      {/* Differential Diagnosis */}
      <SectionCard title="Differential Diagnosis — MT-ATP6 NARP/Leigh vs Key Mimics" borderColor={COLOR3}>
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

      {/* Mandatory empiric treatments */}
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

      {/* CV residual activity guide */}
      <SectionCard title="Complex V Residual Activity — Phenotype Thresholds" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}>
              <th>CV Activity</th><th>Phenotype</th><th>Heteroplasmy</th><th>Retinitis Pigmentosa</th>
            </tr></thead>
            <tbody>
              <tr><td className="fw-semibold" style={{ color: COLOR3 }}>5–30%</td><td>Leigh/MILS (severe)</td><td>≥90%</td><td className="text-muted">Rare (8%)</td></tr>
              <tr><td className="fw-semibold" style={{ color: COLOR2 }}>15–40%</td><td>NARP+Leigh Overlap</td><td>Intermediate</td><td style={{ color: COLOR5 }}>Partial (~35%)</td></tr>
              <tr><td className="fw-semibold" style={{ color: COLOR4 }}>30–60%</td><td>NARP (neurogenic weakness + ataxia + RP)</td><td>70–90%</td><td style={{ color: COLOR5 }}>Present (85%)</td></tr>
              <tr><td className="fw-semibold" style={{ color: '#f9a825' }}>30–55%</td><td>KSS / CPEO (large deletion)</td><td>variable</td><td className="text-muted">12% (deletion context)</td></tr>
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

      <SectionCard title="NARP Definition" borderColor={COLOR4}>
        <p className="small mb-0 fw-semibold" style={{ color: COLOR4 }}>{data.narp_definition}</p>
      </SectionCard>

      <SectionCard title="MILS Definition" borderColor={COLOR3}>
        <p className="small mb-0 fw-semibold text-danger">{data.mils_definition}</p>
      </SectionCard>

      <SectionCard title="Retinitis Pigmentosa — NARP Hallmark" borderColor={COLOR5}>
        <p className="small mb-0" style={{ color: COLOR5 }}>{data.retinitis_pigmentosa_definition}</p>
      </SectionCard>

      <SectionCard title="Heteroplasmy Threshold" borderColor={COLOR2}>
        <p className="small mb-0">{data.heteroplasmy_threshold_definition}</p>
      </SectionCard>

      <SectionCard title="Key Variants" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr style={{ backgroundColor: LIGHT }}><th>Variant</th><th>Details</th></tr></thead>
            <tbody>
              {Object.entries(data.key_variants || {}).map(([variant, detail], i) => (
                <tr key={i}>
                  <td className="fw-semibold">{variant}</td>
                  <td>{detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="m.8993T&gt;G Definition" borderColor={COLOR3}>
        <p className="small mb-0">{data.m8993tg_definition}</p>
      </SectionCard>

      <SectionCard title="m.8993T&gt;C Definition" borderColor={COLOR2}>
        <p className="small mb-0">{data.m8993tc_definition}</p>
      </SectionCard>

      <SectionCard title="m.9176T&gt;C Definition" borderColor={COLOR3}>
        <p className="small mb-0">{data.m9176tc_definition}</p>
      </SectionCard>

      <SectionCard title="NARP Triad Definition" borderColor={COLOR4}>
        <p className="small mb-0">{data.narp_triad_definition}</p>
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
export default function MTATP6Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/mtatp6/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 && !breakdown)
      fetch(`${API}/api/mtatp6/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setErr(e.message));
    if (tab === 3 && !defs)
      fetch(`${API}/api/mtatp6/definitions`)
        .then(r => r.json()).then(setDefs).catch(e => setErr(e.message));
  }, [tab]);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;

  return (
    <div className="container-fluid py-4">
      <h4 className="fw-bold mb-1" style={{ color: COLOR }}>
        MT-ATP6 — Complex V F0 Proton Channel — NARP / Leigh-MILS
      </h4>
      <p className="text-muted small mb-3">
        226 aa · 24.8 kDa · 8 TM helices · H-strand rCRS 8527-9207 · OMIM *516006 ·
        NARP (m.8993T&gt;G/C, 70-90% heteroplasmy) / MILS (≥90%) · Retinitis Pigmentosa hallmark ·
        Isolated CV deficiency · 40-patient cohort (seed-775)
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
      {tab === 1 && <NARPvsTab data={breakdown} overview={overview} />}
      {tab === 2 && <DDxTab overview={overview} breakdown={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
