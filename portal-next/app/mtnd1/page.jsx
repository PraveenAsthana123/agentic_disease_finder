'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'LHON vs Leigh', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — mtDNA CI / maternal inheritance
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#880e4f';   // deep purple — LHON phenotype (intermediate recovery)
const COLOR4 = '#e65100';   // deep orange — Leigh / MELAS-overlap
const COLOR5 = '#2e7d32';   // green — treatments / recovery / normal

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
  const lhon_feats = data.lhon_specific_features || [];
  const mol_feats = data.key_molecular_features || [];

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 MT-ND1 — MOST CONSERVED CI Subunit / LHON #2 Worldwide m.3460G&gt;A / PROXIMAL MEMBRANE ARM ND1-MODULE
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
          🔵 MT-ND1: MOST CONSERVED CI subunit — 318 aa / 36 kDa / 8 TM helices / PROXIMAL MEMBRANE ARM / ND1-MODULE CORE ANCHOR — H-strand rCRS 3307–4262.
          <span style={{ color: COLOR3 }}> LHON: m.3460G&gt;A — #2 primary LHON worldwide (15%); INTERMEDIATE 22% spontaneous recovery; {s.lhon_phenotype_pct}% of cohort.</span>
          <span style={{ color: COLOR4 }}> Leigh: bilateral BG/brainstem — high heteroplasmy ≥70% — CI {s.avg_ci_activity_pct}% avg.</span>
          Tobacco/Ethambutol: ABSOLUTE in LHON. Metformin/VPA/Linezolid/Propofol/Chloramphenicol: ABSOLUTE all phenotypes.
        </p>
      </div>

      {/* LHON + Leigh panels */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100" style={{ borderTop: `4px solid ${COLOR3}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: COLOR3 }}>LHON Phenotype (near-homoplasmic) — #2 worldwide / 22% intermediate recovery</h6>
              <ul className="small mb-0">
                <li>m.3460G&gt;A (p.Ala52Thr) — #2 primary LHON worldwide (~15% of all LHON)</li>
                <li>Subacute painless bilateral optic atrophy; onset 20–30 years</li>
                <li>Sequential bilateral 97%; second eye 6–8 weeks after first</li>
                <li>Male predominance 80–90% (X-linked modifier; incomplete penetrance)</li>
                <li>Peripapillary telangiectatic microangiopathy — NO FFA leak (KEY DDx NAION)</li>
                <li>Red-green dyschromatopsia (KEY DDx OPA1 — blue-yellow)</li>
                <li style={{ color: COLOR3 }}>INTERMEDIATE prognosis: 22% spontaneous visual recovery (vs &lt;4% ND4; 50% ND6)</li>
                <li style={{ color: COLOR5 }}>Idebenone (Raxone) 900 mg/day — Level B (RHODOS 2011)</li>
                <li style={{ color: COLOR4 }}>Tobacco + Ethambutol: ABSOLUTE CI in ALL MT-ND1 genotypes</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100" style={{ borderTop: `4px solid ${COLOR4}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: COLOR4 }}>Leigh Syndrome / MELAS Overlap (high heteroplasmy ≥70%)</h6>
              <ul className="small mb-0">
                <li>m.3697G&gt;A (TM4 proton half-channel) — Leigh/MELAS overlap 20%</li>
                <li>m.3890G&gt;A (TM7 loop) — Leigh severe infantile 15%</li>
                <li>m.4171C&gt;A (TM8 C-terminal) — Leigh / exercise intolerance 10%</li>
                <li>Leigh MRI: bilateral symmetric BG/brainstem T2 {s.leigh_mri_pct}% cohort</li>
                <li>Lactic acidosis {s.lactic_acidosis_pct}%; hypotonia {s.hypotonia_pct}%</li>
                <li>Heteroplasmy threshold: ≥70% → Leigh; &lt;60% → no phenotype</li>
                <li style={{ color: COLOR4 }}>CI severely reduced (5–25%) — entire ND1-module proximal arm collapses</li>
                <li style={{ color: COLOR }}>GIR 6–8 mg/kg/min MANDATORY — NEVER fast in Leigh/MELAS</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* ND1-module molecular features */}
      {mol_feats.length > 0 && (
        <SectionCard title="🔬 ND1-Module / Proximal Membrane Arm — Molecular Distinctives" borderColor={COLOR2}>
          <ul className="mb-0">
            {mol_feats.map((f, i) => (
              <li key={i} className="small mb-1">{f}</li>
            ))}
          </ul>
        </SectionCard>
      )}

      {/* LHON specific features */}
      {lhon_feats.length > 0 && (
        <SectionCard title="👁️ LHON-Specific Features (MT-ND1 cohort)" borderColor={COLOR3}>
          <ul className="mb-0">
            {lhon_feats.map((f, i) => (
              <li key={i} className="small mb-1" style={{ color: f.includes('22%') || f.includes('INTERMEDIATE') ? COLOR5 : 'inherit' }}>{f}</li>
            ))}
          </ul>
        </SectionCard>
      )}

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
                  backgroundColor: p.phenotype.includes('LHON') && !p.phenotype.includes('Leigh') ? COLOR3 :
                                   p.phenotype.includes('Leigh') || p.phenotype.includes('MELAS') ? COLOR4 :
                                   p.phenotype.includes('CPEO') || p.phenotype.includes('KSS') || p.phenotype.includes('Pearson') ? COLOR5 : COLOR,
                }} />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* KPIs — LHON */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR3 }}>LHON Phenotype KPIs</h6>
      <div className="row mb-3">
        <KPI label="LHON (% cohort)" value={`${s.lhon_phenotype_pct}%`} color={COLOR3} />
        <KPI label="Optic Atrophy" value={`${s.optic_atrophy_pct}%`} color={COLOR3} />
        <KPI label="Sequential Bilateral" value={`${s.sequential_bilateral_pct}%`} color={COLOR3} />
        <KPI label="Peripapillary Telangiectasia" value={`${s.peripapillary_telangiectasia_pct}%`} color={COLOR3} />
        <KPI label="Visual Recovery (22% target)" value={`${s.lhon_recovery_pct}%`} color={COLOR5} />
        <KPI label="Red-Green Color Loss" value={`${s.red_green_color_loss_pct}%`} color={COLOR3} />
      </div>

      {/* KPIs — Leigh */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR4 }}>Leigh / MELAS-Overlap Phenotype KPIs</h6>
      <div className="row mb-4">
        <KPI label="Leigh MRI" value={`${s.leigh_mri_pct}%`} color={COLOR4} />
        <KPI label="Avg CI Activity %" value={`${s.avg_ci_activity_pct}%`} color={COLOR4} />
        <KPI label="Lactic Acidosis" value={`${s.lactic_acidosis_pct}%`} color={COLOR4} />
        <KPI label="Avg Lactate (mM)" value={s.avg_lactic_acid_mmolL} color={COLOR4} />
        <KPI label="MELAS Overlap" value={`${s.melas_overlap_pct}%`} color={COLOR4} />
        <KPI label="Mortality" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>

      {/* Feature bars */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Features (% of 40 patients)" borderColor={COLOR}>
            {feats.slice(0, Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={
                  f.feature.includes('Optic') || f.feature.includes('bilateral') || f.feature.includes('telangiectasia') || f.feature.includes('dyschr') ? COLOR3 :
                  f.feature.includes('recovery') ? COLOR5 :
                  f.feature.includes('Leigh') || f.feature.includes('acidosis') || f.feature.includes('MELAS') ? COLOR4 :
                  f.pct > 80 ? COLOR4 : COLOR
                } />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Clinical Features (continued)" borderColor={COLOR}>
            {feats.slice(Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={
                  f.feature.includes('CPEO') || f.feature.includes('Ragged') ? COLOR5 :
                  f.feature.includes('Respiratory') || f.feature.includes('Encepha') ? COLOR4 :
                  f.feature.includes('Dystonia') ? COLOR4 :
                  f.pct > 80 ? COLOR4 : COLOR
                } />
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Onset distribution */}
      <SectionCard title="Onset Distribution" borderColor={COLOR2}>
        <div className="row">
          {Object.entries(onset).map(([k, v]) => (
            <div key={k} className="col-6 col-md-3 mb-2 text-center">
              <div className="fw-bold" style={{
                color: k.includes('young_adult') ? COLOR3 :
                       k.includes('neonatal') || k.includes('infantile') ? COLOR4 :
                       k.includes('childhood') ? COLOR2 : COLOR5
              }}>{v}%</div>
              <div className="text-muted small">{k.replace(/_pct$/, '').replace(/_/g, ' ')}</div>
            </div>
          ))}
        </div>
        <p className="small text-muted mt-2">
          ⚡ m.3460G&gt;A LHON onset: 20–30 years — intermediate between ND6 (teens) and ND4 (20–35 y). Leigh/MELAS: infantile–childhood. Exercise intolerance: adult.
        </p>
      </SectionCard>

      {/* Heteroplasmy thresholds */}
      {data.heteroplasmy_thresholds && (
        <SectionCard title="⚡ Heteroplasmy Threshold — LHON vs Leigh/MELAS Spectrum" borderColor={COLOR3}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0">
              <thead style={{ background: COLOR3, color: '#fff' }}>
                <tr><th>Heteroplasmy / Variant</th><th>Clinical Phenotype</th></tr>
              </thead>
              <tbody>
                {Object.entries(data.heteroplasmy_thresholds).map(([k, v]) => (
                  <tr key={k}>
                    <td className="fw-bold small">{k}</td>
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
              background: a.startsWith('🚫') ? '#ffebee' : a.startsWith('⚠️') ? '#fff8e1' : a.startsWith('🔵') ? '#e3f2fd' : '#e8f5e9',
              borderLeft: `3px solid ${a.startsWith('🚫') ? COLOR3 : a.startsWith('⚠️') ? COLOR4 : a.startsWith('🔵') ? COLOR2 : COLOR5}`,
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
                <th>Heteroplasmy %</th><th>CI %</th><th>Lactate</th><th>Optic Atrophy</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td className="small">{p.patient_id}</td>
                  <td className="small" style={{ color: p.phenotype.includes('LHON') && !p.phenotype.includes('Leigh') ? COLOR3 : p.phenotype.includes('Leigh') || p.phenotype.includes('MELAS') ? COLOR4 : 'inherit' }}>{p.phenotype}</td>
                  <td className="small fw-bold" style={{ color: p.variant.includes('3460') ? COLOR3 : COLOR }}>{p.variant}</td>
                  <td className="small">{p.heteroplasmy_blood_pct}%</td>
                  <td className="small" style={{ color: p.ci_activity_pct < 25 ? COLOR4 : p.ci_activity_pct > 50 ? COLOR5 : COLOR }}>{p.ci_activity_pct}%</td>
                  <td className="small">{p.lactic_acid_mmolL}</td>
                  <td className="small">{p.optic_atrophy ? '✅' : '—'}</td>
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

// ── Tab: LHON vs Leigh ────────────────────────────────────────────────────────
function LhonVsLeighTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.all_variants || [];
  const vdist = data.variant_distribution || [];
  const biochem = data.biochemistry_distribution || {};
  const comparison = data.lhon_vs_other_comparison || {};
  const bnPage = data.bn_page_pattern || {};
  const immuno = data.immunoblot_pattern || {};

  return (
    <div>
      {/* MT-ND1 vs primary LHON peers */}
      {Object.keys(comparison).length > 1 && (
        <SectionCard title="🔀 MT-ND1 vs Primary LHON Peers (ND4 / ND1 / ND6)" borderColor={COLOR}>
          <div className="table-responsive">
            <table className="table table-sm table-bordered mb-0">
              <thead style={{ background: COLOR, color: '#fff' }}>
                <tr>
                  <th>Gene / Mutation</th><th>Rank</th><th>CI Residual</th>
                  <th>Visual Recovery</th><th>Onset Age</th><th>Gene Therapy</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison).filter(([k]) => k !== 'gene').map(([key, info]) => (
                  <tr key={key} style={{
                    background: key === 'mt_nd1_m3460' ? '#fff3e0' :
                                key === 'mt_nd4_m11778' ? '#fce4ec' : '#e8f5e9'
                  }}>
                    <td className="small fw-bold" style={{ color: key === 'mt_nd1_m3460' ? COLOR4 : key === 'mt_nd4_m11778' ? COLOR3 : COLOR5 }}>
                      {key.replace(/_/g, ' ').toUpperCase()}
                    </td>
                    <td className="small">{info.rank}</td>
                    <td className="small">{info.ci_residual}</td>
                    <td className="small fw-bold" style={{ color: key === 'mt_nd6_m14484' ? COLOR5 : key === 'mt_nd4_m11778' ? COLOR3 : COLOR4 }}>
                      {info.recovery_pct}
                    </td>
                    <td className="small">{info.onset_age}</td>
                    <td className="small">{info.gene_therapy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small text-muted mt-2">
            ⚡ MT-ND1 m.3460G&gt;A: INTERMEDIATE recovery (22%) — between ND4 worst (&lt;4%) and ND6 best (50%). CI residual 35–55% — intermediate severity. Proximal arm (ND1-module) vs central (ND4 TM11) vs distal tip (ND6 L-strand).
          </p>
        </SectionCard>
      )}

      {/* Pathogenic variants */}
      <SectionCard title="🧬 Pathogenic Variants in MT-ND1 (H-strand rCRS 3307–4262)" borderColor={COLOR}>
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
                <tr key={i} style={{
                  background: v.phenotype.includes('LHON') && !v.phenotype.includes('Leigh') ? '#fce4ec' :
                              v.phenotype.includes('Leigh') || v.phenotype.includes('MELAS') ? '#fff3e0' : 'inherit'
                }}>
                  <td className="fw-bold small" style={{ color: v.hgvs_mtdna.includes('3460') ? COLOR3 : COLOR }}>{v.hgvs_mtdna}</td>
                  <td className="small">{v.protein}</td>
                  <td className="small">{v.domain}</td>
                  <td className="small">{v.type}</td>
                  <td className="small">{v.phenotype}</td>
                  <td className="small" style={{
                    color: v.severity.includes('#2') || v.severity.includes('primary') ? COLOR3 :
                           v.severity.includes('Severe') ? COLOR4 :
                           v.severity.includes('Intermediate') ? COLOR2 : COLOR5
                  }}>{v.severity}</td>
                  <td className="small">{v.penetrance_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          {variants.map((v, i) => (
            <div key={i} className="small mb-2 p-2 rounded" style={{
              background: v.phenotype.includes('LHON') && !v.phenotype.includes('Leigh') ? '#fce4ec' : LIGHT,
              borderLeft: `3px solid ${v.hgvs_mtdna.includes('3460') ? COLOR5 : v.phenotype.includes('LHON') ? COLOR3 : COLOR}`,
            }}>
              <strong style={{ color: v.hgvs_mtdna.includes('3460') ? COLOR5 : COLOR }}>{v.hgvs_mtdna}</strong>: {v.notes}
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Variant Frequency Distribution (cohort)" borderColor={COLOR}>
            {vdist.map(v => (
              <Bar key={v.variant} label={v.variant} value={v.freq_pct}
                color={v.variant.includes('3460') ? COLOR5 : v.variant.includes('deletion') || v.variant.includes('Large') ? COLOR2 : COLOR4} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="CI Activity Distribution (LHON 35–55%, Leigh 5–25%)" borderColor={COLOR4}>
            <Bar label="CI &lt;25% (Leigh / MELAS-overlap)" value={biochem.ci_below_25_pct} color={COLOR4} />
            <Bar label="CI 25–50% (intermediate / LHON severe)" value={biochem.ci_25_to_50_pct} color={COLOR2} />
            <Bar label="CI 50–75% (LHON / exercise intolerance)" value={biochem.ci_50_to_75_pct} color={COLOR3} />
            <Bar label="CI &gt;75% (near-normal)" value={biochem.ci_above_75_pct} color={COLOR5} />
            <hr />
            <p className="small text-muted mb-1 fw-semibold">Lactate distribution:</p>
            <Bar label="Lactate normal &lt;2 mM (LHON typical)" value={biochem.lactic_normal_below2_pct} color={COLOR5} />
            <Bar label="Lactate 2–5 mM (mild elevation)" value={biochem.lactic_2_to_5_pct} color={COLOR2} />
            <Bar label="Lactate &gt;5 mM (Leigh / MELAS-overlap)" value={biochem.lactic_above_5_pct} color={COLOR4} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Heteroplasmy Distribution (blood)" borderColor={COLOR3}>
        <div className="row">
          {[
            ['Heteroplasmy &lt;70% (exercise intolerance / no phenotype)', 'het_below_70_pct', COLOR5],
            ['70–90% (Leigh / MELAS-overlap range)', 'het_70_to_90_pct', COLOR4],
            ['≥90% (near-homoplasmic — LHON range m.3460G&gt;A)', 'het_above_90_pct', COLOR3],
          ].map(([label, key, color]) => (
            <div key={key} className="col-md-6 mb-2">
              <Bar label={label} value={biochem[key] || 0} color={color} />
            </div>
          ))}
        </div>
        <p className="small text-muted mt-2">
          ⚠️ LHON (m.3460G&gt;A): near-homoplasmic in blood ({'>'}90%) but incomplete penetrance (50% lifetime risk males, 10–15% females).
          Leigh: ≥70% heteroplasmy required for bilateral BG/brainstem Leigh MRI.
          WES does NOT detect mtDNA variants — dedicated mtDNA sequencing required.
          <strong> Check H-strand coverage (rCRS 3307–4262) in NGS QC report.</strong>
        </p>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="BN-PAGE Pattern (phenotype-specific)" borderColor={COLOR2}>
            <p className="small fw-bold mb-1">{bnPage.finding}</p>
            <p className="small mb-1">{bnPage.interpretation}</p>
            <p className="small text-muted">{bnPage.ddx_value}</p>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Immunoblot Pattern (ND1-module subunits)" borderColor={COLOR2}>
            <table className="table table-sm mb-0">
              <tbody>
                {Object.entries(immuno).map(([subunit, finding]) => (
                  <tr key={subunit}>
                    <td className="small fw-bold" style={{ color: subunit.includes('ND1') ? COLOR3 : subunit.includes('NDUFB') ? COLOR4 : COLOR }}>{subunit}</td>
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
                       o.outcome.includes('near-complete') || o.outcome.includes('stable') || o.outcome.includes('exercise') ? COLOR5 :
                       o.outcome.includes('partial') ? COLOR5 :
                       o.outcome.includes('severe') ? COLOR3 :
                       o.outcome.includes('moderate') ? COLOR4 :
                       o.outcome.includes('CPEO') ? COLOR5 : COLOR} />
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Treatment uptake */}
      {data.treatment_uptake && (
        <SectionCard title="Treatment Uptake (cohort — LHON + Leigh + MELAS combined)" borderColor={COLOR5}>
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
      <SectionCard title="🔍 Differential Diagnosis — MT-ND1 vs Key Mimics" borderColor={COLOR3}>
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
          <SectionCard title="🚫 Contraindications" borderColor={COLOR3}>
            {(data.contraindications || []).map((c, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{
                background: c.severity === 'ABSOLUTE CI' ? '#ffebee' : '#fff8e1',
                borderLeft: `3px solid ${c.severity === 'ABSOLUTE CI' ? COLOR3 : COLOR4}`,
              }}>
                <strong className="small" style={{ color: COLOR3 }}>{c.drug}</strong>
                <span className="badge ms-2 small" style={{
                  background: c.severity === 'ABSOLUTE CI' ? COLOR3 : COLOR4, color: '#fff'
                }}>{c.severity}</span>
                <p className="small mb-0 text-muted">{c.reason}</p>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="✅ Treatments" borderColor={COLOR5}>
            {(data.treatments || []).map((t, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{
                background: '#e8f5e9',
                borderLeft: `3px solid ${t.level.includes('Level B') || t.level.includes('Mandatory') ? COLOR5 : COLOR2}`,
              }}>
                <strong className="small" style={{ color: COLOR5 }}>{t.drug}</strong>
                <span className="badge ms-2 small" style={{ background: COLOR5, color: '#fff' }}>{t.level}</span>
                <p className="small mb-0 text-muted">{t.rationale}</p>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🔬 Diagnostic Workup" borderColor={COLOR2}>
        <ul className="mb-0">
          {(data.diagnostic_workup || []).map((step, i) => (
            <li key={i} className="small mb-1"
              style={{ color: step.includes('NOT WES') || step.includes('FFA') || step.includes('H-strand') ? COLOR3 : 'inherit' }}>
              {step}
            </li>
          ))}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <SectionCard title="📖 Gene & Disease Definitions" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm mb-0">
            <tbody>
              {[
                ['Gene', data.gene],
                ['Full name', data.full_name],
                ['Alias', data.alias],
                ['OMIM Gene', data.omim_gene],
                ['Disease name', data.disease_name],
                ['Chromosome / genome position', data.chromosome],
                ['Protein size', `${data.protein_aa} aa, ${data.protein_kDa} kDa`],
                ['TM helices', data.tm_helices],
                ['Inheritance', data.inheritance],
                ['Cohort seed / n', `${data.cohort_seed} / ${data.n_patients}`],
              ].map(([k, v]) => (
                <tr key={k}>
                  <td className="small fw-bold text-nowrap" style={{ color: COLOR, width: '35%' }}>{k}</td>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="💡 Key Concepts" borderColor={COLOR2}>
        {Object.entries(data.key_concepts || {}).map(([term, def]) => (
          <div key={term} className="mb-2 p-2 rounded" style={{ background: LIGHT }}>
            <strong className="small" style={{ color: COLOR }}>{term}</strong>
            <p className="small mb-0">{def}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 Key References" borderColor={COLOR2}>
        {(data.references || []).map((r, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <p className="small mb-1 fw-bold" style={{ color: COLOR }}>{r.citation}</p>
            <p className="small mb-0 text-muted">{r.relevance}</p>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MTND1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [ov, br, df] = await Promise.all([
          fetch(`${API}/api/mtnd1/overview`).then(r => r.json()),
          fetch(`${API}/api/mtnd1/breakdown`).then(r => r.json()),
          fetch(`${API}/api/mtnd1/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(br); setDefinitions(df);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return (
    <div className="container py-5 text-center">
      <div className="spinner-border" style={{ color: COLOR }} role="status" />
      <p className="mt-3 text-muted">Loading MT-ND1 dashboard…</p>
    </div>
  );
  if (error) return (
    <div className="container py-5">
      <div className="alert alert-danger">Error: {error}</div>
    </div>
  );

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <LhonVsLeighTab key="br" data={breakdown} />,
    <DdxTab key="ddx" data={definitions} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold" style={{ color: COLOR }}>
          🧬 MT-ND1 — MOST CONSERVED CI Subunit / LHON #2 Worldwide m.3460G&gt;A / PROXIMAL MEMBRANE ARM ND1-MODULE / Leigh Syndrome
        </h4>
        <p className="text-muted small mb-0">
          318 aa · 36 kDa · 8 TM helices · PROXIMAL MEMBRANE ARM · ND1-MODULE CORE ANCHOR · H-strand rCRS 3307–4262 · OMIM *516000 ·
          m.3460G&gt;A #2 LHON worldwide (15%) · INTERMEDIATE 22% spontaneous visual recovery · Leigh Syndrome high heteroplasmy ≥70% ·
          Maternal inheritance · 40-patient cohort seed-759
        </p>
      </div>

      {/* Tab nav */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tabContent[tab]}
    </div>
  );
}
