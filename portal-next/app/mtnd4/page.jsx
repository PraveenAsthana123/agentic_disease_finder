'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'LHON vs Leigh', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — mtDNA CI / maternal inheritance
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';
const COLOR3 = '#880e4f';   // deep purple — LHON phenotype
const COLOR4 = '#e65100';   // deep orange — Leigh/high heteroplasmy
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
          🧬 MT-ND4 — Central Antiporter Module / DUAL PHENOTYPE: LHON + Leigh Syndrome
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
          🔵 MT-ND4: CENTRAL of 3 antiporter repeats (459 aa / 13 TM helices) — DUAL PHENOTYPE unique among CI subunits.
          <span style={{ color: COLOR3 }}> LHON (m.11778G>A #1 worldwide, ~70% of all LHON): near-homoplasmic, optic atrophy, {s.lhon_phenotype_pct}% of cohort.</span>
          <span style={{ color: COLOR4 }}> Leigh/CI (high heteroplasmy): isolated CI deficiency, Leigh MRI {s.leigh_mri_pct}%, lactic acidosis {s.lactic_acidosis_pct}%.</span>
          Tobacco/Ethambutol/Alcohol/Amiodarone: ABSOLUTE in LHON. Metformin/VPA/Linezolid/Propofol: ABSOLUTE in Leigh.
        </p>
      </div>

      {/* Dual phenotype highlight */}
      <div className="row mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100" style={{ borderTop: `4px solid ${COLOR3}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: COLOR3 }}>LHON Phenotype (near-homoplasmic)</h6>
              <ul className="small mb-0">
                <li>m.11778G>A (p.Arg340His) — #1 LHON mutation worldwide (~70% of LHON)</li>
                <li>Subacute painless bilateral optic atrophy; onset 15–35 years</li>
                <li>Second eye 6–8 weeks after first eye (97%)</li>
                <li>Male predominance 80–90%; {s.male_pct}% male in cohort</li>
                <li>Systemic CI near-normal; optic nerve selectively vulnerable</li>
                <li>Peripapillary microangiopathy — no FFA leak (KEY DDx NAION)</li>
                <li style={{ color: COLOR3 }}>{'<'}4% spontaneous visual recovery (WORST of 3 primary LHON mutations)</li>
                <li style={{ color: COLOR5 }}>Gene therapy: lenadogene nolparvovec (EU 2021, m.11778G>A only)</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100" style={{ borderTop: `4px solid ${COLOR4}` }}>
            <div className="card-body">
              <h6 className="fw-bold" style={{ color: COLOR4 }}>Leigh / CI Deficiency Phenotype (high heteroplasmy)</h6>
              <ul className="small mb-0">
                <li>m.11696G>A (Ser110Asn), m.11253T>C (Phe251Leu) — high heteroplasmy {'>'}80%</li>
                <li>Isolated CI deficiency: {s.avg_ci_activity_pct}% avg CI activity (cohort)</li>
                <li>Lactic acidosis {s.lactic_acidosis_pct}%; hypotonia {s.hypotonia_pct}%</li>
                <li>Leigh MRI bilateral BG/brainstem T2 hyperintensity {s.leigh_mri_pct}%</li>
                <li>Developmental delay/regression {s.developmental_delay_pct}%</li>
                <li>Stroke-like episodes (MELAS overlap) {s.stroke_like_pct}%</li>
                <li>Equal sex distribution (unlike LHON)</li>
                <li style={{ color: COLOR4 }}>No optic atrophy as primary feature (KEY DDx from LHON)</li>
              </ul>
            </div>
          </div>
        </div>
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
                  backgroundColor: p.phenotype.includes('LHON') ? COLOR3 :
                                   p.phenotype.includes('Overlap') || p.phenotype.includes('MELAS') ? COLOR4 :
                                   p.phenotype.includes('Leigh') ? COLOR4 :
                                   p.phenotype.includes('CPEO') || p.phenotype.includes('KSS') ? COLOR5 : COLOR,
                }} />
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* KPIs — LHON row */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR3 }}>LHON Phenotype KPIs</h6>
      <div className="row mb-3">
        <KPI label="LHON (% cohort)" value={`${s.lhon_phenotype_pct}%`} color={COLOR3} />
        <KPI label="Optic Atrophy" value={`${s.optic_atrophy_pct}%`} color={COLOR3} />
        <KPI label="Central Scotoma" value={`${s.central_scotoma_pct}%`} color={COLOR3} />
        <KPI label="Red-Green Dyschromatopsia" value={`${s.dyschromatopsia_pct}%`} color={COLOR3} />
        <KPI label="Peripapillary Microangiopathy" value={`${s.peripapillary_microangiopathy_pct}%`} color={COLOR3} />
        <KPI label="Male (LHON dominant)" value={`${s.male_pct}%`} color={COLOR3} />
      </div>

      {/* KPIs — Leigh row */}
      <h6 className="fw-bold mb-2" style={{ color: COLOR4 }}>Leigh / CI Deficiency Phenotype KPIs</h6>
      <div className="row mb-4">
        <KPI label="Leigh (% cohort)" value={`${s.leigh_phenotype_pct}%`} color={COLOR4} />
        <KPI label="Avg CI Activity %" value={`${s.avg_ci_activity_pct}%`} color={COLOR4} />
        <KPI label="Leigh MRI" value={`${s.leigh_mri_pct}%`} color={COLOR4} />
        <KPI label="Avg Lactate (mM)" value={s.avg_lactic_acid_mmolL} color={COLOR4} />
        <KPI label="Stroke-like Episodes" value={`${s.stroke_like_pct}%`} color={COLOR4} />
        <KPI label="Mortality" value={`${s.deceased_pct}%`} color={COLOR4} />
      </div>

      {/* Feature bars */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Features — LHON & Leigh (% of 40 patients)" borderColor={COLOR}>
            {feats.slice(0, Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={f.feature.includes('LHON') || f.feature.includes('optic') || f.feature.includes('scotoma') || f.feature.includes('Dyschr') || f.feature.includes('micro') ? COLOR3 :
                       f.feature.includes('Stroke') || f.feature.includes('MELAS') ? COLOR4 :
                       f.feature.includes('Ragged') || f.feature.includes('CPEO') ? COLOR5 :
                       f.pct > 80 ? COLOR4 : COLOR} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Clinical Features (continued)" borderColor={COLOR}>
            {feats.slice(Math.ceil(feats.length / 2)).map(f => (
              <Bar key={f.feature} label={f.feature} value={f.pct}
                color={f.feature.includes('LHON') || f.feature.includes('Idebenone') || f.feature.includes('gene') ? COLOR3 :
                       f.feature.includes('Stroke') || f.feature.includes('MELAS') ? COLOR4 :
                       f.feature.includes('CPEO') || f.feature.includes('Ragged') ? COLOR5 :
                       f.pct > 80 ? COLOR4 : COLOR} />
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
                color: k.includes('lhon') || k.includes('young_adult') ? COLOR3 :
                       k.includes('neonatal') || k.includes('infantile') ? COLOR4 : COLOR
              }}>{v}%</div>
              <div className="text-muted small">{k.replace(/_pct$/, '').replace(/_/g, ' ')}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Heteroplasmy thresholds */}
      {data.heteroplasmy_thresholds && (
        <SectionCard title="⚡ Heteroplasmy Threshold — LHON vs Leigh Duality" borderColor={COLOR3}>
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
                <th>Sex</th><th>Heteroplasmy %</th><th>CI %</th><th>Lactate</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id}>
                  <td className="small">{p.patient_id}</td>
                  <td className="small" style={{ color: p.phenotype.includes('LHON') ? COLOR3 : p.phenotype.includes('Leigh') ? COLOR4 : 'inherit' }}>{p.phenotype}</td>
                  <td className="small fw-bold" style={{ color: COLOR }}>{p.variant}</td>
                  <td className="small">{p.sex}</td>
                  <td className="small">{p.heteroplasmy_blood_pct}%</td>
                  <td className="small" style={{ color: p.ci_activity_pct < 25 ? COLOR4 : p.ci_activity_pct < 10 ? COLOR3 : COLOR5 }}>
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

// ── Tab: LHON vs Leigh ────────────────────────────────────────────────────────
function LhonVsLeighTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const variants = data.all_variants || [];
  const vdist = data.variant_distribution || [];
  const biochem = data.biochemistry_distribution || {};
  const comparison = data.lhon_vs_leigh_comparison || {};
  const bnPage = data.bn_page_pattern || {};
  const immuno = data.immunoblot_pattern || {};

  return (
    <div>
      {/* LHON vs Leigh side-by-side */}
      {Object.entries(comparison).length > 0 && (
        <SectionCard title="🔀 MT-ND4 Dual Phenotype — LHON vs Leigh Side-by-Side" borderColor={COLOR}>
          <div className="row">
            {Object.entries(comparison).map(([phenotype, info]) => (
              <div key={phenotype} className="col-md-6 mb-3">
                <div className="p-3 rounded h-100" style={{
                  background: phenotype.includes('LHON') ? '#fce4ec' : '#fff3e0',
                  borderLeft: `4px solid ${phenotype.includes('LHON') ? COLOR3 : COLOR4}`,
                }}>
                  <h6 className="fw-bold mb-2" style={{ color: phenotype.includes('LHON') ? COLOR3 : COLOR4 }}>
                    {phenotype}
                  </h6>
                  <table className="table table-sm mb-0" style={{ background: 'transparent' }}>
                    <tbody>
                      {Object.entries(info).map(([k, v]) => (
                        <tr key={k}>
                          <td className="small fw-bold text-capitalize" style={{ width: '40%', border: 'none' }}>
                            {k.replace(/_/g, ' ')}
                          </td>
                          <td className="small" style={{ border: 'none' }}>{v}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Pathogenic variants */}
      <SectionCard title="🧬 Pathogenic Variants in MT-ND4 (mtDNA rCRS)" borderColor={COLOR}>
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
                <tr key={i} style={{ background: v.phenotype.includes('LHON') ? '#fce4ec' : v.phenotype.includes('Leigh') ? '#fff3e0' : 'inherit' }}>
                  <td className="fw-bold small" style={{ color: v.phenotype.includes('LHON') ? COLOR3 : COLOR }}>{v.hgvs_mtdna}</td>
                  <td className="small">{v.protein}</td>
                  <td className="small">{v.domain}</td>
                  <td className="small">{v.type}</td>
                  <td className="small">{v.phenotype}</td>
                  <td className="small" style={{
                    color: v.severity === 'Severe' || v.severity === 'LHON-primary' ? COLOR3 :
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
            <div key={i} className="small mb-2 p-2 rounded" style={{
              background: v.phenotype.includes('LHON') ? '#fce4ec' : LIGHT,
              borderLeft: `3px solid ${v.phenotype.includes('LHON') ? COLOR3 : COLOR}`,
            }}>
              <strong style={{ color: v.phenotype.includes('LHON') ? COLOR3 : COLOR }}>{v.hgvs_mtdna}</strong>: {v.notes}
            </div>
          ))}
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Variant Frequency Distribution (cohort)" borderColor={COLOR}>
            {vdist.map(v => (
              <Bar key={v.variant} label={v.variant} value={v.freq_pct}
                color={v.variant.includes('11778') ? COLOR3 : COLOR} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="CI Activity Distribution" borderColor={COLOR4}>
            <Bar label="CI <10% (severe Leigh)" value={biochem.ci_below_10_pct} color={COLOR3} />
            <Bar label="CI 10–25% (Leigh range)" value={biochem.ci_10_to_25_pct} color={COLOR4} />
            <Bar label="CI 25–50% (CPEO/mild)" value={biochem.ci_25_to_50_pct} color={COLOR2} />
            <Bar label="CI >50% (LHON systemic normal)" value={biochem.ci_above_50_pct} color={COLOR5} />
            <hr />
            <p className="small text-muted mb-1 fw-semibold">Lactate distribution:</p>
            <Bar label="Lactate normal &lt;2.5 mM (LHON)" value={biochem.lactic_normal_lhon_pct} color={COLOR5} />
            <Bar label="Lactate 2.5–5 mM (mild elevation)" value={biochem.lactic_mild_pct} color={COLOR2} />
            <Bar label="Lactate 5–10 mM (moderate)" value={biochem.lactic_moderate_pct} color={COLOR4} />
            <Bar label="Lactate >10 mM (severe Leigh)" value={biochem.lactic_severe_pct} color={COLOR3} />
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Heteroplasmy Distribution (blood)" borderColor={COLOR3}>
        <div className="row">
          {[
            ['Near-homoplasmic ≥90% (LHON range)', 'het_near_homoplasmic_pct', COLOR3],
            ['80–90% (Leigh severe threshold)', 'het_80_to_90_pct', COLOR4],
            ['60–80% (Leigh moderate)', 'het_60_to_80_pct', COLOR2],
            ['<60% (subclinical/oligosymptomatic)', 'het_below_60_pct', COLOR5],
          ].map(([label, key, color]) => (
            <div key={key} className="col-md-6 mb-2">
              <Bar label={label} value={biochem[key] || 0} color={color} />
            </div>
          ))}
        </div>
        <p className="small text-muted mt-2">
          ⚠️ LHON: near-homoplasmic in blood ({'>'}95%) but incomplete penetrance (50% lifetime risk males, 10% females).
          Leigh: blood heteroplasmy may underestimate muscle by 10–20 ppts.
          WES does NOT detect mtDNA variants — dedicated mtDNA sequencing required.
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
                       o.outcome.includes('severe') ? COLOR3 :
                       o.outcome.includes('partial') || o.outcome.includes('gene') ? COLOR5 :
                       o.outcome.includes('moderate') ? COLOR4 :
                       o.outcome.includes('CPEO') ? COLOR5 : COLOR} />
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Treatment uptake */}
      {data.treatment_uptake && (
        <SectionCard title="Treatment Uptake (cohort — LHON + Leigh combined)" borderColor={COLOR5}>
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
      <SectionCard title="🔍 Differential Diagnosis — MT-ND4 vs Key Mimics" borderColor={COLOR3}>
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
          <SectionCard title="🚫 Absolute Contraindications — LHON" borderColor={COLOR3}>
            {[
              { drug: 'Tobacco (all forms)', reason: 'Strongest precipitating trigger for LHON vision loss; cyanide is CI inhibitor; even passive smoking dangerous in near-homoplasmic carriers' },
              { drug: 'Ethambutol', reason: 'Optic nerve mitochondrial toxin; synergistic with MT-ND4 CI deficiency in RGCs; irreversible blindness; always screen for LHON before TB treatment' },
              { drug: 'Alcohol', reason: 'Acetaldehyde is a CI inhibitor and RGC mitochondrial toxin; precipitates LHON onset in near-homoplasmic carriers' },
              { drug: 'Amiodarone', reason: 'Drug-induced optic neuropathy (DION) synergistic with LHON — catastrophic bilateral blindness' },
              { drug: 'Metformin', reason: 'Complex I inhibitor — additive CI burden on already vulnerable optic nerve; fatal lactic acidosis in Leigh phenotype' },
              { drug: 'Linezolid', reason: 'Inhibits mt-23S rRNA → reduces MT-ND4 synthesis directly; worsens CI in both phenotypes' },
            ].map((c, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#ffebee', borderLeft: `3px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>🚫 {c.drug}</div>
                <div className="small">{c.reason}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🚫 Absolute Contraindications — Leigh/CI Deficiency" borderColor={COLOR4}>
            {[
              { drug: 'Metformin', reason: 'Complex I inhibitor — additive CI failure → fatal lactic acidosis in Leigh phenotype' },
              { drug: 'Valproic acid (VPA)', reason: 'CoA sequestration + POLG inhibition + OXPHOS impairment; hepatotoxicity risk; NOT safe in Leigh' },
              { drug: 'Propofol (PRIS)', reason: 'Directly inhibits CI — compounding MT-ND4 CI deficiency → fatal; use sevoflurane' },
              { drug: 'IV tPA (stroke-like)', reason: 'Stroke-like episodes are METABOLIC not thrombotic — tPA causes harm; treat with thiamine + hydration' },
              { drug: 'Ketogenic diet (severe Leigh)', reason: 'High FADH2 from β-oxidation requires intact CI-dependent OXPHOS; CI deficiency → metabolic crisis' },
              { drug: 'Fasting (any duration)', reason: 'NEVER fast in Leigh phenotype; fasting triggers CI crisis; GIR 6–8 mg/kg/min mandatory' },
            ].map((c, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fff3e0', borderLeft: `3px solid ${COLOR4}` }}>
                <div className="fw-bold small" style={{ color: COLOR4 }}>🚫 {c.drug}</div>
                <div className="small">{c.reason}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="✅ LHON-Specific Treatments" borderColor={COLOR3}>
            {[
              { drug: 'Idebenone (Raxone) 900 mg/day', level: 'Level B', note: 'RHODOS trial (Klopstock 2011 Brain); only RCT evidence in LHON; start early; continue ≥24 months; bypasses CI → CII→CIII→CIV' },
              { drug: 'Lenadogene nolparvovec (Lumevoq)', level: 'Gene therapy (EU 2021)', note: 'AAV2-MT-ND4 intravitreal; m.11778G>A ONLY; bilateral injection; RESCUE+REVERSE trials; refer to specialist centre' },
              { drug: 'CoQ10 / Ubiquinol', level: 'Level C', note: 'Short-chain electron shuttling; ubiquinol preferred; 10–20 mg/kg/day' },
              { drug: 'Riboflavin (B2)', level: 'Level C', note: 'FAD-dependent CI assembly co-factors; empiric supplementation' },
              { drug: 'Low vision rehabilitation', level: 'Level A', note: 'Magnifiers, screen readers, mobility training; essential even when visual prognosis poor' },
            ].map((t, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fce4ec', borderLeft: `3px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>✅ {t.drug} <span className="badge ms-1" style={{ background: COLOR3, fontSize: '0.65em' }}>{t.level}</span></div>
                <div className="small">{t.note}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="✅ Leigh/CI Deficiency Treatments" borderColor={COLOR5}>
            {[
              { drug: 'Thiamine (B1) IV/oral', level: 'Mandatory empiric (Leigh)', note: '10–20 mg/kg IV in acute decompensation; PDH cofactor; CI-associated' },
              { drug: 'GIR 6–8 mg/kg/min dextrose', level: 'Mandatory (Leigh)', note: 'NEVER fast; continuous dextrose prevents CI crisis; increase during illness/surgery' },
              { drug: 'CoQ10 / Ubiquinol', level: 'Level C', note: '10–20 mg/kg/day; CI-to-CIII electron carrier; ubiquinol preferred' },
              { drug: 'NaHCO₃ IV (acute)', level: 'Acute rescue', note: 'Target pH >7.2; 1–2 mEq/kg IV; lactic acidosis crisis' },
              { drug: 'LEV (Levetiracetam)', level: 'Preferred AED', note: 'Renal clearance; no mitochondrial toxicity; avoid VPA/PHT/CBZ' },
              { drug: 'Sevoflurane (anaesthesia)', level: 'Preferred (both phenotypes)', note: 'Over Propofol; all MT-ND4 patients; inform anaesthesiologist' },
            ].map((t, i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#e8f5e9', borderLeft: `3px solid ${COLOR5}` }}>
                <div className="fw-bold small" style={{ color: COLOR5 }}>✅ {t.drug} <span className="badge ms-1" style={{ background: COLOR5, fontSize: '0.65em' }}>{t.level}</span></div>
                <div className="small">{t.note}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🏥 Diagnostic Workup — MT-ND4 (LHON + Leigh CI Deficiency)" borderColor={COLOR2}>
        <ol className="small mb-0">
          {[
            'Detailed ophthalmological exam (LHON): visual acuity, Ishihara/Farnsworth D-15 (red-green loss), Goldmann visual field (centrocaecal scotoma), fundoscopy (disc hyperaemia, peripapillary microangiopathy)',
            'Fluorescein angiography (FFA) — KEY DDx: LHON microangiopathy does NOT leak on FFA; NAION leaks (ischemic disc edema); perform in all acute optic neuropathy',
            'OCT (optical coherence tomography): RNFL thinning in LHON; monitors disease progression and treatment response',
            'Dedicated mtDNA sequencing: targeted m.11778G>A testing first (90% of LHON, 65% of MT-ND4 disease); if negative → full MT-ND4 + MT-ND1 + MT-ND6 panel; WES does NOT cover mtDNA reliably',
            'Blood heteroplasmy quantification: LHON near-homoplasmic (>95%); Leigh: muscle biopsy preferred for accurate heteroplasmy (blood may underestimate by 10–20 ppts)',
            'Maternal family cascade testing: all maternal relatives for heteroplasmy; lifestyle counselling for near-homoplasmic carriers (tobacco, alcohol, ethambutol, amiodarone avoidance)',
            'Brain MRI: Leigh phenotype → bilateral symmetric T2 hyperintensity putamen/brainstem; LHON → typically normal; cortical if MELAS overlap',
            'Plasma lactate + pyruvate + L:P ratio: LHON → normal; Leigh → elevated lactate (>2.5 mM), L:P >20',
            'OXPHOS enzyme activities (muscle or fibroblasts — Leigh only): isolated CI deficiency (CI 10–25%); CII/CIII/CIV normal',
            'BN-PAGE: LHON → CI near-normal; Leigh → CI severely reduced + ND4-lacking sub-complex',
            'Immunoblot: ND4 near-normal in LHON; ND4 reduced/absent in Leigh; secondary ND5/ND2/ND4L reduction in Leigh',
          ].map((step, i) => (
            <li key={i} className="mb-1">{step}</li>
          ))}
        </ol>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="👁️ LHON Acute Management" borderColor={COLOR3}>
            {[
              { step: '1. Stop tobacco immediately — irreversible CI inhibitor + precipitating trigger', color: COLOR3 },
              { step: '2. Stop alcohol — acetaldehyde toxin in at-risk tissue', color: COLOR3 },
              { step: '3. Start idebenone 900 mg/day ASAP — do not wait for bilateral involvement', color: COLOR3 },
              { step: '4. Refer to LHON gene therapy centre (m.11778G>A: lenadogene nolparvovec)', color: COLOR3 },
              { step: '5. LOW VISION REFERRAL — mandatory regardless of prognosis', color: COLOR5 },
              { step: '6. Maternal family testing + counselling (tobacco/ethambutol/amiodarone)', color: COLOR },
              { step: '7. Second eye monitoring weekly — 97% bilateral within 6–8 weeks', color: COLOR4 },
            ].map((s, i) => (
              <div key={i} className="mb-2 small p-2 rounded fw-semibold"
                style={{ background: '#fce4ec', borderLeft: `3px solid ${s.color}`, color: s.color }}>
                {s.step}
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🧠 Leigh Acute Crisis Management" borderColor={COLOR4}>
            {[
              { step: '1. IV thiamine 10–20 mg/kg IMMEDIATELY', color: COLOR4 },
              { step: '2. GIR 6–8 mg/kg/min (10% dextrose) — NEVER fast', color: COLOR4 },
              { step: '3. NaHCO₃ 1–2 mEq/kg if pH <7.2', color: COLOR4 },
              { step: '4. NIV/BiPAP for respiratory support — avoid Propofol', color: COLOR4 },
              { step: '5. Seizures → LEV IV (NOT VPA); benzodiazepines for acute status', color: COLOR },
              { step: '6. Stroke-like: NO tPA; IV thiamine + hydration + avoid fasting', color: COLOR4 },
              { step: '7. Metabolic genetics + neurology consult URGENTLY', color: COLOR5 },
            ].map((s, i) => (
              <div key={i} className="mb-2 small p-2 rounded fw-semibold"
                style={{ background: '#fff3e0', borderLeft: `3px solid ${s.color}`, color: s.color }}>
                {s.step}
              </div>
            ))}
          </SectionCard>
        </div>
      </div>
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
                <tr key={i} style={{ background: c.severity.includes('LHON') ? '#fce4ec' : 'inherit' }}>
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
export default function MtND4Page() {
  const [tab, setTab] = useState(0);
  const [overviewData, setOverviewData] = useState(null);
  const [variantData,  setVariantData]  = useState(null);
  const [defsData,     setDefsData]     = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/mtnd4/overview`).then(r => r.json()),
      fetch(`${API}/api/mtnd4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtnd4/definitions`).then(r => r.json()),
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
          4
        </div>
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
            MT-ND4 — LHON (#1 worldwide) + Isolated CI Deficiency / Leigh Syndrome
          </h4>
          <p className="text-muted mb-0 small">
            Central Antiporter Module · 459 aa / 13 TM helices · Dual Phenotype (LHON near-homoplasmic / Leigh high-heteroplasmy) · Maternal Inheritance · 40-patient cohort (seed 739)
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
          {tab === 1 && <LhonVsLeighTab data={variantData} />}
          {tab === 2 && <DdxTab data={variantData} />}
          {tab === 3 && <DefinitionsTab data={defsData} />}
        </>
      )}
    </div>
  );
}
