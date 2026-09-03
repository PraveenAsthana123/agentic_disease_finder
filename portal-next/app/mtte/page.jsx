'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#004d40';   // deep teal-green — tRNA-Glu / CPEO+MIDM
const LIGHT  = '#e0f2f1';
const COLOR2 = '#00695c';   // medium teal — CPEO / cardiomyopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / severe
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#6a1b9a';   // deep purple — reversible neonatal / MIDM

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
  const hmap = data.heteroplasmy_clinical_map || [];

  return (
    <div>
      {/* Gene header */}
      <div className="p-3 mb-4 rounded" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <div className="d-flex flex-wrap align-items-start gap-3">
          <div>
            <h4 className="fw-bold mb-1" style={{ color: COLOR }}>MT-TE — tRNA-Glu</h4>
            <div className="text-muted small">
              <span className="badge me-1" style={{ background: COLOR }}>OMIM *590025</span>
              <span className="badge me-1" style={{ background: COLOR2 }}>Combined CI+CIV Deficiency</span>
              <span className="badge me-1 bg-dark">Glu codons: GAA/GAG · CUC anticodon</span>
              <span className="badge me-1" style={{ background: COLOR5 }}>CPEO / MIDM / Reversible-Neonatal</span>
            </div>
            <p className="mt-2 mb-0 small">
              MT-TE encodes mitochondrial tRNA-Glu (CUC anticodon, 69 nt) — <strong>L-strand</strong> rCRS 14674–14742 —
              between MT-ND6 (ends 14673, L-strand) and MT-CYB (starts 14747, H-strand).
              Mutations cause <strong>combined CI + CIV deficiency</strong> (mt-translation fingerprint: CII NORMAL).
              <strong className="text-danger"> L-STRAND ENCODED</strong> — verify L-strand NGS coverage to avoid false-negative ·
              <strong style={{ color: COLOR5 }}> MIDM (maternally inherited diabetes)</strong> — one of only two mt-tRNA DM genes (other: MT-TL1 MIDD) ·
              <strong style={{ color: COLOR5 }}> m.14674T&gt;C = REVERSIBLE infantile COX deficiency</strong> — unique: neonatal crisis → near-normal adult in ~60%.
            </p>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Patients" value={s.n_patients} />
        <KPI label="Mean Heteroplasmy (blood)" value={`${s.avg_heteroplasmy_blood_pct}%`} color={COLOR2} />
        <KPI label="Mean CI Activity" value={`${s.avg_ci_activity_pct_normal}%`} color={COLOR3} />
        <KPI label="Mean CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR3} />
        <KPI label="CII (NORMAL)" value={`${s.avg_cii_activity_pct_normal}%`} color={COLOR4} />
        <KPI label="CPEO" value={`${s.pct_cpeo}%`} color={COLOR2} />
        <KPI label="Myopathy" value={`${s.pct_myopathy}%`} color={COLOR} />
        <KPI label="Diabetes (MIDM)" value={`${s.pct_diabetes_mellitus}%`} color={COLOR5} />
        <KPI label="Cardiomyopathy" value={`${s.pct_cardiomyopathy}%`} color={COLOR3} />
        <KPI label="SNHL" value={`${s.pct_snhl}%`} color={COLOR2} />
        <KPI label="Reversible CIV" value={`${s.pct_reversible_civ_deficiency}%`} color={COLOR5} />
        <KPI label="Mean Onset (yr)" value={s.avg_age_onset_yr} />
      </div>

      {/* Two unique MT-TE features */}
      <SectionCard title="Two Unique MT-TE Signatures — MIDM + Reversible Neonatal COX Deficiency" borderColor={COLOR5}>
        <div className="row">
          <div className="col-md-6">
            <div className="p-3 rounded mb-2" style={{ background: '#f3e5f5', border: '2px solid #6a1b9a' }}>
              <div className="fw-bold small" style={{ color: '#6a1b9a' }}>MIDM — Maternally Inherited Diabetes Mellitus (m.14709T&gt;C)</div>
              <div className="small mt-1">Beta-cell mitochondrial ATP failure → impaired GSIS → DM</div>
              <div className="small">Presents as 'type 1.5' or 'atypical type 2 DM' with maternal history of DM + myopathy</div>
              <div className="small text-danger fw-bold mt-1">METFORMIN ABSOLUTE CI — use INSULIN</div>
              <div className="small text-muted">Same DM mechanism as MT-TL1 MIDD; key DDx = presence/absence of stroke-like episodes</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded mb-2" style={{ background: '#e8f5e9', border: '2px solid #2e7d32' }}>
              <div className="fw-bold small" style={{ color: '#2e7d32' }}>Reversible Infantile COX Deficiency (m.14674T&gt;C/G) — UNIQUE</div>
              <div className="small mt-1">Neonatal lactic acidosis + hypotonia + respiratory failure → spontaneous improvement</div>
              <div className="small">~60% achieve near-normal adult function by school age (nuclear EF-Tu/TARS2 compensation)</div>
              <div className="small text-danger fw-bold mt-1">DO NOT WITHDRAW NICU SUPPORT — improvement expected</div>
              <div className="small text-muted">Only mt-tRNA mutation with documented spontaneous biochemical normalisation</div>
            </div>
          </div>
        </div>
        <p className="small text-muted mb-0 mt-2">
          Both features are UNIQUE to MT-TE among all 22 mt-tRNA genes.
          MT-TE should be on the differential for any maternal DM + myopathy, or for unexplained neonatal COX deficiency with recovery.
        </p>
      </SectionCard>

      {/* L-strand sequencing alert */}
      <SectionCard title="L-Strand Encoding — NGS Coverage Pitfall" borderColor={COLOR3}>
        <div className="p-3 rounded" style={{ background: '#fce4ec', border: `1px solid ${COLOR3}` }}>
          <div className="fw-bold small mb-2" style={{ color: COLOR3 }}>⚠ MT-TE IS L-STRAND ENCODED — VERIFY NGS COVERAGE</div>
          <div className="small">
            MT-TE (rCRS 14674–14742) is encoded on the <strong>LIGHT (L) strand</strong>, like MT-ND6.
            H-strand-dominant capture panels (e.g., TruSight Mitochondrial, SureSelect mtDNA) may have reduced read depth at this position.
            Before reporting MT-TE as negative, confirm adequate L-strand coverage in the sequencing QC report.
            <strong> False-negative MT-TE reports are a known clinical pitfall</strong> that delays diagnosis of MIDM and reversible neonatal COX deficiency.
          </div>
          <div className="small mt-2 text-muted">
            Adjacent L-strand genes: MT-ND6 (14149–14673) and MT-TQ, MT-TA (elsewhere); H-strand flanks: MT-CYB (14747–15887).
          </div>
        </div>
      </SectionCard>

      {/* Heteroplasmy map */}
      <SectionCard title="Heteroplasmy–Phenotype Map (m.14709T>C; blood underestimates by 10–15%)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Blood heteroplasmy / Mutation</th><th>Clinical phenotype</th><th>Management</th>
            </tr></thead>
            <tbody>
              {hmap.map((r, i) => (
                <tr key={i} style={{ background: r.range.includes('14674') ? '#e8f5e9' : 'inherit' }}>
                  <td className="fw-bold">{r.range}</td>
                  <td>{r.phenotype}</td>
                  <td className="text-muted">{r.management}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts — Contraindications" borderColor={COLOR3}>
        <div className="row">
          {alerts.map((a, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#fce4ec', border: '1px solid #c62828' }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{a.alert}</div>
                <div className="small text-muted">{a.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="Phenotype Distribution (40-patient cohort, seed-797)" borderColor={COLOR}>
        {pheno_dist.map((p, i) => (
          <Bar key={i} label={`${p.phenotype} (n=${p.count})`} value={p.pct} color={COLOR} />
        ))}
      </SectionCard>

      {/* Molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR4}>
        <ul className="small mb-0">
          {mol_feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Cohort summary */}
      <SectionCard title="Cohort Summary" borderColor={COLOR2}>
        <ul className="small mb-0">
          {feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Cohort ─────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vs = data.variant_summaries || [];
  const pts = data.per_patient || [];
  const triggers = data.trigger_rates || [];
  const txs = data.treatment_info || [];
  const bf = data.biochemical_fingerprint || {};

  return (
    <div>
      {/* Variant summaries */}
      <SectionCard title="Variant Summaries by Mutation" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Variant</th><th>Region</th><th>N</th>
              <th>Het%</th><th>CI%</th><th>CIV%</th>
              <th>CPEO%</th><th>Myopathy%</th><th>DM%</th><th>Cardio%</th><th>RevCIV%</th>
            </tr></thead>
            <tbody>
              {vs.map((v, i) => (
                <tr key={i} style={{ background: v.variant === 'm.14674T>C' ? '#e8f5e9' : 'inherit' }}>
                  <td className="fw-bold">{v.variant}</td>
                  <td>{v.region}</td>
                  <td>{v.n}</td>
                  <td>{v.avg_heteroplasmy_blood_pct}</td>
                  <td style={{ color: COLOR3 }}>{v.avg_ci_activity_pct}</td>
                  <td style={{ color: COLOR3 }}>{v.avg_civ_activity_pct}</td>
                  <td>{v.pct_cpeo}%</td>
                  <td>{v.pct_myopathy}%</td>
                  <td style={{ color: COLOR5 }}>{v.pct_diabetes_mellitus}%</td>
                  <td>{v.pct_cardiomyopathy}%</td>
                  <td style={{ color: '#2e7d32' }}>{v.pct_reversible_civ}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          {vs.map((v, i) => (
            <div key={i} className="mb-2 p-2 rounded" style={{ background: v.variant === 'm.14674T>C' ? '#e8f5e9' : LIGHT }}>
              <span className="fw-bold small me-2">{v.variant}</span>
              <span className="text-muted small">{v.note}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint (BN-PAGE / Spectrophotometry)" borderColor={COLOR4}>
        <div className="row">
          {[
            { label: 'CI Activity', val: bf.CI_pct_normal, color: COLOR3 },
            { label: 'CIV Activity', val: bf.CIV_pct_normal, color: COLOR3 },
            { label: 'CII Activity (NORMAL)', val: bf.CII_pct_normal, color: COLOR4 },
          ].map((x, i) => (
            <div className="col-md-4 mb-2" key={i}>
              <div className="p-2 rounded text-center" style={{ background: LIGHT }}>
                <div className="fw-bold" style={{ color: x.color }}>{x.val}%</div>
                <div className="small text-muted">{x.label}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="small mt-2">
          <div><strong>Pattern:</strong> {bf.pattern}</div>
          <div><strong>BN-PAGE:</strong> {bf.BN_PAGE}</div>
          <div><strong>Histochem:</strong> {bf.muscle_histochemistry}</div>
          {bf.L_strand_NGS_note && (
            <div className="mt-2 p-2 rounded" style={{ background: '#fce4ec' }}>
              <strong style={{ color: COLOR3 }}>⚠ NGS Note:</strong> {bf.L_strand_NGS_note}
            </div>
          )}
        </div>
      </SectionCard>

      {/* Per-patient table (first 20) */}
      <SectionCard title="Per-Patient Cohort Data (40 patients, seed-797 — first 20 shown)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>ID</th><th>Variant</th><th>Sex</th><th>Onset</th>
              <th>Het%</th><th>CI%</th><th>CIV%</th><th>CII%</th>
              <th>Lactate</th><th>CPEO</th><th>Myo</th><th>DM</th><th>Cardio</th><th>RevCIV</th><th>RRF</th>
            </tr></thead>
            <tbody>
              {pts.slice(0, 20).map((p, i) => (
                <tr key={i} style={{ background: p.reversible_civ ? '#e8f5e9' : 'inherit' }}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.variant}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_yr === 0 ? 'Neo' : `${p.age_onset_yr}yr`}</td>
                  <td>{p.heteroplasmy_blood_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.ci_pct}</td>
                  <td style={{ color: COLOR3 }}>{p.civ_pct}</td>
                  <td style={{ color: COLOR4 }}>{p.cii_pct}</td>
                  <td>{p.lactate_mmol_L}</td>
                  <td>{p.cpeo ? '✓' : '—'}</td>
                  <td>{p.myopathy ? '✓' : '—'}</td>
                  <td style={{ color: COLOR5 }}>{p.diabetes_mellitus ? '✓' : '—'}</td>
                  <td>{p.cardiomyopathy ? '✓' : '—'}</td>
                  <td style={{ color: '#2e7d32' }}>{p.reversible_civ ? '↑Rev' : '—'}</td>
                  <td>{p.ragged_red_fibres ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Trigger rates */}
      <SectionCard title="Crisis Trigger Rates" borderColor={COLOR3}>
        {triggers.map((t, i) => (
          <Bar key={i} label={t.trigger} value={t.pct} color={COLOR3} />
        ))}
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Treatment Summary" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Agent</th><th>Evidence</th><th>Note</th>
            </tr></thead>
            <tbody>
              {txs.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.agent}</td>
                  <td><span className="badge" style={{ background: COLOR }}>{t.evidence}</span></td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Management ──────────────────────────────────────────────────────
function DDxTab({ bdata }) {
  if (!bdata) return <p className="text-muted">Loading…</p>;
  const ddx = bdata.ddx_comparison || [];
  const ci_info = bdata.contraindication_info || [];

  return (
    <div>
      {/* DDx table */}
      <SectionCard title="Differential Diagnosis — MT-TE vs Key Mimics" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Gene / Disease</th><th>Primary Disease</th><th>OXPHOS</th><th>Key Distinguisher from MT-TE</th>
            </tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} style={{ background: d.gene.includes('MT-TL1') ? '#fff8e1' : 'inherit' }}>
                  <td className="fw-bold">{d.gene}</td>
                  <td>{d.disease}</td>
                  <td>{d.oxphos}</td>
                  <td className="text-muted">{d.distinguisher}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small text-warning fw-bold mt-2 mb-0">
          ⚠ MT-TL1 (MIDD/MELAS) is the MOST CRITICAL DDx — both MT-TL1 and MT-TE cause maternal DM.
          MT-TL1 = MIDD (DM+deafness) / MELAS (stroke-like); MT-TE = MIDM (DM+CPEO+myopathy); NO stroke-like episodes in MT-TE.
        </p>
      </SectionCard>

      {/* Contraindication details */}
      <SectionCard title="Contraindications — Detailed Rationale" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Agent</th><th>Category</th><th>Rationale</th>
            </tr></thead>
            <tbody>
              {ci_info.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold">{c.agent}</td>
                  <td><span className="badge" style={{ background: c.category.includes('ABSOLUTE') ? COLOR3 : '#e65100' }}>{c.category}</span></td>
                  <td className="text-muted">{c.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* MIDM management */}
      <SectionCard title="MIDM Management — Maternally Inherited Diabetes Mellitus" borderColor={COLOR5}>
        <div className="row">
          {[
            { title: 'Insulin (basal-bolus)', text: 'First-line for MT-TE MIDM — pancreatic beta-cell mitochondrial failure; basal-bolus regimen; HbA1c target <7%' },
            { title: 'METFORMIN ABSOLUTE CI', text: 'Complex I inhibitor → fatal lactic acidosis in CI-deficient skeletal muscle; even small doses contraindicated; inform ALL MT-TE carriers including subclinical' },
            { title: 'DPP4 inhibitors', text: 'Sitagliptin, saxagliptin — cautiously used (limited mt-toxicity evidence); monitor lactate; do not use if renal impairment' },
            { title: 'SGLT2 inhibitors', text: 'Investigational — euglycaemic DKA risk in mt-disease; caution; not yet standard for MT-TE MIDM' },
            { title: 'Endocrinology co-management', text: 'Joint neurology + endocrinology care mandatory; annual HbA1c, fasting glucose; DM screening in all maternal relatives' },
            { title: 'Maternal family screening', text: 'All maternal relatives need DM + myopathy screening; cascade mtDNA testing; early MIDM diagnosis prevents metformin harm' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#f3e5f5', border: `1px solid ${COLOR5}` }}>
                <div className="fw-bold small" style={{ color: COLOR5 }}>{item.title}</div>
                <div className="small text-muted">{item.text}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Reversible neonatal management */}
      <SectionCard title="Reversible Infantile COX Deficiency — m.14674T>C NICU Protocol" borderColor={COLOR4}>
        <div className="row">
          {[
            { step: 'DO NOT WITHDRAW SUPPORT', detail: '~60% of m.14674T>C patients achieve near-normal adult function — support improves outcome; false impression of irreversibility leads to inappropriate withdrawal' },
            { step: 'Respiratory support', detail: 'NICU ventilation + BiPAP as needed; spontaneous improvement in CIV activity over months to years; track serial ABG and lactate' },
            { step: 'Prevent fasting — NG feeds', detail: 'Continuous NG or TPN; GIR 6–8 mg/kg/min; NO fasting → triggers acute lactic crisis in neonatal period' },
            { step: 'IV Thiamine empiric', detail: '10–20 mg/kg/dose IV; PDH cofactor; empiric while workup proceeding; BTBGD exclusion first' },
            { step: 'Serial CIV measurement', detail: 'Repeat muscle CIV activity every 12–18 months to track recovery; blood CIV unreliable; document normalisation trajectory' },
            { step: 'Genetic counselling — maternal', detail: 'Confirm maternal heteroplasmy; cascade testing of siblings; recurrence risk discussion with parents; heteroplasmy segregation modelling' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#e8f5e9', border: `1px solid #2e7d32` }}>
                <div className="fw-bold small" style={{ color: '#2e7d32' }}>{item.step}</div>
                <div className="small text-muted">{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Emergency protocol */}
      <SectionCard title="Acute Crisis Protocol" borderColor={COLOR3}>
        <div className="row">
          {[
            { step: '1. GIR 6–8 mg/kg/min', detail: 'IV dextrose — NEVER fast; maintain normoglycaemia; prevents catabolism and lactate surge' },
            { step: '2. IV Thiamine 10-20 mg/kg', detail: 'Empiric — PDH/KGDH cofactor; give BEFORE glucose if Wernicke risk; continue oral 100-300 mg/day maintenance' },
            { step: '3. Avoid Propofol', detail: 'Use sevoflurane — PRIS risk amplified in CI+CIV deficiency; especially critical in m.14674T>C neonates' },
            { step: '4. Bicarbonate if pH <7.2', detail: 'Target pH >7.2; do not fully correct lactic acidosis; treat underlying trigger; isotonic bicarbonate preferred' },
            { step: '5. STOP Metformin immediately', detail: 'If patient on metformin (MIDM misdiagnosed as type 2 DM) — stop immediately; switch to insulin; monitor lactate hourly' },
            { step: '6. Use LEV for seizures', detail: 'VPA ABSOLUTE CI; LEV first-line; if already on VPA → taper rapidly, transition to LEV + bicarb cover for LA risk' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#fce4ec', border: `1px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{item.step}</div>
                <div className="small text-muted">{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const gb = data.gene_biology || {};
  const ct = data.clinical_terms || {};
  const ph = data.pharmacology || {};
  const refs = data.key_references || [];

  return (
    <div>
      <SectionCard title="Gene Biology" borderColor={COLOR4}>
        {Object.entries(gb).map(([k, v], i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR4 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Terms" borderColor={COLOR2}>
        {Object.entries(ct).map(([k, v], i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR2 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Pharmacology" borderColor={COLOR3}>
        {ph.preferred_aed && (
          <div className="mb-2">
            <span className="fw-bold small">Preferred AED: </span>
            <span className="small">{ph.preferred_aed}</span>
          </div>
        )}
        {ph.dm_management && (
          <div className="mb-2">
            <span className="fw-bold small" style={{ color: COLOR5 }}>MIDM / DM Management: </span>
            <span className="small">{ph.dm_management}</span>
          </div>
        )}
        {ph.emergency_protocol && (
          <div className="mb-2">
            <span className="fw-bold small">Emergency Protocol: </span>
            <span className="small">{ph.emergency_protocol}</span>
          </div>
        )}
        {ph.anaesthetic_guidance && (
          <div className="mb-2">
            <span className="fw-bold small">Anaesthetic Guidance: </span>
            <span className="small">{ph.anaesthetic_guidance}</span>
          </div>
        )}
        {ph.absolute_ci && (
          <div className="mt-3">
            <div className="fw-bold small mb-2" style={{ color: COLOR3 }}>Absolute Contraindications:</div>
            {Object.entries(ph.absolute_ci).map(([drug, reason], i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                <span className="fw-bold small me-2" style={{ color: COLOR3 }}>{drug}:</span>
                <span className="small text-muted">{reason}</span>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR}>
        <ol className="small mb-0">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function MTTEPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtte/overview`).then(r => r.json()),
      fetch(`${API}/api/mtte/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtte/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOverview(o);
      setBreakdown(b);
      setDefinitions(d);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: COLOR }}>
        🧬 MT-TE — tRNA-Glu Dashboard
      </h2>
      <p className="text-muted mb-3 small">
        Combined CI+CIV Deficiency · CPEO / Myopathy / MIDM (Maternally-Inherited-Diabetes) ·
        m.14709T&gt;C MIDM most common · m.14674T&gt;C REVERSIBLE-Infantile-COX-Deficiency-UNIQUE ·
        L-strand rCRS 14674–14742 · Between MT-ND6 (ends 14673) and MT-CYB (starts 14747)
      </p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DDxTab bdata={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
