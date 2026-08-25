'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Cohort & ER Stress', 'Treatment & Comparison', 'Definitions'];

// MODY10 colour scheme — deep purple/red (INS misfolding; ER stress; apoptosis; dominant-negative)
const ACCENT  = '#7b1fa2';   // deep purple — misfolded proinsulin; dominant-negative; UPR
const ACCENT2 = '#1565c0';   // deep blue — genetics; INS gene; OMIM; disulfide bonds
const ACCENT3 = '#c62828';   // deep red — insulin required; progressive apoptosis; structural loss
const ACCENT4 = '#e65100';   // deep orange — misdiagnosis T1D; ER stress overload; CHOP apoptosis
const ACCENT5 = '#880e4f';   // deep pink — R46Q founder; dominant-negative mechanism
const ACCENT6 = '#37474f';   // dark slate — epidemiology; no ethnic enrichment; global
const ACCENT7 = '#4a148c';   // deep violet — ER stress pathway; UPR; BiP; PERK/IRE1/ATF6
const ACCENT8 = '#bf360c';   // deep red-orange — comparison table; vs MODY8/9

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

function Alert({ color, children }) {
  return (
    <div className="alert mb-2" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function Badge({ text, color }) {
  return <span className="badge me-1" style={{ background: color, fontSize: '0.72em' }}>{text}</span>;
}

export default function MODY10Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody10/overview`).then(r => r.json()),
      fetch(`${API}/api/mody10/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody10/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const patients = overview?.patients || [];
  const keyFacts = overview?.key_facts || [];
  const alerts = overview?.alerts || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT2}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY10 — INS-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 10 · Chr 11p15.5 · OMIM #613370 · ~1% MODY · Dominant-Negative Misfolded Proinsulin · ER Stress UPR · Progressive Beta-Cell Apoptosis · Insulin Required · C-Peptide Falls · Antibody-Negative · R46Q Founder · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="INS *176730" color={ACCENT2} />
            <Badge text="Dom-neg ER stress" color={ACCENT7} />
            <Badge text="Insulin required" color={ACCENT3} />
            <Badge text="C-pep falls" color={ACCENT3} />
            <Badge text="R46Q founder" color={ACCENT5} />
            <Badge text="Global dist." color={ACCENT6} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Mean Duration (yr)" value={kpis.mean_duration_years?.toFixed(1)} color={ACCENT2} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(3)} color={ACCENT3} />
        <KPI label="Insulin Req (%)" value={`${kpis.pct_insulin_required?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Misdiagnosed (%)" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="T1D Misdiag (%)" value={`${kpis.pct_misdiagnosed_t1d?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT2} />
        <KPI label="Advanced Stage (%)" value={`${kpis.pct_advanced_stage?.toFixed(0)}%`} color={ACCENT7} />
        <KPI label="R46Q (%)" value={`${kpis.pct_r46q?.toFixed(0)}%`} color={ACCENT5} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab 0: Overview */}
      {tab === 0 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="🧬 Disease Overview" color={ACCENT}>
              <Alert color={ACCENT7}>
                <strong>MODY10 — INS DOMINANT-NEGATIVE MISFOLDED PROINSULIN → ER STRESS → BETA-CELL APOPTOSIS.</strong> Heterozygous INS missense mutations (especially those disrupting disulfide bond cysteines) produce misfolded mutant proinsulin that accumulates in the ER. Unlike simple haploinsufficiency, the mutant protein <em>actively triggers</em> chronic UPR (PERK/IRE1/ATF6), oxidative stress, and progressive beta-cell apoptosis — a dominant-negative mechanism.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>C-PEPTIDE FALLS PROGRESSIVELY — STRUCTURAL LOSS, NOT FUNCTIONAL DEFICIT.</strong> Unlike MODY9 (PAX4 haploinsufficiency → functional GSIS impairment, C-peptide preserved) or MODY2 (GCK → stable C-peptide over decades), MODY10 causes structural beta-cell apoptosis. C-peptide is detectable early (teens–late 20s) but declines decade-by-decade. Insulin is required in ~70–80% of patients; SU alone cannot arrest apoptotic progression.
              </Alert>
              <Alert color={ACCENT4}>
                <strong>MISDIAGNOSIS T1D ~40% — ANTIBODY-NEGATIVE RULES MODY10 IN.</strong> Young-onset DM + falling C-peptide + family history + antibody-negative = strong MODY10 signal. The most dangerous error: labelling as T1D and never testing INS. Expanded MODY NGS panel including INS resolves the diagnosis. De novo rate ~10–15% — absence of family history does not exclude MODY10.
              </Alert>
              <Alert color={ACCENT5}>
                <strong>DISTINGUISH MODY10 FROM PNDM-INS — ONSET AGE AND FAMILY HISTORY.</strong> Both are caused by INS mutations. PNDM-INS: de novo dominant or biallelic, severe misfolding, onset &lt; 6 months, very low C-peptide from birth. MODY10: familial heterozygous AD, dominant-negative, onset teens–40s, C-peptide present early.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>INS (Insulin; preproinsulin 110 aa) · Chr 11p15.5 · *176730</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#613370 (MODY10)</td></tr>
                  <tr><td className="fw-bold">Protein function</td><td>Preproinsulin → proinsulin (signal cleavage) → insulin (A+B chains, 3 disulfide bonds) + C-peptide. Mutations disrupt disulfide formation → misfolding → ER retention → UPR</td></tr>
                  <tr><td className="fw-bold">Mutation type</td><td>Heterozygous dominant-negative missense; disulfide-disrupting cysteine substitutions and core hydrophobic mutations; R46Q (A-chain), R89C (B-chain), C96Y (disulfide loop), L68M (B-chain core)</td></tr>
                  <tr><td className="fw-bold">Key mechanism</td><td>Dominant-negative misfolded proinsulin → ER stress (BiP sequesters → PERK/IRE1/ATF6 active) → chronic UPR → CHOP-mediated beta-cell apoptosis → progressive structural beta-cell loss → falling C-peptide</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% transmission · heterozygous dominant-negative → MODY10; de novo ~10–15%</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1% of all MODY; global distribution; no major ethnic enrichment</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Teens–early 40s (mean ~26–32 yr); overlaps MODY3; earlier than MODY7</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>FALLS progressively — structural apoptotic loss; early = preserved; late = low; unlike MODY9 (functional/preserved) or MODY2 (stable)</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — always; mandatory test to exclude T1D before INS sequencing</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🔬 INS Dominant-Negative ER Stress Pathway" color={ACCENT7}>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT7 + '10', border: `1px solid ${ACCENT7}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT7 }}>Misfolded Proinsulin → UPR → Apoptosis Cascade</div>
                <ol className="small mb-0">
                  <li><strong>Normal:</strong> Wild-type INS → correct disulfide bonds (A7-B7; A20-B19; A6-A11) → proinsulin folds → exits ER → Golgi → mature insulin secreted</li>
                  <li><strong>MODY10 mutation:</strong> Heterozygous missense (e.g. C96Y) → one allele produces misfolded proinsulin → cannot form correct disulfides → ER retention</li>
                  <li><strong>BiP sequestration:</strong> Misfolded proinsulin binds and sequesters BiP/GRP78 → BiP dissociates from PERK/IRE1/ATF6 → UPR activated (dominant-negative amplification)</li>
                  <li><strong>PERK arm:</strong> PERK phosphorylates eIF2α → reduces new proinsulin synthesis (protective short-term) → chronic activation → ATF4 → CHOP → BAX/BCL-2 → apoptosis</li>
                  <li><strong>IRE1 arm:</strong> Splices XBP1 mRNA → ERAD genes; chronic IRE1 → RIDD (degrades insulin mRNA) → secondary insulin deficiency</li>
                  <li><strong>ATF6 arm:</strong> Cleaved → activates ERAD and ER chaperones; eventual CHOP upregulation if ER stress unresolved</li>
                  <li><strong>Result:</strong> Progressive beta-cell mass loss → C-peptide decline → insulin dependency; structural (not functional) unlike MODY9</li>
                </ol>
              </div>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT4 + '10', border: `1px solid ${ACCENT4}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT4 }}>MODY10 Diagnostic Algorithm</div>
                <ol className="small mb-0">
                  <li>Young DM, antibody-negative (GADA/ZnT8/IA-2 all negative) → do NOT label T1D</li>
                  <li>Check: family history of DM (50% AD transmission); C-peptide (present early)</li>
                  <li>Progressive HbA1c + falling C-peptide + family history → high MODY10 pre-test probability</li>
                  <li>Order: expanded MODY NGS panel including INS (full coding + splice sites)</li>
                  <li>Novel variant → functional ER stress assay (CHOP-reporter; BiP induction; proinsulin-mCherry aggregation)</li>
                  <li>Confirmed MODY10 → cascade screen first-degree relatives; start insulin early (C-peptide monitoring q6–12 mo)</li>
                </ol>
              </div>
            </Section>
            <Section title="📋 Key Clinical Facts" color={ACCENT2}>
              <ul className="list-group list-group-flush">
                {keyFacts.map((f, i) => (
                  <li key={i} className="list-group-item py-1 small">{f}</li>
                ))}
              </ul>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 1: Cohort & ER Stress */}
      {tab === 1 && (
        <div className="row g-3">
          <div className="col-lg-5">
            <Section title="🔬 INS Variant Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td>
                          <span className="badge me-1" style={{ background: v.includes('R46Q') ? ACCENT5 : v.includes('C96Y') || v.includes('R89C') ? ACCENT3 : ACCENT2, fontSize: '0.72em' }}>{v}</span>
                          {v.includes('C96Y') && <span className="badge" style={{ background: ACCENT7, fontSize: '0.68em' }}>Strongest ER stress</span>}
                          {v.includes('R46Q') && <span className="badge" style={{ background: ACCENT5, fontSize: '0.68em' }}>Molven 2008 founder</span>}
                        </td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="🌍 Ethnicity Distribution" color={ACCENT6}>
              <div className="small text-muted mb-1">No major ethnic enrichment — global distribution (unlike MODY9 East Asian enrichment or MODY8 Norwegian)</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Ethnicity</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.ethnicity_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td>{v}</td></tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="⚠️ Prior Misdiagnosis" color={ACCENT4}>
              <div className="small text-muted mb-1">T1D misdiagnosis ~40%: antibody-negative DM + progressive HbA1c + family Hx → test INS</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Misdiagnosis</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.misdiagnosis_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-7">
            <Section title="📊 HbA1c Distribution (Progressive)" color={ACCENT}>
              <div className="small text-muted mb-1">Progressive HbA1c — NOT stable like MODY2 GCK; rises with duration as beta-cell mass falls</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>HbA1c Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 C-Peptide Distribution (FALLS — structural apoptosis)" color={ACCENT3}>
              <div className="small text-muted mb-1">C-peptide falls progressively — structural ER-stress apoptosis (unlike MODY9 functional/preserved). Low C-peptide ≠ T1D if antibody-negative.</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>C-Pep (nmol/L)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.c_peptide_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Disease Stage" color={ACCENT7}>
              <div className="small text-muted mb-1">Stage tracks C-peptide trajectory: early = preserved; intermediate = declining; advanced = insulin-dependent</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Stage</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.disease_stage_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔥 ER Stress Level by Variant" color={ACCENT7}>
              <div className="small text-muted mb-1">C96Y and R89C = highest ER stress (strong disulfide disruption); earliest insulin dependency</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>ER Stress</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.er_stress_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="⏱️ Disease Duration" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Duration</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.disease_duration_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 2: Treatment & Comparison */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="💊 Treatment Strategy" color={ACCENT3}>
              <Alert color={ACCENT3}>
                <strong>INSULIN REQUIRED IN ~70–80% — SU CANNOT ARREST STRUCTURAL APOPTOSIS.</strong> Unlike MODY1/3/6 where SU restores GSIS via a functional route, MODY10 beta-cell loss is <em>structural apoptotic</em> — SU (sulfonylurea) cannot regenerate apoptotic cells. Early-phase SU may augment residual GSIS transiently (C-peptide &gt; 0.30 nmol/L) but insulin is inevitable as beta-cell mass declines. CGM strongly recommended to guide basal-bolus titration.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.treatment || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔬 Genetics & Testing" color={ACCENT2}>
              <Alert color={ACCENT2}>
                <strong>EXPANDED MODY NGS PANEL REQUIRED — INS NOT IN OLDEST PANELS.</strong> Standard 4-gene panels (HNF1A/HNF4A/GCK/HNF1B) miss MODY10. Full INS coding sequence + splice sites required. Novel variants → functional ER stress assay (CHOP-reporter; BiP induction; proinsulin-mCherry aggregation microscopy). PNDM-INS differentiation: de novo or biallelic = PNDM; familial heterozygous AD = MODY10.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.genetics_testing || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT2 }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🔄 MODY8 vs MODY9 vs MODY10 Comparison" color={ACCENT8}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.comparison_mody8_9_10 || {}).map(([type, props]) => (
                    <tr key={type}>
                      <td className="fw-bold small" style={{ color: ACCENT8 }}>{type}</td>
                      <td className="small">
                        {typeof props === 'object' ? Object.entries(props).map(([k, v]) => (
                          <div key={k}><span className="fw-bold">{k.replace(/_/g, ' ')}:</span> {v}</div>
                        )) : props}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="💊 Treatment Distribution" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Treatment</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.treatment_distribution || {}).map(([k, v]) => (
                    <tr key={k}><td className="small">{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Summary Flags" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(breakdown?.summary_flags || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k.replace(/_/g, ' ')}</td><td>{typeof v === 'number' ? (k.includes('pct') ? `${v}%` : v) : v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔔 Clinical Alerts" color={ACCENT4}>
              {Object.entries(alerts).map(([k, v]) => (
                <Alert key={k} color={ACCENT4}><strong>{k.replace(/_/g, ' ')}:</strong> {v}</Alert>
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="📚 Disease Definition" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.disease || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🧬 Genes & Proteins" color={ACCENT2}>
              {Object.entries(definitions?.genes_and_proteins || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="fw-bold small" style={{ color: ACCENT2 }}>{k}</div>
                  <div className="small text-muted">{typeof v === 'object' ? JSON.stringify(v) : v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="📖 Clinical Terminology" color={ACCENT6}>
              {Object.entries(definitions?.clinical_terms || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <span className="badge me-1" style={{ background: ACCENT6 }}>{k}</span>
                  <span className="small text-muted">{v}</span>
                </div>
              ))}
            </Section>
            <Section title="🔬 Lab Thresholds" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.lab_thresholds || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT3 }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-12">
            <Section title="👥 Patient Sample" color={ACCENT6}>
              <div className="table-responsive" style={{ maxHeight: 320 }}>
                <table className="table table-sm table-bordered table-hover">
                  <thead>
                    <tr>
                      <th>ID</th><th>Sex</th><th>Age</th><th>Dx Age</th><th>HbA1c%</th>
                      <th>C-pep</th><th>BMI</th><th>Variant</th><th>ER Stress</th><th>Stage</th><th>Misdiagnosis</th><th>FamHx</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.slice(0, 15).map(p => (
                      <tr key={p.patient_id}>
                        <td className="small">{p.patient_id}</td>
                        <td>{p.sex}</td>
                        <td>{p.age}</td>
                        <td>{p.age_at_diagnosis}</td>
                        <td>{p.hba1c_percent}</td>
                        <td>{p.c_peptide_nmol_L}</td>
                        <td>{p.bmi_kg_m2}</td>
                        <td><span className="badge" style={{ background: p.variant?.includes('R46Q') ? ACCENT5 : p.variant?.includes('C96Y') || p.variant?.includes('R89C') ? ACCENT3 : ACCENT2, fontSize: '0.65em' }}>{p.variant}</span></td>
                        <td><span className="badge" style={{ background: p.er_stress_level === 'High (ER overload)' ? ACCENT7 : ACCENT6, fontSize: '0.65em' }}>{p.er_stress_level}</span></td>
                        <td><span className="badge" style={{ background: p.disease_stage?.includes('Advanced') ? ACCENT3 : p.disease_stage?.includes('Intermediate') ? ACCENT4 : ACCENT2, fontSize: '0.65em' }}>{p.disease_stage?.split(' ')[0]}</span></td>
                        <td>{p.prior_misdiagnosis !== 'None' ? <span className="badge" style={{ background: ACCENT4, fontSize: '0.65em' }}>{p.prior_misdiagnosis}</span> : '—'}</td>
                        <td>{p.family_history_positive ? '✓' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}
    </div>
  );
}
