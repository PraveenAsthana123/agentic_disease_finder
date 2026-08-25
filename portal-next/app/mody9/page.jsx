'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Cohort & KPD', 'Treatment & Comparison', 'Definitions'];

// MODY9 colour scheme — warm saffron/amber (PAX4; East Asian enrichment; KPD-DKA; ARX repression)
const ACCENT  = '#e65100';   // deep orange — PAX4 TF; ARX repression; East Asian enrichment
const ACCENT2 = '#1565c0';   // deep blue — genetics; OMIM; paired domain
const ACCENT3 = '#2e7d32';   // deep green — SU first-line; 75-80% response; remission
const ACCENT4 = '#6a1b9a';   // deep purple — KPD; DKA at onset; misdiagnosis T1D
const ACCENT5 = '#c62828';   // deep red — homeodomain R121W; founder mutation
const ACCENT6 = '#37474f';   // dark slate — epidemiology; East Asian enrichment
const ACCENT7 = '#00695c';   // deep teal — ARX pathway; islet differentiation
const ACCENT8 = '#f57f17';   // amber — comparison vs MODY8/MODY10; structural vs functional

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

export default function MODY9Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody9/overview`).then(r => r.json()),
      fetch(`${API}/api/mody9/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody9/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY9 — PAX4-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 9 · Chr 7q32.1 · OMIM #612225 · ~1–2% MODY · PAX4 Haploinsufficiency · ARX De-repression · Alpha-cell Bias · KPD at Onset · SU First-line · East Asian Enriched · R121W Founder · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="PAX4 *167413" color={ACCENT2} />
            <Badge text="ARX repression" color={ACCENT7} />
            <Badge text="KPD-DKA onset" color={ACCENT4} />
            <Badge text="SU 75–80%" color={ACCENT3} />
            <Badge text="R121W founder" color={ACCENT5} />
            <Badge text="East Asian" color={ACCENT6} />
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
        <KPI label="KPD Onset (%)" value={`${kpis.pct_kdp_presentation?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="SU Response (%)" value={`${kpis.pct_su_response?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT2} />
        <KPI label="Misdiagnosed (%)" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="East Asian (%)" value={`${kpis.pct_east_asian?.toFixed(0)}%`} color={ACCENT6} />
        <KPI label="R121W (%)" value={`${kpis.pct_r121w?.toFixed(0)}%`} color={ACCENT5} />
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
                <strong>MODY9 — PAX4 HAPLOINSUFFICIENCY → ARX DE-REPRESSION → ALPHA-CELL BIAS.</strong> PAX4 (Paired Box 4) is a transcription factor whose primary function is to <em>repress ARX</em> (the master alpha-cell fate determinant) in islet progenitors. PAX4 LOF → insufficient ARX repression → alpha-cell programme is not fully suppressed → impaired beta-cell differentiation balance → reduced functional beta-cell mass → progressive GSIS impairment → hyperglycaemia.
              </Alert>
              <Alert color={ACCENT4}>
                <strong>MODY9-UNIQUE: KETOSIS-PRONE DIABETES (KPD) at ONSET.</strong> The R121W (Arg121Trp) homeodomain founder mutation (Thai/Korean/Japanese) causes transient near-complete GSIS failure → DKA at acute onset. After insulin stabilisation, C-peptide <em>recovers</em> in 50–70% → insulin withdrawal trial → switch to sulfonylurea. This DKA-then-remission pattern is unique to MODY9 among all MODY subtypes. Misdiagnosis as T1D (antibody-negative DKA) is the most dangerous error.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>SULFONYLUREA FIRST-LINE — 75–80% RESPOND.</strong> SU (glibenclamide/gliclazide) closes K-ATP channels independently of the transcriptional deficit. Non-KPD patients start on SU directly. KPD patients start on insulin → C-peptide recovery check at 3 months → switch to SU if C-peptide ≥ 0.20 nmol/L stimulated. No exocrine involvement, no renal cysts, no structural pancreatic destruction — SU mechanism intact.
              </Alert>
              <Alert color={ACCENT6}>
                <strong>EAST ASIAN ENRICHMENT — NOT IN OLDEST MODY PANELS.</strong> R121W is a Thai/Korean/Japanese/Chinese founder mutation. Standard 4-gene MODY panels (HNF1A/HNF4A/GCK/HNF1B) miss MODY9 entirely. Expanded NGS panels including PAX4 are mandatory. European lineages carry R37W and other rare variants. ~10–15% de novo.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>PAX4 (Paired Box 4) · Chr 7q32.1 · *167413</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#612225 (MODY9)</td></tr>
                  <tr><td className="fw-bold">Protein function</td><td>Paired-domain + homeodomain TF expressed in islet progenitors; transcriptionally represses ARX (alpha-cell fate determinant); activates GLUT2 and insulin promoter directly</td></tr>
                  <tr><td className="fw-bold">Mutation type</td><td>Heterozygous LOF — missense, splice, truncating; haploinsufficiency; R121W founder in homeodomain loop 1 (East Asian); R37W in paired domain (European)</td></tr>
                  <tr><td className="fw-bold">Key mechanism</td><td>PAX4 haploinsufficiency → partial ARX de-repression → alpha-cell bias → reduced functional beta-cell mass → impaired GSIS (functional, NOT structural like MODY8)</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% transmission · heterozygous LOF → MODY9</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1–2% of all MODY; East Asian enrichment (Thai, Korean, Japanese, Chinese); rare in Europeans</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Teens–40s (mean ~30–35 yr); KPD variants may present with acute DKA at any age</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>PRESERVED at diagnosis (functional, not structural deficit); may transiently dip in KPD then recover; falls with long duration (progressive beta-cell reduction)</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — always; mandatory test to exclude T1D, especially in KPD presentation</td></tr>
                  <tr><td className="fw-bold">KPD</td><td>Ketosis-Prone Diabetes: DKA at onset, antibody-negative, C-peptide recovery after insulin, 50–70% achieve SU/diet remission; predominantly R121W carriers</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🔬 PAX4 → ARX Repression Pathway" color={ACCENT7}>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT7 + '10', border: `1px solid ${ACCENT7}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT7 }}>PAX4 → ARX Axis — Beta vs Alpha Cell Fate Decision</div>
                <ol className="small mb-0">
                  <li><strong>Normal:</strong> PAX4 expressed in islet progenitors → binds Pax/ATTA elements in ARX promoter → transcriptionally silences ARX → beta-cell differentiation proceeds</li>
                  <li><strong>MODY9 mutation:</strong> Heterozygous PAX4 LOF → haploinsufficiency → one copy insufficient to fully repress ARX</li>
                  <li><strong>ARX de-repression:</strong> ARX partially active → alpha-cell fate programme not fully suppressed → islet progenitor pool shifted towards alpha-cell fate</li>
                  <li><strong>Beta-cell mass reduction:</strong> Reduced functional beta-cell mass → GSIS impairment → progressive hyperglycaemia</li>
                  <li><strong>R121W KPD variant:</strong> Homeodomain loop 1 disruption → markedly reduced ARX binding affinity → transient near-complete GSIS failure → DKA → C-peptide recovery after stabilisation</li>
                  <li><strong>SU rescue:</strong> Closes K-ATP channels directly → compensates for reduced beta-cell mass; intact structural pancreas = SU response</li>
                </ol>
              </div>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT4 + '10', border: `1px solid ${ACCENT4}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT4 }}>KPD-MODY9 — Clinical Decision Algorithm</div>
                <ol className="small mb-0">
                  <li>DKA at onset, young patient, antibody-negative → <strong>do NOT label T1D</strong></li>
                  <li>Check: GADA, ZnT8, IA-2 (all negative in MODY9)</li>
                  <li>Stabilise with insulin; obtain PAX4 sequencing + C-peptide</li>
                  <li>At 3 months: check stimulated C-peptide (≥ 0.20 nmol/L = recovery)</li>
                  <li>If C-peptide recovered: trial insulin withdrawal → start SU (glibenclamide 0.5 mg/day; titrate)</li>
                  <li>50–70% of R121W KPD achieve SU or diet remission; monitor HbA1c + SMBG q3mo</li>
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

      {/* Tab 1: Cohort & KPD */}
      {tab === 1 && (
        <div className="row g-3">
          <div className="col-lg-5">
            <Section title="🔬 PAX4 Variant Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td>
                          <span className="badge me-1" style={{ background: v.includes('R121W') ? ACCENT5 : ACCENT2, fontSize: '0.72em' }}>{v}</span>
                          {v.includes('R121W') && <span className="badge" style={{ background: ACCENT6, fontSize: '0.68em' }}>Thai/Korean/JP founder</span>}
                        </td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="🌏 Ethnicity Distribution" color={ACCENT6}>
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
            <Section title="🔥 KPD Presentation" color={ACCENT4}>
              <div className="small text-muted mb-1">KPD = Ketosis-Prone DM: DKA at onset, antibody-negative, C-peptide recovery; predominantly R121W carriers</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>KPD Status</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.kdp_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="⚠️ Prior Misdiagnosis" color={ACCENT4}>
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
            <Section title="📊 HbA1c Distribution" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>HbA1c Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 C-Peptide Distribution (PRESERVED — functional deficit)" color={ACCENT3}>
              <div className="small text-muted mb-1">C-peptide is PRESERVED in MODY9 (functional GSIS deficit, not structural loss); unlike LOW C-peptide in MODY8 (structural)</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>C-Pep (nmol/L)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.c_peptide_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Age at Diagnosis" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Age at Dx</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.age_at_diagnosis_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 BMI Distribution" color={ACCENT6}>
              <div className="small text-muted mb-1">BMI typically normal (20–30 kg/m²); unlike T2D (obese) or MODY8 (can be underweight due to EPI)</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>BMI (kg/m²)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.bmi_tiers || {}).map(([k, v]) => (
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
                <strong>SULFONYLUREA FIRST-LINE (75–80%) — FUNCTIONAL DEFICIT, INTACT STRUCTURE.</strong> Non-KPD patients: SU (glibenclamide 0.5–2.5 mg/day or gliclazide MR 30 mg/day) — start low (MODY more sensitive than T2D). KPD patients: insulin acute → C-peptide check 3 months → switch to SU if recovery. No PERT, no vitamin supplementation, no exocrine management — unlike MODY8.
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
                <strong>EXPANDED MODY NGS PANEL REQUIRED — PAX4 NOT IN OLDEST PANELS.</strong> Standard 4-gene panels (HNF1A/HNF4A/GCK/HNF1B) miss MODY9. Expanded NGS including PAX4 is mandatory. Functional validation (ARX reporter assay) recommended for novel variants. Family cascade: all first-degree relatives (50% AD transmission); KPD relatives may present as T1D — antibody test + PAX4 sequencing resolves.
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
            <Section title="🔑 KPD Pathway" color={ACCENT4}>
              {Object.entries(definitions?.kdp_mody9 || {}).map(([k, v]) => (
                <div key={k} className="mb-2">
                  <div className="fw-bold small" style={{ color: ACCENT4 }}>{k.replace(/_/g, ' ')}</div>
                  <div className="small text-muted">{v}</div>
                </div>
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="👥 Patient Sample" color={ACCENT6}>
              <div className="table-responsive" style={{ maxHeight: 320 }}>
                <table className="table table-sm table-bordered table-hover">
                  <thead>
                    <tr>
                      <th>ID</th><th>Sex</th><th>Age</th><th>Dx Age</th><th>HbA1c%</th>
                      <th>C-pep</th><th>BMI</th><th>Variant</th><th>KPD</th><th>Ethnicity</th><th>Misdiagnosis</th><th>FamHx</th>
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
                        <td><span className="badge" style={{ background: p.variant?.includes('R121W') ? ACCENT5 : ACCENT2, fontSize: '0.65em' }}>{p.variant}</span></td>
                        <td>{p.kdp_presentation ? <span className="badge" style={{ background: ACCENT4, fontSize: '0.65em' }}>KPD</span> : '—'}</td>
                        <td><span className="badge" style={{ background: ACCENT6, fontSize: '0.65em' }}>{p.ethnicity}</span></td>
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
