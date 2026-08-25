'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Cohort & EPI', 'Treatment & Comparison', 'Definitions'];

// MODY8 colour scheme — deep teal/forest (CEL exocrine enzyme; dual pancreatic failure; lipomatosis)
const ACCENT  = '#00695c';   // deep teal — CEL exocrine enzyme; dual failure; EPI-first
const ACCENT2 = '#1a237e';   // deep indigo — genetics; OMIM; VNTR analysis
const ACCENT3 = '#e65100';   // deep amber — insulin mandatory; no SU; treatment urgency
const ACCENT4 = '#880e4f';   // deep magenta — misdiagnosis; T1D/pancreatitis confusion
const ACCENT5 = '#1b5e20';   // deep green — family history; cascade testing
const ACCENT6 = '#37474f';   // dark slate — epidemiology; Norwegian founder
const ACCENT7 = '#4a148c';   // deep purple — fat-soluble vitamin deficiency; EPI severity
const ACCENT8 = '#b71c1c';   // deep red — lipomatosis imaging; structural destruction

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

export default function MODY8Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody8/overview`).then(r => r.json()),
      fetch(`${API}/api/mody8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody8/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY8 — CEL-MODY / BSSL-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 8 · Chr 9q34.3 · OMIM #609812 · ~1–2% MODY · CEL VNTR Frameshift · Pancreatic Lipomatosis · Exocrine + Endocrine Failure · NO SU · Insulin Mandatory · Norwegian Founder p.V698Lfs*5 · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="CEL *114840" color={ACCENT2} />
            <Badge text="VNTR frameshift" color={ACCENT} />
            <Badge text="Lipomatosis" color={ACCENT8} />
            <Badge text="EPI + PERT" color={ACCENT7} />
            <Badge text="NO SU — Insulin" color={ACCENT3} />
            <Badge text="Norwegian founder" color={ACCENT6} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Mean Duration (yr)" value={kpis.mean_duration_years?.toFixed(1)} color={ACCENT2} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT3} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(3)} color={ACCENT4} />
        <KPI label="Mean FEL-1 (µg/g)" value={kpis.mean_fel1_ug_g?.toFixed(0)} color={ACCENT7} />
        <KPI label="Pancreatic Fat (%)" value={kpis.mean_pancreatic_fat_fraction_pct?.toFixed(1)} color={ACCENT8} />
        <KPI label="On Insulin (%)" value={`${kpis.pct_on_insulin?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="SU Response (%)" value={`${kpis.pct_su_response?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT5} />
        <KPI label="Misdiagnosed (%)" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT4} />
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
              <Alert color={ACCENT8}>
                <strong>MODY8 — UNIQUE DUAL PANCREATIC FAILURE: Exocrine + Endocrine.</strong> CEL VNTR frameshift → misfolded carboxyl ester lipase → acinar cell ER stress → progressive <em>pancreatic lipomatosis</em> (fat replacement of exocrine parenchyma) → secondary beta-cell structural loss. <em>Only MODY type caused by an exocrine enzyme gene</em>. Exocrine insufficiency (EPI) often precedes diabetes by years.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>NO SULFONYLUREA — INSULIN MANDATORY:</strong> Beta-cell loss is STRUCTURAL (embedded in destroyed exocrine tissue). There is no functional K-ATP pool to stimulate. SU is contraindicated. ALL MODY8 patients require insulin (basal ± bolus) plus PERT (pancreatic enzyme replacement) with every meal.
              </Alert>
              <Alert color={ACCENT7}>
                <strong>PANCREATIC LIPOMATOSIS — PATHOGNOMONIC IMAGING FINDING:</strong> CT/MRI shows diffuse fat infiltration of the pancreas (pancreatic fat fraction typically ≥ 40–85%). Distinct from MODY5 (HNF1B) which shows atrophy without fat. No renal cysts (MODY5 hallmark), no hypomagnesaemia, no Mullerian anomalies — key differentiators from MODY5.
              </Alert>
              <Alert color={ACCENT4}>
                <strong>HIGH MISDIAGNOSIS — T1D (35%) + CHRONIC PANCREATITIS (20%):</strong> Insulin requirement → T1D misdiagnosis. Steatorrhoea + abdominal symptoms → chronic pancreatitis diagnosis. CFRD (10%) due to EPI overlap. Antibody-negative status + CEL VNTR testing resolves. Scandinavian ancestry is a key clinical clue.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>CEL / BSSL (Carboxyl Ester Lipase) · Chr 9q34.3 · *114840</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#609812 (MODY8)</td></tr>
                  <tr><td className="fw-bold">Protein function</td><td>Pancreatic exocrine enzyme — hydrolyzes cholesterol esters, triglycerides, fat-soluble vitamins (ADEK) in duodenum (bile salt-stimulated)</td></tr>
                  <tr><td className="fw-bold">Mutation type</td><td>Single nt deletion in VNTR (11-bp tandem repeat) in CEL exon 11 → frameshift → misfolded C-terminal domain → ER aggregation → acinar cell toxicity</td></tr>
                  <tr><td className="fw-bold">Key mechanism</td><td>Misfolded CEL → pancreatic acinar cell apoptosis → lipomatosis → secondary beta-cell structural loss (NOT primary transcriptional/secretory defect)</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence · heterozygous VNTR frameshift → MODY8</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1–2% of all MODY; Norwegian/Scandinavian enrichment due to founder p.V698Lfs*5</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Late 20s–50s (mean ~35–45 yr); EPI may precede diabetes by years</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>LOW at diagnosis (structural loss); falls progressively; unlike MODY3/6/7 where C-peptide preserved early</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — mandatory to exclude T1D</td></tr>
                  <tr><td className="fw-bold">Founder variant</td><td>p.V698Lfs*5 — single nt deletion in VNTR repeat unit 16 (Johansson et al. 2011, Nat Genet)</td></tr>
                  <tr><td className="fw-bold">FEL-1 stool test</td><td>ALL MODY8: FEL-1 &lt; 200 µg/g (EPI confirmed); often &lt; 100 µg/g (severe EPI)</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🔬 VNTR Frameshift → Lipomatosis Pathway" color={ACCENT8}>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT8 + '10', border: `1px solid ${ACCENT8}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT8 }}>Progressive Pancreatic Destruction — MODY8 Unique Mechanism</div>
                <ol className="small mb-0">
                  <li><strong>Normal:</strong> CEL (11-bp VNTR repeats in exon 11) secreted into duodenum → bile salt-stimulated fat digestion</li>
                  <li><strong>MODY8 mutation:</strong> Single nt deletion in VNTR repeat → frameshift → misfolded C-terminal domain</li>
                  <li><strong>ER aggregation:</strong> Misfolded CEL accumulates in ER of acinar cells → ER stress → acinar cell apoptosis</li>
                  <li><strong>Lipomatosis:</strong> Exocrine parenchyma progressively replaced by adipocytes (fat infiltration) → visible on CT/MRI</li>
                  <li><strong>Secondary beta-cell loss:</strong> Islets embedded in fat-replaced parenchyma → structural beta-cell destruction → insulin-requiring DM</li>
                  <li><strong>Dual failure:</strong> EPI (steatorrhoea + vitamin ADEK deficiency) + MODY diabetes — both progressive</li>
                </ol>
              </div>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT7 + '10', border: `1px solid ${ACCENT7}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT7 }}>Exocrine Insufficiency (EPI) — Clinical Features</div>
                <ul className="small mb-0">
                  <li>Steatorrhoea (oily/greasy stools, floating, foul-smelling)</li>
                  <li>Fat-soluble vitamin deficiency: Vitamin D (↓25-OH-D), Vitamin A, Vitamin K (↑INR), Vitamin E</li>
                  <li>Weight loss / failure to maintain weight despite adequate caloric intake</li>
                  <li>FEL-1 (faecal elastase-1) &lt; 200 µg/g stool — confirms EPI; not affected by PERT timing</li>
                  <li>Often precedes diabetes by years — EPI as first presenting feature of MODY8</li>
                </ul>
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

      {/* Tab 1: Cohort & EPI */}
      {tab === 1 && (
        <div className="row g-3">
          <div className="col-lg-5">
            <Section title="🔬 CEL VNTR Variant Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td>
                          <span className="badge me-1" style={{ background: v === 'p.V698Lfs*5' ? ACCENT8 : ACCENT2, fontSize: '0.72em' }}>{v}</span>
                          {v === 'p.V698Lfs*5' && <span className="badge" style={{ background: ACCENT6, fontSize: '0.68em' }}>Norwegian founder</span>}
                        </td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 FEL-1 (Faecal Elastase-1) Severity" color={ACCENT7}>
              <div className="small text-muted mb-1">FEL-1 &lt; 200 µg/g = EPI confirmed; &lt; 100 = severe EPI requiring high-dose PERT</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>FEL-1 Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.fel1_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Pancreatic Fat Fraction (MRI)" color={ACCENT8}>
              <div className="small text-muted mb-1">Fat fraction &gt; 40% = lipomatosis; normal &lt; 15%</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Fat Fraction (%)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.pancreatic_fat_fraction_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🧪 Fat-Soluble Vitamin Deficiency" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Vitamin</th><th>n deficient</th><th>%</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.vitamin_deficiency_counts || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td><td>{((v / 40) * 100).toFixed(0)}%</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-7">
            <Section title="📊 HbA1c Distribution" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>HbA1c Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 C-Peptide Distribution (low — structural loss)" color={ACCENT4}>
              <div className="small text-muted mb-1">C-peptide is LOW in MODY8 (structural beta-cell destruction); unlike preserved C-peptide in MODY3/6/7</div>
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
              <div className="small text-muted mb-1">BMI typically lower than T2D due to fat malabsorption and weight loss; underweight (BMI &lt; 18.5) possible</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>BMI (kg/m²)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.bmi_tiers || {}).map(([k, v]) => (
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
        </div>
      )}

      {/* Tab 2: Treatment & Comparison */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="💉 Treatment Strategy" color={ACCENT3}>
              <Alert color={ACCENT3}>
                <strong>DUAL TREATMENT — INSULIN + PERT MANDATORY for ALL MODY8:</strong> (1) Insulin (basal ± bolus) for diabetes — no SU ever. (2) PERT (Creon/Pancreaze) with every meal for EPI. (3) Fat-soluble vitamins (ADEK) supplementation. SU is contraindicated — structural beta-cell absence means no functional pool to stimulate.
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
                <strong>VNTR SEQUENCING CRITICAL NOTE:</strong> Standard NGS may MISS MODY8 — single nt deletion in CEL exon 11 VNTR (repetitive 11-bp tandem repeat) requires VNTR-aware PCR, long-read sequencing, or repeat-specific assay. CEL-CELP pseudogene homology can also confound results.
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
            <Section title="🔄 MODY7 vs MODY8 vs MODY9 Comparison" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.comparison_mody7_8_9 || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT6 }}>{k}</td><td className="small">{v}</td></tr>
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
            <Section title="📊 Summary Flags" color={ACCENT8}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(breakdown?.summary_flags || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k.replace(/_/g, ' ')}</td><td>{typeof v === 'number' ? (k.includes('pct') ? `${v}%` : v) : v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔔 Clinical Alerts" color={ACCENT8}>
              {Object.entries(alerts).map(([k, v]) => (
                <Alert key={k} color={ACCENT8}><strong>{k.replace(/_/g, ' ')}:</strong> {v}</Alert>
              ))}
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
                  <div className="small text-muted">{v}</div>
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
            <Section title="🔬 Lab Thresholds" color={ACCENT5}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.lab_thresholds || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT5 }}>{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
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
                      <th>C-pep</th><th>FEL-1</th><th>Fat%</th><th>Variant</th><th>Misdiagnosis</th><th>FamHx</th><th>Scan.</th>
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
                        <td>{p.fel1_ug_g}</td>
                        <td>{p.pancreatic_fat_fraction_pct}</td>
                        <td><span className="badge" style={{ background: p.variant === 'p.V698Lfs*5' ? ACCENT8 : ACCENT2, fontSize: '0.65em' }}>{p.variant}</span></td>
                        <td>{p.prior_misdiagnosis !== 'None' ? <span className="badge" style={{ background: ACCENT4, fontSize: '0.65em' }}>{p.prior_misdiagnosis}</span> : '—'}</td>
                        <td>{p.family_history_positive ? '✓' : '—'}</td>
                        <td>{p.scandinavian_ancestry ? '✓' : '—'}</td>
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
