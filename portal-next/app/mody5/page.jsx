'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variant & Cohort', 'Treatment & Renal', 'Definitions'];

// MODY5 colour scheme — deep teal/indigo (multi-organ, complex, insulin-requiring)
const ACCENT  = '#1a237e';   // deep indigo — HNF1B; complex multi-organ
const ACCENT2 = '#006064';   // dark teal — renal cysts; kidney phenotype
const ACCENT3 = '#b71c1c';   // deep red — misdiagnosis; de-novo; insulin required
const ACCENT4 = '#e65100';   // deep orange — hypomagnesaemia; exocrine insufficiency
const ACCENT5 = '#4a148c';   // purple — genetics; 17q12 deletion
const ACCENT6 = '#1565c0';   // blue — CKD; eGFR; renal monitoring
const ACCENT7 = '#37474f';   // dark slate — epidemiology; de-novo rate
const ACCENT8 = '#00695c';   // deep teal — pancreatic atrophy; imaging

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

export default function MODY5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody5/overview`).then(r => r.json()),
      fetch(`${API}/api/mody5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody5/definitions`).then(r => r.json()),
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

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT2}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY5 — HNF1B-MODY / RCAD Syndrome</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 5 · Chr 17q12 · OMIM #137920 · ~5% of all MODY · Multi-organ (Renal Cysts + Diabetes) · Insulin required · ~50% de-novo mutations · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="HNF1B *189907" color={ACCENT5} />
            <Badge text="Renal cysts" color={ACCENT2} />
            <Badge text="Pancreatic atrophy" color={ACCENT8} />
            <Badge text="Insulin required" color={ACCENT3} />
            <Badge text="~50% de-novo" color={ACCENT7} />
            <Badge text="Hypomagnesaemia" color={ACCENT4} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT7} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT3} />
        <KPI label="Mean FG (mmol/L)" value={kpis.mean_fasting_glucose_mmol?.toFixed(1)} color={ACCENT3} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(3)} color={ACCENT8} />
        <KPI label="Mean Mg (mmol/L)" value={kpis.mean_serum_mg_mmol_L?.toFixed(2)} color={ACCENT4} />
        <KPI label="Mean eGFR" value={kpis.mean_egfr?.toFixed(0)} color={ACCENT6} />
        <KPI label="Insulin Rx (%)" value={`${kpis.pct_insulin_treated?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Renal Abn (%)" value={`${kpis.pct_renal_abnormality?.toFixed(0)}%`} color={ACCENT2} />
        <KPI label="Pancr. Atrophy (%)" value={`${kpis.pct_pancreatic_atrophy?.toFixed(0)}%`} color={ACCENT8} />
        <KPI label="De-novo (%)" value={`${kpis.pct_de_novo?.toFixed(0)}%`} color={ACCENT7} />
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
              <Alert color={ACCENT2}>
                <strong>MODY5-UNIQUE — Renal Cysts PRECEDE Diabetes:</strong> Structural kidney abnormalities (multicystic dysplastic kidneys, renal cysts, hypoplasia) are often detected antenatally or in childhood — BEFORE diabetes onset. ~70% of MODY5 patients have renal structural anomalies. Suspect MODY5 when any young person with renal cysts develops diabetes.
              </Alert>
              <Alert color={ACCENT8}>
                <strong>Pancreatic Atrophy (CT/MRI):</strong> Both exocrine AND endocrine pancreas are affected — visible as reduced pancreatic volume on imaging. Exocrine insufficiency causes steatorrhoea and fat-soluble vitamin deficiency. Beta-cell loss drives progressive, insulin-requiring diabetes.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>NO SULFONYLURE RESPONSE — INSULIN REQUIRED:</strong> Unlike MODY1/MODY3 (85–90% SU response), MODY5 beta-cells are structurally lost (atrophy). Sulfonylure cannot recruit absent cells. Insulin from early in the disease course. C-peptide falls progressively.
              </Alert>
              <Alert color={ACCENT7}>
                <strong>~50% DE-NOVO MUTATIONS:</strong> Unlike other MODY types (75–90% family history), HNF1B LOF is de-novo in ~50% — mostly whole-gene 17q12 microdeletions. Negative family history does NOT exclude MODY5. MLPA/aCGH mandatory alongside sequencing.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>HNF1B (Hepatocyte Nuclear Factor 1 Beta) · Chr 17q12 · *189907</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#137920 (MODY5 / RCAD Syndrome)</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence · ~50% de-novo (17q12 deletion)</td></tr>
                  <tr><td className="fw-bold">Mechanism</td><td>HNF1B LOF → disrupts kidney tubulogenesis (cysts) + pancreatic development (atrophy) + Müllerian duct + FXYD2 (Mg wasting)</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1:10,000–1:50,000 (~5% of all MODY; underdiagnosed)</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Late teens – early 30s (mean ~26 yr); renal findings often pre-date DM by years</td></tr>
                  <tr><td className="fw-bold">HbA1c</td><td>Typically 7.0–11.0% (poorly controlled without insulin); progressive</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>LOW and falling (structural pancreatic atrophy) — key differentiator from MODY2 (normal)</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — but low C-pep + insulin need → often misdiagnosed T1D</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="📋 Key Clinical Facts" color={ACCENT5}>
              <ul className="list-group list-group-flush">
                {keyFacts.map((f, i) => (
                  <li key={i} className="list-group-item py-1 small">{f}</li>
                ))}
              </ul>
            </Section>
            <Section title="🩺 Diagnostic Criteria" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(overview?.diagnostic_criteria || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 1: Variant & Cohort */}
      {tab === 1 && (
        <div className="row g-3">
          <div className="col-lg-5">
            <Section title="🔬 Variant / Mutation Distribution" color={ACCENT5}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant / CNV</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td><span className="badge" style={{ background: ACCENT5, fontSize: '0.72em' }}>{v}</span></td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="🧫 Mutation Origin (de-novo vs inherited)" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Origin</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.mutation_origin_distribution || {}).map(([o, n]) => (
                    <tr key={o}><td>{o.replace(/_/g, ' ')}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 HbA1c Tiers" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="💊 C-Peptide Tiers (low = structural atrophy)" color={ACCENT8}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>C-peptide (nmol/L)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.c_peptide_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🧂 Serum Magnesium Tiers" color={ACCENT4}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Mg (mmol/L)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.serum_mg_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-7">
            <Section title="👩‍⚕️ Patient Cohort (40 patients, seed 309)" color={ACCENT}>
              <div style={{ maxHeight: 580, overflowY: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.71em' }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Dx Age</th><th>HbA1c%</th>
                      <th>C-pep</th><th>Mg</th><th>eGFR</th><th>Variant</th><th>Origin</th><th>Renal</th><th>Rx</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => (
                      <tr key={p.patient_id}>
                        <td>{p.patient_id}</td>
                        <td>{p.age}</td>
                        <td>{p.sex}</td>
                        <td>{p.age_at_diagnosis}</td>
                        <td>
                          <span style={{ color: p.hba1c_percent > 9.0 ? ACCENT3 : p.hba1c_percent > 7.5 ? ACCENT4 : ACCENT8, fontWeight: 600 }}>
                            {p.hba1c_percent?.toFixed(1)}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: p.c_peptide_nmol_L < 0.25 ? ACCENT3 : p.c_peptide_nmol_L < 0.40 ? ACCENT4 : ACCENT8, fontWeight: 600 }}>
                            {p.c_peptide_nmol_L?.toFixed(2)}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: p.serum_mg_mmol_L < 0.55 ? ACCENT3 : p.serum_mg_mmol_L < 0.70 ? ACCENT4 : ACCENT8, fontWeight: 600 }}>
                            {p.serum_mg_mmol_L?.toFixed(2)}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: p.egfr_ml_min_1_73m2 < 60 ? ACCENT3 : p.egfr_ml_min_1_73m2 < 90 ? ACCENT4 : ACCENT, fontWeight: 600 }}>
                            {p.egfr_ml_min_1_73m2}
                          </span>
                        </td>
                        <td style={{ fontSize: '0.68em' }}>{p.variant}</td>
                        <td style={{ fontSize: '0.68em' }}>{p.mutation_origin?.replace(/_/g, ' ')}</td>
                        <td style={{ fontSize: '0.65em' }}>{p.renal_phenotype?.replace(/_/g, ' ')}</td>
                        <td style={{ fontSize: '0.65em' }}>{p.current_treatment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 2: Treatment & Renal */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-lg-6">
            <Section title="💊 Treatment Distribution" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Treatment</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.treatment_distribution || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🏥 Misdiagnosis Distribution" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Prior Diagnosis</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.misdiagnosis_distribution || {}).map(([m, n]) => (
                    <tr key={m}><td>{m.replace(/_/g, ' ')}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="👥 Age Groups at Current Visit" color={ACCENT7}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Age group</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.age_groups || {}).map(([g, n]) => (
                    <tr key={g}><td>{g}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Summary Flags" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(breakdown?.summary_flags || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k.replace(/pct_/,'').replace(/_/g,' ')}</td><td className="fw-bold" style={{ color: ACCENT3 }}>{v}%</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🫘 Renal Phenotype Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Renal Phenotype</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.renal_phenotype_distribution || {}).map(([r, n]) => (
                    <tr key={r}><td>{r.replace(/_/g, ' ')}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔬 eGFR / CKD Stage Distribution" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>CKD Stage (eGFR)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.egfr_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🆚 MODY5 vs MODY3 — Key Differentiators" color={ACCENT}>
              <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.83em' }}>
                <thead className="table-dark"><tr><th>Feature</th><th>MODY5 (HNF1B)</th><th>MODY3 (HNF1A)</th></tr></thead>
                <tbody>
                  <tr><td>Renal glycosuria</td><td style={{ color: ACCENT8 }}>ABSENT (0%)</td><td style={{ color: ACCENT2 }}>PRESENT (50%)</td></tr>
                  <tr><td>Renal cysts</td><td style={{ color: ACCENT3 }}>PRESENT (~70%)</td><td style={{ color: ACCENT8 }}>ABSENT</td></tr>
                  <tr><td>Pancreatic atrophy</td><td style={{ color: ACCENT3 }}>PRESENT (CT/MRI)</td><td style={{ color: ACCENT8 }}>ABSENT</td></tr>
                  <tr><td>Exocrine insufficiency</td><td style={{ color: ACCENT3 }}>PRESENT (~40%)</td><td style={{ color: ACCENT8 }}>ABSENT</td></tr>
                  <tr><td>Sulfonylure response</td><td style={{ color: ACCENT3 }}>NO (atrophy)</td><td style={{ color: ACCENT }}>YES (85–90%)</td></tr>
                  <tr><td>De-novo mutations</td><td style={{ color: ACCENT3 }}>~50%</td><td style={{ color: ACCENT8 }}>Rare (&lt;5%)</td></tr>
                  <tr><td>Family history</td><td style={{ color: ACCENT4 }}>~50%</td><td style={{ color: ACCENT8 }}>~90%</td></tr>
                  <tr><td>C-peptide</td><td style={{ color: ACCENT3 }}>Low / falling</td><td style={{ color: ACCENT8 }}>Preserved early</td></tr>
                  <tr><td>Hypomagnesaemia</td><td style={{ color: ACCENT3 }}>PRESENT (~40–50%)</td><td style={{ color: ACCENT8 }}>ABSENT</td></tr>
                  <tr><td>Genital malform.</td><td style={{ color: ACCENT4 }}>~25% females</td><td style={{ color: ACCENT8 }}>ABSENT</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && (
        <div className="row g-3">
          {Object.entries(definitions || {}).map(([section, entries]) => (
            <div key={section} className="col-lg-6">
              <Section title={`📖 ${section.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`} color={ACCENT}>
                <table className="table table-sm table-bordered">
                  <tbody>
                    {Object.entries(entries || {}).map(([k, v]) => (
                      <tr key={k}>
                        <td className="fw-bold small" style={{ width: '35%' }}>{k.replace(/_/g, ' ')}</td>
                        <td className="small">{typeof v === 'string' ? v : JSON.stringify(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Section>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
