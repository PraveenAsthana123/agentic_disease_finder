'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variant & Cohort', 'Treatment & Comparison', 'Definitions'];

// MODY7 colour scheme — deep amber/copper (KLF11 zinc finger; oxidative/ROS mechanism; contested type)
const ACCENT  = '#bf360c';   // deep burnt-orange — KLF11 zinc finger; oxidative stress hallmark
const ACCENT2 = '#1a237e';   // deep indigo — genetics; OMIM; zinc finger structure
const ACCENT3 = '#1b5e20';   // deep green — SU-responsive; functional beta-cells remain early
const ACCENT4 = '#880e4f';   // deep magenta — misdiagnosis; T2D confusion; late onset
const ACCENT5 = '#006064';   // dark teal — C-peptide preserved early; functional before loss
const ACCENT6 = '#37474f';   // dark slate — epidemiology; family history; contested literature
const ACCENT7 = '#b71c1c';   // deep red — MAO-A excess; ROS; oxidative beta-cell apoptosis
const ACCENT8 = '#e65100';   // amber — progressive HbA1c; oxidative loss; insulin requirement

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

export default function MODY7Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody7/overview`).then(r => r.json()),
      fetch(`${API}/api/mody7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody7/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY7 — KLF11-MODY / TIEG2-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 7 · Chr 2p25.1 · OMIM #610508 · ~1–2% MODY · KLF11 Zinc Finger Repressor · MAO-A/ROS Oxidative Mechanism · SU-Responsive · Most Contested MODY · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="KLF11 *603301" color={ACCENT2} />
            <Badge text="Zinc finger repressor" color={ACCENT} />
            <Badge text="MAO-A/ROS" color={ACCENT7} />
            <Badge text="SU first-line" color={ACCENT3} />
            <Badge text="Q62R+A347S founding" color={ACCENT6} />
            <Badge text="Contested" color={ACCENT4} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Mean Duration (yr)" value={kpis.mean_duration_years?.toFixed(1)} color={ACCENT2} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT8} />
        <KPI label="Mean FG (mmol/L)" value={kpis.mean_fasting_glucose_mmol?.toFixed(1)} color={ACCENT8} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(3)} color={ACCENT5} />
        <KPI label="Mean BMI" value={kpis.mean_bmi_kg_m2?.toFixed(1)} color={ACCENT6} />
        <KPI label="Mean MDA (nmol/mL)" value={kpis.mean_mda_oxidative_stress_nmol_mL?.toFixed(2)} color={ACCENT7} />
        <KPI label="On SU (%)" value={`${kpis.pct_on_sulfonylurea?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="SU Responders (%)" value={`${kpis.pct_su_responders_of_su_treated?.toFixed(0)}%`} color={ACCENT3} />
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
              <Alert color={ACCENT7}>
                <strong>MODY7 — UNIQUE OXIDATIVE MECHANISM: MAO-A/H₂O₂ → Beta-Cell Apoptosis.</strong> KLF11 (Krüppel-Like Factor 11 / TIEG2) normally represses MAO-A (Monoamine Oxidase A) via mSin3A co-repressor recruitment. LOF → excess MAO-A → elevated H₂O₂ → progressive oxidative beta-cell apoptosis. This ROS-driven mechanism is <em>distinct from all other MODY types</em> (which operate via TF or enzyme haploinsufficiency). Antioxidant therapy (NAC) is under research investigation.
              </Alert>
              <Alert color={ACCENT4}>
                <strong>MOST CONTESTED MODY TYPE — Functional Validation Required:</strong> Originally described by Neve et al. (Nat Genet 2005) in two French families with Q62R and A347S variants. Subsequent population studies have questioned penetrance and causality in some cohorts. Before clinical labelling of novel KLF11 variants as MODY7, functional assay (mSin3A/MAO-A repression reporter) is recommended. The Q62R and A347S founding variants have strong functional support.
              </Alert>
              <Alert color={ACCENT4}>
                <strong>LATER ONSET → T2D MISDIAGNOSIS TRAP:</strong> Mean onset ~38–42 yr (later than MODY3 at ~24 yr and MODY6 at ~35 yr). This adult onset, combined with BMI overlap with T2D populations, leads to ~32% T2D misdiagnosis rate — the highest T2D misdiagnosis among all MODY types after MODY2.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>SULFONYLUREA FIRST-LINE (~75–80% response):</strong> Residual functional beta-cells respond to SU (K-ATP closure bypasses oxidative deficit). Response rate slightly lower than MODY1/3/4 (~85–90%) because progressive oxidative apoptosis reduces the SU-responsive beta-cell pool with duration. Earlier treatment = better long-term outcomes.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>KLF11 / TIEG2 (Krüppel-Like Factor 11) · Chr 2p25.1 · *603301</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#610508 (MODY7)</td></tr>
                  <tr><td className="fw-bold">Protein family</td><td>Krüppel-like factor (KLF) — SP1-type C2H2 zinc finger transcriptional repressor (513 aa)</td></tr>
                  <tr><td className="fw-bold">Key mechanism</td><td>KLF11 LOF → excess MAO-A → H₂O₂ → beta-cell oxidative apoptosis; also ↑SHP → ↓HNF4A/HNF1A axis</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence · heterozygous LOF → MODY7</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1–2% of all MODY; extremely rare; most contested causality in literature</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Late 20s–50s (mean ~38–42 yr) — later than MODY3/6; overlaps T2D age range</td></tr>
                  <tr><td className="fw-bold">HbA1c</td><td>Progressive (rising with duration — oxidative loss cumulative); NOT stable like MODY2/GCK</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>PRESERVED at diagnosis; falls progressively as oxidative apoptosis accumulates</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — mandatory to exclude T1D</td></tr>
                  <tr><td className="fw-bold">Founding variants</td><td>Q62R (PCNLS/mSin3A domain) + A347S (zinc finger 1) — Neve et al. 2005</td></tr>
                  <tr><td className="fw-bold">BMI</td><td>Slightly elevated (22–34) — higher T2D overlap compared to MODY3/6</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="⚡ Oxidative Stress Mechanism" color={ACCENT7}>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT7 + '10', border: `1px solid ${ACCENT7}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT7 }}>MAO-A/H₂O₂ Pathway — Unique to MODY7</div>
                <ol className="small mb-0">
                  <li><strong>Normal:</strong> KLF11 binds GT-box in MAO-A promoter → recruits mSin3A HDAC → represses MAO-A transcription</li>
                  <li><strong>MODY7 LOF:</strong> KLF11 Q62R/A347S → mSin3A not recruited → MAO-A derepressed → excess MAO-A protein</li>
                  <li><strong>Oxidative injury:</strong> MAO-A deaminates serotonin/noradrenaline → excess H₂O₂ → lipid peroxidation → beta-cell apoptosis</li>
                  <li><strong>Progressive:</strong> Oxidative loss accumulates over years → C-peptide falls → insulin dependence</li>
                  <li><strong>Research target:</strong> NAC (N-acetylcysteine) scavenges H₂O₂ → potential to slow beta-cell loss (under investigation)</li>
                </ol>
              </div>
              <div className="p-2 rounded mb-2" style={{ background: ACCENT2 + '10', border: `1px solid ${ACCENT2}33` }}>
                <div className="small fw-bold mb-1" style={{ color: ACCENT2 }}>SHP/HNF Axis — Secondary MODY7 Mechanism</div>
                <ol className="small mb-0">
                  <li>KLF11 normally represses SHP (NR0B2, Small Heterodimer Partner)</li>
                  <li>KLF11 LOF → elevated SHP → SHP represses HNF4A (MODY1) and HNF1A (MODY3) targets</li>
                  <li>Result: secondary impairment of INS/GCK/GLUT2 transcription via HNF axis</li>
                  <li>MODY7 is linked to MODY1/3 regulatory axis — distinguishing from pure beta-cell oxidative damage alone</li>
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

      {/* Tab 1: Variant & Cohort */}
      {tab === 1 && (
        <div className="row g-3">
          <div className="col-lg-5">
            <Section title="🔬 KLF11 Variant Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td>
                          <span className="badge me-1" style={{ background: (v === 'Q62R' || v === 'A347S') ? ACCENT7 : ACCENT2, fontSize: '0.72em' }}>{v}</span>
                          {v === 'Q62R' && <span className="badge" style={{ background: ACCENT, fontSize: '0.68em' }}>founding PCNLS</span>}
                          {v === 'A347S' && <span className="badge" style={{ background: ACCENT, fontSize: '0.68em' }}>founding ZnF1</span>}
                        </td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 HbA1c Distribution" color={ACCENT8}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>HbA1c Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 C-Peptide Distribution" color={ACCENT5}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>C-Pep (nmol/L)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.c_peptide_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-7">
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
            <Section title="⚡ MDA Oxidative Stress Distribution" color={ACCENT7}>
              <div className="small text-muted mb-1">MDA (malondialdehyde) nmol/mL — surrogate oxidative stress marker (research use; elevated in MODY7 vs other MODY types)</div>
              <table className="table table-sm table-bordered">
                <thead><tr><th>MDA (nmol/mL)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.mda_oxidative_stress_tiers || {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 BMI Distribution" color={ACCENT6}>
              <div className="small text-muted mb-1">BMI (kg/m²) — higher T2D overlap compared to MODY3/6; contributes to T2D misdiagnosis</div>
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
            <Section title="💊 Treatment Distribution" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Treatment</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.treatment_distribution || {}).map(([k, v]) => (
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
                <strong>SULFONYLUREA FIRST-LINE (~75–80% response)</strong> — early MODY7 retains functional beta-cells; SU closes K-ATP → depolarization → Ca²⁺ → insulin exocytosis. Response rate slightly lower than MODY1/3/4 due to progressive oxidative loss reducing the SU-responsive pool.
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
              <Alert color={ACCENT4}>
                <strong>MODY7 CONTROVERSY NOTE:</strong> Functional validation (mSin3A/MAO-A repression assay) is recommended before clinically labelling novel KLF11 missense variants as pathogenic. Founding variants Q62R and A347S have strong functional evidence.
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
            <Section title="🔄 MODY6 vs MODY7 vs MODY8 Comparison" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(definitions?.comparison_mody6_7_8 || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small" style={{ color: ACCENT6 }}>{k}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Summary Flags" color={ACCENT8}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(breakdown?.summary_flags || {}).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold small">{k.replace(/_/g, ' ')}</td><td>{typeof v === 'number' ? `${v}%` : v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🔔 Clinical Alerts" color={ACCENT7}>
              {Object.entries(alerts).map(([k, v]) => (
                <Alert key={k} color={ACCENT7}><strong>{k.replace(/_/g, ' ')}:</strong> {v}</Alert>
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
                      <th>C-pep</th><th>BMI</th><th>MDA</th><th>Variant</th><th>Tx</th><th>Misd.</th><th>FamHx</th>
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
                        <td>{p.mda_oxidative_stress_nmol_mL}</td>
                        <td><span className="badge" style={{ background: ACCENT2, fontSize: '0.65em' }}>{p.variant}</span></td>
                        <td className="small">{p.current_treatment.split(' ')[0]}</td>
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
