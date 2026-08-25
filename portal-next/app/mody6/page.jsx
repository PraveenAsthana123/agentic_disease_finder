'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variant & Cohort', 'Treatment & Comparison', 'Definitions'];

// MODY6 colour scheme — deep violet/purple (NEUROD1 bHLH neuronal TF; cerebellar spectrum; rare)
const ACCENT  = '#4a148c';   // deep violet — NEUROD1 bHLH master; neuronal identity
const ACCENT2 = '#1a237e';   // deep indigo — genetics; OMIM; E-box binding
const ACCENT3 = '#1b5e20';   // deep green — SU-responsive; functional beta-cells
const ACCENT4 = '#880e4f';   // deep magenta — misdiagnosis; underdiagnosis; old panels
const ACCENT5 = '#006064';   // dark teal — C-peptide preserved; functional defect
const ACCENT6 = '#37474f';   // dark slate — epidemiology; family history; variable expressivity
const ACCENT7 = '#b71c1c';   // deep red — neurological alert; cerebellar ataxia; deafness
const ACCENT8 = '#e65100';   // amber — progressive HbA1c; insulin failure risk

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

export default function MODY6Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody6/overview`).then(r => r.json()),
      fetch(`${API}/api/mody6/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody6/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY6 — NEUROD1-MODY / BETA2-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 6 · Chr 2q31.3 · OMIM #606394 · ~1–2% of all MODY · NEUROD1 bHLH E-box TF · Cooperates with PDX1 · SU-Responsive · Neurological Spectrum · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="NEUROD1 *601724" color={ACCENT2} />
            <Badge text="bHLH E-box TF" color={ACCENT} />
            <Badge text="SU first-line" color={ACCENT3} />
            <Badge text="R111L founding" color={ACCENT6} />
            <Badge text="Neuro alert" color={ACCENT7} />
            <Badge text="~1–2% MODY" color={ACCENT4} />
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
        <KPI label="On SU (%)" value={`${kpis.pct_on_sulfonylurea?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="SU Responders (%)" value={`${kpis.pct_su_responders_of_su_treated?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT6} />
        <KPI label="Misdiagnosed (%)" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="Neuro Features (%)" value={`${kpis.pct_neurological_features?.toFixed(0)}%`} color={ACCENT7} />
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
              <Alert color={ACCENT}>
                <strong>MODY6-NEUROD1/BETA2: bHLH Transcription Factor — E-box Insulin Promoter Binding.</strong> NEUROD1 (Neurogenic Differentiation 1, alias BETA2) is a basic helix-loop-helix TF that dimerizes with E12/E47 and binds E-box elements (CANNTG) in the insulin gene promoter. It cooperates synergistically with <em>PDX1</em> (MODY4, A-box) and <em>MafA</em> (C1/RIPE3b) to achieve full beta-cell INS transcription. Haploinsufficiency → reduced INS/GCK transcription → impaired GSIS → progressive diabetes.
              </Alert>
              <Alert color={ACCENT7}>
                <strong>NEUROLOGICAL SPECTRUM — UNIQUE AMONG MODY TYPES:</strong> NEUROD1 is expressed in cerebellar granule neurons, hippocampus, and inner ear hair cells. Heterozygous LOF → pure MODY6 (no neurological features in most). Biallelic/compound-het LOF → <em>cerebellar ataxia + sensorineural deafness</em> + severe DM. Screen audiology and neurology if unexplained ataxia or hearing loss in a patient with MODY6 or their family members.
              </Alert>
              <Alert color={ACCENT4}>
                <strong>NOT IN OLDEST MODY PANELS — UNDERDIAGNOSED:</strong> Pre-2010 MODY panels targeted only HNF1A/HNF4A/GCK/HNF1B. NEUROD1 was added in expanded NGS panels. Many MODY6 families labelled T1D (antibody-negative, young, insulin-requiring) or T2D (older relatives) for years without genetic testing with an appropriate panel. Must use <em>expanded MODY NGS panel</em> including NEUROD1.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>SULFONYLUREA FIRST-LINE (~80–85% response):</strong> Beta-cells present and functionally impaired — not structurally lost (unlike MODY5). SU closes K-ATP channels → depolarization → Ca²⁺ influx → insulin exocytosis → bypasses the transcriptional deficit. Start low dose (glibenclamide 2.5 mg or gliclazide 40 mg); titrate up.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>NEUROD1 / BETA2 (Neurogenic Differentiation 1) · Chr 2q31.3 · *601724</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#606394 (MODY6)</td></tr>
                  <tr><td className="fw-bold">TF family</td><td>bHLH (basic helix-loop-helix) — Class B tissue-specific; dimerizes with E12/E47</td></tr>
                  <tr><td className="fw-bold">Binding motif</td><td>E-box CANNTG — insulin promoter E1 element; also GCK, GLP1R promoters</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence · biallelic LOF → ataxia + deafness + severe DM</td></tr>
                  <tr><td className="fw-bold">Mechanism</td><td>NEUROD1 haploinsufficiency → reduced INS/GCK/GLP1R transcription → impaired GSIS → progressive DM</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1–2% of all MODY; underdiagnosed; absent from oldest gene panels</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Late teens to 50s; variable expressivity within families</td></tr>
                  <tr><td className="fw-bold">HbA1c</td><td>Progressive (rising with duration); responds well to sulfonylurea</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>PRESERVED at diagnosis (functional defect, not structural loss); falls with duration</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — mandatory to exclude T1D</td></tr>
                  <tr><td className="fw-bold">Neurological</td><td>Usually ABSENT (heterozygous); PRESENT (biallelic/severe LOF) → ataxia + deafness</td></tr>
                  <tr><td className="fw-bold">Founding variant</td><td>R111L (Arg111Leu) bHLH domain — Malecki et al. Nat Genet 1999</td></tr>
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="📋 Key Clinical Facts" color={ACCENT2}>
              <ul className="list-group list-group-flush">
                {keyFacts.map((f, i) => (
                  <li key={i} className="list-group-item py-1 small">{f}</li>
                ))}
              </ul>
            </Section>
            <Section title="🩺 Diagnostic Criteria" color={ACCENT6}>
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
            <Section title="🔬 NEUROD1 Variant Distribution" color={ACCENT2}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td>
                          <span className="badge me-1" style={{ background: v === 'R111L' ? ACCENT7 : ACCENT2, fontSize: '0.72em' }}>{v}</span>
                          {v === 'R111L' && <span className="badge" style={{ background: ACCENT, fontSize: '0.68em' }}>founding bHLH</span>}
                        </td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 HbA1c Tiers" color={ACCENT8}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="💊 C-Peptide Tiers (preserved vs falling)" color={ACCENT5}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>C-peptide (nmol/L)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.c_peptide_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="🕐 Age at Diagnosis Tiers" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Age at Dx</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.age_at_diagnosis_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="⏳ Disease Duration Tiers" color={ACCENT2}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Duration</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.disease_duration_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-7">
            <Section title="👩‍⚕️ Patient Cohort (40 patients, seed 313)" color={ACCENT}>
              <div style={{ maxHeight: 580, overflowY: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.71em' }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Dx Age</th><th>Dur</th><th>HbA1c%</th>
                      <th>C-pep</th><th>Variant</th><th>Treatment</th><th>Fam Hx</th><th>Prior Dx</th><th>Neuro</th>
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map(p => (
                      <tr key={p.patient_id}>
                        <td>{p.patient_id}</td>
                        <td>{p.age}</td>
                        <td>{p.sex}</td>
                        <td>{p.age_at_diagnosis}</td>
                        <td>{p.duration_years}</td>
                        <td>
                          <span style={{ color: p.hba1c_percent > 8.5 ? ACCENT8 : p.hba1c_percent > 7.5 ? ACCENT : ACCENT3, fontWeight: 600 }}>
                            {p.hba1c_percent?.toFixed(1)}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: p.c_peptide_nmol_L < 0.30 ? ACCENT7 : p.c_peptide_nmol_L < 0.60 ? ACCENT8 : ACCENT5, fontWeight: 600 }}>
                            {p.c_peptide_nmol_L?.toFixed(2)}
                          </span>
                        </td>
                        <td style={{ fontSize: '0.68em' }}>
                          {p.variant === 'R111L'
                            ? <span style={{ color: ACCENT7, fontWeight: 600 }}>{p.variant}</span>
                            : p.variant}
                        </td>
                        <td style={{ fontSize: '0.65em' }}>{p.current_treatment}</td>
                        <td style={{ color: p.family_history_positive ? ACCENT6 : ACCENT4 }}>
                          {p.family_history_positive ? 'Yes' : 'No'}
                        </td>
                        <td style={{ fontSize: '0.65em', color: p.prior_misdiagnosis !== 'None' ? ACCENT4 : '' }}>
                          {p.prior_misdiagnosis}
                        </td>
                        <td style={{ color: p.neurological_features ? ACCENT7 : '' }}>
                          {p.neurological_features ? '⚠ Yes' : 'No'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 2: Treatment & Comparison */}
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
            <Section title="🏥 Misdiagnosis Distribution" color={ACCENT4}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Prior Diagnosis</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.misdiagnosis_distribution || {}).map(([m, n]) => (
                    <tr key={m}><td>{m.replace(/_/g, ' ')}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="⚡ SU Hypoglycaemia (patients on sulfonylurea)" color={ACCENT3}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Episodes/yr (SU patients only)</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.su_hypoglycaemia_tiers_on_su_patients || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="👥 Current Age Groups" color={ACCENT6}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Age group</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.age_groups_current || {}).map(([g, n]) => (
                    <tr key={g}><td>{g}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 Summary Flags" color={ACCENT}>
              <table className="table table-sm table-bordered">
                <tbody>
                  {Object.entries(breakdown?.summary_flags || {}).map(([k, v]) => (
                    <tr key={k}>
                      <td className="fw-bold small">{k.replace(/pct_/,'').replace(/_/g,' ')}</td>
                      <td className="fw-bold" style={{ color: ACCENT }}>{v}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>
          </div>
          <div className="col-lg-6">
            <Section title="🆚 MODY6 vs Other MODY Types — Key Differentiators" color={ACCENT}>
              <div style={{ overflowX: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.80em' }}>
                  <thead className="table-dark">
                    <tr><th>Feature</th><th>MODY6 (NEUROD1)</th><th>MODY4 (PDX1)</th><th>MODY3 (HNF1A)</th><th>MODY5 (HNF1B)</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>TF family</td>
                      <td style={{ color: ACCENT, fontWeight: 600 }}>bHLH (E-box)</td>
                      <td>Homeodomain (A-box)</td>
                      <td>Homeodomain</td>
                      <td>Homeodomain</td>
                    </tr>
                    <tr>
                      <td>MODY frequency</td>
                      <td style={{ color: ACCENT4 }}>~1–2%</td>
                      <td style={{ color: ACCENT4 }}>~1% (rarest)</td>
                      <td>~35%</td>
                      <td>~5%</td>
                    </tr>
                    <tr>
                      <td>Neurological features</td>
                      <td style={{ color: ACCENT7, fontWeight: 600 }}>POSSIBLE (biallelic)</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Renal glycosuria</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT3 }}>PRESENT (50%)</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Renal cysts</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT7 }}>PRESENT (~70%)</td>
                    </tr>
                    <tr>
                      <td>Pancreatic atrophy</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT5 }}>ABSENT</td>
                      <td style={{ color: ACCENT7 }}>PRESENT (CT/MRI)</td>
                    </tr>
                    <tr>
                      <td>SU response</td>
                      <td style={{ color: ACCENT3, fontWeight: 600 }}>YES (~80–85%)</td>
                      <td style={{ color: ACCENT3 }}>YES (85–90%)</td>
                      <td style={{ color: ACCENT3 }}>YES (85–90%)</td>
                      <td style={{ color: ACCENT7 }}>NO (atrophy)</td>
                    </tr>
                    <tr>
                      <td>HbA1c pattern</td>
                      <td>Progressive ↑</td>
                      <td>Progressive ↑</td>
                      <td>Progressive ↑</td>
                      <td>Progressive ↑</td>
                    </tr>
                    <tr>
                      <td>C-peptide at Dx</td>
                      <td style={{ color: ACCENT5 }}>Preserved → falls</td>
                      <td style={{ color: ACCENT5 }}>Preserved → falls</td>
                      <td style={{ color: ACCENT5 }}>Preserved → falls</td>
                      <td style={{ color: ACCENT7 }}>Low / falling</td>
                    </tr>
                    <tr>
                      <td>In oldest MODY panels</td>
                      <td style={{ color: ACCENT7, fontWeight: 600 }}>NO — expanded only</td>
                      <td style={{ color: ACCENT7 }}>NO — expanded only</td>
                      <td style={{ color: ACCENT3 }}>YES (original panel)</td>
                      <td style={{ color: ACCENT3 }}>YES (original panel)</td>
                    </tr>
                    <tr>
                      <td>Founding variant</td>
                      <td style={{ color: ACCENT }}>R111L (bHLH domain)</td>
                      <td>P63fsdelC (exon 1)</td>
                      <td>P291fsinsC</td>
                      <td>17q12 deletion</td>
                    </tr>
                    <tr>
                      <td>Family history</td>
                      <td>~75–80%</td>
                      <td>~80–90%</td>
                      <td>~90%</td>
                      <td>~50% (50% de-novo)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Section>
            <Section title="🧠 NEUROD1 Neurological Alert — When to Screen" color={ACCENT7}>
              <Alert color={ACCENT7}>
                <strong>Cerebellar ataxia + sensorineural deafness in a MODY6 family = biallelic/compound-het LOF risk.</strong> NEUROD1 is essential for cerebellar granule neuron differentiation and cochlear hair cell development. If a patient with MODY6 has unexplained cerebellar ataxia or progressive sensorineural hearing loss, or if a family member has these features alongside diabetes, test for a <em>second NEUROD1 pathogenic allele</em>. Refer to neurology + audiology. Biallelic NEUROD1 LOF → neonatal/early-onset severe DM + progressive ataxia + deafness.
              </Alert>
              <table className="table table-sm table-bordered" style={{ fontSize: '0.82em' }}>
                <thead className="table-dark">
                  <tr><th>NEUROD1 Dosage</th><th>Phenotype</th><th>Features</th><th>Action</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td>1 copy LOF (heterozygous)</td>
                    <td>MODY6</td>
                    <td>Progressive DM; no neuro in most</td>
                    <td>SU first-line; diet if mild; monitor</td>
                  </tr>
                  <tr>
                    <td>2 copies LOF (biallelic/comp-het)</td>
                    <td>Severe NEUROD1 LOF syndrome</td>
                    <td>Early DM + cerebellar ataxia + deafness</td>
                    <td>Insulin; neurology + audiology referral; URGENT</td>
                  </tr>
                </tbody>
              </table>
              <Alert color={ACCENT4}>
                <strong>Panel note — EXPANDED MODY panel required:</strong> NEUROD1 is not in the oldest 4-gene panels (HNF1A / HNF4A / GCK / HNF1B). If clinical suspicion for MODY6 and standard panel is negative, request expanded MODY NGS panel explicitly including NEUROD1, PDX1, KLF11, CEL, PAX4, INS, BLK.
              </Alert>
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
