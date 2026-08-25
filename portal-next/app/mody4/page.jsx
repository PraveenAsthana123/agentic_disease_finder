'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variant & Cohort', 'Treatment & Comparison', 'Definitions'];

// MODY4 colour scheme — rich amber/saffron (rarest MODY; PDX1 master TF; SU-responsive)
const ACCENT  = '#e65100';   // deep amber-orange — PDX1 master TF; rarest MODY
const ACCENT2 = '#4e342e';   // dark brown — exon 1 GC-rich; two-hit dosage
const ACCENT3 = '#1b5e20';   // deep green — SU-responsive; functional defect
const ACCENT4 = '#880e4f';   // deep magenta — misdiagnosis risk; variable expressivity
const ACCENT5 = '#1a237e';   // deep indigo — genetics; OMIM; panel testing
const ACCENT6 = '#37474f';   // dark slate — epidemiology; rarest; family history
const ACCENT7 = '#b71c1c';   // deep red — two-hit risk; homozygous → PNDM
const ACCENT8 = '#006064';   // dark teal — C-peptide preserved; functional beta-cells

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

export default function MODY4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/mody4/overview`).then(r => r.json()),
      fetch(`${API}/api/mody4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mody4/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MODY4 — PDX1-MODY / IPF1-MODY</h4>
            <div className="text-muted small">Maturity-Onset Diabetes of the Young Type 4 · Chr 13q12.2 · OMIM #606392 · ~1% of all MODY (rarest) · PDX1 Master Beta-Cell TF · SU-Responsive · Two-Hit Dosage Effect · Autosomal Dominant</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="PDX1 *600733" color={ACCENT5} />
            <Badge text="Rarest ~1% MODY" color={ACCENT6} />
            <Badge text="SU first-line" color={ACCENT3} />
            <Badge text="Two-hit → PNDM" color={ACCENT7} />
            <Badge text="C-pep preserved" color={ACCENT8} />
            <Badge text="P63fsdelC founding" color={ACCENT2} />
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Cohort" value={kpis.cohort_size || 40} color={ACCENT} />
        <KPI label="Mean Age (yr)" value={kpis.mean_age_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Dx Age (yr)" value={kpis.mean_age_at_diagnosis_years?.toFixed(1)} color={ACCENT6} />
        <KPI label="Mean Duration (yr)" value={kpis.mean_duration_years?.toFixed(1)} color={ACCENT2} />
        <KPI label="Mean HbA1c (%)" value={kpis.mean_hba1c_percent?.toFixed(1)} color={ACCENT4} />
        <KPI label="Mean FG (mmol/L)" value={kpis.mean_fasting_glucose_mmol?.toFixed(1)} color={ACCENT4} />
        <KPI label="Mean C-pep (nmol/L)" value={kpis.mean_c_peptide_nmol_L?.toFixed(3)} color={ACCENT8} />
        <KPI label="On SU (%)" value={`${kpis.pct_on_sulfonylurea?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="SU Responders (%)" value={`${kpis.pct_su_responders_of_su_treated?.toFixed(0)}%`} color={ACCENT3} />
        <KPI label="Family Hx (%)" value={`${kpis.pct_family_hx_positive?.toFixed(0)}%`} color={ACCENT6} />
        <KPI label="Misdiagnosed (%)" value={`${kpis.pct_prior_misdiagnosis?.toFixed(0)}%`} color={ACCENT4} />
        <KPI label="Insulin Required (%)" value={`${kpis.pct_insulin_treated?.toFixed(0)}%`} color={ACCENT7} />
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
              <Alert color={ACCENT6}>
                <strong>MODY4-RAREST: ~1% of all MODY.</strong> PDX1/IPF1 (Pancreatic and Duodenal Homeobox 1) is the <em>master transcription factor</em> for beta-cell identity — it directly drives insulin gene transcription, GCK, GLUT2, PC1/PCSK1, MafA, and Nkx6.1. Heterozygous LOF → haploinsufficiency → reduced insulin secretion → progressive diabetes. Fewer than 200 families described in early literature; prevalence underestimated.
              </Alert>
              <Alert color={ACCENT7}>
                <strong>TWO-HIT DOSAGE EFFECT — Critical Family Risk:</strong> Heterozygous PDX1 LOF → MODY4 (moderate, adult onset). Compound heterozygous or homozygous PDX1 LOF → <em>pancreatic agenesis</em> or <em>PNDM</em> (neonatal-onset, no pancreas, insulin-requiring from birth). Finding a pathogenic PDX1 variant MANDATES second-hit family screening — a sibling with two hits may have PNDM.
              </Alert>
              <Alert color={ACCENT3}>
                <strong>SULFONYLUREA FIRST-LINE (like MODY1/MODY3):</strong> Beta-cells are present and functionally impaired (not structurally lost, unlike MODY5). SU closes K-ATP channels → depolarization → Ca²⁺ influx → insulin exocytosis → bypasses the secretory defect. 85–90% excellent response rate; start low dose (glibenclamide 2.5 mg or gliclazide 40 mg), titrate up.
              </Alert>
              <Alert color={ACCENT2}>
                <strong>P63fsdelC — THE FOUNDING MODY4 MUTATION:</strong> Pro63 frameshift deletion (c.186delC) in the GC-rich region of PDX1 exon 1 was reported by Stoffers DA et al. (Nat Genet 1997) — the first human PDX1 variant causing MODY. GC-rich exon 1 is a mutational hotspot; standard NGS may fail here — verify coverage or use Sanger.
              </Alert>
              <table className="table table-sm table-bordered">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>PDX1 / IPF1 (Pancreatic and Duodenal Homeobox 1) · Chr 13q12.2 · *600733</td></tr>
                  <tr><td className="fw-bold">Disease OMIM</td><td>#606392 (MODY4)</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Dominant · 50% recurrence · homozygous LOF → pancreatic agenesis</td></tr>
                  <tr><td className="fw-bold">Mechanism</td><td>PDX1 haploinsufficiency → reduced INS/GCK/GLUT2/PC1 transcription → impaired GSIS → progressive diabetes</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>~1% of all MODY (rarest classical MODY; likely underdiagnosed)</td></tr>
                  <tr><td className="fw-bold">Onset age</td><td>Teens to 50s (variable; wider range than MODY3); mean ~35 yr in some series</td></tr>
                  <tr><td className="fw-bold">HbA1c</td><td>Progressive (not stable like MODY2); 5.8–9.5%; responds well to sulfonylurea</td></tr>
                  <tr><td className="fw-bold">C-peptide</td><td>PRESERVED at diagnosis (functional defect, not structural loss); falls with duration</td></tr>
                  <tr><td className="fw-bold">Autoantibodies</td><td>NEGATIVE (GADA, ZnT8, IA-2) — mandatory to exclude T1D</td></tr>
                  <tr><td className="fw-bold">Pancreatic atrophy</td><td>ABSENT (unlike MODY5) — exocrine function preserved in heterozygous state</td></tr>
                  <tr><td className="fw-bold">Renal cysts</td><td>ABSENT (PDX1 not expressed in kidney; no renal phenotype)</td></tr>
                  <tr><td className="fw-bold">Renal glycosuria</td><td>ABSENT (PDX1 does not regulate SGLT2)</td></tr>
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
            <Section title="🔬 PDX1 Variant Distribution" color={ACCENT5}>
              <table className="table table-sm table-bordered table-hover">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.variant_distribution || {})
                    .sort((a, b) => b[1] - a[1])
                    .map(([v, n]) => (
                      <tr key={v}>
                        <td>
                          <span className="badge me-1" style={{ background: v === 'P63fsdelC' ? ACCENT7 : ACCENT5, fontSize: '0.72em' }}>{v}</span>
                          {v === 'P63fsdelC' && <span className="badge" style={{ background: ACCENT2, fontSize: '0.68em' }}>founding</span>}
                        </td>
                        <td>{n}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </Section>
            <Section title="📊 HbA1c Tiers" color={ACCENT4}>
              <table className="table table-sm table-bordered">
                <thead><tr><th>Tier</th><th>n</th></tr></thead>
                <tbody>
                  {Object.entries(breakdown?.hba1c_tiers || {}).map(([t, n]) => (
                    <tr key={t}><td>{t}</td><td>{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>
            <Section title="💊 C-Peptide Tiers (preserved vs falling)" color={ACCENT8}>
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
            <Section title="👩‍⚕️ Patient Cohort (40 patients, seed 311)" color={ACCENT}>
              <div style={{ maxHeight: 580, overflowY: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.71em' }}>
                  <thead className="table-dark">
                    <tr>
                      <th>ID</th><th>Age</th><th>Sex</th><th>Dx Age</th><th>Dur</th><th>HbA1c%</th>
                      <th>C-pep</th><th>Variant</th><th>Treatment</th><th>Fam Hx</th><th>Prior Dx</th>
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
                          <span style={{ color: p.hba1c_percent > 8.5 ? ACCENT4 : p.hba1c_percent > 7.5 ? ACCENT : ACCENT3, fontWeight: 600 }}>
                            {p.hba1c_percent?.toFixed(1)}
                          </span>
                        </td>
                        <td>
                          <span style={{ color: p.c_peptide_nmol_L < 0.30 ? ACCENT7 : p.c_peptide_nmol_L < 0.60 ? ACCENT : ACCENT8, fontWeight: 600 }}>
                            {p.c_peptide_nmol_L?.toFixed(2)}
                          </span>
                        </td>
                        <td style={{ fontSize: '0.68em' }}>
                          {p.variant === 'P63fsdelC'
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
            <Section title="🆚 MODY4 vs Other MODY Types — Key Differentiators" color={ACCENT}>
              <div style={{ overflowX: 'auto' }}>
                <table className="table table-sm table-bordered table-hover" style={{ fontSize: '0.80em' }}>
                  <thead className="table-dark">
                    <tr><th>Feature</th><th>MODY4 (PDX1)</th><th>MODY3 (HNF1A)</th><th>MODY5 (HNF1B)</th><th>MODY2 (GCK)</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>MODY frequency</td>
                      <td style={{ color: ACCENT7, fontWeight: 600 }}>~1% (rarest)</td>
                      <td>~35%</td><td>~5%</td><td>~25–35%</td>
                    </tr>
                    <tr>
                      <td>Renal glycosuria</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT3 }}>PRESENT (50%)</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Renal cysts</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT7 }}>PRESENT (~70%)</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Pancreatic atrophy</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT7 }}>PRESENT (CT/MRI)</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Sulfonylure response</td>
                      <td style={{ color: ACCENT3, fontWeight: 600 }}>YES (85–90%)</td>
                      <td style={{ color: ACCENT3 }}>YES (85–90%)</td>
                      <td style={{ color: ACCENT7 }}>NO (atrophy)</td>
                      <td style={{ color: ACCENT7 }}>NO (causes hypoglycaemia)</td>
                    </tr>
                    <tr>
                      <td>HbA1c pattern</td>
                      <td>Progressive ↑</td>
                      <td>Progressive ↑</td>
                      <td>Progressive ↑</td>
                      <td style={{ color: ACCENT3, fontWeight: 600 }}>STABLE 5.6–7.6%</td>
                    </tr>
                    <tr>
                      <td>C-peptide at Dx</td>
                      <td style={{ color: ACCENT8 }}>Preserved → falls</td>
                      <td style={{ color: ACCENT8 }}>Preserved → falls</td>
                      <td style={{ color: ACCENT7 }}>Low / falling</td>
                      <td style={{ color: ACCENT3 }}>NORMAL</td>
                    </tr>
                    <tr>
                      <td>Macrosomia/TNH</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Exocrine insufficiency</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                      <td style={{ color: ACCENT7 }}>PRESENT (~40%)</td>
                      <td style={{ color: ACCENT8 }}>ABSENT</td>
                    </tr>
                    <tr>
                      <td>Two-hit → neonatal DM</td>
                      <td style={{ color: ACCENT7, fontWeight: 600 }}>YES (PNDM risk)</td>
                      <td style={{ color: ACCENT8 }}>Not known</td>
                      <td style={{ color: ACCENT }}>MODY → PNDM (severe)</td>
                      <td style={{ color: ACCENT3 }}>YES (GCK homoz. → PNDM)</td>
                    </tr>
                    <tr>
                      <td>Family history</td>
                      <td>~80–90%</td>
                      <td>~90%</td>
                      <td>~50% (50% de-novo)</td>
                      <td>~75–80%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Section>
            <Section title="🏥 PDX1 Two-Hit Risk — Critical Family Screening Alert" color={ACCENT7}>
              <Alert color={ACCENT7}>
                <strong>If a pathogenic PDX1 variant is found:</strong> Screen ALL first-degree relatives. If a sibling carries the <em>same variant</em> AND a second pathogenic PDX1 variant (compound heterozygous), they may develop <strong>pancreatic agenesis or PNDM</strong> (neonatal-onset diabetes, no pancreas). This is a medical emergency — refer urgently to specialist. PDX1 homozygous = no PDX1 protein = no pancreas.
              </Alert>
              <table className="table table-sm table-bordered" style={{ fontSize: '0.82em' }}>
                <thead className="table-dark"><tr><th>PDX1 Dosage</th><th>Phenotype</th><th>Onset</th><th>Action</th></tr></thead>
                <tbody>
                  <tr><td>1 copy LOF (heterozygous)</td><td>MODY4</td><td>Teens–50s</td><td>SU first-line; diet if mild</td></tr>
                  <tr><td>0 copies (homozygous / comp-het)</td><td>Pancreatic agenesis / PNDM</td><td>Neonatal</td><td>Insulin lifelong; Creon; URGENT referral</td></tr>
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
