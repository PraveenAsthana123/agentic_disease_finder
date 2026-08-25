'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// Joubert Syndrome colour scheme — teal-purple-amber-crimson (TZ ciliopathy; MTS; retinal; renal-hepatic)
const ACCENT  = '#00695c';   // dark teal — JBTS; TZ ciliopathy; primary
const ACCENT2 = '#4527a0';   // deep purple — MTS; cerebellar; molar tooth sign
const ACCENT3 = '#e65100';   // deep orange — renal/NPHP; transplant
const ACCENT4 = '#b71c1c';   // dark crimson — hepatic/CHF; portal hypertension
const ACCENT5 = '#1565c0';   // dark blue — retinal dystrophy; LCA-like; CEP290
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR; cohort
const ACCENT7 = '#4e342e';   // dark brown — neonatal apnoea; breathing dysrhythmia
const ACCENT8 = '#558b2f';   // dark olive-green — NOT primary DM; secondary only

const _COHORT_SIZE = 40;

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

function Bar({ label, value, max, color }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{value}</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function JoubertPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/joubert/overview`).then(r => r.json()),
      fetch(`${API}/api/joubert/breakdown`).then(r => r.json()),
      fetch(`${API}/api/joubert/definitions`).then(r => r.json()),
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
        <div className="d-flex align-items-center gap-3 flex-wrap">
          <div>
            <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>🧬 Joubert Syndrome</h4>
            <div className="text-muted small">JBTS — Ciliary Transition Zone (TZ) Ciliopathy · Molar Tooth Sign · CEP290 / AHI1 / INPP5E / TMEM67</div>
            <div className="mt-1">
              <Badge text="CEP290 *610142" color={ACCENT5} />
              <Badge text="OMIM #213300" color={ACCENT} />
              <Badge text="12q21.32 (CEP290)" color={ACCENT2} />
              <Badge text="AR Biallelic LOF" color={ACCENT6} />
              <Badge text="~1/80,000–1/100,000" color={ACCENT3} />
              <Badge text="TZ Ciliopathy" color={ACCENT} />
              <Badge text={`n=${_COHORT_SIZE}`} color={ACCENT6} />
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)} style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Tab 0: Overview */}
      {tab === 0 && (
        <div>
          {/* Alerts */}
          <div className="mb-3">
            <Alert color={ACCENT2}><strong>🧠 Brain MRI MANDATORY:</strong> {alerts.mts_mandatory}</Alert>
            <Alert color={ACCENT7}><strong>🫁 Neonatal Apnoea:</strong> {alerts.neonatal_apnoea}</Alert>
            <Alert color={ACCENT3}><strong>🫘 Renal Screen:</strong> {alerts.renal_screen}</Alert>
            <Alert color={ACCENT4}><strong>🫀 Hepatic Screen:</strong> {alerts.hepatic_screen}</Alert>
          </div>

          {/* KPIs */}
          <div className="row g-2 mb-4">
            <KPI label="Cohort" value={kpis.cohort_n} color={ACCENT6} />
            <KPI label="Median Age" value={`${kpis.median_age} yr`} color={ACCENT6} />
            <KPI label="Mean BMI" value={`${kpis.mean_bmi} kg/m²`} color={ACCENT6} />
            <KPI label="Mean eGFR" value={`${kpis.mean_egfr} ml/min`} color={ACCENT3} />
            <KPI label="Retinal Involvement" value={`${kpis.pct_retinal}%`} color={ACCENT5} />
            <KPI label="Renal Disease" value={`${kpis.pct_renal}%`} color={ACCENT3} />
            <KPI label="ESRD" value={`${kpis.pct_esrd}%`} color={ACCENT3} />
            <KPI label="Hepatic / CHF" value={`${kpis.pct_hepatic}%`} color={ACCENT4} />
            <KPI label="Polydactyly" value={`${kpis.pct_polydactyly}%`} color={ACCENT8} />
            <KPI label="T2D (secondary)" value={`${kpis.pct_dm}%`} color={ACCENT8} />
            <KPI label="CEP290 %" value={`${kpis.pct_cep290}%`} color={ACCENT5} />
            <KPI label="Mean IQ est." value={kpis.mean_iq} color={ACCENT2} />
          </div>

          {/* Mechanism box */}
          <Section title="🔬 Mechanism: Transition Zone (TZ) Ciliopathy" color={ACCENT}>
            <div className="card shadow-sm mb-3">
              <div className="card-body small">
                <div className="row">
                  <div className="col-md-6">
                    <strong style={{ color: ACCENT }}>TZ ciliopathy (JBTS):</strong>
                    <ul className="mt-1 ps-3">
                      <li>JBTS proteins (CEP290, AHI1, INPP5E, CC2D2A, TMEM67) = TZ structural components</li>
                      <li>TZ = gating compartment at ciliary base — controls protein entry/exit</li>
                      <li>TZ loss → SHH, PDGF-Rα, Wnt, PI(4,5)P2 signalling failure</li>
                      <li>Cerebellar granule cell migration failure → <strong>Molar Tooth Sign</strong></li>
                    </ul>
                    <strong style={{ color: ACCENT2 }}>Molar Tooth Sign (MTS) anatomy:</strong>
                    <ul className="mt-1 ps-3">
                      <li>SCP elongation (horizontal course → 'roots')</li>
                      <li>Cerebellar vermis aplasia/hypoplasia → deepened IP fossa ('pulp chamber')</li>
                      <li>4th ventricle bat-wing deformation ('crown')</li>
                      <li>Together = <em>molar tooth</em> on axial T2 MRI — PATHOGNOMONIC</li>
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <strong style={{ color: ACCENT5 }}>vs. BBS (BBSome IFT mis-trafficking):</strong>
                    <ul className="mt-1 ps-3">
                      <li>BBS: retrograde IFT cargo failure → LepR, SSTR3, GPCRs fail to enter cilia</li>
                      <li>JBTS: TZ gating collapse → SHH/Wnt/PDGF fail → developmental malformation</li>
                      <li>BBS: NO MTS; brain MRI normal. JBTS: MTS always present</li>
                      <li>BBS: obesity dominant, T2D 50%. JBTS: DM secondary only</li>
                    </ul>
                    <strong style={{ color: ACCENT3 }}>CEP290 allele spectrum:</strong>
                    <ul className="mt-1 ps-3">
                      <li>Severe truncating → Meckel-Gruber (lethal)</li>
                      <li>IVS26 / mild missense → <strong>JBTS</strong> (with/without retinal)</li>
                      <li>IVS26 biallelic → LCA10 (retinal only; NO MTS)</li>
                      <li>Mild → BBS14 (BBSome cargo defect)</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          {/* Key Facts */}
          <Section title="🔑 Key Clinical Facts" color={ACCENT2}>
            <ul className="small ps-3">
              {keyFacts.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </Section>

          {/* Patients preview */}
          <Section title={`👥 Cohort Preview (n=${_COHORT_SIZE}, seed 335)`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm table-striped small">
                <thead>
                  <tr>
                    <th>#</th><th>Age</th><th>Sex</th><th>Gene</th><th>Subtype</th><th>Retinal</th><th>Renal eGFR</th><th>Hepatic</th><th>HbA1c</th><th>Misdiagnosis</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td>{p.age}</td>
                      <td>{p.sex}</td>
                      <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.gene}>{p.gene.split('(')[0].trim()}</td>
                      <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.jbts_subtype}>{p.jbts_subtype.split('(')[0].trim()}</td>
                      <td style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.retinal_status}>{p.retinal_status.split('(')[0].trim()}</td>
                      <td>{p.egfr_ml_min} ml/min</td>
                      <td style={{ maxWidth: 110, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.hepatic_status}>{p.hepatic_status.split('(')[0].trim()}</td>
                      <td>{p.hba1c}%</td>
                      <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={p.prior_misdiagnosis}>{p.prior_misdiagnosis.split('(')[0].trim()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* Tab 1: Multi-System Breakdown */}
      {tab === 1 && breakdown && (
        <div className="row">
          <div className="col-md-6">
            <Section title="🧬 Gene Distribution" color={ACCENT5}>
              {Object.entries(breakdown.gene_distribution || {}).sort((a,b)=>b[1]-a[1]).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
            <Section title="🧠 JBTS Subtype" color={ACCENT2}>
              {Object.entries(breakdown.jbts_subtype || {}).sort((a,b)=>b[1]-a[1]).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
            <Section title="👁️ Retinal Status" color={ACCENT5}>
              {Object.entries(breakdown.retinal_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
            <Section title="🫘 Renal / eGFR Tiers" color={ACCENT3}>
              {Object.entries(breakdown.egfr_tiers || {}).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="🫀 Hepatic / CHF Status" color={ACCENT4}>
              {Object.entries(breakdown.hepatic_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="📊 HbA1c Distribution (T2D secondary only)" color={ACCENT8}>
              {Object.entries(breakdown.hba1c_tiers || {}).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
              ))}
            </Section>
            <Section title="⚖️ BMI Distribution" color={ACCENT6}>
              {Object.entries(breakdown.bmi_tiers || {}).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="🔁 Prior Misdiagnosis" color={ACCENT7}>
              {Object.entries(breakdown.misdiagnosis || {}).sort((a,b)=>b[1]-a[1]).map(([k,v]) => (
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
          </div>

          {/* Summary flags */}
          {breakdown.summary_flags && (
            <div className="col-12 mt-2">
              <Section title="📋 Cohort Summary Flags" color={ACCENT}>
                <div className="row g-2">
                  {Object.entries(breakdown.summary_flags).map(([k, v]) => (
                    <div key={k} className="col-6 col-md-3 col-lg-2">
                      <div className="card text-center shadow-sm">
                        <div className="card-body py-2">
                          <div className="fw-bold" style={{ color: ACCENT }}>{typeof v === 'number' ? `${v}%` : v}</div>
                          <div className="text-muted" style={{ fontSize: '0.68em' }}>{k.replace(/_/g, ' ')}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </Section>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Treatment & Diagnostics */}
      {tab === 2 && (
        <div className="row">
          <div className="col-md-6">
            <Section title="🩺 Diagnostic Algorithm" color={ACCENT2}>
              <div className="card shadow-sm mb-3">
                <div className="card-body small">
                  <ol className="ps-3 mb-0">
                    <li><strong>Brain MRI (axial T2):</strong> Look for Molar Tooth Sign — MANDATORY FIRST TEST</li>
                    <li><strong>Gene panel (≥35 JBTS genes):</strong> after MTS confirmed — CEP290, AHI1, INPP5E, CC2D2A, TMEM67, RPGRIP1L, KIF7, TCTN1-3, NPHP1/4</li>
                    <li><strong>If panel negative:</strong> Whole Exome Sequencing (WES) — 20-30% JBTS unexplained</li>
                    <li><strong>ERG:</strong> documents retinal involvement (rod-cone pattern); baseline</li>
                    <li><strong>Renal USS + urine ACR + eGFR:</strong> NPHP screening — annually from diagnosis</li>
                    <li><strong>LFTs + liver USS:</strong> CHF / Caroli screening (TMEM67/CC2D2A subtypes)</li>
                    <li><strong>Ophthalmology:</strong> fundus photography; OCT; visual fields — 6-monthly</li>
                    <li><strong>Polysomnography (neonatal):</strong> apnoea/hyperpnoea monitoring until resolved</li>
                    <li><strong>Parental karyotype:</strong> if chromosomal rearrangement suspected (rare)</li>
                  </ol>
                </div>
              </div>
            </Section>

            <Section title="🧠 Neurological Management" color={ACCENT2}>
              <div className="card shadow-sm">
                <div className="card-body small">
                  <ul className="ps-3 mb-0">
                    <li>Physiotherapy (hypotonia → walking support; most walk by age 6 with support)</li>
                    <li>Occupational therapy (fine motor; ADL); speech-language therapy</li>
                    <li>Early educational intervention; special education programme</li>
                    <li>Neonatal apnoea: caffeine citrate (loading 20 mg/kg; maintenance 5-10 mg/kg/day); O2 supplement; resolves by 2-3 years</li>
                    <li>Developmental milestones monitoring; annual neurodevelopmental assessment</li>
                    <li>Brain MRI: baseline then if clinically indicated (progressive symptoms)</li>
                  </ul>
                </div>
              </div>
            </Section>

            <Section title="👁️ Retinal Management (CEP290/AHI1)" color={ACCENT5}>
              <div className="card shadow-sm">
                <div className="card-body small">
                  <ul className="ps-3 mb-0">
                    <li>Low-vision aids; orientiation + mobility training; Braille if VA below 0.1</li>
                    <li>ERG baseline + annual; OCT (retinal thinning); visual fields (ring scotoma)</li>
                    <li><strong>CEP290 ASO therapy (sepofarsen/QR-110):</strong> for IVS26+1655A>G allele (LCA10/JBTS-retinal); intravitreal 3-monthly; ILLUMINATE trial +10-15 letters BCVA</li>
                    <li>CRISPR-Cas9 CEP290 editing: pre-clinical phase; clinical trial recruiting</li>
                    <li>Genetic counselling for gene therapy trial eligibility (CEP290 IVS26 allele required)</li>
                  </ul>
                </div>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="🫘 Renal Management (NPHP subtype)" color={ACCENT3}>
              <div className="card shadow-sm mb-3">
                <div className="card-body small">
                  <ul className="ps-3 mb-0">
                    <li>Annual USS + urine ACR + eGFR from diagnosis</li>
                    <li>ACE-I/ARB: if proteinuria or hypertension (renoprotective)</li>
                    <li>Avoid NSAIDs; avoid contrast nephropathy (prehydration if needed)</li>
                    <li>Nephrology referral: eGFR &lt; 60 or proteinuria &gt; 0.5 g/day</li>
                    <li>Pre-emptive transplant listing: CKD stage 3-4 (eGFR &lt; 30-45)</li>
                    <li><strong>Kidney transplantation:</strong> EXCELLENT outcomes (no recurrence — NPHP is cell-autonomous); living related donors preferred (screen siblings first — 25% risk)</li>
                    <li>Dialysis: haemodialysis or peritoneal — bridge to transplant</li>
                  </ul>
                </div>
              </div>
            </Section>

            <Section title="🫀 Hepatic Management (CHF — TMEM67/CC2D2A)" color={ACCENT4}>
              <div className="card shadow-sm mb-3">
                <div className="card-body small">
                  <ul className="ps-3 mb-0">
                    <li>Annual LFTs + liver USS; gastroscopy (varices) when portal hypertension detected</li>
                    <li>Portal hypertension: propranolol (non-selective beta-blocker); endoscopic variceal banding</li>
                    <li>Caroli disease: ursodeoxycholic acid (UDCA 10-15 mg/kg/day); cholangitis prophylaxis</li>
                    <li>Liver transplant: decompensated CHF (ascites + encephalopathy + variceal bleeding)</li>
                    <li><strong>Combined liver-kidney transplant:</strong> ESRD + decompensated CHF simultaneously; sequential (liver first → 3-6 months → kidney) or simultaneous</li>
                    <li>No recurrence of CHF in transplanted liver (ductal plate malformation is developmental)</li>
                  </ul>
                </div>
              </div>
            </Section>

            <Section title="💊 Diabetes / Metabolic (secondary only)" color={ACCENT8}>
              <div className="card shadow-sm">
                <div className="card-body small">
                  <div className="alert mb-2" style={{ background: ACCENT8+'15', borderLeft: `3px solid ${ACCENT8}` }}>
                    <strong>T2D is NOT a primary JBTS feature</strong> — unlike BBS (50%), Alström (~80%), Wolfram (100%).
                    JBTS DM is secondary to ESRD or obesity. C-peptide PRESERVED (insulin resistance). Autoantibodies NEGATIVE.
                  </div>
                  <ul className="ps-3 mb-0">
                    <li>Metformin (adjust dose for renal function; stop if eGFR &lt; 30)</li>
                    <li>GLP-1RA (semaglutide/liraglutide): if obesity + DM; not primary indication</li>
                    <li>SGLT2i: AVOID if eGFR &lt; 30 (renal safety); consider if eGFR 30-60 with caution</li>
                    <li>Insulin: dialysis patients or exhausted oral therapy</li>
                    <li>Annual HbA1c + fasting glucose (ESRD patients: HbA1c unreliable → fasting glucose + fructosamine)</li>
                  </ul>
                </div>
              </div>
            </Section>
          </div>

          {/* Subtype-gene-management matrix */}
          <div className="col-12 mt-2">
            <Section title="🗺️ JBTS Subtype → Gene → Priority Management" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead style={{ background: ACCENT + '22' }}>
                    <tr>
                      <th>Subtype</th><th>Predominant Gene(s)</th><th>Priority Surveillance</th><th>Key Treatment</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>Pure JBTS (cerebellar only)</td><td>Various; mild alleles</td><td>Neuro; motor delay</td><td>Physiotherapy; early intervention</td></tr>
                    <tr><td>JBTS + Retinal (JSRD)</td><td>CEP290, AHI1</td><td>ERG + OCT annually</td><td>Low-vision aids; CEP290 ASO (IVS26 allele)</td></tr>
                    <tr><td>JBTS + Renal (NPHP)</td><td>CC2D2A, NPHP1, INPP5E</td><td>Annual eGFR + USS</td><td>ACE-I; pre-emptive transplant; EXCELLENT post-Tx outcomes</td></tr>
                    <tr><td>JBTS + Hepatic (CHF)</td><td>TMEM67, CC2D2A, RPGRIP1L</td><td>LFTs + USS + gastroscopy</td><td>Propranolol; UDCA; liver transplant</td></tr>
                    <tr><td>JBTS + Renal + Hepatic</td><td>TMEM67, CC2D2A</td><td>Dual-organ surveillance</td><td>Combined liver-kidney transplant</td></tr>
                    <tr><td>JBTS + Polydactyly (JSOFD)</td><td>KIF7, TCTN1, CC2D2A</td><td>Skeletal X-ray; orthopaedic</td><td>Surgical correction; orofacial management</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && definitions && (
        <div>
          <div className="row">
            <div className="col-md-6">
              <Section title="🧬 Disease Overview" color={ACCENT}>
                {definitions.disease && Object.entries(definitions.disease).map(([k, v]) => (
                  <div key={k} className="mb-2">
                    <div className="fw-semibold small text-uppercase" style={{ color: ACCENT, letterSpacing: '0.03em' }}>{k.replace(/_/g,' ')}</div>
                    <div className="small text-muted">{String(v)}</div>
                  </div>
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="🔬 Genes & Proteins" color={ACCENT5}>
                {definitions.genes_and_proteins && Object.entries(definitions.genes_and_proteins).map(([k, v]) => (
                  <div key={k} className="mb-3">
                    <div className="fw-semibold small" style={{ color: ACCENT5 }}>{k}</div>
                    <div className="small text-muted">{String(v)}</div>
                  </div>
                ))}
              </Section>
            </div>
          </div>
          <div className="row mt-2">
            <div className="col-md-6">
              <Section title="📚 Clinical Terms & Differentials" color={ACCENT2}>
                {definitions.clinical_terms && Object.entries(definitions.clinical_terms).map(([k, v]) => (
                  <div key={k} className="mb-3 p-2 rounded" style={{ background: ACCENT2+'08', border: `1px solid ${ACCENT2}22` }}>
                    <div className="fw-semibold small" style={{ color: ACCENT2 }}>{k}</div>
                    <div className="small text-muted mt-1">{String(v)}</div>
                  </div>
                ))}
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="💡 Management Pearls" color={ACCENT3}>
                {definitions.management_pearls && Object.entries(definitions.management_pearls).map(([k, v]) => (
                  <div key={k} className="mb-3 p-2 rounded" style={{ background: ACCENT3+'08', border: `1px solid ${ACCENT3}22` }}>
                    <div className="fw-semibold small" style={{ color: ACCENT3 }}>{k.replace(/_/g,' ')}</div>
                    <div className="small text-muted mt-1">{String(v)}</div>
                  </div>
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-2 border-top text-muted small d-flex flex-wrap gap-3">
        <span>🧬 Joubert Syndrome · JBTS · Transition Zone Ciliopathy · Molar Tooth Sign</span>
        <span>CEP290 *610142 · OMIM #213300 · 12q21.32 · AR Biallelic LOF</span>
        <span>n={_COHORT_SIZE} synthetic cohort · seed 335</span>
        <Link href="/bbs" className="text-decoration-none" style={{ color: ACCENT }}>← BBS (BBS1 BBSome)</Link>
        <Link href="/expert-dashboards-catalog" className="text-decoration-none" style={{ color: ACCENT }}>Dashboard Catalog →</Link>
      </div>
    </div>
  );
}
