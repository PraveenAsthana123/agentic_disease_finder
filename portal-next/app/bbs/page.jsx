'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Treatment & Diagnostics', 'Definitions'];

// BBS / BBS1 colour scheme — indigo-amber-green-teal (ciliopathy; BBSome; rod-cone; polydactyly)
const ACCENT  = '#4527a0';   // deep indigo — BBS1/BBSome; ciliopathy; primary colour
const ACCENT2 = '#e65100';   // deep orange — polydactyly; cardinal feature
const ACCENT3 = '#1b5e20';   // dark green — C-peptide preserved; insulin resistance; NOT falling
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal; night blindness
const ACCENT5 = '#006064';   // dark teal — renal cysts/anomalies; structural renal disease
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance; cohort
const ACCENT7 = '#4e342e';   // dark brown — cognitive/learning disability; neuronal cilia
const ACCENT8 = '#bf360c';   // burnt orange — obesity; leptin resistance; BBSome LepR mis-trafficking

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

export default function BBSPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/bbs/overview`).then(r => r.json()),
      fetch(`${API}/api/bbs/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bbs/definitions`).then(r => r.json()),
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
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT4}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>Bardet-Biedl Syndrome (BBS1)</h4>
            <div className="text-muted small">Rod-Cone Dystrophy · Post-Axial Polydactyly · Obesity · Learning Disability · Renal Cysts · Hypogonadism · BBS1 BBSome Subunit · Chr 11q13.2 · OMIM *209901/#209900 · Ciliopathy · C-Peptide Preserved · Autosomal Recessive · ~1/125,000</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="BBS1 *209901" color={ACCENT} />
            <Badge text="Polydactyly" color={ACCENT2} />
            <Badge text="Rod-FIRST ERG" color={ACCENT4} />
            <Badge text="C-pep PRESERVED" color={ACCENT3} />
            <Badge text="Renal Cysts" color={ACCENT5} />
            <Badge text="Autosomal Recessive" color={ACCENT6} />
            <Badge text="M390R founder" color={ACCENT} />
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ─── TAB 0: Overview ─── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Gene" value={kpis.gene} color={ACCENT} />
            <KPI label="Chromosome" value={kpis.chromosome} color={ACCENT} />
            <KPI label="Inheritance" value="AR (biallelic)" color={ACCENT6} />
            <KPI label="Mean BMI" value={kpis.mean_bmi} color={ACCENT8} />
            <KPI label="Polydactyly" value={kpis.pct_polydactyly} color={ACCENT2} />
            <KPI label="Legal Blindness" value={kpis.pct_legal_blind} color={ACCENT4} />
            <KPI label="Renal Anomaly" value={kpis.pct_renal_anomaly} color={ACCENT5} />
            <KPI label="Learning Dis." value={kpis.pct_learning_dis} color={ACCENT7} />
            <KPI label="DM (~50%)" value={kpis.pct_dm} color={ACCENT} />
            <KPI label="C-Peptide" value="Preserved" color={ACCENT3} />
            <KPI label="Anosmia" value={kpis.pct_anosmia} color={ACCENT6} />
            <KPI label="OMIM Disease" value={kpis.omim_disease} color={ACCENT} />
          </div>

          {/* Critical Alerts */}
          <Section title="⚠ Critical Clinical Alerts" color={ACCENT2}>
            {Object.entries(alerts).map(([k, v]) => {
              const color = k.includes('rod') ? ACCENT4 : k.includes('poly') ? ACCENT2 : k.includes('cognitive') ? ACCENT7 : k.includes('renal') ? ACCENT5 : k.includes('c_peptide') ? ACCENT3 : k.includes('panel') ? ACCENT : ACCENT8;
              return (
                <Alert key={k} color={color}>
                  <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
                </Alert>
              );
            })}
          </Section>

          {/* BBSome Mechanism */}
          <Section title="🔬 BBS1 — BBSome Ciliopathy / IFT Cargo Mis-trafficking Mechanism" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>Normal BBSome Function</div>
                    <ol className="small mb-0">
                      <li>BBS1 (593 aa) is the cargo-recognition beta-propeller subunit of the BBSome octamer</li>
                      <li>BBSome coats ciliary vesicles and traffics GPCRs (LepR, SSTR3, MCHR1) INTO cilia</li>
                      <li>Leptin receptor (LepR) in hypothalamic neuronal cilia senses leptin → satiety signal → appetite control</li>
                      <li>Olfactory receptor cilia traffic via BBSome → olfactory signal transduction</li>
                      <li>Photoreceptor outer segment proteins trafficked via BBSome → rod/cone integrity</li>
                    </ol>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>BBS1 Biallelic LOF → BBSome Failure → Multi-System Ciliopathy</div>
                    <ol className="small mb-0">
                      <li><strong>M390R disrupts BBS1–BBS7 interaction</strong> → BBSome cannot assemble → no IFT cargo loading</li>
                      <li>LepR fails to enter hypothalamic cilia → leptin signal failure → hyperphagia → obesity</li>
                      <li>Photoreceptor outer segment proteins mis-trafficked → ROD-CONE degeneration (rods FIRST)</li>
                      <li>Renal tubular cilia structural defect → cysts / dysplasia / calyceal clubbing</li>
                      <li>Neuronal cilia dysfunction → learning disability / cognitive impairment</li>
                    </ol>
                  </div>
                </div>
                <div className="alert mt-2 mb-0 small" style={{ background: ACCENT3 + '18', borderLeft: `3px solid ${ACCENT3}` }}>
                  <strong>C-peptide PRESERVED</strong> — DM is T2D-like insulin resistance (LepR mis-trafficking → satiety failure → obesity → peripheral IR).
                  NOT beta-cell apoptosis (unlike Wolfram ER-stress or MODY10). Compare: Alström also preserved — different mechanism (ALMS1 basal body scaffold vs BBS1 BBSome IFT cargo).
                </div>
              </div>
            </div>
          </Section>

          {/* Cardinal Features */}
          <Section title="🎯 BBS Cardinal Features (6 Primary — Classic Diagnosis Requires All 6)" color={ACCENT2}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { n: '①', feature: 'Rod-Cone Retinal Dystrophy', detail: 'Rod-FIRST (night blindness → ring scotoma → legal blindness); NOT cone-first; ERG: scotopic extinguished >> photopic', color: ACCENT4 },
                    { n: '②', feature: 'Post-Axial Polydactyly', detail: 'Extra digit(s) hands/feet; post-axial most common; ~70%; surgical excision early childhood; distinguishes from Alström (absent)', color: ACCENT2 },
                    { n: '③', feature: 'Obesity (Truncal)', detail: 'Hyperphagia from infancy; LepR mis-trafficking; BMI 30–56; leptin resistance; GLP-1RA + metformin; bariatric if BMI ≥ 40', color: ACCENT8 },
                    { n: '④', feature: 'Learning Disability', detail: 'Mild–moderate cognitive impairment (IQ 55–84); neuronal cilia BBSome dysfunction; early educational support; absent in Alström', color: ACCENT7 },
                    { n: '⑤', feature: 'Renal Anomalies (Structural)', detail: 'Cysts / calyceal clubbing / horseshoe kidney; NOT tubular nephropathy (contrast Alström); annual USS + urine ACR + eGFR', color: ACCENT5 },
                    { n: '⑥', feature: 'Hypogonadism', detail: 'Males: cryptorchidism, small genitalia; Females: irregular menses, hypogenitalism; hormone replacement at puberty', color: ACCENT6 },
                  ].map((item, i) => (
                    <div key={i} className="col-md-4">
                      <div className="card h-100 border-0" style={{ background: item.color + '0d', borderLeft: `4px solid ${item.color}` }}>
                        <div className="card-body py-2 px-2">
                          <div className="fw-bold small" style={{ color: item.color }}>{item.n} {item.feature}</div>
                          <div className="small text-muted">{item.detail}</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Section>

          {/* Key Facts */}
          <Section title="📋 Key Clinical Facts" color={ACCENT}>
            <div className="row g-2">
              {keyFacts.map((f, i) => (
                <div key={i} className="col-md-6">
                  <div className="small p-2 rounded" style={{ background: ACCENT + '0d', borderLeft: `3px solid ${ACCENT}` }}>
                    {f}
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* BBS1 Mutations Table */}
          <Section title="🧪 BBS1 Key Mutations (biallelic LOF — common genotypes)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Mutation</th><th>Exon</th><th>Population</th><th>Type</th><th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>M390R (c.1169T&gt;G)</strong></td><td>Exon 12</td><td>European (founder)</td><td>Missense</td><td>~70% European BBS1 alleles; disrupts BBS1–BBS7 interaction; BBSome assembly failure</td></tr>
                  <tr><td><strong>Y222C (c.665A&gt;G)</strong></td><td>Exon 8</td><td>European</td><td>Missense</td><td>Reduced BBS1 stability; less common; compound het with M390R</td></tr>
                  <tr><td>R228* (c.682C&gt;T)</td><td>Exon 8</td><td>European</td><td>Nonsense (truncating)</td><td>Complete LOF; common compound het partner with M390R</td></tr>
                  <tr><td>L518* (c.1553T&gt;A)</td><td>Exon 14</td><td>Northern European</td><td>Nonsense (truncating)</td><td>Complete LOF; Northern European enrichment</td></tr>
                  <tr><td>c.1303C&gt;T (p.Gln435*)</td><td>Exon 12</td><td>European</td><td>Nonsense (truncating)</td><td>Compound het with M390R; exon 12 cluster</td></tr>
                  <tr><td>IVS12+1G&gt;A</td><td>Intron 12</td><td>European</td><td>Splice-site</td><td>Aberrant splicing; frameshift consequence</td></tr>
                  <tr><td>Del exon5-6 (large deletion)</td><td>Exons 5–6</td><td>Pan-ethnic</td><td>CNV (MLPA)</td><td>Copy number variant; requires MLPA or CGH array for detection</td></tr>
                  <tr><td>p.Arg160Gln (c.479G&gt;A)</td><td>Exon 5</td><td>Kuwaiti</td><td>Missense</td><td>BBSome binding interface; Kuwaiti/Middle Eastern enrichment</td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient Preview Table */}
          <Section title="👥 Cohort Preview (first 12 patients, seed=333)" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT6 + '18' }}>
                    <th>#</th><th>Mutation</th><th>Retinal</th><th>Poly.</th><th>Renal</th><th>Cognitive</th><th>DM</th><th>HbA1c%</th><th>C-Pep (nmol/L)</th><th>BMI</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td>{p.id}</td>
                      <td><code style={{ fontSize: '0.68em' }}>{p.mutation}</code></td>
                      <td><span className="small">{p.retinal}</span></td>
                      <td><Badge text={p.poly} color={p.poly === 'Yes' ? ACCENT2 : ACCENT6} /></td>
                      <td><span className="small">{p.renal}</span></td>
                      <td><span className="small">{p.cognitive}</span></td>
                      <td><Badge text={p.dm} color={p.dm === 'Yes' ? ACCENT : ACCENT3} /></td>
                      <td>{p.hba1c}</td>
                      <td style={{ color: ACCENT3 }}>{p.c_pep}</td>
                      <td>{p.bmi}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 1: Multi-System Breakdown ─── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Retinal / Vision Status (Rod-FIRST)" color={ACCENT4}>
              {Object.entries(breakdown.retinal_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
              <div className="small text-muted mt-1">Rod-cone RP-like: rods extinguished first (ERG scotopic); distinguishes from Alström (cone-rod, cones first)</div>
            </Section>
            <Section title="Renal Status (Structural Cysts)" color={ACCENT5}>
              {Object.entries(breakdown.renal_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
              <div className="small text-muted mt-1">Structural anomalies (cysts/clubbing/horseshoe); NOT tubular nephropathy as in Alström</div>
            </Section>
            <Section title="Cognitive / Learning Disability" color={ACCENT7}>
              {Object.entries(breakdown.cognitive_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
              <div className="small text-muted mt-1">50–60% mild–moderate learning disability; absent in Alström Syndrome</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="BBS1 Mutation Distribution" color={ACCENT}>
              {Object.entries(breakdown.mutation_distribution || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
              ))}
              <div className="small text-muted mt-1">M390R/M390R (founder homozygous) and M390R/compound-het predominate in European cohorts</div>
            </Section>
            <Section title="Diabetes Status" color={ACCENT3}>
              {Object.entries(breakdown.diabetes_status || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
              ))}
              <div className="small text-muted mt-1">DM in ~50% (T2D-like insulin resistance; C-pep preserved; metformin + GLP-1RA)</div>
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {Object.entries(breakdown.ethnicity_distribution || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <div className="row g-3">
              <div className="col-md-3">
                <Section title="HbA1c Tiers" color={ACCENT}>
                  {Object.entries(breakdown.hba1c_tiers || {}).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="C-Peptide Tiers (Preserved)" color={ACCENT3}>
                  {Object.entries(breakdown.c_peptide_tiers || {}).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT3} />
                  ))}
                  <div className="small text-muted mt-1">Majority normal–elevated (insulin resistance, not apoptosis)</div>
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="BMI Tiers (Obesity)" color={ACCENT8}>
                  {Object.entries(breakdown.bmi_tiers || {}).map(([k,v])=>(
                    <Bar key={k} label={k} value={v} max={_COHORT_SIZE} color={ACCENT8} />
                  ))}
                </Section>
              </div>
              <div className="col-md-3">
                <Section title="Prior Misdiagnosis" color={ACCENT2}>
                  {Object.entries(breakdown.misdiagnosis || {}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>(
                    <Bar key={k} label={k.length > 40 ? k.slice(0,40)+'…' : k} value={v} max={_COHORT_SIZE} color={ACCENT2} />
                  ))}
                </Section>
              </div>
            </div>
            {/* Summary flags */}
            {breakdown.summary_flags && (
              <Section title="📊 Summary Statistics" color={ACCENT6}>
                <div className="row g-2">
                  {Object.entries(breakdown.summary_flags).map(([k,v]) => (
                    <div key={k} className="col-6 col-md-3">
                      <div className="card border-0 shadow-sm text-center py-2">
                        <div className="fw-bold fs-5" style={{ color: ACCENT }}>{typeof v === 'number' ? v + (k.includes('m390r') ? '% alleles' : '%') : v}</div>
                        <div className="text-muted small">{k.replace(/_/g,' ')}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </Section>
            )}
          </div>
        </div>
      )}

      {/* ─── TAB 2: Treatment & Diagnostics ─── */}
      {tab === 2 && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="🔍 Diagnostic Pathway" color={ACCENT}>
                <div className="card border-0 shadow-sm">
                  <div className="card-body">
                    {[
                      { step: '1. Gene Panel (≥24 BBS genes)', detail: 'BBS1, BBS2, ARL6, BBS4-10, MKKS, TRIM32, MKS1, CEP290 — mandatory; single BBS1 testing misses 75% of BBS', color: ACCENT },
                      { step: '2. ERG at Diagnosis', detail: 'Rod-cone pattern (scotopic extinguished >> photopic); baseline; annual; differentiates from Alström (cone-rod) and Usher', color: ACCENT4 },
                      { step: '3. Renal USS', detail: 'Structural anomalies: cysts, calyceal clubbing, horseshoe kidney; annual surveillance; ACR + eGFR', color: ACCENT5 },
                      { step: '4. Skeletal X-ray', detail: 'Hands + feet: confirm post-axial polydactyly; count digits; surgical planning referral', color: ACCENT2 },
                      { step: '5. Neuropsychological Assessment', detail: 'IQ + learning profile; speech/language; OT referral; educational plan', color: ACCENT7 },
                      { step: '6. Metabolic Screen', detail: 'Fasting glucose, HbA1c, insulin, C-peptide (preserved/high), TG, lipids; annual from diagnosis', color: ACCENT3 },
                      { step: '7. Olfactory Testing (UPSIT)', detail: 'University of Pennsylvania Smell ID Test; hyposmia/anosmia (~65%); safety counselling', color: ACCENT6 },
                      { step: '8. Echocardiogram', detail: 'Congenital HD screen at diagnosis; ASD/VSD/BAV; cardiology if abnormal', color: ACCENT8 },
                      { step: '9. Hormonal Assessment', detail: 'Males: testosterone/FSH/LH/testicular USS; Females: oestradiol/FSH/pelvic USS/PCOS screen', color: ACCENT6 },
                      { step: '10. Brain MRI (if CNS features)', detail: 'Hippocampal / cortical structural anomalies; rare; CNS BBS features', color: ACCENT7 },
                    ].map((item, i) => (
                      <div key={i} className="d-flex gap-2 mb-2">
                        <div className="fw-bold small" style={{ color: item.color, minWidth: 200 }}>{item.step}</div>
                        <div className="small text-muted">{item.detail}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="💊 Treatment Summary (Multi-System; No Disease-Modifying Rx 2026)" color={ACCENT3}>
                <div className="card border-0 shadow-sm">
                  <div className="card-body">
                    {[
                      { sys: 'Retina', tx: 'Low-vision aids; ERG annual; avoid strong light; CRISPR-BBSome gene therapy trials (recruiting)', color: ACCENT4 },
                      { sys: 'Obesity', tx: 'GLP-1RA (semaglutide 2.4mg/week or liraglutide 3.0mg/day) + metformin; bariatric surgery (BMI ≥ 40, adult, comorbidities)', color: ACCENT8 },
                      { sys: 'Diabetes (T2D-like)', tx: 'Metformin first-line; GLP-1RA (obesity + IR dual benefit); SGLT2i (renal/cardiac); insulin ONLY if HbA1c > 9% uncontrolled', color: ACCENT3 },
                      { sys: 'Polydactyly', tx: 'Surgical excision 6–12 months (before weight-bearing); orthopaedic + plastic surgery referral; X-ray planning', color: ACCENT2 },
                      { sys: 'Renal', tx: 'Annual USS + urine ACR + eGFR; ACE-I/ARB if proteinuria; avoid NSAIDs; nephrology if eGFR < 60', color: ACCENT5 },
                      { sys: 'Learning Disability', tx: 'Early educational support (IEP); speech/language therapy; occupational therapy; neuropsychological review 3-yearly', color: ACCENT7 },
                      { sys: 'Hypogonadism', tx: 'Males: testosterone replacement (after growth complete); Females: oestrogen/progesterone puberty induction + maintenance', color: ACCENT6 },
                      { sys: 'Anosmia', tx: 'UPSIT annually; safety counselling (gas leaks, fire alarm, food spoilage); smell training (limited evidence)', color: ACCENT6 },
                      { sys: 'Cardiac (CHD)', tx: 'Echo at diagnosis; annual if CHD confirmed; standard HF therapy if haemodynamically significant', color: ACCENT8 },
                      { sys: 'Hepatic', tx: 'Annual LFTs; fibrates if TG > 5 mmol/L; hepatology referral if elevated ALT/AST; FibroScan if NAFLD', color: ACCENT5 },
                    ].map((item, i) => (
                      <div key={i} className="d-flex gap-2 mb-2">
                        <Badge text={item.sys} color={item.color} />
                        <div className="small text-muted">{item.tx}</div>
                      </div>
                    ))}
                    <div className="alert mt-2 mb-0 small" style={{ background: ACCENT + '18', borderLeft: `3px solid ${ACCENT}` }}>
                      <strong>No disease-modifying therapy</strong> as of 2026. GLP-1RA + metformin central to metabolic management.
                      CRISPR-Cas9 BBSome gene therapy trials ongoing for retinal disease.
                      BBS UK Registry + BBS Foundation + Global BBS Registry — enrol all patients.
                    </div>
                  </div>
                </div>
              </Section>

              <Section title="⚖ BBS vs Alström — Clinical Differential" color={ACCENT2}>
                <div className="table-responsive">
                  <table className="table table-sm small mb-0">
                    <thead>
                      <tr style={{ background: ACCENT + '18' }}>
                        <th>Feature</th><th style={{ color: ACCENT }}>BBS (BBS1)</th><th style={{ color: ACCENT4 }}>Alström (ALMS1)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td><strong>Polydactyly</strong></td><td className="text-danger fw-bold">YES (~70%)</td><td className="text-success fw-bold">NO (absent)</td></tr>
                      <tr><td><strong>ERG pattern</strong></td><td>Rod-FIRST (scotopic &gt;&gt; photopic)</td><td>Cone-FIRST (photopic &gt;&gt; scotopic)</td></tr>
                      <tr><td><strong>Learning disability</strong></td><td className="text-danger fw-bold">YES (50–60%)</td><td className="text-success fw-bold">NO (cognition normal)</td></tr>
                      <tr><td><strong>Renal anomaly</strong></td><td>Cysts / structural</td><td>Tubular nephropathy (functional)</td></tr>
                      <tr><td><strong>Infantile DCM</strong></td><td>No / rare (5%)</td><td className="text-danger fw-bold">YES (60%; life-threatening)</td></tr>
                      <tr><td><strong>Anosmia</strong></td><td className="text-danger fw-bold">YES (65%)</td><td>Uncommon</td></tr>
                      <tr><td><strong>DM prevalence</strong></td><td>~50% (T2D-like)</td><td>~80% (T2D-like)</td></tr>
                      <tr><td><strong>C-peptide</strong></td><td>PRESERVED / HIGH</td><td>PRESERVED / HIGH</td></tr>
                      <tr><td><strong>Mechanism</strong></td><td>BBSome IFT cargo failure</td><td>Cilia basal body scaffold LOF</td></tr>
                      <tr><td><strong>Causative gene</strong></td><td>BBS1 (of ≥24 BBS genes)</td><td>ALMS1 (sole gene)</td></tr>
                    </tbody>
                  </table>
                </div>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ─── TAB 3: Definitions ─── */}
      {tab === 3 && definitions && (
        <div>
          {Object.entries(definitions).map(([section, content]) => (
            <Section key={section} title={section.replace(/_/g,' ').toUpperCase()} color={ACCENT}>
              {typeof content === 'object' && !Array.isArray(content)
                ? Object.entries(content).map(([k, v]) => (
                    <div key={k} className="mb-3 p-3 rounded" style={{ background: ACCENT + '06', border: `1px solid ${ACCENT}22` }}>
                      <div className="fw-bold mb-1" style={{ color: ACCENT }}>{k}</div>
                      <div className="small text-muted" style={{ whiteSpace: 'pre-wrap' }}>{typeof v === 'string' ? v : JSON.stringify(v, null, 2)}</div>
                    </div>
                  ))
                : <div className="small text-muted p-2">{String(content)}</div>
              }
            </Section>
          ))}
        </div>
      )}
    </div>
  );
}
