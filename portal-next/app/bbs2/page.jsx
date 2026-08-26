'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS2 colour scheme — deep violet-amber-teal-rose (ciliopathy; BBSome core scaffold; R631P Bedouin)
const ACCENT  = '#4a148c';   // deep violet — BBS2/BBSome structural core; ciliopathy
const ACCENT2 = '#e65100';   // deep orange — polydactyly; cardinal feature
const ACCENT3 = '#1b5e20';   // dark green — metabolic; obesity; GLP-1RA
const ACCENT4 = '#880e4f';   // dark rose — rod-cone degeneration; retinal
const ACCENT5 = '#006064';   // dark teal — renal anomaly; structural renal
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance
const ACCENT7 = '#4e342e';   // dark brown — cognitive/learning disability
const ACCENT8 = '#bf360c';   // burnt orange — obesity; hyperphagia; LepR mis-trafficking

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
        <span>{label}</span><span className="fw-bold">{value} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function BBS2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/bbs2/overview`).then(r => r.json()),
      fetch(`${API}/api/bbs2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bbs2/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const ageDist = overview?.age_distribution || {};
  const bbsomeTable = overview?.bbsome_subunit_table || [];
  const retinalDist = overview?.retinal_distribution || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT4}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>Bardet-Biedl Syndrome Type 2 (BBS2)</h4>
            <div className="text-muted small">Rod-Cone Dystrophy · Post-Axial Polydactyly · Obesity · Cognitive Impairment · Renal Anomalies · Hypogonadism · BBS2 BBSome Core Scaffold · Chr 16q13 · OMIM *606151/#209900 · R631P Bedouin Founder · Autosomal Recessive · ~10–20% of BBS</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="BBS2 *606151" color={ACCENT} />
            <Badge text="Polydactyly 70%" color={ACCENT2} />
            <Badge text="Rod-FIRST ERG" color={ACCENT4} />
            <Badge text="R631P Bedouin" color={ACCENT6} />
            <Badge text="Renal 50%" color={ACCENT5} />
            <Badge text="BBSome Core" color={ACCENT} />
            <Badge text="AR biallelic" color={ACCENT6} />
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
            <KPI label="Cohort (N)" value={`${_COHORT_SIZE} patients`} color={ACCENT6} />
            <KPI label="Polydactyly" value={`${kpis.polydactyly_n} (${kpis.polydactyly_pct}%)`} color={ACCENT2} />
            <KPI label="Obesity" value={`${kpis.obesity_n} (${kpis.obesity_pct}%)`} color={ACCENT8} />
            <KPI label="Cognitive LD" value={`${kpis.cognitive_n} (${kpis.cognitive_pct}%)`} color={ACCENT7} />
            <KPI label="Renal Anomaly" value={`${kpis.renal_any_n} (${kpis.renal_any_pct}%)`} color={ACCENT5} />
            <KPI label="Hypogonadism" value={`${kpis.hypogonadism_n} (${kpis.hypogonadism_pct}%)`} color={ACCENT6} />
            <KPI label="Anosmia" value={`${kpis.anosmia_n} (${kpis.anosmia_pct}%)`} color={ACCENT6} />
            <KPI label="Retinal End-Stage" value={`${kpis.retinal_endstage_n} (${kpis.retinal_endstage_pct}%)`} color={ACCENT4} />
            <KPI label="CHD" value={`${kpis.chd_n} (${kpis.chd_pct}%)`} color={ACCENT4} />
            <KPI label="Tri-allelic BBS" value={`${kpis.triallelic_n} (${kpis.triallelic_pct}%)`} color={ACCENT} />
            <KPI label="Misdiagnosed" value={`${kpis.misdiagnosis_n} (${kpis.misdiagnosis_pct}%)`} color={ACCENT2} />
            <KPI label="ESRD" value={`${kpis.esrd_n}`} color={ACCENT5} />
          </div>

          {/* Critical Alerts */}
          <Section title="⚠ Critical Clinical Alerts — BBS2" color={ACCENT2}>
            <Alert color={ACCENT4}>
              <strong>Rod-cone dystrophy (rod FIRST):</strong> Scotopic ERG extinguishes before photopic — distinguishes BBS2 from Alstrom (cone-rod). Annual ERG + OCT from age 3.
            </Alert>
            <Alert color={ACCENT2}>
              <strong>No M390R European founder in BBS2:</strong> BBS1 M390R accounts for 70% of European BBS1 alleles — BBS2 lacks this. BBS2 allele spectrum is heterogeneous; full sequencing mandatory. R631P is the Bedouin/Arabian Peninsula founder (~42% of this cohort).
            </Alert>
            <Alert color={ACCENT}>
              <strong>BBSome core scaffold:</strong> BBS2–BBS7 dimer is the structural spine of BBSome. BBS2 LOF → catastrophic BBSome misassembly at step 0 (unlike peripheral subunit LOF). Full BBS panel (≥20 genes) mandatory; single-gene BBS2 testing misses tri-allelic cases.
            </Alert>
            <Alert color={ACCENT8}>
              <strong>Obesity (LepR mis-trafficking):</strong> 88% of BBS2 patients develop obesity by adolescence via LepR mis-trafficking. GLP-1 RA (semaglutide/liraglutide) + metformin; bariatric surgery for BMI ≥ 40 in adults.
            </Alert>
            <Alert color={ACCENT5}>
              <strong>Renal anomalies (50%):</strong> Structural (cysts, calyceal clubbing, horseshoe kidney) — NOT tubular nephropathy (contrast Alstrom). Annual USS + GFR essential.
            </Alert>
          </Section>

          {/* Mechanism */}
          <Section title="🔬 BBS2 — BBSome Core Scaffold / IFT Cargo Mis-trafficking Mechanism" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>BBS2 Protein Structure &amp; Function</div>
                    <ul className="small mb-0">
                      <li><strong>WD40 beta-propeller (aa 1–400):</strong> structural scaffold base; stacks onto BBS7 WD40 face → BBS2–BBS7 dimer (BBSome structural spine)</li>
                      <li><strong>WD40/CC transition (aa 350–500):</strong> BBS9 (PTHB1) contact site; BBS9 bridges BBS2 → BBS1 (cargo recognition); disruption → cargo recognition lost</li>
                      <li><strong>C-terminal coiled-coil (aa 500–721):</strong> packs against BBS7 CC + BBS8 (TTC8) N-TPR; stabilises outer BBSome coat; <em>R631P Bedouin founder site</em></li>
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>BBS2 Biallelic LOF → BBSome Failure → Multi-System Ciliopathy</div>
                    <ol className="small mb-0">
                      <li><strong>BBS2 LOF → BBS2–BBS7 dimer fails</strong> → BBSome scaffold cannot assemble at step 0</li>
                      <li>LepR cannot enter hypothalamic cilia → satiety failure → hyperphagia → obesity (88%)</li>
                      <li>Photoreceptor outer segment proteins mis-trafficked → ROD-CONE degeneration (rods first)</li>
                      <li>Olfactory cilia BBSome absent → anosmia/hyposmia (62%)</li>
                      <li>Renal tubular cilia structural defect → cysts / calyceal clubbing (50%)</li>
                      <li>Neuronal cilia dysfunction → learning disability (55%)</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          {/* Cardinal Features */}
          <Section title="🎯 BBS2 Cardinal Features (6 Primary)" color={ACCENT2}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { n: '①', feature: 'Rod-Cone Retinal Dystrophy', detail: 'Rod-FIRST (night blindness → ring scotoma → legal blindness); NOT cone-first; ERG: scotopic extinguished >> photopic; same rod-cone pattern as BBS1', color: ACCENT4 },
                    { n: '②', feature: 'Post-Axial Polydactyly', detail: 'Extra digit(s) hands/feet; post-axial most common; ~70%; surgical excision early childhood; distinguishes from Alstrom (absent)', color: ACCENT2 },
                    { n: '③', feature: 'Obesity (Truncal)', detail: 'Hyperphagia from infancy; LepR mis-trafficking; BMI 30–50 by adolescence; GLP-1RA + metformin; bariatric if BMI ≥ 40', color: ACCENT8 },
                    { n: '④', feature: 'Cognitive / Learning Disability', detail: '~55% mild–moderate; neuronal cilia BBSome dysfunction; similar profile to BBS1; early educational support; absent in Alstrom', color: ACCENT7 },
                    { n: '⑤', feature: 'Renal Anomalies (Structural)', detail: '~50% — cysts / calyceal clubbing / horseshoe kidney; NOT tubular nephropathy; annual USS + ACR + eGFR', color: ACCENT5 },
                    { n: '⑥', feature: 'Hypogonadism', detail: '~70% — males: cryptorchidism, small testes; females: irregular menses, ovarian cysts; hormone replacement if needed', color: ACCENT6 },
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

          {/* BBSome Subunit Table */}
          <Section title="🔩 BBSome Octameric Complex — All 8 Subunits (BBS2 = Core Scaffold)" color={ACCENT}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Subunit</th><th>Role in BBSome</th><th>OMIM Gene</th><th>BBS Freq (%)</th>
                  </tr>
                </thead>
                <tbody>
                  {bbsomeTable.map((row, i) => (
                    <tr key={i} style={row.subunit.includes('THIS') ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                      <td><strong style={{ color: row.subunit.includes('THIS') ? ACCENT : undefined }}>{row.subunit}</strong></td>
                      <td className="small">{row.role}</td>
                      <td><code>{row.omim}</code></td>
                      <td>~{row.frequency_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          {/* Retinal Distribution + Age at Dx */}
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="👁 Retinal Stage Distribution (Rod-Cone Progression)" color={ACCENT4}>
                {retinalDist.map((r, i) => (
                  <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
                ))}
                <div className="small text-muted mt-1">Rod-FIRST: scotopic ERG extinguished before photopic. Contrasts with Alstrom (cone-rod).</div>
              </Section>
            </div>
            <div className="col-md-6">
              <Section title="📅 Age at Diagnosis Distribution" color={ACCENT6}>
                {Object.entries(ageDist).map(([k, v]) => (
                  <Bar key={k} label={k.replace(/_/g,' ')} value={v} max={_COHORT_SIZE} color={ACCENT6} />
                ))}
                <div className="small text-muted mt-1">Earlier diagnosis driven by polydactyly + retinal symptoms in childhood.</div>
              </Section>
            </div>
          </div>

          {/* Mechanism Highlight: BBS2 vs BBS1 */}
          <Section title="🔍 BBS2 vs BBS1 — Key Distinctions" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Feature</th>
                    <th style={{ color: ACCENT }}>BBS2 (this dashboard)</th>
                    <th style={{ color: '#1a237e' }}>BBS1 (comparison)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>Founder mutation</strong></td><td className="text-danger">R631P — Bedouin/Arabian Peninsula (C-terminal CC)</td><td className="text-danger">M390R — European founder (70% of European BBS1 alleles)</td></tr>
                  <tr><td><strong>Allele spectrum</strong></td><td>Heterogeneous (no dominant single allele in European)</td><td>M390R predominant in European (homozygous or compound-het)</td></tr>
                  <tr><td><strong>BBSome role</strong></td><td className="fw-bold text-primary">CORE scaffold — BBS2–BBS7 dimer (structural spine)</td><td>Cargo recognition beta-propeller + ARM (periphery)</td></tr>
                  <tr><td><strong>LOF consequence</strong></td><td>Catastrophic BBSome misassembly at step 0</td><td>BBSome assembles but cargo recognition impaired</td></tr>
                  <tr><td><strong>Tri-allelic BBS</strong></td><td className="fw-bold">~9% (BBS2 × 2 + BBS7/9 modifier — core partners)</td><td>~5% (less common)</td></tr>
                  <tr><td><strong>Clinical phenotype</strong></td><td>Clinically indistinguishable from BBS1 (gene panel required)</td><td>Clinically indistinguishable from BBS2 (gene panel required)</td></tr>
                  <tr><td><strong>OMIM gene</strong></td><td><code>*606151</code></td><td><code>*209901</code></td></tr>
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
            <Section title="Retinal Stage (Rod-FIRST, Rod-Cone)" color={ACCENT4}>
              {(breakdown.retinal_stage_distribution || []).map((r, i) => (
                <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
            </Section>
            <Section title="Renal Anomaly Distribution" color={ACCENT5}>
              {(breakdown.renal_distribution || []).map((r, i) => (
                <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT5} />
              ))}
            </Section>
            <Section title="Polydactyly Types" color={ACCENT2}>
              {(breakdown.polydactyly_distribution || []).slice(0, 2).map((r, i) => (
                <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
              <div className="small text-muted mt-1">Post-axial dominant; surgical excision in infancy preferred.</div>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Allele Class Distribution" color={ACCENT}>
              {(breakdown.allele_class_summary || []).map((r, i) => (
                <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT6}>
              {(breakdown.ethnicity_distribution || []).map((r, i) => (
                <Bar key={i} label={r.ethnicity} value={r.n} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Age at Diagnosis" color={ACCENT7}>
              {(breakdown.presentation_distribution || []).map((r, i) => (
                <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>
          </div>
          <div className="col-12">
            <Section title="🏥 Multi-System Feature Burden" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm small mb-0">
                  <thead>
                    <tr style={{ background: ACCENT + '18' }}>
                      <th>System Feature</th><th>N</th><th>%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.systemic_burden || []).map((r, i) => (
                      <tr key={i}>
                        <td>{r.feature}</td>
                        <td><strong>{r.n}</strong></td>
                        <td>{r.pct}%{r.avg_bmi ? ` (avg BMI ${r.avg_bmi})` : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
            <Section title="Prior Misdiagnosis" color={ACCENT2}>
              {(breakdown.misdiagnosis_distribution || []).map((r, i) => (
                <Bar key={i} label={r.label.length > 60 ? r.label.slice(0, 60) + '…' : r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT2} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ─── TAB 2: Variants & Diagnostics ─── */}
      {tab === 2 && breakdown && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <Section title="🧪 BBS2 Key Pathogenic Variants" color={ACCENT}>
                <div className="card border-0 shadow-sm">
                  <div className="card-body">
                    {(breakdown.top_variants || []).map((v, i) => (
                      <div key={i} className="mb-2 p-2 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                        <div className="fw-bold small" style={{ color: ACCENT }}>{v.variant.split('—')[0].trim()}</div>
                        <div className="small text-muted">{v.variant.split('—').slice(1).join('—').trim()}</div>
                        <div className="small"><Badge text={`n=${v.n}`} color={ACCENT6} /></div>
                      </div>
                    ))}
                    <div className="small text-muted mt-2">
                      <strong>Note:</strong> BBS2 lacks a dominant single founder in European populations (unlike BBS1 M390R). Full gene sequencing + MLPA for CNVs mandatory.
                    </div>
                  </div>
                </div>
              </Section>

              <Section title="⚖ BBS2 vs Alstrom — Clinical Differential" color={ACCENT2}>
                <div className="table-responsive">
                  <table className="table table-sm small mb-0">
                    <thead>
                      <tr style={{ background: ACCENT + '18' }}>
                        <th>Feature</th><th style={{ color: ACCENT }}>BBS2</th><th style={{ color: ACCENT4 }}>Alstrom (ALMS1)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td><strong>Polydactyly</strong></td><td className="text-danger fw-bold">YES (~70%)</td><td className="text-success fw-bold">NO</td></tr>
                      <tr><td><strong>ERG pattern</strong></td><td>Rod-FIRST (scotopic {'>>'}  photopic)</td><td>Cone-FIRST (photopic {'>>'}  scotopic)</td></tr>
                      <tr><td><strong>Learning disability</strong></td><td className="text-danger fw-bold">YES (55%)</td><td className="text-success fw-bold">NO</td></tr>
                      <tr><td><strong>Infantile DCM</strong></td><td>NO / rare (7%)</td><td className="text-danger fw-bold">YES (60%)</td></tr>
                      <tr><td><strong>Renal anomaly type</strong></td><td>Structural (cysts/clubbing)</td><td>Tubular nephropathy (functional)</td></tr>
                      <tr><td><strong>Anosmia</strong></td><td className="text-danger fw-bold">YES (62%)</td><td>Uncommon</td></tr>
                      <tr><td><strong>Mechanism</strong></td><td>BBSome IFT cargo failure</td><td>Cilia basal body scaffold LOF</td></tr>
                    </tbody>
                  </table>
                </div>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="🔍 Diagnostic Pathway — BBS2" color={ACCENT}>
                <div className="card border-0 shadow-sm">
                  <div className="card-body">
                    {[
                      { step: '1. Full BBS Gene Panel (≥20 genes)', detail: 'BBS1–BBS22; BBS2 single-gene testing misses tri-allelic cases + other BBS genes; MLPA for CNVs', color: ACCENT },
                      { step: '2. ERG at Diagnosis', detail: 'Rod-cone: scotopic extinguished >> photopic; annual from age 3; distinguishes BBS2 from Alstrom (cone-rod)', color: ACCENT4 },
                      { step: '3. Renal USS + ACR + eGFR', detail: 'Structural anomalies (cysts, clubbing) from birth; annual surveillance; ACEi/ARB if proteinuria', color: ACCENT5 },
                      { step: '4. Metabolic Screen', detail: 'Fasting glucose, HbA1c, insulin, lipids; annual from diagnosis; GLP-1RA + metformin if obesity', color: ACCENT3 },
                      { step: '5. Neuropsychological Assessment', detail: 'IQ + learning profile; early IEP; speech-language therapy; 55% learning disability', color: ACCENT7 },
                      { step: '6. Smell Testing (UPSIT)', detail: 'Olfactory testing; hyposmia/anosmia (62%); gas safety counselling', color: ACCENT6 },
                      { step: '7. Hormonal Panel', detail: 'LH/FSH/testosterone (males) or oestradiol/FSH (females); cryptorchidism — orchidopexy in infancy', color: ACCENT6 },
                      { step: '8. Echo', detail: 'Congenital HD screen (7%); NOT infantile cardiomyopathy (Alstrom); standard cardiac assessment', color: ACCENT8 },
                      { step: '9. Tri-allelic BBS screen', detail: 'If more severe than expected for biallelic BBS2 → expand panel to BBS7/BBS9 for modifier allele', color: ACCENT },
                    ].map((item, i) => (
                      <div key={i} className="d-flex gap-2 mb-2">
                        <div className="fw-bold small" style={{ color: item.color, minWidth: 220 }}>{item.step}</div>
                        <div className="small text-muted">{item.detail}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </Section>
            </div>
          </div>

          {/* Treatment */}
          <Section title="💊 Treatment Summary — BBS2 (Multi-System; No Disease-Modifying Rx 2026)" color={ACCENT3}>
            <div className="card border-0 shadow-sm">
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { sys: 'Retina', tx: 'Annual ERG + OCT; low-vision aids; orientation/mobility training; avoid high-intensity light; gene therapy trials (recruiting)', color: ACCENT4 },
                    { sys: 'Obesity', tx: 'GLP-1RA (semaglutide/liraglutide) + metformin; bariatric surgery (BMI ≥ 40, adult, comorbidities); dietitian from diagnosis', color: ACCENT8 },
                    { sys: 'Polydactyly', tx: 'Surgical excision (infancy, 6–12 months); orthopaedic + plastic surgery; X-ray planning; weight-bearing digit correction', color: ACCENT2 },
                    { sys: 'Renal', tx: 'Annual USS + ACR + eGFR; ACEi/ARB if proteinuria; avoid NSAIDs; nephrology if eGFR < 60; ESRD → transplant planning', color: ACCENT5 },
                    { sys: 'Cognitive LD', tx: 'Neuropsychological testing at school age; IEP; special education support; speech-language + occupational therapy', color: ACCENT7 },
                    { sys: 'Hypogonadism', tx: 'Testosterone replacement (males) or oestrogen-progesterone (females); cryptorchidism → orchidopexy in infancy', color: ACCENT6 },
                    { sys: 'Anosmia', tx: 'UPSIT annually; gas detector (LPG/CO); occupational counselling; smell training (limited evidence)', color: ACCENT6 },
                    { sys: 'Cardiac', tx: 'Echo at diagnosis; annual if CHD; NOT DCM (Alstrom); standard structural heart disease management', color: ACCENT8 },
                    { sys: 'Tri-allelic BBS', tx: 'Intensified multi-system monitoring; genetic counselling (reduced predictability); full family pedigree analysis', color: ACCENT },
                  ].map((item, i) => (
                    <div key={i} className="col-md-6 mb-1">
                      <div className="d-flex gap-2">
                        <Badge text={item.sys} color={item.color} />
                        <div className="small text-muted">{item.tx}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="alert mt-3 mb-0 small" style={{ background: ACCENT + '18', borderLeft: `3px solid ${ACCENT}` }}>
                  <strong>No disease-modifying therapy as of 2026.</strong> GLP-1RA (semaglutide/tirzepatide) central to metabolic management.
                  BBSome gene therapy trials ongoing for retinal disease. BBS Foundation + Global BBS Registry — enrol all patients.
                </div>
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 3: Definitions ─── */}
      {tab === 3 && definitions && (
        <div>
          {Object.entries(definitions).map(([section, content]) => (
            <Section key={section} title={section.replace(/_/g, ' ').toUpperCase()} color={ACCENT}>
              {Array.isArray(content)
                ? content.map((item, i) => (
                    <div key={i} className="mb-3 p-3 rounded" style={{ background: ACCENT + '06', border: `1px solid ${ACCENT}22` }}>
                      {typeof item === 'string'
                        ? <div className="small text-muted">{item}</div>
                        : Object.entries(item).map(([k, v]) => (
                            <div key={k} className="mb-1">
                              <span className="fw-bold small" style={{ color: ACCENT }}>{k}: </span>
                              <span className="small text-muted">{typeof v === 'string' ? v : JSON.stringify(v)}</span>
                            </div>
                          ))
                      }
                    </div>
                  ))
                : typeof content === 'object'
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
