'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Multi-System Breakdown', 'Variants & Diagnostics', 'Definitions'];

// BBS1 colour scheme — deep indigo-amber-teal-rose (BBSome cargo arm; Met390Arg; rod-cone; European founder)
const ACCENT  = '#1a237e';   // deep indigo — BBS1/BBSome; cargo-recognition subunit
const ACCENT2 = '#e65100';   // deep orange — polydactyly; cardinal feature
const ACCENT3 = '#1b5e20';   // dark green — metabolic; obesity; GLP-1RA
const ACCENT4 = '#880e4f';   // dark rose — rod-cone retinal degeneration
const ACCENT5 = '#006064';   // dark teal — renal structural anomaly
const ACCENT6 = '#37474f';   // dark slate — epidemiology; AR inheritance
const ACCENT7 = '#4e342e';   // dark brown — cognitive/learning disability
const ACCENT8 = '#bf360c';   // burnt orange — obesity; hyperphagia; LepR

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

export default function BBS1Page() {
  const [tab, setTab]           = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [err, setErr]           = useState('');

  useEffect(() => {
    fetch(`${API}/api/bbs1/overview`).then(r => r.json()).then(setOverview).catch(e => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (tab === 1 && !breakdown)
      fetch(`${API}/api/bbs1/breakdown`).then(r => r.json()).then(setBreakdown).catch(e => setErr(String(e)));
    if (tab === 3 && !defs)
      fetch(`${API}/api/bbs1/definitions`).then(r => r.json()).then(setDefs).catch(e => setErr(String(e)));
  }, [tab]);

  const kpis     = overview?.kpis || {};
  const alerts   = overview?.alerts || {};
  const patients = overview?.patients || [];

  const retinalDist  = breakdown?.retinal_stage_distribution || [];
  const ageDist      = breakdown?.age_distribution || [];
  const ethDist      = breakdown?.ethnicity_distribution || [];
  const acDist       = breakdown?.allele_class_distribution || [];
  const dxRouteDist  = breakdown?.dx_route_distribution || [];
  const bbsomeTable  = breakdown?.bbsome_table || [];
  const notableVars  = breakdown?.notable_variants || [];
  const glossary     = defs?.glossary || [];

  return (
    <div className="container-fluid py-3">
      <div className="mb-2">
        <Link href="/" className="btn btn-sm btn-outline-secondary">&larr; Back to Portal</Link>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT4}11)`, border: `1px solid ${ACCENT}44` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: '1.6rem' }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>Bardet-Biedl Syndrome Type 1 (BBS1)</h4>
            <div className="text-muted small">Rod-Cone Dystrophy (Rod-FIRST) · Post-Axial Polydactyly · Obesity (LepR) · Cognitive · Renal · Hypogonadism · BBS1 BBSome Cargo-Recognition · Chr 11q13.2 · OMIM *209901/#209900 · Met390Arg European Founder · Most Common BBS Gene · Autosomal Recessive</div>
          </div>
          <div className="ms-auto d-flex gap-1 flex-wrap">
            <Badge text="BBS1 *209901" color={ACCENT} />
            <Badge text="Met390Arg Founder" color={ACCENT6} />
            <Badge text="Rod-FIRST ERG" color={ACCENT4} />
            <Badge text="Obesity LepR 91%" color={ACCENT8} />
            <Badge text="Polydactyly 65%" color={ACCENT2} />
            <Badge text="Renal 50%" color={ACCENT5} />
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
            <KPI label="Cohort (N)"        value={`${_COHORT_SIZE} patients`}                                     color={ACCENT6} />
            <KPI label="Rod-Cone Retinal"  value={`${kpis.rod_cone_pct ?? '—'}%`}                                color={ACCENT4} />
            <KPI label="Obesity (LepR)"    value={`${kpis.obesity_pct ?? '—'}%`}                                 color={ACCENT8} />
            <KPI label="Polydactyly"       value={`${kpis.poly_pct ?? '—'}%`}                                    color={ACCENT2} />
            <KPI label="Renal Anomaly"     value={`${kpis.renal_pct ?? '—'}%`}                                   color={ACCENT5} />
            <KPI label="Hypogonadism"      value={`${kpis.hypogonadism_pct ?? '—'}%`}                            color={ACCENT6} />
            <KPI label="Anosmia"           value={`${kpis.anosmia_pct ?? '—'}%`}                                 color={ACCENT6} />
            <KPI label="Cognitive LD"      value={`${kpis.cognitive_pct ?? '—'}%`}                               color={ACCENT7} />
            <KPI label="Tri-allelic BBS"   value={`${kpis.triallelic_pct ?? '—'}%`}                              color={ACCENT} />
            <KPI label="Misdiagnosed"      value={`${kpis.misdiagnosed_pct ?? '—'}%`}                            color={ACCENT2} />
            <KPI label="ESRD"              value={`${kpis.esrd_pct ?? '—'}%`}                                    color={ACCENT5} />
            <KPI label="No MKS Tier"       value={kpis.no_mks_tier ? 'Confirmed' : '—'}                          color={ACCENT3} />
          </div>

          {/* Critical Alerts */}
          <Section title="⚠ Critical Clinical Alerts — BBS1" color={ACCENT2}>
            <Alert color={ACCENT}>
              <strong>Met390Arg (c.1169T>G) — most common BBS mutation worldwide:</strong> European founder at BBS9-bridge interface (aa 390). Carrier frequency ~1:70–100 in Northern European BBS families. Homozygous → moderate BBS1. Full BBS multigene panel mandatory outside European ancestry — Met390Arg is NOT a universal BBS1 allele.
            </Alert>
            <Alert color={ACCENT4}>
              <strong>Rod-cone dystrophy — rod FIRST (~93%):</strong> Night blindness before central vision loss; ERG scotopic extinguished &gt;&gt; photopic. Distinguishes from Alstrom (cone-rod). Annual ERG + OCT from age 3. CRISPR-based BBS1 retinal therapy in preclinical development (2023–2024).
            </Alert>
            <Alert color={ACCENT8}>
              <strong>Obesity — LepR mis-trafficking / hypothalamic leptin resistance (~91%):</strong> Hyperphagia from infancy. GLP-1 RA (semaglutide) + metformin first-line. Bariatric surgery for BMI ≥ 40 (sleeve gastrectomy bypasses ciliary LepR via peripheral GLP-1/PYY axis). Early GLP-1 RA trials in paediatric BBS ongoing.
            </Alert>
            <Alert color={ACCENT6}>
              <strong>BBS1 is the most common BBS gene (~20–25% of all BBS in European populations):</strong> Full BBS panel (≥20 genes) mandatory. Tri-allelic BBS (BBS1 × 2 + BBS4/BBS5/BBS9 modifier) in ~5% of families — gene panel must include all BBSome subunits.
            </Alert>
          </Section>

          {/* Mechanism */}
          <Section title="🔬 BBS1 — BBSome Cargo-Recognition Arm / Rab8-GTP Docking Mechanism" color={ACCENT}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-3">
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT }}>BBS1 Protein Domain Architecture (~593 aa)</div>
                    <ul className="small mb-0">
                      <li><strong>N-terminal beta-propeller platform (aa 1–300):</strong> scaffold platform layer; contacts BBS4 (TPR) and BBS5 (PH domain); anchors BBSome lattice formation</li>
                      <li><strong>Central scaffold / BBS9-bridge region (aa 301–450):</strong> BBS9 (PTHB1) docking interface; <em>Met390 — BBS9 hydrophobic contact residue; most common BBS mutation site worldwide</em></li>
                      <li><strong>C-terminal ARM cargo-recognition domain (aa 451–593):</strong> Rab8-GTP binding surface; LHFPL4 / Arl6 adaptor contact; final cargo-docking step for IFT-B loading</li>
                    </ul>
                  </div>
                  <div className="col-md-6">
                    <div className="fw-bold mb-1" style={{ color: ACCENT2 }}>BBS1 LOF → Cargo-Recognition Failure → Multi-System Ciliopathy</div>
                    <ol className="small mb-0">
                      <li><strong>BBS1 LOF → BBS1–BBS9 bridge destabilised</strong> → BBSome assembles structurally (BBS2–BBS7 core intact) but cargo-recognition ARM non-functional</li>
                      <li>Rab8-GTP cannot recruit BBSome to cilia base → GPCRs cannot load onto IFT-B</li>
                      <li>LepR mis-trafficked from hypothalamic cilia → hyperphagia → obesity (91%)</li>
                      <li>Photoreceptor membrane proteins mis-trafficked → rod-FIRST degeneration (93%)</li>
                      <li>Olfactory cilia proteins mis-trafficked → anosmia (62%)</li>
                      <li>Neuronal cilia GPCR dysfunction → learning disability (63%); renal cilia → structural anomalies (50%)</li>
                    </ol>
                  </div>
                </div>
              </div>
            </div>
          </Section>

          {/* Cardinal Features */}
          <Section title="🎯 BBS1 Cardinal Features (6 Primary + Ancillary)" color={ACCENT2}>
            <div className="card border-0 shadow-sm mb-2">
              <div className="card-body">
                <div className="row g-2">
                  {[
                    { n: '①', feature: 'Rod-Cone Retinal Dystrophy', detail: 'Rod-FIRST (night blindness → ring scotoma → legal blindness); ERG: scotopic extinguished >> photopic; NOT cone-rod (Alstrom); annual ERG + OCT age 3+', color: ACCENT4 },
                    { n: '②', feature: 'Post-Axial Polydactyly', detail: 'Extra digit(s) — post-axial, hands/feet; ~65%; surgical excision early childhood; distinguishes from Alstrom (polydactyly absent); less frequent than BBS4/BBS6', color: ACCENT2 },
                    { n: '③', feature: 'Obesity (LepR mis-trafficking)', detail: 'Hyperphagia from infancy; LepR not trafficked into hypothalamic cilia; BMI 30–50 by adolescence; GLP-1 RA + metformin; bariatric for BMI ≥ 40', color: ACCENT8 },
                    { n: '④', feature: 'Cognitive / Learning Disability', detail: '~63% mild–moderate; neuronal cilia GPCR dysfunction; early educational support; similar to BBS2; absent in Alstrom', color: ACCENT7 },
                    { n: '⑤', feature: 'Renal Anomalies (Structural)', detail: '~50% — cysts, calyceal clubbing, horseshoe kidney, vesicoureteric reflux; NOT tubular nephropathy; annual USS + ACR + eGFR; ESRD ~10%', color: ACCENT5 },
                    { n: '⑥', feature: 'Hypogonadism', detail: '~77% — males: cryptorchidism, micropenis, small testes; females: irregular menses, ovarian cysts; hormone replacement age-appropriate; genital anomalies at birth → referral', color: ACCENT6 },
                    { n: 'Anc', feature: 'Anosmia / Hyposmia', detail: '~62% — olfactory cilia GPCR (OR) mis-trafficking; UPSIT testing; counselling implications (gas leak safety); Scratch-and-Sniff test in children', color: ACCENT6 },
                    { n: 'Anc', feature: 'Congenital Heart Disease', detail: '~7% — septal defects (VSD/ASD), pulmonary stenosis; neonatal cardiac echo in all BBS1; less frequent than in BBS4/BBS6', color: ACCENT4 },
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

          {/* BBS1 vs BBS2 distinction */}
          <Section title="🔍 BBS1 vs BBS2 — Key Molecular Distinctions" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Feature</th>
                    <th style={{ color: ACCENT }}>BBS1 (this dashboard)</th>
                    <th style={{ color: '#4a148c' }}>BBS2 (comparison)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>Founder mutation</strong></td><td className="text-danger fw-bold">M390R — European; most common BBS mutation worldwide (70–80% of European BBS1 alleles)</td><td>R631P — Bedouin/Arabian Peninsula (C-terminal CC)</td></tr>
                  <tr><td><strong>BBSome subunit role</strong></td><td className="fw-bold" style={{color:ACCENT}}>CARGO-RECOGNITION — ARM domain (Rab8-GTP docking)</td><td className="fw-bold" style={{color:'#4a148c'}}>STRUCTURAL CORE — BBS2–BBS7 WD40 dimer spine</td></tr>
                  <tr><td><strong>LOF consequence</strong></td><td>BBSome assembles structurally; cargo-docking arm fails</td><td>Catastrophic BBSome misassembly at step 0</td></tr>
                  <tr><td><strong>BBS frequency</strong></td><td className="fw-bold">~20–25% of all BBS (most common in European)</td><td>~10–20% of all BBS (second/third most common)</td></tr>
                  <tr><td><strong>Tri-allelic BBS</strong></td><td>~5% (BBS1 × 2 + BBS4/BBS5/BBS9 modifier)</td><td>~9% (BBS2 × 2 + BBS7/9 modifier — core partners)</td></tr>
                  <tr><td><strong>Clinical phenotype</strong></td><td>Indistinguishable from BBS2 — gene panel mandatory</td><td>Indistinguishable from BBS1 — gene panel mandatory</td></tr>
                  <tr><td><strong>OMIM gene</strong></td><td><code>*209901</code></td><td><code>*606151</code></td></tr>
                </tbody>
              </table>
            </div>
          </Section>

          {/* Patient table */}
          <Section title={`📋 Educational Cohort — ${_COHORT_SIZE} BBS1 Patients (seed 461)`} color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th><th>Variant</th>
                    <th>Retinal</th><th>Obese</th><th>Poly</th><th>Renal</th><th>Cogn</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.slice(0, 20).map((p, i) => (
                    <tr key={i}>
                      <td><code>{p.id}</code></td>
                      <td>{p.age}</td>
                      <td>{p.sex}</td>
                      <td className="small">{p.ethnicity}</td>
                      <td><code className="small">{p.variant}</code></td>
                      <td>{p.rod_cone ? <span className="text-danger">✓</span> : '—'}</td>
                      <td>{p.obesity ? <span className="text-warning">✓</span> : '—'}</td>
                      <td>{p.poly ? <span style={{color:ACCENT2}}>✓</span> : '—'}</td>
                      <td>{p.renal ? <span style={{color:ACCENT5}}>✓</span> : '—'}</td>
                      <td>{p.cognitive ? <span style={{color:ACCENT7}}>✓</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="text-muted small mt-1">Showing first 20 of {_COHORT_SIZE} patients. All data synthetic educational seed.</div>
          </Section>
        </div>
      )}

      {/* ─── TAB 1: Multi-System Breakdown ─── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="👁 Retinal Stage (Rod-FIRST, Rod-Cone Progression)" color={ACCENT4}>
              {retinalDist.map((r, i) => (
                <Bar key={i} label={r.label} value={r.n} max={_COHORT_SIZE} color={ACCENT4} />
              ))}
              <div className="small text-muted mt-1">Rod-FIRST: scotopic ERG extinguished before photopic. Contrasts with Alstrom (cone-rod).</div>
            </Section>

            <Section title="📊 Ethnicity Distribution" color={ACCENT6}>
              {ethDist.map((e, i) => (
                <Bar key={i} label={e.ethnicity} value={e.count} max={_COHORT_SIZE} color={ACCENT6} />
              ))}
            </Section>

            <Section title="📅 Age Distribution" color={ACCENT}>
              {ageDist.map((a, i) => (
                <Bar key={i} label={a.label} value={a.n} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="🧬 Allele Class Distribution" color={ACCENT}>
              {acDist.map((a, i) => (
                <Bar key={i} label={a.allele_class} value={a.count} max={_COHORT_SIZE} color={ACCENT} />
              ))}
            </Section>

            <Section title="🏥 Diagnosis Route Distribution" color={ACCENT7}>
              {dxRouteDist.map((d, i) => (
                <Bar key={i} label={d.route} value={d.n} max={_COHORT_SIZE} color={ACCENT7} />
              ))}
            </Section>

            <Section title="📊 Phenotype Summary" color={ACCENT2}>
              {[
                { k: 'rod_cone',     label: 'Rod-Cone Retinal',   color: ACCENT4 },
                { k: 'obesity',      label: 'Obesity (LepR)',     color: ACCENT8 },
                { k: 'poly',         label: 'Polydactyly',        color: ACCENT2 },
                { k: 'renal',        label: 'Renal Anomaly',      color: ACCENT5 },
                { k: 'cognitive',    label: 'Cognitive LD',       color: ACCENT7 },
                { k: 'hypogonadism', label: 'Hypogonadism',       color: ACCENT6 },
                { k: 'anosmia',      label: 'Anosmia',            color: ACCENT6 },
                { k: 'chd',          label: 'CHD',                color: ACCENT4 },
                { k: 'hepatic',      label: 'Hepatic',            color: ACCENT3 },
                { k: 'esrd',         label: 'ESRD',               color: ACCENT5 },
              ].map((f, i) => {
                const d = breakdown.phenotype_summary?.[f.k];
                return d ? <Bar key={i} label={f.label} value={d.n} max={_COHORT_SIZE} color={f.color} /> : null;
              })}
            </Section>
          </div>

          {/* BBSome table */}
          <div className="col-12">
            <Section title="🔩 BBSome Octameric Complex — All Subunits (BBS1 = Cargo-Recognition)" color={ACCENT}>
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
          </div>
        </div>
      )}
      {tab === 1 && !breakdown && <div className="text-center py-4 text-muted">Loading breakdown…</div>}

      {/* ─── TAB 2: Variants & Diagnostics ─── */}
      {tab === 2 && (
        <div>
          <Section title="🧬 Notable BBS1 Variants" color={ACCENT}>
            {(breakdown?.notable_variants || notableVars).length === 0 ? (
              <div className="text-muted small">Load breakdown tab first to see variants.</div>
            ) : (
              (breakdown?.notable_variants || notableVars).map((v, i) => (
                <div key={i} className="card border-0 shadow-sm mb-2">
                  <div className="card-body py-2">
                    <div className="d-flex align-items-baseline gap-2 mb-1">
                      <strong style={{ color: ACCENT }}>{v.name}</strong>
                      <code className="small">{v.cdna}</code>
                      <span className="badge" style={{ background: ACCENT + 'aa', fontSize: '0.7em' }}>{v.severity}</span>
                    </div>
                    <div className="small text-muted mb-1"><strong>Domain:</strong> {v.domain}</div>
                    <div className="small text-muted mb-1"><strong>Population:</strong> {v.population}</div>
                    <div className="small">{v.mechanism}</div>
                  </div>
                </div>
              ))
            )}
          </Section>

          <Section title="🩺 Diagnostic Protocol — BBS1" color={ACCENT6}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT + '18' }}>
                    <th>Investigation</th><th>Frequency</th><th>Rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Full BBS multigene panel (≥20 genes)', 'At diagnosis', 'BBS1 M390R targeted + full panel; rules out tri-allelic BBS (BBS4/BBS5/BBS9 third allele)'],
                    ['ERG (Electroretinography) + OCT', 'Annual from age 3', 'Rod-FIRST rod-cone dystrophy; scotopic ERG extinguishes first; baseline before symptoms'],
                    ['Abdominal USS', 'Annual', 'Renal cysts, calyceal clubbing, horseshoe kidney, hepatic anomaly screening'],
                    ['BMI + metabolic panel', 'Every 6 months', 'LepR mis-trafficking obesity; GLP-1 RA initiation decision; dyslipidaemia / NASH risk'],
                    ['eGFR + ACR', 'Annual', 'CKD progression monitoring; ESRD risk stratification (10% BBS1)'],
                    ['Genital / hormone assessment', 'Annual paediatric', 'Cryptorchidism / micropenis (males); irregular menses / ovarian cysts (females); hormone replacement'],
                    ['UPSIT olfactory testing', 'At diagnosis + 5y', 'Anosmia / hyposmia; safety counselling (gas); quality-of-life'],
                    ['Neonatal cardiac echo', 'At birth / diagnosis', 'CHD (septal defects, PS) in ~7%; BBS1 lower than BBS4/BBS6 but mandatory'],
                    ['Developmental assessment', 'Annual to age 18', 'Learning disability (63%); early educational support programme; IQ testing age 5+'],
                    ['BBS1 carrier screening (partner)', 'At diagnosis', 'M390R European ancestry — carrier frequency ~1:150–200 general population; reproductive planning'],
                  ].map((r, i) => (
                    <tr key={i}>
                      <td><strong>{r[0]}</strong></td>
                      <td>{r[1]}</td>
                      <td className="small">{r[2]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section title="⚖️ DDx — BBS1 vs Alstrom vs Laurence-Moon vs non-syndromic RP" color={ACCENT4}>
            <div className="table-responsive">
              <table className="table table-sm small mb-0">
                <thead>
                  <tr style={{ background: ACCENT4 + '18' }}>
                    <th>Feature</th><th style={{color:ACCENT}}>BBS1</th><th>Alstrom (ALMS1)</th><th>Laurence-Moon</th><th>Non-syndromic RP</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td><strong>Retinal pattern</strong></td><td className="fw-bold text-danger">Rod-FIRST (rod-cone)</td><td>Cone-FIRST (cone-rod)</td><td>Rod-cone</td><td>Rod-cone or rod-only</td></tr>
                  <tr><td><strong>Polydactyly</strong></td><td>65% post-axial</td><td className="text-muted">ABSENT</td><td className="text-muted">ABSENT</td><td className="text-muted">ABSENT</td></tr>
                  <tr><td><strong>Obesity</strong></td><td>91% (LepR)</td><td>90% (LepR + ALMS1)</td><td>Variable</td><td className="text-muted">ABSENT</td></tr>
                  <tr><td><strong>Renal</strong></td><td>50% structural</td><td>Tubular nephropathy</td><td>Rare</td><td className="text-muted">ABSENT</td></tr>
                  <tr><td><strong>Cognitive</strong></td><td>63% mild-mod</td><td>Normal IQ</td><td>~50%</td><td className="text-muted">ABSENT</td></tr>
                  <tr><td><strong>CHD</strong></td><td>7%</td><td>70–80% cardiomyopathy</td><td>Rare</td><td className="text-muted">ABSENT</td></tr>
                  <tr><td><strong>Gene/inheritance</strong></td><td>BBS1 / AR</td><td>ALMS1 / AR</td><td>PNPLA6 / AR</td><td>Multiple / variable</td></tr>
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ─── TAB 3: Definitions ─── */}
      {tab === 3 && defs && (
        <div>
          <div className="card border-0 shadow-sm mb-3">
            <div className="card-body">
              <div className="row g-2 small">
                <div className="col-md-4"><strong>Gene:</strong> BBS1 (OMIM *{defs.omim_gene})</div>
                <div className="col-md-4"><strong>Disease:</strong> BBS (OMIM #{defs.omim_disease})</div>
                <div className="col-md-4"><strong>Chromosome:</strong> {defs.chromosome}</div>
                <div className="col-12"><strong>Protein:</strong> {defs.protein_size}</div>
                <div className="col-12"><strong>Inheritance:</strong> {defs.inheritance}</div>
              </div>
            </div>
          </div>

          <div className="alert mb-3" style={{ background: ACCENT + '11', borderLeft: `4px solid ${ACCENT}` }}>
            <strong>BBSome Cargo-Recognition Rule:</strong> {defs.bbsome_cargo_recognition_rule}
          </div>

          <Section title="📖 Glossary" color={ACCENT}>
            {glossary.map((g, i) => (
              <div key={i} className="card border-0 shadow-sm mb-2">
                <div className="card-body py-2">
                  <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{g.term}</div>
                  <div className="small">{g.definition}</div>
                </div>
              </div>
            ))}
          </Section>
        </div>
      )}
      {tab === 3 && !defs && <div className="text-center py-4 text-muted">Loading definitions…</div>}
    </div>
  );
}
