'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'WDR35 IFT-A Complex & Cilia Architecture', 'Definitions'];

// SRTD7 colour scheme — WDR35/IFT121 / IFT-A complex / direct IFT140 partner / short/stubby EM
const ACCENT  = '#00695c';   // dark teal-green — IFT-A complex; WDR35/IFT121
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#1565c0';   // deep blue — renal TIN; ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#2e7d32';   // dark green — IFT-A complex; molecular biology; WD40 scaffold
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM short cilia vs club; diagnostic
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial/preaxial; VEPTR surgery

const SEED = 395;

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

function SimpleBar({ label, n, total, color }) {
  const pct = total > 0 ? Math.round(n / total * 100) : 0;
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="fw-bold">{n} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function SRTD7Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd7/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd7/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOver(o); setBreak(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err)       return <div className="alert alert-danger m-4">API error: {err}</div>;
  if (!overview) return <div className="text-center p-5"><div className="spinner-border" /></div>;

  const k = overview.kpis;
  const N = overview.cohort_n;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <Link href="/" className="btn btn-sm btn-outline-secondary">← Back</Link>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
            WDR35 Short-Rib Thoracic Dysplasia 7 (SRTD7 / ATD7)
          </h4>
          <div className="text-muted small">
            OMIM #614091 · *613602 · 2p24.1 · 1181 aa · IFT-A Complex WD40 Subunit / IFT121 · AR · ~1/300K–700K · seed={SEED}
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe (null) alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
      </Alert>
      <Alert color={ACCENT}>
        <strong>WDR35/IFT121 is an IFT-A COMPLEX subunit — NOT a dynein-2 motor subunit.</strong> This is the
        critical mechanistic distinction from SRTD3/8/11/15/17. WDR35 (1181 aa; 14-blade WD40 β-propeller)
        is the <strong>DIRECT structural binding partner of IFT140 (SRTD9)</strong> within the IFT-A complex:
        WDR35 C-face ↔ IFT140 N-terminal WD40 domain. Loss → IFT-A destabilisation → retrograde IFT
        failure + IFT-B import defect → SHORT/STUBBY CILIA → Hedgehog signal failure → narrow thorax.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert — KEY DISTINCTION:</strong> WDR35 (SRTD7) = <strong>SHORT/STUBBY CILIA</strong>{' '}
        with IFT-particle accumulation at the cilia base/transition zone. This is DIFFERENT from
        dynein-2 SRTDs (SRTD3/8/11/15/17) which show BULGING/CLUB CILIA TIPS. Within IFT-A SRTDs,
        WDR35 (SRTD7) and IFT140 (SRTD9) and WDR19 (SRTD5) all show SHORT/STUBBY CILIA — gene panel
        is mandatory to distinguish them.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <>
          <div className="row g-2 mb-3">
            <KPI label={`Cohort (seed=${SEED})`}  value={N}             color={ACCENT}  />
            <KPI label="Thorax Severe"     value={`${k.thorax_severe_n} (${k.thorax_severe_pct}%)`} color={ACCENT2} />
            <KPI label="Polydactyly"       value={`${k.polydactyly_n} (${k.polydactyly_pct}%)`}   color={ACCENT8} />
            <KPI label="Renal Involved"    value={`${k.renal_any_n} (${k.renal_any_pct}%)`}        color={ACCENT3} />
            <KPI label="Retinal Dystrophy" value={`${k.retinal_any_n} (${k.retinal_any_pct}%)`}   color={ACCENT4} />
            <KPI label="CHF (hepatic)"     value={`${k.hepatic_chf_n} (${k.hepatic_chf_pct}%)`}   color={ACCENT5} />
            <KPI label="VEPTR Surgery"     value={`${k.veptr_any_n} (${k.veptr_any_pct}%)`}       color={ACCENT8} />
            <KPI label="Renal Transplant ✓" value={k.transplant_done_n}                            color={ACCENT3} />
            <KPI label="Prior Misdiagnosis" value={`${k.misdiagnosis_n} (${k.misdiagnosis_pct}%)`} color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Disease Summary" color={ACCENT}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>WDR35 (*613602) / IFT121 — WD-Repeat Domain 35 — 2p24.1 — 1181 aa</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#614091 — SRTD7 (ATD7)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>14-blade WD40 β-propeller: N-terminal extension (aa 1–120) + blades 1–7 (aa 121–580; IFT-B bridge) + blades 8–14/C-face (aa 581–1181; DIRECT IFT140 SRTD9 binding)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/300,000–700,000 · ~30–50 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>~0.3–0.5% of all SRTD; 3rd most common IFT-A SRTD (after WDR19/SRTD5, IFT140/SRTD9)</td></tr>
                    <tr><td className="fw-bold">IFT complex</td><td>IFT-A complex (NOT dynein-2 motor); DIRECT IFT140 binding partner; retrograde IFT adapter + IFT-B bridge</td></tr>
                    <tr><td className="fw-bold">EM finding</td><td>SHORT/STUBBY CILIA with IFT-particle accumulation at transition zone/base — distinct from dynein-2 SRTDs (club tip); SAME CLASS as SRTD5 and SRTD9</td></tr>
                    <tr><td className="fw-bold">No situs inversus</td><td>✓ Primary non-motile cilia (9+0), not nodal motile cilia</td></tr>
                    <tr><td className="fw-bold">No Joubert MTS</td><td>✓ No molar tooth sign — brainstem/vermis not involved</td></tr>
                    <tr><td className="fw-bold">Renal Tx curative</td><td>✓ Cell-autonomous IFT defect; no recurrence post-transplant</td></tr>
                  </tbody>
                </table>
              </Section>

              <Section title="Age at Diagnosis" color={ACCENT6}>
                {breakdown && (() => {
                  const age = overview.age_distribution;
                  return (
                    <>
                      <SimpleBar label="0–1 yr (neonatal)"  n={age.dx_0_1yr}   total={N} color={ACCENT2} />
                      <SimpleBar label="2–5 yr (infant)"    n={age.dx_2_5yr}   total={N} color={ACCENT}  />
                      <SimpleBar label="6–10 yr (child)"    n={age.dx_6_10yr}  total={N} color={ACCENT3} />
                      <SimpleBar label="11–16 yr (teen)"    n={age.dx_11_16yr} total={N} color={ACCENT6} />
                    </>
                  );
                })()}
              </Section>

              <Section title="Sex Split" color={ACCENT6}>
                <div className="d-flex gap-4">
                  <span>♂ M: <strong>{overview.sex_split.M}</strong></span>
                  <span>♀ F: <strong>{overview.sex_split.F}</strong></span>
                </div>
              </Section>
            </div>

            <div className="col-md-6">
              <Section title="Molecular Mechanism" color={ACCENT}>
                <p className="small mb-2">{overview.mechanism}</p>
                <Alert color={ACCENT6}>
                  <strong>Direct IFT140 binding — molecular linchpin:</strong> The C-face of WDR35's
                  14-blade WD40 β-propeller directly docks onto the N-terminal WD40 domain of IFT140.
                  This is the most critical binary interaction in the WDR35–IFT140 half of the IFT-A
                  complex. Pathogenic variants in WDR35 blades 8–14 preferentially disrupt this contact,
                  destabilising the entire IFT-A complex and causing SRTD7.
                </Alert>
              </Section>

              <Section title="Key Clinical Distinction" color={ACCENT2}>
                <p className="small">{overview.key_distinction}</p>
              </Section>

              <Section title="IFT-A Complex — SRTD Subunit Table" color={ACCENT6}>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered small">
                    <thead><tr>
                      <th>Subunit</th><th>SRTD</th><th>Chr</th><th>Freq</th>
                    </tr></thead>
                    <tbody>
                      {overview.ifta_subunit_table.map((r, i) => (
                        <tr key={i} style={r.srtd === 'SRTD7' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                          <td>{r.subunit}</td><td>{r.srtd}</td><td>{r.chr}</td><td>{r.freq}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
            </div>
          </div>
        </>
      )}

      {/* ── TAB 1: Diagnostic Breakdown ── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Thorax Severity" color={ACCENT2}>
              {breakdown.thorax_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Polydactyly" color={ACCENT8}>
              {breakdown.polydactyly_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT8} />
              ))}
            </Section>
            <Section title="Renal Status" color={ACCENT3}>
              {breakdown.renal_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT3} />
              ))}
            </Section>
            <Section title="CKD Stage" color={ACCENT3}>
              {breakdown.ckd_stage_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Retinal Status" color={ACCENT4}>
              {breakdown.retinal_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT4} />
              ))}
            </Section>
            <Section title="Hepatic / CHF Status" color={ACCENT5}>
              {breakdown.hepatic_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT5} />
              ))}
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="VEPTR / Surgical Management" color={ACCENT8}>
              {breakdown.veptr_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT8} />
              ))}
            </Section>
            <Section title="Presentation Mode" color={ACCENT}>
              {breakdown.presentation_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT} />
              ))}
            </Section>
            <Section title="Prior Misdiagnoses" color={ACCENT7}>
              {breakdown.misdiagnosis_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Allele Class Distribution" color={ACCENT6}>
              {breakdown.allele_class_summary.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Ethnicity Distribution" color={ACCENT}>
              {breakdown.ethnicity_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.ethnicity} n={r.n} total={N} color={ACCENT} />
              ))}
            </Section>
            <Section title="Respiratory Management" color={ACCENT2}>
              {breakdown.respiratory_management.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT2} />
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: WDR35 IFT-A Complex & Cilia Architecture ── */}
      {tab === 2 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Top Pathogenic Variants (WDR35)" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Variant</th><th>n</th></tr></thead>
                <tbody>
                  {breakdown.top_variants.map((v, i) => (
                    <tr key={i}><td>{v.variant}</td><td className="fw-bold">{v.n}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="WDR35 Domain Architecture" color={ACCENT6}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  <tr><td className="fw-bold">N-terminal extension (aa 1–120)</td><td>IFT-A complex entry; IFT43 contact; stabilises N-terminal propeller blades; hypomorphic variant zone</td></tr>
                  <tr><td className="fw-bold">WD40 blades 1–7 (aa 121–580)</td><td>IFT-A assembly platform; IFT-B anterograde bridge; TTC21B/IFT139 and IFT122 contacts within IFT-A; central pathogenic variant cluster</td></tr>
                  <tr style={{ background: ACCENT + '18', fontWeight: 'bold' }}><td>WD40 blades 8–14 / C-face (aa 581–1181)</td><td>DIRECT IFT140 (SRTD9) N-terminal WD40 binding — molecular linchpin; most severe clinical variants here; complete loss → SRPS spectrum</td></tr>
                </tbody>
              </table>
              <Alert color={ACCENT6}>
                <strong>WDR35 C-face ↔ IFT140 WD40:</strong> The C-face (blades 8–14) of WDR35's β-propeller
                docks onto the N-terminal WD40 domain of IFT140. This is the most critical protein-protein
                interaction for maintaining the structural integrity of this IFT-A sub-complex. Variants
                disrupting this interface cause moderate to severe SRTD7.
              </Alert>
            </Section>

            <Section title="IFT-A vs Dynein-2 — EM Distinction" color={ACCENT7}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Group</th><th>EM Finding</th><th>Gene Examples</th></tr></thead>
                <tbody>
                  <tr style={{ background: ACCENT + '18' }}><td className="fw-bold">IFT-A SRTDs</td><td>SHORT/STUBBY CILIA; IFT-particle accumulation at cilia base / transition zone</td><td>WDR35 (SRTD7), IFT140 (SRTD9), WDR19 (SRTD5)</td></tr>
                  <tr style={{ background: '#b71c1c18' }}><td className="fw-bold">Dynein-2 SRTDs</td><td>BULGING/CLUB CILIA TIPS; IFT-B accumulates at ciliary tip (retrograde motor absent)</td><td>DYNC2H1 (SRTD3), WDR60 (SRTD8), WDR34 (SRTD11), DYNC2LI1 (SRTD15), TCTEX1D2 (SRTD17)</td></tr>
                </tbody>
              </table>
              <Alert color={ACCENT7}>
                <strong>Within IFT-A SRTDs:</strong> WDR35 (SRTD7), IFT140 (SRTD9), and WDR19 (SRTD5) all
                show SHORT/STUBBY CILIA on EM — they CANNOT be distinguished from each other by EM alone.
                Gene panel with WDR35, IFT140, WDR19 sequencing is mandatory to differentiate.
              </Alert>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="IFT-A Complex — Full Subunit Table" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr>
                    <th>Subunit</th><th>Role</th><th>SRTD</th><th>Chr</th><th>Freq</th>
                  </tr></thead>
                  <tbody>
                    {overview.ifta_subunit_table.map((r, i) => (
                      <tr key={i} style={r.srtd === 'SRTD7' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                        <td>{r.subunit}</td>
                        <td className="small">{r.role}</td>
                        <td>{r.srtd}</td>
                        <td>{r.chr}</td>
                        <td className="small">{r.freq}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>

            <Section title="Differential Diagnosis — DDx Table" color={ACCENT7}>
              {defs && (
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>Disease</th><th>Key Difference from WDR35 (SRTD7)</th></tr></thead>
                  <tbody>
                    {defs.ddx_table.map((r, i) => (
                      <tr key={i}>
                        <td className="fw-bold text-nowrap">{r.disease}</td>
                        <td className="small">{r.key_difference}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Section>

            <Section title="Renal Transplant Outcomes" color={ACCENT3}>
              {breakdown.treatment_renal.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT3} />
              ))}
              <Alert color={ACCENT3}>
                <strong>Renal transplant is CURATIVE</strong> — WDR35/IFT121 IFT defect is cell-autonomous.
                The donor kidney (donor WDR35+) functions normally post-Tx; no recurrence.
              </Alert>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene Card — WDR35 / IFT121" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD7 / ATD7" color={ACCENT2}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.disease_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Diagnostic Workup" color={ACCENT6}>
              <ol className="small ps-3">
                {defs.diagnostic_workup.map((step, i) => (
                  <li key={i} className="mb-1">{step.replace(/^\d+\.\s/, '')}</li>
                ))}
              </ol>
            </Section>
          </div>
          <div className="col-md-6">
            <Section title="Mechanism Glossary" color={ACCENT}>
              {defs.mechanism_glossary.map((g, i) => (
                <div key={i} className="mb-2">
                  <span className="fw-bold small" style={{ color: ACCENT }}>{g.term}: </span>
                  <span className="small">{g.definition}</span>
                </div>
              ))}
            </Section>

            <Section title="Key Variants — WDR35" color={ACCENT6}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Variant</th><th>Domain</th><th>Consequence</th><th>Ethnicity</th></tr></thead>
                <tbody>
                  {defs.key_variants.map((v, i) => (
                    <tr key={i}>
                      <td className="fw-bold text-nowrap">{v.variant}</td>
                      <td className="small">{v.domain}</td>
                      <td className="small">{v.consequence}</td>
                      <td className="small">{v.ethnicity}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Treatment Summary" color={ACCENT3}>
              <ol className="small ps-3">
                {defs.treatment_summary.map((step, i) => (
                  <li key={i} className="mb-1">{step.replace(/^\d+\.\s/, '')}</li>
                ))}
              </ol>
            </Section>
          </div>
        </div>
      )}
    </div>
  );
}
