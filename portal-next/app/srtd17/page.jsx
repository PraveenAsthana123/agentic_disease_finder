'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'TCTEX1D2 Dynein-2 Light Chain & Retrograde IFT', 'Definitions'];

// SRTD17 colour scheme — TCTEX1D2 / dynein-2 light chain / T-complex fold / narrow thorax / Jeune ATD17
const ACCENT  = '#1a237e';   // deep indigo — TCTEX1D2 light chain; dynein-2 tail light subunit
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#1565c0';   // deep blue — renal TIN; ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#37474f';   // dark slate — T-complex fold; molecular architecture
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM club cilia; diagnostic
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; VEPTR surgery

const SEED = 389;

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

export default function SRTD17Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd17/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd17/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd17/definitions`).then(r => r.json()),
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
            TCTEX1D2 Short-Rib Thoracic Dysplasia 17 (SRTD17 / ATD17)
          </h4>
          <div className="text-muted small">
            OMIM #617405 · *617819 · 3q21.3 · 142 aa · Dynein-2 Light Chain (T-Complex 1-Like) · AR · ~1/2M–5M · seed={SEED}
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe (null) alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
      </Alert>
      <Alert color={ACCENT}>
        <strong>TCTEX1D2 is the SMALLEST subunit of the dynein-2 complex (142 aa)</strong> — a light chain with
        a T-complex protein 1-like domain that docks onto the WDR60 (SRTD8) β-propeller C-face.
        Loss → WDR60 destabilisation → retrograde IFT failure → Hedgehog signalling failure → narrow thorax.
        Extremely rare: ~5–15 families worldwide (2026). TCTEX1D2 must be on the SRTD gene panel.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert:</strong> Same bulging/club ciliary tips as SRTD3, 8, 11, 15 (IFT-B pile-up at tip) —
        cannot distinguish SRTD17 from other dynein-2 SRTDs by EM alone. Gene panel is mandatory.
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
            <KPI label="Cohort (seed=389)" value={N}             color={ACCENT}  />
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
                    <tr><td className="fw-bold">Gene</td><td>TCTEX1D2 (*617819) — T-Complex Protein 1-Like Domain Containing 2 — 3q21.3 — 142 aa</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#617405 — SRTD17 (ATD17)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>Dynein-2 light chain: N-terminal T-complex dimerisation fold + core WDR60-interface domain + C-terminal regulatory helix</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/2,000,000–5,000,000 · ~5–15 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>Extremely rare; fifth known dynein-2-subunit SRTD gene</td></tr>
                    <tr><td className="fw-bold">Dynein-2 position</td><td>Light chain; contacts WDR60 (SRTD8) β-propeller C-face; stabilises WDR60 within the dynein-2 tail</td></tr>
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
                <Alert color={ACCENT2}>
                  <strong>Severity vs SRTD8/WDR60:</strong> Loss of TCTEX1D2 secondarily destabilises
                  WDR60 (its direct binding partner). SRTD17 and SRTD8 share overlapping molecular basis
                  but differ in frequency (~5–15 families vs ~50–75 families). Biallelic TCTEX1D2 null
                  → SRPS spectrum. Phenotype is generally moderate-to-mild: secondary organ involvement
                  (renal, retinal, hepatic) is less frequent than SRTD3/8/11.
                </Alert>
              </Section>

              <Section title="Key Clinical Distinction" color={ACCENT2}>
                <p className="small">{overview.key_distinction}</p>
              </Section>

              <Section title="Dynein-2 Complex — SRTD Subunit Table" color={ACCENT6}>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered small">
                    <thead><tr>
                      <th>Subunit</th><th>Role</th><th>SRTD</th><th>Chr</th><th>Freq</th>
                    </tr></thead>
                    <tbody>
                      {overview.dynein2_subunit_table.map((r, i) => (
                        <tr key={i} style={r.srtd === 'SRTD17' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
                          <td>{r.subunit}</td><td>{r.role}</td><td>{r.srtd}</td><td>{r.chr}</td><td>{r.freq}</td>
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
            <Section title="VEPTR Surgery" color={ACCENT8}>
              {breakdown.veptr_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT8} />
              ))}
            </Section>
            <Section title="First Presentation" color={ACCENT7}>
              {breakdown.presentation_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Prior Misdiagnosis" color={ACCENT7}>
              {breakdown.misdiagnosis_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT7} />
              ))}
            </Section>
            <Section title="Allele Class" color={ACCENT6}>
              {breakdown.allele_class_summary.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT6} />
              ))}
            </Section>
            <Section title="Ethnicity" color={ACCENT}>
              {breakdown.ethnicity_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.ethnicity} n={r.n} total={N} color={ACCENT} />
              ))}
            </Section>
            <Section title="Renal Treatment" color={ACCENT3}>
              {breakdown.treatment_renal.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT3} />
              ))}
            </Section>
            <Section title="Respiratory Management" color={ACCENT2}>
              {breakdown.respiratory_management.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT2} />
              ))}
            </Section>
            <Section title="Top Pathogenic Variants (TCTEX1D2 — cohort seed=389)" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>Variant</th><th>n</th></tr></thead>
                  <tbody>
                    {breakdown.top_variants.map((r, i) => (
                      <tr key={i}><td>{r.variant}</td><td>{r.n}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: TCTEX1D2 Dynein-2 Light Chain & Retrograde IFT ── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="TCTEX1D2 Protein Structure (142 aa)" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>TCTEX1D2 is the SMALLEST dynein-2 subunit (142 aa)</strong> — a light chain with a
                T-complex protein 1-like fold. Unlike WDR34/WDR60 (β-propeller adapters) or DYNC2LI1
                (RAS-fold scaffold), TCTEX1D2 is a compact light chain docking onto the WDR60
                β-propeller C-face. Its loss secondarily destabilises WDR60 (SRTD8).
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Domain</th><th>Residues</th><th>Function</th></tr></thead>
                <tbody>
                  <tr><td>N-terminal T-complex fold</td><td>aa 1–60</td><td>Dimerisation interface; dynein-2 complex docking platform; loss → null phenotype</td></tr>
                  <tr><td>Core WDR60-interface domain</td><td>aa 61–110</td><td>Direct contact with WDR60 (SRTD8) β-propeller C-face; missense cluster aa 70–100; most pathogenic variants here</td></tr>
                  <tr><td>C-terminal regulatory helix</td><td>aa 111–142</td><td>IFT train assembly regulation; phospho-site; hypomorphic missense → mild SRTD17</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="Dynein-2 Complex Assembly — TCTEX1D2 Position" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>#</th><th>Subunit</th><th>SRTD</th><th>TCTEX1D2 contact?</th></tr></thead>
                  <tbody>
                    <tr><td>1</td><td>DYNC2H1 (×2)</td><td>SRTD3</td><td>Indirect (contacts WDR60 which contacts DYNC2H1)</td></tr>
                    <tr><td>2</td><td>DYNC2LI1</td><td>SRTD15</td><td>Indirect (contacts WDR60 side of tail)</td></tr>
                    <tr><td>3</td><td>WDR34</td><td>SRTD11</td><td>Indirect (opposite side of dynein-2 tail)</td></tr>
                    <tr><td>4</td><td>WDR60</td><td>SRTD8</td><td>✓ Direct (WDR60 β-propeller C-face; key interaction)</td></tr>
                    <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}><td>5</td><td>TCTEX1D2</td><td>SRTD17 ◀</td><td>— (self; light chain; WDR60 C-face dock)</td></tr>
                    <tr><td>6</td><td>DYNLRB1/2</td><td>—</td><td>Roadblock-type; no direct TCTEX1D2 contact</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="Retrograde IFT Failure Cascade (TCTEX1D2 Loss)" color={ACCENT2}>
              {[
                ['TCTEX1D2 absent', 'T-complex light chain cannot dock onto WDR60 β-propeller C-face → WDR60 destabilised within the dynein-2 tail'],
                ['WDR60 (SRTD8) partially decoupled', 'WDR60 loses its TCTEX1D2 stabiliser → partial dynein-2 tail assembly failure (less complete than DYNC2LI1/DYNC2H1 loss)'],
                ['Dynein-2 activity reduced', 'Retrograde IFT from ciliary tip to cell body impaired (incomplete arrest — milder than null DYNC2H1)'],
                ['IFT-B accumulation at ciliary tip', '"Bulging/club" cilia on EM — identical to SRTD3/8/11/15; EM cannot distinguish'],
                ['Hedgehog (Ihh/Shh) failure', 'PTCH1/SMO/GLI trafficking from tip impaired → GLI3R processing fails → Hh target genes dysregulated'],
                ['Chondrocyte failure', 'Short ribs, NARROW THORAX (primary), short long bones'],
                ['Renal tubular (secondary)', 'TIN + corticomedullary cysts → ESRD in survivors (~15–20%; less than SRTD3)'],
                ['Retinal photoreceptors (secondary)', 'Connecting cilium IFT failure → rod-cone dystrophy (~8%; less than SRTD8)'],
                ['Biliary cholangiocytes (minority)', 'CHF — ductal plate malformation (~5%; least common in SRTD17)'],
              ].map(([step, desc], i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-2" style={{ background: ACCENT }}>Step {i + 1}</span>
                  <strong>{step}:</strong> {desc}
                </div>
              ))}
            </Section>

            <Section title="TCTEX1D2 vs WDR60 — Key Differences" color={ACCENT7}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Feature</th><th>TCTEX1D2 (SRTD17)</th><th>WDR60 (SRTD8)</th></tr></thead>
                <tbody>
                  <tr><td>Protein size</td><td>142 aa (smallest dynein-2 subunit)</td><td>1173 aa</td></tr>
                  <tr><td>Fold type</td><td>T-complex 1-like domain (light chain)</td><td>7-blade WD40 β-propeller</td></tr>
                  <tr><td>Chromosome</td><td>3q21.3</td><td>7q36.3</td></tr>
                  <tr><td>Dynein-2 role</td><td>LIGHT CHAIN — stabilises WDR60 within dynein-2 tail</td><td>INTERMEDIATE CHAIN — major structural adapter (opposite WDR34)</td></tr>
                  <tr><td>SRTD frequency</td><td>Extremely rare (~5–15 families)</td><td>~5–10% of SRTD</td></tr>
                  <tr><td>Contact partner</td><td>WDR60 β-propeller C-face</td><td>DYNC2H1 + TCTEX1D2 + DYNC2LI1</td></tr>
                  <tr><td>OMIM gene</td><td>*617819</td><td>*615462</td></tr>
                  <tr><td>OMIM disease</td><td>#617405</td><td>#615503</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="Why No Situs Inversus / No Joubert MTS" color={ACCENT4}>
              <p className="small mb-1">
                <strong>No situs inversus:</strong> TCTEX1D2 functions in PRIMARY non-motile cilia (9+0 axoneme).
                Situs inversus requires MOTILE nodal cilia (9+2). TCTEX1D2 is not expressed in nodal motile cilia.
              </p>
              <p className="small mb-0">
                <strong>No Joubert MTS:</strong> The molar tooth sign requires brainstem/cerebellar vermis
                decussation defects (JBTS ciliopathies). TCTEX1D2/SRTD17 affects skeletal chondrocytes
                and secondary organs — NOT the CNS midline structures.
              </p>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene Card — TCTEX1D2" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD17" color={ACCENT2}>
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
                {defs.diagnostic_workup.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
              </ol>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="Mechanism Glossary" color={ACCENT}>
              {defs.mechanism_glossary.map((g, i) => (
                <div key={i} className="mb-2 small">
                  <strong style={{ color: ACCENT }}>{g.term}:</strong> {g.definition}
                </div>
              ))}
            </Section>

            <Section title="Key Pathogenic Variants (TCTEX1D2)" color={ACCENT}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>Variant</th><th>Domain</th><th>Consequence</th><th>Ethnicity</th></tr></thead>
                  <tbody>
                    {defs.key_variants.map((v, i) => (
                      <tr key={i}>
                        <td><code>{v.variant}</code></td>
                        <td>{v.domain}</td>
                        <td>{v.consequence}</td>
                        <td>{v.ethnicity}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>

            <Section title="Treatment Summary" color={ACCENT3}>
              <ol className="small ps-3">
                {defs.treatment_summary.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
              </ol>
            </Section>

            <Section title="Differential Diagnosis Table" color={ACCENT7}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>Disease</th><th>Key Differentiator from SRTD17</th></tr></thead>
                  <tbody>
                    {defs.ddx_table.map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold">{d.disease}</td>
                        <td>{d.key_difference}</td>
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
