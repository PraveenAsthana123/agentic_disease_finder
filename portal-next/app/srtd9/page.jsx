'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT140 IFT-A Complex & Cilia Architecture', 'Definitions'];

// SRTD9 colour scheme — IFT140 / IFT-A complex / cilia / short/stubby EM / narrow thorax
const ACCENT  = '#006064';   // dark teal — IFT-A complex; intraflagellar transport
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#1565c0';   // deep blue — renal TIN; ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#1b5e20';   // dark green — IFT-A complex; molecular biology; WD40 scaffold
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM short cilia vs club; diagnostic
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial/preaxial; VEPTR surgery

const SEED = 391;

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

export default function SRTD9Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd9/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd9/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd9/definitions`).then(r => r.json()),
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
            IFT140 Short-Rib Thoracic Dysplasia 9 (SRTD9 / ATD9)
          </h4>
          <div className="text-muted small">
            OMIM #266920 · *614620 · 16p13.3 · 1462 aa · IFT-A Complex Core Scaffold · AR · ~1/200K–500K · seed={SEED}
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe (null) alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
      </Alert>
      <Alert color={ACCENT}>
        <strong>IFT140 is an IFT-A COMPLEX subunit — NOT a dynein-2 motor subunit.</strong> This is the
        critical mechanistic distinction from SRTD3/8/11/15/17. IFT140 (1462 aa) scaffolds the
        IFT-A complex: N-terminal WD40 β-propeller binds WDR35/IFT121 (SRTD7); C-terminal TPR
        domain binds WDR19/IFT144 (SRTD5/NPHP13). Loss → IFT-A instability → retrograde IFT
        failure + IFT-B import defect → SHORT/STUBBY CILIA → Hedgehog signal failure → narrow thorax.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert — KEY DISTINCTION:</strong> IFT140 (SRTD9) = <strong>SHORT/STUBBY CILIA</strong>{' '}
        with IFT-particle accumulation at the cilia base/transition zone. This is DIFFERENT from
        dynein-2 SRTDs (SRTD3/8/11/15/17) which show BULGING/CLUB CILIA TIPS. Recognising the EM
        difference guides gene panel interpretation — both groups require gene panel for definitive diagnosis.
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
                    <tr><td className="fw-bold">Gene</td><td>IFT140 (*614620) — Intraflagellar Transport Protein 140 — 16p13.3 — 1462 aa</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#266920 — SRTD9 (ATD9)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>IFT-A complex core scaffold: N-terminal WD40 β-propeller (aa 1–500) + central linker (aa 501–900) + C-terminal TPR domain (aa 901–1462)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/200,000–500,000 · ~50–100 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>~0.5–1% of all SRTD; 2nd most common IFT-A SRTD (after WDR19/SRTD5)</td></tr>
                    <tr><td className="fw-bold">IFT complex</td><td>IFT-A complex (NOT dynein-2 motor); retrograde IFT adapter + IFT-B import at cilia base</td></tr>
                    <tr><td className="fw-bold">EM finding</td><td>SHORT/STUBBY CILIA with IFT-particle accumulation at transition zone/base — distinct from dynein-2 SRTDs (club tip)</td></tr>
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
                  <strong>IFT-A complex role:</strong> IFT140 bridges WDR35/IFT121 (WD40 N-terminal face)
                  and WDR19/IFT144 (TPR C-terminal domain) within the 6-subunit IFT-A complex.
                  Without IFT140, the IFT-A complex cannot assemble properly at the cilia base,
                  retrograde IFT fails, and IFT-B cargo cannot be imported into the axoneme →
                  short cilia → Hedgehog signalling failure → SRTD9.
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
                        <tr key={i} style={r.srtd === 'SRTD9' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
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
            <Section title={`Top Pathogenic Variants (IFT140 — cohort seed=${SEED})`} color={ACCENT}>
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

      {/* ── TAB 2: IFT140 IFT-A Complex & Cilia Architecture ── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="IFT140 Protein Structure (1462 aa)" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>IFT140 is the structural BRIDGE of the IFT-A complex.</strong> Its N-terminal
                WD40 β-propeller contacts WDR35/IFT121 (SRTD7); its C-terminal TPR domain contacts
                WDR19/IFT144 (SRTD5/NPHP13). Loss of IFT140 splits the IFT-A complex into two
                disconnected halves → complete IFT-A assembly failure.
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Domain</th><th>Residues</th><th>Function</th></tr></thead>
                <tbody>
                  <tr><td>N-terminal WD40 β-propeller</td><td>aa 1–500</td><td>~10 WD40 repeats; WDR35/IFT121 (SRTD7) binding interface; most common missense variant region; loss → severe SRTD9</td></tr>
                  <tr><td>Central linker / IFT-B contact</td><td>aa 501–900</td><td>Connects WD40 to TPR; IFT-B complex interface; dynein-2 DYNC2H1 (SRTD3) heavy-chain docking; moderate variant zone</td></tr>
                  <tr><td>C-terminal TPR domain</td><td>aa 901–1462</td><td>Tetratricopeptide repeat scaffold; WDR19/IFT144 (SRTD5) binding; IFT-A complex nucleation; hypomorphic missense → mild SRTD9</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="IFT-A Complex Contacts — IFT140 Central Role" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>#</th><th>IFT-A Subunit</th><th>SRTD</th><th>IFT140 contact?</th></tr></thead>
                  <tbody>
                    <tr><td>1</td><td>WDR19/IFT144</td><td>SRTD5 / NPHP13</td><td>✓ Direct — C-terminal TPR domain (key interaction)</td></tr>
                    <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}><td>2</td><td>IFT140</td><td>SRTD9 ◀</td><td>— (self; structural bridge)</td></tr>
                    <tr><td>3</td><td>WDR35/IFT121</td><td>SRTD7</td><td>✓ Direct — N-terminal WD40 β-propeller (key interaction)</td></tr>
                    <tr><td>4</td><td>TTC21B/IFT139</td><td>SRTD4 / NPHP12</td><td>Indirect (contacts WDR19/WDR35 side)</td></tr>
                    <tr><td>5</td><td>IFT122</td><td>CED2-like</td><td>Indirect (outer IFT-A shell)</td></tr>
                    <tr><td>6</td><td>IFT43</td><td>—</td><td>Indirect (peripheral; small subunit)</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="IFT-A vs Dynein-2 SRTD — Critical EM Distinction" color={ACCENT7}>
              <Alert color={ACCENT7}>
                <strong>This is the most important diagnostic pearl for IFT140 (SRTD9) vs SRTD3/8/11/15/17:</strong>
                EM biopsy morphology differs between IFT-A and dynein-2 SRTDs.
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Feature</th><th>IFT-A SRTDs (incl. IFT140 SRTD9)</th><th>Dynein-2 SRTDs (SRTD3/8/11/15/17)</th></tr></thead>
                <tbody>
                  <tr><td>EM cilia morphology</td><td><strong>SHORT/STUBBY CILIA</strong>; IFT particles at cilia base / transition zone</td><td><strong>BULGING/CLUB CILIA TIPS</strong>; IFT-B piles up at tip</td></tr>
                  <tr><td>IFT-B accumulation</td><td>At cilia base / transition zone (cannot enter)</td><td>At ciliary TIP (cannot exit)</td></tr>
                  <tr><td>Cilia length</td><td>Shorter than normal; truncated axoneme</td><td>Near-normal length but swollen tips</td></tr>
                  <tr><td>Affected complex</td><td>IFT-A (retrograde adapter)</td><td>Dynein-2 (retrograde motor)</td></tr>
                  <tr><td>Genes</td><td>WDR19, IFT140, WDR35, TTC21B</td><td>DYNC2H1, WDR60, WDR34, DYNC2LI1, TCTEX1D2</td></tr>
                  <tr><td>Gene panel needed?</td><td>YES — EM guides but cannot distinguish WDR19 from IFT140</td><td>YES — club cilia cannot distinguish SRTD3 from SRTD8/11/15/17</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="IFT-A Failure Cascade (IFT140 Loss)" color={ACCENT2}>
              {[
                ['IFT140 absent', 'N-terminal WD40 cannot contact WDR35/IFT121; C-terminal TPR cannot contact WDR19/IFT144 → IFT-A complex splits into disconnected halves'],
                ['IFT-A complex assembly fails', 'Incomplete IFT-A complex cannot bind dynein-2 (DYNC2H1) at the cilia base for retrograde IFT; also cannot scaffold IFT-B import at the transition zone'],
                ['IFT-B import defect at cilia base', 'IFT-B cargo cannot enter the axoneme properly at the transition zone → accumulates at cilia base → SHORT/STUBBY CILIA (not full-length axoneme)'],
                ['Retrograde IFT impaired', 'IFT trains at the ciliary tip cannot recruit dynein-2 via IFT-A → retrograde run fails; further cilia shortening'],
                ['Hedgehog (Ihh/Shh) failure', 'Gli2 activation and Gli3R processing at short cilia tip impaired → Hedgehog target genes dysregulated in chondrocytes'],
                ['Chondrocyte failure', 'Short ribs, NARROW THORAX (primary, pathognomonic), shortened tubular bones (short limbs)'],
                ['Renal tubular (secondary)', 'TIN + corticomedullary cysts → ESRD in ~20–30% of survivors'],
                ['Retinal photoreceptors (secondary)', 'Connecting cilium IFT failure → rod-cone dystrophy (~15–20%)'],
                ['Biliary cholangiocytes (secondary)', 'CHF — ductal plate malformation (~8–12%)'],
              ].map(([step, desc], i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-2" style={{ background: ACCENT }}>Step {i + 1}</span>
                  <strong>{step}:</strong> {desc}
                </div>
              ))}
            </Section>

            <Section title="Why No Situs Inversus / No Joubert MTS" color={ACCENT4}>
              <p className="small mb-1">
                <strong>No situs inversus:</strong> IFT140 functions in PRIMARY non-motile cilia (9+0 axoneme).
                Situs inversus requires MOTILE nodal cilia (9+2 with inner dynein arms). IFT140 is
                not expressed in nodal motile cilia — purely primary ciliopathy.
              </p>
              <p className="small mb-0">
                <strong>No Joubert MTS:</strong> The molar tooth sign requires decussation defects of brainstem/
                cerebellar vermis (JBTS ciliopathies affecting transition zone). IFT140/SRTD9 affects
                skeletal chondrocytes and secondary organs — NOT the CNS midline structures.
              </p>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene Card — IFT140" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD9" color={ACCENT2}>
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

            <Section title="Key Pathogenic Variants (IFT140)" color={ACCENT}>
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
                  <thead><tr><th>Disease</th><th>Key Differentiator from SRTD9</th></tr></thead>
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
