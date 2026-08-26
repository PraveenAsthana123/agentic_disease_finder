'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'WDR19 IFT-A Complex, Ectodermal & Allelic Spectrum', 'Definitions'];

// SRTD5 colour scheme — WDR19/IFT144 / IFT-A complex / most common IFT-A SRTD / CED-ectodermal overlap
const ACCENT  = '#00695c';   // dark teal-green — IFT-A complex; WDR19/IFT144 largest subunit
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; primary severity
const ACCENT3 = '#1565c0';   // deep blue — renal TIN / NPHP13 / ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary ciliopathy
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#1b5e20';   // dark green — IFT-A molecular biology; WD40 scaffold; subunit table
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM short cilia vs club; diagnostic pearl
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial/preaxial; VEPTR surgery
const ACCENT9 = '#4e342e';   // brown — ectodermal CED-like features; hair/teeth; unique to WDR19

const SEED = 393;

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

export default function SRTD5Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd5/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd5/definitions`).then(r => r.json()),
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
            WDR19 Short-Rib Thoracic Dysplasia 5 (SRTD5 / ATD5)
          </h4>
          <div className="text-muted small">
            OMIM #614376 · *608151 · 4p14 · 1342 aa · IFT-A Complex LARGEST Subunit (IFT144) · AR · ~1/150K–300K · seed={SEED}
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe (null) alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
      </Alert>
      <Alert color={ACCENT}>
        <strong>WDR19/IFT144 is the LARGEST IFT-A COMPLEX subunit (1342 aa) — the central scaffold of the IFT-A complex.</strong>{' '}
        IFT140 (SRTD9) C-terminal TPR domain docks directly onto WDR19. WDR19 is the MOST COMMON
        IFT-A SRTD (~1% of all SRTD; ~2–3× more frequent than IFT140/SRTD9). Loss → IFT-A assembly
        failure → SHORT/STUBBY CILIA → Hedgehog signal failure → SRTD5. Also causes NPHP13 (#614377).
      </Alert>
      <Alert color={ACCENT9}>
        <strong>ECTODERMAL FEATURES (CED-like) — UNIQUE TO WDR19 among SRTD genes:</strong>{' '}
        Sparse hair (hypotrichosis) + hypodontia/small peg-shaped teeth in ~20–30% of SRTD5 patients.
        This CED1/Sensenbrenner overlap is specific to WDR19 (and IFT122). If SRTD phenotype + sparse
        hair/hypodontia: WDR19 is the leading IFT-A gene to investigate.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert:</strong> WDR19 (SRTD5) = <strong>SHORT/STUBBY CILIA</strong> (IFT-A class,
        same as IFT140/SRTD9, WDR35/SRTD7). DIFFERENT from dynein-2 SRTDs (SRTD3/8/11/15/17) which
        show BULGING/CLUB CILIA TIPS. Gene panel mandatory to distinguish within IFT-A and dynein-2 groups.
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
            <KPI label={`Cohort (seed=${SEED})`}   value={N}             color={ACCENT}  />
            <KPI label="Thorax Severe"      value={`${k.thorax_severe_n} (${k.thorax_severe_pct}%)`}  color={ACCENT2} />
            <KPI label="Polydactyly"        value={`${k.polydactyly_n} (${k.polydactyly_pct}%)`}     color={ACCENT8} />
            <KPI label="Renal Involved"     value={`${k.renal_any_n} (${k.renal_any_pct}%)`}         color={ACCENT3} />
            <KPI label="Retinal Dystrophy"  value={`${k.retinal_any_n} (${k.retinal_any_pct}%)`}     color={ACCENT4} />
            <KPI label="CHF (hepatic)"      value={`${k.hepatic_chf_n} (${k.hepatic_chf_pct}%)`}     color={ACCENT5} />
            <KPI label="Ectodermal (CED)"   value={`${k.ectodermal_any_n} (${k.ectodermal_any_pct}%)`} color={ACCENT9} />
            <KPI label="VEPTR Surgery"      value={`${k.veptr_any_n} (${k.veptr_any_pct}%)`}         color={ACCENT8} />
            <KPI label="Renal Transplant ✓" value={k.transplant_done_n}                               color={ACCENT3} />
            <KPI label="Prior Misdiagnosis" value={`${k.misdiagnosis_n} (${k.misdiagnosis_pct}%)`}   color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Disease Summary" color={ACCENT}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>WDR19 (*608151) — also IFT144, KIAA1638 — 4p14 — 1342 aa — IFT-A LARGEST subunit</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#614376 — SRTD5 (ATD5); allelic: NPHP13 (#614377)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>Multi-blade WD40 β-propeller (aa 1–900; IFT140 + IFT122 scaffold) + C-terminal α-helical tail (aa 901–1342; TTC21B contact; CED alleles)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/150,000–300,000 · ~100–200 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>MOST COMMON IFT-A SRTD (~1% of all SRTD); ~2–3× more common than IFT140 (SRTD9)</td></tr>
                    <tr><td className="fw-bold">IFT complex</td><td>IFT-A complex LARGEST subunit (NOT dynein-2 motor); central assembly scaffold</td></tr>
                    <tr><td className="fw-bold">EM finding</td><td>SHORT/STUBBY CILIA with IFT-particle accumulation at transition zone/base — same as all IFT-A SRTDs; distinct from dynein-2 SRTDs (club tip)</td></tr>
                    <tr><td className="fw-bold">Ectodermal (CED)</td><td>~20–30% — sparse hair, hypodontia, small teeth — UNIQUE to WDR19 (and IFT122) among SRTD genes</td></tr>
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
                  <strong>WDR19 as IFT-A central scaffold:</strong> WDR19/IFT144 (1342 aa) is the
                  largest IFT-A subunit. Its central WD40 platform directly docks IFT140 (SRTD9)
                  via its C-terminal TPR domain. Loss of WDR19 → IFT140 cannot dock → IFT-A
                  cannot form → short cilia → Hedgehog failure → SRTD5.
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
                        <tr key={i} style={r.srtd.startsWith('SRTD5') ? { background: ACCENT + '22', fontWeight: 'bold' } : {}}>
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
            <Section title="Ectodermal (CED-like) Features — UNIQUE to WDR19" color={ACCENT9}>
              {breakdown.ectodermal_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT9} />
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
            <Section title={`Top Pathogenic Variants (WDR19 — cohort seed=${SEED})`} color={ACCENT}>
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

      {/* ── TAB 2: WDR19 IFT-A Complex, Ectodermal & Allelic Spectrum ── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="WDR19/IFT144 Protein Structure (1342 aa)" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>WDR19 is the CENTRAL SCAFFOLD of the IFT-A complex.</strong> Its WD40
                central platform (aa 501–900) directly docks IFT140 (SRTD9) via IFT140's C-terminal
                TPR domain — this is the critical interaction linking the two most common IFT-A SRTDs.
                Without WDR19, the IFT-A complex collapses from the core outward.
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Domain</th><th>Residues</th><th>Function</th></tr></thead>
                <tbody>
                  <tr><td>N-terminal WD40 β-propeller cluster</td><td>aa 1–500</td><td>~12 WD40 repeats; IFT43 + IFT122 scaffolding; most common SRTD5 missense region; truncation here → SRPS</td></tr>
                  <tr><td>Central WD40 platform (IFT140-contact)</td><td>aa 501–900</td><td>IFT140 C-TPR domain binding surface; IFT-B complex contacts; DYNC2H1 heavy-chain docking; NPHP13 hypomorphic alleles cluster here</td></tr>
                  <tr><td>C-terminal α-helical regulatory tail</td><td>aa 901–1342</td><td>IFT-A complex nucleation; TTC21B/IFT139 (SRTD4) contact; CED-overlap ectodermal variants here; NPHP13-dominant mild alleles</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="WDR19 Allelic Disease Spectrum" color={ACCENT3}>
              <Alert color={ACCENT3}>
                <strong>WDR19 is unique among SRTD genes — same gene, three clinical phenotypes
                depending on allele severity:</strong>
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Phenotype</th><th>OMIM</th><th>Allele type</th><th>Dominant feature</th></tr></thead>
                <tbody>
                  <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}>
                    <td>SRTD5 (ATD5) ◀</td><td>#614376</td><td>Biallelic LOF (comp het or homozygous)</td><td>NARROW THORAX + polydactyly + secondary renal/retinal</td>
                  </tr>
                  <tr>
                    <td>NPHP13</td><td>#614377</td><td>Hypomorphic C-terminal (esp. aa 900–1342)</td><td>Nephronophthisis renal-dominant; NO/mild thorax; ESRD in 2nd–3rd decade</td>
                  </tr>
                  <tr>
                    <td>CED1-overlap</td><td>Partial</td><td>Null/severe + C-terminal combination</td><td>Ectodermal features (hair + teeth) ± thorax; overlap with IFT122-CED2</td>
                  </tr>
                </tbody>
              </table>
            </Section>

            <Section title="IFT-A Complex Contacts — WDR19 Central Role" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>#</th><th>IFT-A Subunit</th><th>SRTD</th><th>WDR19 contact?</th></tr></thead>
                  <tbody>
                    <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}><td>1</td><td>WDR19/IFT144</td><td>SRTD5 ◀</td><td>— (self; LARGEST subunit; central scaffold)</td></tr>
                    <tr><td>2</td><td>IFT140</td><td>SRTD9</td><td>✓ Direct — Central WD40 platform (IFT140 C-TPR docks here; CRITICAL interaction)</td></tr>
                    <tr><td>3</td><td>WDR35/IFT121</td><td>SRTD7</td><td>Indirect (contacts IFT140 WD40 side; WDR19 does not contact WDR35 directly)</td></tr>
                    <tr><td>4</td><td>TTC21B/IFT139</td><td>SRTD4 / NPHP12</td><td>✓ Direct — C-terminal α-helical tail (aa 901–1342)</td></tr>
                    <tr><td>5</td><td>IFT122</td><td>CED2-like</td><td>✓ Direct — N-terminal WD40 platform (aa 1–500)</td></tr>
                    <tr><td>6</td><td>IFT43</td><td>—</td><td>Indirect (N-terminal WD40 periphery)</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="IFT-A vs Dynein-2 SRTD — Critical EM Distinction" color={ACCENT7}>
              <Alert color={ACCENT7}>
                <strong>WDR19 (SRTD5) EM finding: SHORT/STUBBY CILIA</strong> — same as all IFT-A SRTDs.
                DISTINCT from dynein-2 SRTDs which show BULGING/CLUB CILIA TIPS.
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Feature</th><th>IFT-A SRTDs (WDR19 SRTD5 + IFT140 SRTD9 + WDR35 SRTD7)</th><th>Dynein-2 SRTDs (SRTD3/8/11/15/17)</th></tr></thead>
                <tbody>
                  <tr><td>EM cilia</td><td><strong>SHORT/STUBBY CILIA</strong>; IFT particles at cilia base/TZ</td><td><strong>BULGING/CLUB TIPS</strong>; IFT-B piles at tip</td></tr>
                  <tr><td>IFT-B site</td><td>Cilia base / transition zone (cannot enter)</td><td>Ciliary TIP (cannot exit)</td></tr>
                  <tr><td>Ectodermal</td><td><strong>WDR19 ONLY: sparse hair + hypodontia (~20–30%)</strong></td><td>Absent in all dynein-2 SRTDs</td></tr>
                  <tr><td>NPHP alleles</td><td><strong>WDR19 ONLY: NPHP13 (renal-dominant)</strong></td><td>None (dynein-2 SRTDs no NPHP alleles)</td></tr>
                  <tr><td>Gene panel</td><td>Mandatory — EM cannot distinguish WDR19 from IFT140</td><td>Mandatory — club cilia cannot distinguish SRTD3 from SRTD8/11/15/17</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="WDR19 IFT-A Failure Cascade" color={ACCENT2}>
              {[
                ['WDR19 absent', 'Central WD40 platform cannot dock IFT140 (C-TPR) or scaffold IFT122 (N-terminal) → IFT-A complex collapses from the core outward'],
                ['IFT-A complex fails', 'Incomplete IFT-A cannot bind dynein-2 (DYNC2H1) for retrograde IFT; cannot scaffold IFT-B import at cilia base transition zone'],
                ['IFT-B import defect', 'IFT-B cargo cannot enter axoneme at TZ → accumulates at cilia base → SHORT/STUBBY CILIA (truncated axoneme, not full-length)'],
                ['Retrograde IFT impaired', 'IFT trains at tip cannot recruit dynein-2 via IFT-A → retrograde run fails; further axoneme shortening'],
                ['Hedgehog (Ihh/Shh) failure', 'Gli2 activation + Gli3R processing impaired at short cilia → Hedgehog target genes dysregulated in chondrocytes'],
                ['Chondrocyte failure', 'SHORT RIBS, NARROW THORAX (primary/pathognomonic), shortened tubular bones (short limbs)'],
                ['Renal tubular (secondary)', 'TIN + corticomedullary cysts → ESRD ~25–35% survivors; NPHP13 alleles: renal may dominate without thorax'],
                ['Retinal photoreceptors (secondary)', 'Connecting cilium IFT failure → rod-cone dystrophy (~15–25%)'],
                ['Ectodermal (unique to WDR19)', 'Hair follicle + dental lamina primary cilia depend on WDR19/IFT-A → sparse hair (hypotrichosis), hypodontia (~20–30%)'],
              ].map(([step, desc], i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-2" style={{ background: ACCENT }}>Step {i + 1}</span>
                  <strong>{step}:</strong> {desc}
                </div>
              ))}
            </Section>

            <Section title="Why WDR19 is the Most Clinically Complex IFT-A SRTD" color={ACCENT4}>
              <p className="small mb-1">
                <strong>Three-disease gene:</strong> WDR19 causes SRTD5 (thoracic-dominant),
                NPHP13 (renal-dominant), AND CED1-overlap (ectodermal) — the allele class determines
                the clinical presentation. This genotype–phenotype correlation is unique among SRTD genes.
              </p>
              <p className="small mb-1">
                <strong>NPHP13 alleles:</strong> Hypomorphic C-terminal WDR19 variants (aa 901–1342)
                may present as isolated nephronophthisis in adolescence/young adulthood — WDR19
                should be included in NPHP gene panels even without skeletal features.
              </p>
              <p className="small mb-0">
                <strong>Ectodermal features as clinical guide:</strong> Sparse hair + hypodontia in
                an SRTD patient = WDR19 first hypothesis. Referral to dermatology + paediatric
                dentistry early reduces diagnostic delay. CED1 (Sensenbrenner) in differential.
              </p>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene Card — WDR19/IFT144" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD5" color={ACCENT2}>
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

            <Section title="Key Pathogenic Variants (WDR19)" color={ACCENT}>
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
                  <thead><tr><th>Disease</th><th>Key Differentiator from SRTD5</th></tr></thead>
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
