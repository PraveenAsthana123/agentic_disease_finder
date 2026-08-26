'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'TTC21B IFT-A TPR Adaptor & Three Phenotypes', 'Definitions'];

// SRTD4 colour scheme — TTC21B / IFT-A TPR adaptor / three phenotypes / Jeune ATD4
const ACCENT  = '#1b5e20';   // deep green — IFT-A complex; TPR adaptor; IFT139
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#0d47a1';   // deep blue — renal TIN; ESRD; NPHP12; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#37474f';   // dark slate — TPR protein architecture; molecular
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM short stubby cilia; diagnostic
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; VEPTR surgery
const ACCENT9 = '#6a1b9a';   // violet — JBTS12 / Joubert phenotype (unique TTC21B feature)

const SEED = 397;

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

export default function SRTD4Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd4/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd4/definitions`).then(r => r.json()),
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
            TTC21B Short-Rib Thoracic Dysplasia 4 (SRTD4 / ATD4)
          </h4>
          <div className="text-muted small">
            OMIM #613819 · *612014 · 2q24.3 · 1315 aa · IFT-A TPR Adaptor (IFT139) · AR · ~1/300K–1M · seed={SEED}
            &nbsp;·&nbsp;<span style={{ color: ACCENT9 }}>Also: NPHP12 (#613820) · JBTS12</span>
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe (null) alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
      </Alert>
      <Alert color={ACCENT}>
        <strong>TTC21B (IFT139) is an IFT-A COMPLEX adaptor (TPR protein)</strong> — bridges WDR19/IFT144 (SRTD5) C-tail
        to IFT122 within the IFT-A core. Loss → IFT-A assembly failure → SHORT STUBBY CILIA (not club/bulging;
        that is dynein-2). Second most common IFT-A SRTD after WDR19 (~50–100 families, 2026).
      </Alert>
      <Alert color={ACCENT9}>
        <strong>UNIQUE — THREE phenotypes from one gene:</strong> SRTD4 (severe biallelic LOF, narrow thorax) ·
        NPHP12 (hypomorphic C-terminal alleles, renal-only, no thorax) · JBTS12 (Joubert MTS, cerebellar
        vermis hypoplasia, hypomorphic alleles). TTC21B bridges the SRTD and Joubert ciliopathy spectra.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert:</strong> SHORT STUBBY cilia — IFT-A class signature. Clinically identical
        to SRTD5 (WDR19), SRTD9 (IFT140), SRTD7 (WDR35) within IFT-A group. Gene panel is mandatory;
        ectodermal features absent (ectodermal is UNIQUE to WDR19/SRTD5 among IFT-A SRTDs).
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
            <KPI label="Cohort (seed=397)" value={N}             color={ACCENT}  />
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
                    <tr><td className="fw-bold">Gene</td><td>TTC21B (*612014) — Tetratricopeptide Repeat Domain 21B — 2q24.3 — 1315 aa (IFT139)</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#613819 — SRTD4 (ATD4) · #613820 — NPHP12 · JBTS12</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>IFT-A TPR adaptor: N-terminal TPR I (IFT122 contact) + central TPR II (WDR19 C-tail binding; pathogenic hotspot) + C-terminal TPR III (allele severity) + unstructured tail</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/300,000–1,000,000 · ~50–100 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>~1–2% of SRTD; second most common IFT-A SRTD after WDR19/SRTD5</td></tr>
                    <tr><td className="fw-bold">IFT-A position</td><td>Bridges WDR19/IFT144 (SRTD5) C-tail to IFT122; central adaptor in IFT-A complex core</td></tr>
                    <tr><td className="fw-bold">Three phenotypes</td><td>SRTD4 (thoracic) · NPHP12 (renal-only, hypomorphic) · JBTS12 (Joubert MTS, hypomorphic)</td></tr>
                    <tr><td className="fw-bold">No ectodermal</td><td>✓ Ectodermal features (CED1) unique to WDR19/SRTD5; absent in TTC21B/SRTD4</td></tr>
                    <tr><td className="fw-bold">No situs inversus</td><td>✓ Primary non-motile cilia (9+0), not nodal motile cilia</td></tr>
                    <tr><td className="fw-bold">Joubert MTS</td><td>Present in JBTS12 (hypomorphic alleles only); absent in full SRTD4 null alleles</td></tr>
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
                <Alert color={ACCENT9}>
                  <strong>TTC21B phenotype spectrum:</strong> Allele severity determines phenotype.
                  Biallelic null → SRPS/SRTD4 (perinatal lethal / severe narrow thorax). Compound het
                  missense+truncating → moderate SRTD4. Hypomorphic C-terminal missense alone →
                  NPHP12 (renal-only, no thorax — same paradigm as WDR19/NPHP13). Specific
                  hypomorphic alleles → JBTS12 (Joubert MTS, cerebellar vermis). Gene panel
                  mandatory — full allele characterisation drives phenotype prediction.
                </Alert>
              </Section>

              <Section title="Key Clinical Distinction" color={ACCENT2}>
                <p className="small">{overview.key_distinction}</p>
              </Section>

              <Section title="IFT-A Complex — SRTD Subunit Table" color={ACCENT6}>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered small">
                    <thead><tr>
                      <th>Subunit</th><th>Role</th><th>SRTD</th><th>Chr</th><th>Freq</th>
                    </tr></thead>
                    <tbody>
                      {overview.ift_a_subunit_table.map((r, i) => (
                        <tr key={i} style={r.srtd === 'SRTD4' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
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
            <Section title="Top Pathogenic Variants (TTC21B — cohort seed=397)" color={ACCENT}>
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

      {/* ── TAB 2: TTC21B IFT-A TPR Adaptor & Three Phenotypes ── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="TTC21B Protein Structure (1315 aa — IFT-A TPR Adaptor)" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>TTC21B is an IFT-A COMPLEX adaptor (not dynein-2)</strong> — a TPR (tetratricopeptide
                repeat) domain protein that bridges WDR19/IFT144 (SRTD5) C-tail to IFT122 (CED2) within
                the IFT-A complex. SRTD4 EM shows SHORT STUBBY cilia — the IFT-A signature, distinct
                from the club/bulging tips of dynein-2 SRTDs (SRTD3/8/11/15/17).
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Domain</th><th>Residues</th><th>Function</th></tr></thead>
                <tbody>
                  <tr><td>N-terminal TPR cluster I</td><td>aa 1–300</td><td>IFT122 (CED2) contact surface; scaffold docking; N-terminal TPR missense → severe null-like phenotype</td></tr>
                  <tr><td>Central TPR cluster II</td><td>aa 301–700</td><td>WDR19/IFT144 C-tail binding surface; IFT-A complex core; pathogenic hotspot (aa 450–650); most SRTD4 missense here</td></tr>
                  <tr><td>C-terminal TPR cluster III</td><td>aa 701–1100</td><td>WDR19 C-tail junction; ALLELE SEVERITY DETERMINANT — hypomorphic missense here → NPHP12 (renal-only) or JBTS12 (Joubert)</td></tr>
                  <tr><td>C-terminal unstructured tail</td><td>aa 1101–1315</td><td>IFT train regulation; dispensable for SRTD4; hypomorphic variants → mildest phenotype only</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="IFT-A Complex Assembly — TTC21B Position" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>#</th><th>Subunit</th><th>SRTD</th><th>TTC21B contact?</th></tr></thead>
                  <tbody>
                    <tr><td>1</td><td>IFT140</td><td>SRTD9</td><td>Indirect (IFT-A scaffold; contacts WDR35 + WDR19)</td></tr>
                    <tr><td>2</td><td>WDR19 / IFT144</td><td>SRTD5</td><td>✓ Direct (WDR19 C-tail contacts TTC21B central TPR II)</td></tr>
                    <tr><td>3</td><td>WDR35 / IFT121</td><td>SRTD7</td><td>Indirect (contacts IFT140; TTC21B opposite side)</td></tr>
                    <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}><td>4</td><td>TTC21B / IFT139</td><td>SRTD4 ◀</td><td>— (self; adaptor bridging WDR19 C-tail ↔ IFT122)</td></tr>
                    <tr><td>5</td><td>IFT122</td><td>CED2</td><td>✓ Direct (TTC21B N-terminal TPR I contacts IFT122)</td></tr>
                    <tr><td>6</td><td>IFT43</td><td>CED3</td><td>Indirect (contacts WDR35; stability role)</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="TTC21B Three Phenotype Spectrum — Allele Severity" color={ACCENT9}>
              <Alert color={ACCENT9}>
                <strong>TTC21B is unique</strong> among SRTD genes in causing three ciliopathy
                phenotypes from a single gene, depending on which domains are disrupted.
              </Alert>
              {[
                ['SRTD4 / ATD4 (#613819)', 'Severe biallelic LOF — null × null, or truncating × severe missense (central TPR I–II). Narrow thorax is PRIMARY. Full IFT-A failure. SRPS spectrum with biallelic null alleles.', ACCENT2],
                ['NPHP12 (#613820)', 'Hypomorphic C-terminal TTC21B alleles (aa 701–1100) — residual IFT-A function preserved. Renal-only: TIN + corticomedullary cysts → ESRD. NO narrow thorax. Same paradigm as WDR19/NPHP13.', ACCENT3],
                ['JBTS12 (Joubert syndrome type 12)', 'Specific hypomorphic TTC21B alleles → Molar Tooth Sign (MTS) on MRI, cerebellar vermis hypoplasia, oculomotor apraxia, breathing irregularity. ABSENT in full SRTD4 null alleles. TTC21B expressed in cerebellar granule cells — explains CNS involvement.', ACCENT9],
              ].map(([title, desc, col], i) => (
                <div key={i} className="mb-3 p-2 rounded" style={{ border: `2px solid ${col}`, background: col + '10' }}>
                  <div className="fw-bold small mb-1" style={{ color: col }}>{title}</div>
                  <p className="small mb-0">{desc}</p>
                </div>
              ))}
            </Section>

            <Section title="IFT-A vs Dynein-2 SRTD — EM Distinction" color={ACCENT7}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Feature</th><th>IFT-A SRTDs (SRTD4/5/7/9)</th><th>Dynein-2 SRTDs (SRTD3/8/11/15/17)</th></tr></thead>
                <tbody>
                  <tr><td>Cilia EM</td><td><strong>SHORT STUBBY</strong> — no bulging tip</td><td><strong>CLUB / BULGING TIP</strong> — IFT-B piles up at tip</td></tr>
                  <tr><td>IFT-B location</td><td>Fails to enter at base (transition zone block)</td><td>Accumulates at ciliary tip (retrograde motor failure)</td></tr>
                  <tr><td>Primary defect</td><td>IFT-A complex failure → IFT-B import blocked</td><td>Dynein-2 motor failure → retrograde IFT blocked</td></tr>
                  <tr><td>Ectodermal</td><td>Only WDR19/SRTD5 (CED1) — absent in SRTD4/7/9</td><td>Absent in all dynein-2 SRTDs</td></tr>
                  <tr><td>Joubert MTS</td><td>Only TTC21B/SRTD4 (JBTS12 alleles)</td><td>Absent in all dynein-2 SRTDs</td></tr>
                  <tr><td>Renal-only alleles</td><td>TTC21B (NPHP12) + WDR19 (NPHP13)</td><td>None</td></tr>
                  <tr><td>Can distinguish by EM?</td><td>Within IFT-A group: NO — gene panel mandatory</td><td>Within dynein-2 group: NO — gene panel mandatory</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="Why Short Stubby Cilia — IFT-A Failure Cascade" color={ACCENT2}>
              {[
                ['TTC21B absent', 'TPR adaptor cannot bridge WDR19 C-tail to IFT122 → IFT-A complex assembly disrupted at the WDR19–TTC21B–IFT122 junction'],
                ['IFT-A complex unstable', 'IFT-A cannot form competent retrograde IFT trains → retrograde transport from ciliary tip impaired'],
                ['IFT-B import blocked', 'IFT-B complex (anterograde cargo) CANNOT be imported at the cilia base / transition zone (requires IFT-A) → cilia cannot elongate normally'],
                ['Short stubby cilia', 'Shortened, structurally deficient cilia (not club tip) — IFT-B components absent within cilia'],
                ['Hedgehog (Ihh/Shh) failure', 'PTCH1/SMO/GLI cannot traffic within stunted cilia → GLI3R processing fails → Hh target genes off in chondrocytes'],
                ['Chondrocyte failure', 'Short ribs, NARROW THORAX (primary pathognomonic), short long bones (secondary limb shortening)'],
                ['Secondary organs', 'Renal TIN (tubular cilia IFT-A failure) → ESRD in ~20–30%; rod-cone dystrophy (~10–15%); CHF (~8%)'],
              ].map(([step, desc], i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-2" style={{ background: ACCENT }}>Step {i + 1}</span>
                  <strong>{step}:</strong> {desc}
                </div>
              ))}
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene Card — TTC21B (IFT139)" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD4 / NPHP12 / JBTS12" color={ACCENT2}>
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

            <Section title="Key Pathogenic Variants (TTC21B)" color={ACCENT}>
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
                  <thead><tr><th>Disease</th><th>Key Differentiator from SRTD4</th></tr></thead>
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
