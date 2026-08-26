'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'WDR34 Dynein-2 & Retrograde IFT', 'Definitions'];

// SRTD11 colour scheme — WDR34 / dynein-2 opposite to WDR60 / narrow thorax / Jeune ATD11
const ACCENT  = '#1a237e';   // deep indigo — WDR34 6-blade β-propeller; dynein-2 opposite chain
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#2e7d32';   // deep green — renal TIN; secondary ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; ductal plate malformation
const ACCENT6 = '#37474f';   // dark slate — 6-blade propeller structure; molecular architecture
const ACCENT7 = '#f57f17';   // amber — misdiagnosis alerts; EM club cilia; diagnostic
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; VEPTR surgery

const SEED = 385;

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

export default function SRTD11Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd11/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd11/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd11/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOver(o); setBreak(b); setDefs(d); })
      .catch(e => setErr(e.message));
  }, []);

  if (err)      return <div className="alert alert-danger m-4">API error: {err}</div>;
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
            WDR34 Short-Rib Thoracic Dysplasia 11 (SRTD11 / ATD11)
          </h4>
          <div className="text-muted small">
            OMIM #615633 · *604126 · 9q34.11 · 536 aa · Dynein-2 WD40 6-blade β-propeller (opposite to WDR60/SRTD8) · AR · ~1/500k–1M · seed={SEED}
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
      </Alert>
      <Alert color={ACCENT}>
        <strong>WDR34 is on the OPPOSITE side of the dynein-2 tail from WDR60 (SRTD8).</strong>{' '}
        WDR34 forms a 6-blade WD40 β-propeller; directly contacts DYNC2H1 (SRTD3) via N-terminal stem
        and DYNC2LI1 (SRTD15) via C-terminus. Loss collapses the entire dynein-2 tail →
        retrograde IFT failure → Hedgehog signalling failure → narrow thorax.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert:</strong> Same bulging/club ciliary tips as SRTD3 and SRTD8 (IFT-B pile-up at tip) —
        distinct from NPHP (TZ defect) and PCD (dynein-arm defect). Include WDR34 on all skeletal ciliopathy panels.
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
            <KPI label="Cohort (seed=385)" value={N}             color={ACCENT}  />
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
                    <tr><td className="fw-bold">Gene</td><td>WDR34 (*604126) — WD Repeat Domain 34 — 9q34.11 — 536 aa</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#615633 — SRTD11 (ATD11)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>6-blade WD40 β-propeller + N-stem (DYNC2H1 dock) + C-helix (DYNC2LI1 contact)</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/500,000–1,000,000 · ~30–50 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>~3–5% of all molecularly confirmed SRTD</td></tr>
                    <tr><td className="fw-bold">Dynein-2 position</td><td>OPPOSITE side of tail from WDR60 (SRTD8); contacts DYNC2LI1 (SRTD15) directly</td></tr>
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
                  <strong>Severity vs SRTD8/WDR60:</strong> WDR34 loss more completely collapses the dynein-2 tail
                  than WDR60 loss (WDR34 also destabilises WDR60 in trans). Biallelic WDR34 null → SRPS-spectrum
                  perinatal lethality more reliably than biallelic WDR60 null. Severity rank: DYNC2H1 ≥ WDR34 &gt; WDR60.
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
                        <tr key={i} style={r.srtd === 'SRTD11' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
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
            <Section title="Top Pathogenic Variants (WDR34 — cohort seed=385)" color={ACCENT}>
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

      {/* ── TAB 2: WDR34 Dynein-2 & Retrograde IFT ── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="WDR34 Protein Structure (536 aa)" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>WDR34 (536 aa) is substantially shorter than WDR60 (1,173 aa)</strong> — it forms a
                6-blade WD40 β-propeller (vs WDR60's 7-blade). Despite being smaller, WDR34 occupies a
                critical structural position on the OPPOSITE side of the dynein-2 tail from WDR60.
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Domain</th><th>Residues</th><th>Function</th></tr></thead>
                <tbody>
                  <tr><td>N-terminal stem / coiled-coil</td><td>aa 1–80</td><td>Docks onto DYNC2H1 heavy chain tail; required for dynein-2 complex assembly</td></tr>
                  <tr><td>WD repeat 1–6 / 6-blade β-propeller</td><td>aa ~80–480</td><td>Core scaffold; trans contact with WDR60 across dynein-2 tail; most pathogenic missense in WD3–WD5</td></tr>
                  <tr><td>DYNC2LI1 interface helix</td><td>aa ~480–536</td><td>C-terminal helix directly contacts DYNC2LI1 (SRTD15); truncating variants here → severe SRTD11</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="Dynein-2 Complex Assembly — WDR34 Position" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>#</th><th>Subunit</th><th>SRTD</th><th>WDR34 contact?</th></tr></thead>
                  <tbody>
                    <tr><td>1</td><td>DYNC2H1 (×2)</td><td>SRTD3</td><td>✓ Via N-terminal stem (direct)</td></tr>
                    <tr><td>2</td><td>DYNC2LI1</td><td>SRTD15</td><td>✓ Via C-terminal helix (direct)</td></tr>
                    <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}><td>3</td><td>WDR34</td><td>SRTD11 ◀</td><td>— (self; sits opposite WDR60)</td></tr>
                    <tr><td>4</td><td>WDR60</td><td>SRTD8</td><td>✓ Trans contact across tail (indirect; stabilised by WDR34)</td></tr>
                    <tr><td>5</td><td>TCTEX1D2</td><td>SRTD17</td><td>Indirect (contacts WDR60, not WDR34)</td></tr>
                    <tr><td>6</td><td>DYNLRB1/2</td><td>—</td><td>Roadblock-type; no direct WDR34 contact</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="Retrograde IFT Failure Cascade (WDR34 Loss)" color={ACCENT2}>
              {[
                ['WDR34 absent', 'N-terminal stem cannot dock onto DYNC2H1 → dynein-2 tail module begins to collapse'],
                ['WDR60 secondarily destabilised', 'WDR34 trans contact with WDR60 lost → WDR60 detaches → more complete tail collapse than WDR60 alone'],
                ['Dynein-2 inactivated', 'Retrograde IFT from ciliary tip to cell body stalls'],
                ['IFT-B pile-up at ciliary tip', '"Bulging/club" cilia tip on EM — same as SRTD3 and SRTD8'],
                ['Hedgehog (Ihh/Shh) failure', 'PTCH1/SMO/GLI cannot traffic from tip → GLI3R processing fails → Hh target genes dysregulated'],
                ['Chondrocyte failure', 'Short ribs, NARROW THORAX (primary), short long bones (rhizomelia ± mesomelia)'],
                ['Renal tubular (secondary)', 'TIN + corticomedullary cysts → ESRD in survivors (~30–40%)'],
                ['Retinal photoreceptors (secondary)', 'Connecting cilium IFT failure → rod-cone dystrophy (~15–20%)'],
                ['Biliary cholangiocytes (minority)', 'CHF — ductal plate malformation (~10%)'],
              ].map(([step, desc], i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-2" style={{ background: ACCENT }}>Step {i + 1}</span>
                  <strong>{step}:</strong> {desc}
                </div>
              ))}
            </Section>

            <Section title="WDR34 vs WDR60 — Key Differences" color={ACCENT7}>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Feature</th><th>WDR34 (SRTD11)</th><th>WDR60 (SRTD8)</th></tr></thead>
                <tbody>
                  <tr><td>Protein size</td><td>536 aa</td><td>1,173 aa</td></tr>
                  <tr><td>β-propeller blades</td><td>6-blade</td><td>7-blade</td></tr>
                  <tr><td>Chromosome</td><td>9q34.11</td><td>7q36.3</td></tr>
                  <tr><td>Dynein-2 tail side</td><td>One side (contacts DYNC2LI1)</td><td>Opposite side (contacts TCTEX1D2)</td></tr>
                  <tr><td>SRTD frequency</td><td>~3–5%</td><td>~5–10%</td></tr>
                  <tr><td>Null lethality</td><td>More reliably SRPS-spectrum</td><td>SRPS-spectrum, but less reliably</td></tr>
                  <tr><td>OMIM gene</td><td>*604126</td><td>*615462</td></tr>
                  <tr><td>OMIM disease</td><td>#615633</td><td>#615503</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="Why No Situs Inversus / No Joubert MTS" color={ACCENT4}>
              <p className="small mb-1">
                <strong>No situs inversus:</strong> WDR34 functions in PRIMARY non-motile cilia (9+0 axoneme).
                Situs inversus requires MOTILE nodal cilia (9+2). WDR34 is not expressed in nodal motile cilia.
              </p>
              <p className="small mb-0">
                <strong>No Joubert MTS:</strong> The molar tooth sign requires brainstem/cerebellar vermis
                decussation defects (seen in JBTS ciliopathies). WDR34/SRTD11 affects skeletal chondrocytes
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
            <Section title="Gene Card — WDR34" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD11" color={ACCENT2}>
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

            <Section title="Key Pathogenic Variants (WDR34)" color={ACCENT}>
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
                  <thead><tr><th>Disease</th><th>Key Differentiator from SRTD11</th></tr></thead>
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
