'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'IFT43 IFT-A Satellite & Campomelia', 'Definitions'];

// SRTD18 colour scheme — IFT43 / IFT-A satellite / campomelia / narrow thorax / CED3 dual phenotype
const ACCENT  = '#1b5e20';   // deep green — IFT43 smallest IFT-A subunit; satellite anchor
const ACCENT2 = '#b71c1c';   // deep red — narrow thorax / neonatal respiratory; severity
const ACCENT3 = '#0d47a1';   // deep blue — renal TIN/ESRD; transplant outcome
const ACCENT4 = '#4a148c';   // deep purple — retinal rod-cone dystrophy; secondary
const ACCENT5 = '#e65100';   // burnt orange — CHF / hepatic fibrosis; CED3 hepatic overlap
const ACCENT6 = '#37474f';   // dark slate — IFT-A satellite architecture; molecular structure
const ACCENT7 = '#f57f17';   // amber — campomelia; misdiagnosis alerts; distinctive feature
const ACCENT8 = '#880e4f';   // deep pink — polydactyly; postaxial; VEPTR/osteotomy surgery
const ACCENT9 = '#006064';   // dark teal — CED3/Sensenbrenner dual phenotype

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

export default function SRTD18Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOver]   = useState(null);
  const [breakdown, setBreak] = useState(null);
  const [defs, setDefs]       = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/srtd18/overview`).then(r => r.json()),
      fetch(`${API}/api/srtd18/breakdown`).then(r => r.json()),
      fetch(`${API}/api/srtd18/definitions`).then(r => r.json()),
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
            IFT43 Short-Rib Thoracic Dysplasia 18 (SRTD18 / ATD18)
          </h4>
          <div className="text-muted small">
            OMIM #617866 · *614068 · 14q24 · 378 aa · IFT-A Satellite Core (Smallest IFT-A Subunit) · AR · ~1/1M–3M · seed={SEED}
          </div>
        </div>
      </div>

      {/* Key alerts */}
      <Alert color={ACCENT2}>
        <strong>PRIMARY: NARROW THORAX</strong> — pathognomonic; neonatal respiratory failure in severe (null) alleles;
        NOT secondary (unlike renal in NPHP). VEPTR/MAGEC growing rods = first-line surgical treatment.
        Campomelia (bowing of long bones) present in ~50% — DISTINCTIVE SRTD18 feature.
      </Alert>
      <Alert color={ACCENT}>
        <strong>IFT43 is the SMALLEST IFT-A subunit (378 aa) — a satellite peripheral anchor</strong> that directly
        contacts IFT121/WDR35 (SRTD7) N-terminal WD40 blades. Loss → IFT-A satellite collapse →
        retrograde IFT failure → SHORT-STUBBY CILIA (IFT-A class on EM). Second IFT-A satellite SRTD
        gene after WDR35/IFT121 (SRTD7). W174R founder allele (Arabian Peninsula).
      </Alert>
      <Alert color={ACCENT9}>
        <strong>DUAL PHENOTYPE — allele class governs:</strong> Biallelic null → SRTD18 / SRPS; Hypomorphic →
        CED3 (Cranioectodermal Dysplasia 3 / Sensenbrenner syndrome: craniosynostosis + sparse hair +
        dental anomalies + narrow thorax + hepatic fibrosis). Same gene — different severity alleles.
      </Alert>
      <Alert color={ACCENT7}>
        <strong>EM alert:</strong> SHORT-STUBBY CILIA (IFT-A class) — identical to SRTD2/4/5/7/9.
        Cannot distinguish SRTD18 from other IFT-A SRTDs by EM alone. Gene panel is mandatory.
        Campomelia on X-ray is the best radiographic differentiator from other IFT-A types.
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
            <KPI label="Cohort (seed=391)" value={N}             color={ACCENT}  />
            <KPI label="Thorax Severe"     value={`${k.thorax_severe_n} (${k.thorax_severe_pct}%)`} color={ACCENT2} />
            <KPI label="Campomelia"        value={`${k.campomelia_n} (${k.campomelia_pct}%)`}       color={ACCENT7} />
            <KPI label="Polydactyly"       value={`${k.polydactyly_n} (${k.polydactyly_pct}%)`}   color={ACCENT8} />
            <KPI label="Renal Involved"    value={`${k.renal_any_n} (${k.renal_any_pct}%)`}        color={ACCENT3} />
            <KPI label="Retinal Dystrophy" value={`${k.retinal_any_n} (${k.retinal_any_pct}%)`}   color={ACCENT4} />
            <KPI label="CHF (hepatic)"     value={`${k.hepatic_chf_n} (${k.hepatic_chf_pct}%)`}   color={ACCENT5} />
            <KPI label="CED3 Features"     value={k.ced3_features_n}                                color={ACCENT9} />
            <KPI label="VEPTR Surgery"     value={`${k.veptr_any_n} (${k.veptr_any_pct}%)`}       color={ACCENT8} />
            <KPI label="Renal Transplant ✓" value={k.transplant_done_n}                            color={ACCENT3} />
            <KPI label="Prior Misdiagnosis" value={`${k.misdiagnosis_n} (${k.misdiagnosis_pct}%)`} color={ACCENT7} />
          </div>

          <div className="row g-3">
            <div className="col-md-6">
              <Section title="Disease Summary" color={ACCENT}>
                <table className="table table-sm table-bordered small">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>IFT43 (*614068) — Intraflagellar Transport Protein 43 — 14q24 — 378 aa</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>#617866 — SRTD18 with Polydactyly (ATD18)</td></tr>
                    <tr><td className="fw-bold">Protein</td><td>IFT-A satellite peripheral subunit: N-terminal adaptor (IFT121-binding) + central linker (W174R cluster) + C-terminal anchoring domain</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                    <tr><td className="fw-bold">Prevalence</td><td>~1/1,000,000–3,000,000 · ~10–25 families worldwide (2026)</td></tr>
                    <tr><td className="fw-bold">SRTD frequency</td><td>Very rare; second IFT-A satellite SRTD gene after SRTD7 (IFT121/WDR35)</td></tr>
                    <tr><td className="fw-bold">Dual phenotype</td><td>Biallelic null → SRTD18 / SRPS spectrum; Hypomorphic → CED3 (Sensenbrenner syndrome)</td></tr>
                    <tr><td className="fw-bold">Distinctive feature</td><td>Campomelia (bowing of long bones) in ~50% — highest among IFT-A SRTDs</td></tr>
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
                <Alert color={ACCENT7}>
                  <strong>Campomelia mechanism:</strong> IFT-A satellite failure → chondrocyte Hh signalling
                  failure → asymmetric periosteal remodelling in long bones → bowing/campomelia.
                  This is DISTINCTIVE to SRTD18 among IFT-A SRTDs — use campomelia on prenatal USS
                  or postnatal X-ray to prioritise IFT43 in differential within IFT-A panel.
                </Alert>
              </Section>

              <Section title="Key Clinical Distinction" color={ACCENT2}>
                <p className="small">{overview.key_distinction}</p>
              </Section>

              <Section title="IFT-A Complex — SRTD Subunit Table" color={ACCENT6}>
                <div className="table-responsive">
                  <table className="table table-sm table-bordered small">
                    <thead><tr>
                      <th>Subunit</th><th>Module</th><th>SRTD</th><th>Chr</th><th>Freq</th>
                    </tr></thead>
                    <tbody>
                      {overview.ift_a_satellite_table.map((r, i) => (
                        <tr key={i} style={r.srtd === 'SRTD18' ? { background: ACCENT + '18', fontWeight: 'bold' } : {}}>
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
            <Section title="Campomelia (Bowing — DISTINCTIVE SRTD18)" color={ACCENT7}>
              {breakdown.campomelia_distribution.map((r, i) => (
                <SimpleBar key={i} label={r.label} n={r.n} total={N} color={ACCENT7} />
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
            <Section title="CED3 / Sensenbrenner Features" color={ACCENT9}>
              {breakdown.ced3_distribution.map((r, i) => (
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
            <Section title="Top Pathogenic Variants (IFT43 — cohort seed=391)" color={ACCENT}>
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

      {/* ── TAB 2: IFT43 IFT-A Satellite & Campomelia ── */}
      {tab === 2 && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="IFT43 Protein Structure (378 aa — IFT-A Satellite)" color={ACCENT}>
              <Alert color={ACCENT}>
                <strong>IFT43 is the SMALLEST IFT-A subunit (378 aa)</strong> — the peripheral anchor of the
                IFT-A satellite module. It directly contacts the N-terminal WD40 blades 1–3 of
                IFT121/WDR35 (SRTD7). Loss of IFT43 destabilises the entire satellite module —
                IFT121 loses its peripheral anchor and IFT-A retrograde function collapses.
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Domain</th><th>Residues</th><th>Function</th></tr></thead>
                <tbody>
                  <tr><td>N-terminal adaptor module</td><td>aa 1–110</td><td>IFT121 N-terminal WD40 blade-1-3 binding; main IFT-A satellite contact; missense cluster aa 70–110</td></tr>
                  <tr><td>Central linker</td><td>aa 111–200</td><td>IFT121-contact interface; W174R pathogenic cluster; allosteric communication to IFT-A ARM hub</td></tr>
                  <tr><td>C-terminal anchoring domain</td><td>aa 201–378</td><td>Positions IFT43 at IFT-A periphery; CED3/Sensenbrenner hypomorphic allele cluster</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="IFT-A Complex Assembly — IFT43 Position" color={ACCENT6}>
              <div className="table-responsive">
                <table className="table table-sm table-bordered small">
                  <thead><tr><th>Module</th><th>Subunit</th><th>SRTD</th><th>IFT43 contact?</th></tr></thead>
                  <tbody>
                    <tr><td>ARM hub</td><td>IFT144 (WDR19)</td><td>SRTD5</td><td>Indirect (IFT122 bridges hub to satellite)</td></tr>
                    <tr><td>ARM hub</td><td>IFT140</td><td>SRTD9</td><td>Indirect (ARM hub peripheral contact)</td></tr>
                    <tr><td>ARM hub</td><td>IFT139 (TTC21B)</td><td>SRTD4</td><td>Indirect (ARM hub retrograde arm)</td></tr>
                    <tr><td>ARM hub</td><td>IFT122</td><td>SRTD2</td><td>Indirect (bridges ARM hub to satellite)</td></tr>
                    <tr><td>Satellite</td><td>IFT121 (WDR35)</td><td>SRTD7</td><td>✓ Direct (IFT121 N-WD40 blades 1–3 = IFT43 binding surface)</td></tr>
                    <tr style={{ background: ACCENT + '22', fontWeight: 'bold' }}><td>Satellite</td><td>IFT43</td><td>SRTD18 ◀</td><td>— (self; peripheral anchor; contacts IFT121)</td></tr>
                  </tbody>
                </table>
              </div>
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="Retrograde IFT Failure Cascade (IFT43 Loss)" color={ACCENT2}>
              {[
                ['IFT43 absent', 'Satellite peripheral anchor missing → IFT121 (WDR35, SRTD7) N-terminal WD40 loses its stabilising contact'],
                ['IFT-A satellite collapses', 'Without IFT43 anchor, IFT121 destabilises within the satellite module → IFT-A complex peripheral integrity fails'],
                ['Retrograde IFT impaired', 'IFT-A-mediated retrograde transport (tip-to-base) is compromised → cilia content recycling impaired'],
                ['SHORT-STUBBY CILIA on EM', 'IFT-A class morphology: short, stubby, slightly widened cilia (NOT club/bulging) — same as SRTD2/4/5/7/9'],
                ['IFT-B precursor accumulation at base', 'Anterograde cargoes pile up at cilia base (opposite to dynein-2 SRTDs where IFT-B piles at tip)'],
                ['Hedgehog (Ihh/Shh) failure', 'PTCH1/SMO/GLI trafficking impaired → GLI3R not processed → Hh target genes dysregulated in chondrocytes'],
                ['Chondrocyte failure', 'Short ribs, NARROW THORAX (primary), shortened limbs'],
                ['Campomelia (DISTINCTIVE)', 'Asymmetric periosteal remodelling from Hh failure → long bone bowing (femur/tibia/humerus) in ~50%'],
                ['Secondary organ failure', 'Renal TIN/cysts (~27%), retinal rod-cone dystrophy (~13%), CHF/hepatic fibrosis (~9%)'],
              ].map(([step, desc], i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-2" style={{ background: ACCENT }}>Step {i + 1}</span>
                  <strong>{step}:</strong> {desc}
                </div>
              ))}
            </Section>

            <Section title="CED3 / Sensenbrenner Dual Phenotype" color={ACCENT9}>
              <Alert color={ACCENT9}>
                <strong>Allele class governs phenotype:</strong> Biallelic null alleles → SRTD18 (severe skeletal ciliopathy).
                Hypomorphic alleles (residual IFT43 function) → CED3 / Sensenbrenner syndrome:
                craniosynostosis + sparse/thin hair + hypodontia/dental anomalies + narrow thorax +
                hepatic fibrosis. IFT43 is one of three CED genes (CED1=IFT122, CED2=WDR35, CED3=IFT43).
              </Alert>
              <table className="table table-sm table-bordered small">
                <thead><tr><th>Feature</th><th>SRTD18 (null alleles)</th><th>CED3 (hypomorphic)</th></tr></thead>
                <tbody>
                  <tr><td>Thorax</td><td>Narrow — neonatal respiratory</td><td>Narrow — moderate; survivable</td></tr>
                  <tr><td>Campomelia</td><td>~50% (more common)</td><td>Rare (&lt;10%)</td></tr>
                  <tr><td>Craniosynostosis</td><td>Rare (&lt;5%)</td><td>Present (~70%)</td></tr>
                  <tr><td>Hair</td><td>Normal</td><td>Sparse / fine</td></tr>
                  <tr><td>Dentition</td><td>Normal</td><td>Hypodontia / abnormal shape</td></tr>
                  <tr><td>Hepatic</td><td>CHF (ductal plate) ~9%</td><td>Fibrosis present ~30%</td></tr>
                  <tr><td>Renal</td><td>TIN/ESRD ~22%</td><td>TIN/cysts ~15%</td></tr>
                </tbody>
              </table>
            </Section>

            <Section title="Why No Situs Inversus / No Joubert MTS" color={ACCENT4}>
              <p className="small mb-1">
                <strong>No situs inversus:</strong> IFT43 functions in PRIMARY non-motile cilia (9+0 axoneme).
                Situs inversus requires MOTILE nodal cilia (9+2). IFT43 is not expressed in nodal motile cilia.
              </p>
              <p className="small mb-0">
                <strong>No Joubert MTS:</strong> The molar tooth sign requires brainstem/cerebellar vermis
                decussation defects (JBTS ciliopathies). IFT43/SRTD18 affects skeletal chondrocytes
                and secondary organs — NOT the CNS midline structures. Compare: CEP290 biallelic null
                → MKS4 (lethal) or JBTS5 (midline defect) — very different from IFT43.
              </p>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Gene Card — IFT43" color={ACCENT}>
              <table className="table table-sm table-bordered small">
                <tbody>
                  {Object.entries(defs.gene_card).map(([k, v]) => (
                    <tr key={k}><td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td><td>{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </Section>

            <Section title="Disease Card — SRTD18" color={ACCENT2}>
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

            <Section title="Key Pathogenic Variants (IFT43)" color={ACCENT}>
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
                  <thead><tr><th>Disease</th><th>Key Differentiator from SRTD18</th></tr></thead>
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
