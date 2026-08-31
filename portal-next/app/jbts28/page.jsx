'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'MKS1 B9-Complex Pearls', 'Definitions'];

// JBTS28 colour scheme — MKS1 / B9-Complex / TZ Inner-Leaflet Gate / MKS Tier / Finnish Founder
// Deep amber MKS-tier tones; teal B9-complex; crimson for CHF/lethal allele warnings; slate for gate
const ACCENT   = '#e65100';   // deep orange — MKS tier lethal allele warning / high-severity
const ACCENT2  = '#00695c';   // dark teal — B9-complex TZ gate structure
const ACCENT3  = '#006064';   // darker teal — TZ inner leaflet / Y-link scaffold
const ACCENT4  = '#0277bd';   // sky blue — renal NPHP-like (B9-NPHP4 axis)
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#b71c1c';   // dark crimson — MKS1 perinatal lethal / MKS tier
const ACCENT7  = '#f57f17';   // amber — hepatic CHF (high ~30%; MKS1-TMEM67 biliary axis)
const ACCENT8  = '#4a148c';   // deep purple — retinal rod-cone / connecting cilia
const ACCENT9  = '#1b5e20';   // forest green — cerebellar / neurological
const ACCENT10 = '#880e4f';   // maroon — Finnish founder / allele severity

const SEED = 469;
const N_COHORT = 40;

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
    <div className="alert mb-3" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
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

export default function JBTS28Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts28/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts28/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts28/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="alert alert-danger m-4">Error: {error}</div>;

  const kpis = overview?.kpis || {};
  const alerts = overview?.alerts || {};
  const facts = overview?.key_facts || [];
  const patients = overview?.patients || [];

  return (
    <div className="container-fluid py-3 px-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT2}22)`, border: `1px solid ${ACCENT}55` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: 28 }}>&#x1f9ec;</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>MKS1 — Joubert Syndrome Type 28 (JBTS28)</h4>
            <div className="small text-muted">
              B9-Complex Central Scaffold · TZ Inner-Leaflet Y-Link Gate · MKS TIER (MKS1 Allelic) · Finnish Founder IVS14 · 17q22 · OMIM *609883 / #617122
            </div>
            <div className="mt-1">
              <span className="badge me-1" style={{ background: ACCENT6 }}>&#x26a0;&#xfe0f; MKS TIER</span>
              <span className="badge me-1" style={{ background: ACCENT2 }}>B9-Complex Scaffold</span>
              <span className="badge me-1" style={{ background: ACCENT7 }}>HIGH Hepatic CHF ~30%</span>
              <span className="badge me-1" style={{ background: ACCENT10 }}>Finnish Founder IVS14</span>
              <span className="badge me-1 bg-secondary">No Coloboma</span>
              <span className="badge" style={{ background: ACCENT4 }}>Renal ~35% (NPHP4 axis)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort (JBTS28)" value={kpis.total_patients ?? N_COHORT} color={ACCENT} />
            <KPI label="MTS %" value={`${kpis.mts_pct ?? 100}%`} color={ACCENT2} />
            <KPI label="Cerebellar Ataxia" value={`${kpis.ataxia_pct ?? '—'}%`} color={ACCENT9} />
            <KPI label="Hepatic CHF" value={`${kpis.hepatic_pct ?? '—'}%`} color={ACCENT7} />
            <KPI label="Renal NPHP" value={`${kpis.renal_pct ?? '—'}%`} color={ACCENT4} />
            <KPI label="Retinal" value={`${kpis.retinal_pct ?? '—'}%`} color={ACCENT8} />
            <KPI label="Polydactyly" value={`${kpis.poly_pct ?? '—'}%`} color={ACCENT5} />
            <KPI label="Hypotonia" value={`${kpis.hypotonia_pct ?? '—'}%`} color={ACCENT3} />
            <KPI label="OMA" value={`${kpis.oma_pct ?? '—'}%`} color={ACCENT} />
            <KPI label="Breathing" value={`${kpis.breathing_pct ?? '—'}%`} color={ACCENT2} />
            <KPI label="ID" value={`${kpis.id_pct ?? '—'}%`} color={ACCENT5} />
            <KPI label="ESRD" value={`${kpis.esrd_pct ?? '—'}%`} color={ACCENT6} />
          </div>

          {/* MKS TIER alert */}
          <Alert color={ACCENT6}>
            <strong style={{ color: ACCENT6 }}>&#x26a0;&#xfe0f; MKS TIER — MKS1 BIALLELIC NULL → MECKEL-GRUBER SYNDROME TYPE 1 (PERINATAL LETHAL):</strong>{' '}
            {alerts.mks_tier_allelic}
          </Alert>

          <Alert color={ACCENT2}>
            <strong style={{ color: ACCENT2 }}>B9-COMPLEX CENTRAL SCAFFOLD (MKS1 vs B9D1/JBTS19 vs B9D2/JBTS34):</strong>{' '}
            {alerts.b9_complex_scaffold}
          </Alert>

          <Alert color={ACCENT7}>
            <strong style={{ color: ACCENT7 }}>HIGH HEPATIC CHF (~30%) — MKS1-MKS3/TMEM67 BILIARY AXIS:</strong>{' '}
            {alerts.high_hepatic_chf}
          </Alert>

          <Alert color={ACCENT10}>
            <strong style={{ color: ACCENT10 }}>FINNISH FOUNDER — c.1408-7_1408-3delATTTT IVS14 SPLICE:</strong>{' '}
            {alerts.finnish_founder}
          </Alert>

          {/* Key facts */}
          <Section title="Key Clinical Facts — JBTS28 (MKS1)" color={ACCENT}>
            <ul className="list-unstyled mb-0">
              {facts.map((f, i) => (
                <li key={i} className="mb-1 small"><span style={{ color: ACCENT }}>&#x25b8;</span> {f}</li>
              ))}
            </ul>
          </Section>

          {/* Patient table */}
          <Section title={`40-Patient JBTS28 Educational Cohort (Seed ${SEED})`} color={ACCENT5}>
            <div style={{ overflowX: 'auto' }}>
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th>
                    <th>Variant</th><th>MTS</th><th>Ataxia</th><th>Hepatic</th><th>Renal</th><th>Retinal</th><th>Poly</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.map(p => (
                    <tr key={p.id}>
                      <td className="fw-bold" style={{ color: ACCENT }}>{p.id}</td>
                      <td>{p.age}</td>
                      <td>{p.sex}</td>
                      <td className="small">{p.ethnicity}</td>
                      <td className="small font-monospace">{p.variant}</td>
                      <td>{p.mts ? <span style={{ color: ACCENT2 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.ataxia ? <span style={{ color: ACCENT9 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.hepatic ? <span style={{ color: ACCENT7 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.renal ? <span style={{ color: ACCENT4 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.retinal ? <span style={{ color: ACCENT8 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.poly ? <span style={{ color: ACCENT5 }}>&#x2713;</span> : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 1: DIAGNOSTIC BREAKDOWN ── */}
      {tab === 1 && breakdown && (
        <div className="row g-3">
          <div className="col-md-6">
            <Section title="Ethnicity Distribution" color={ACCENT2}>
              {breakdown.ethnicity_distribution?.map((e, i) => (
                <Bar key={i} label={e.ethnicity} value={e.count} max={N_COHORT} color={ACCENT2} />
              ))}
            </Section>

            <Section title="Allele Class Distribution" color={ACCENT}>
              {breakdown.allele_class_distribution?.map((a, i) => (
                <Bar key={i} label={a.allele_class} value={a.count} max={N_COHORT} color={ACCENT} />
              ))}
            </Section>
          </div>

          <div className="col-md-6">
            <Section title="Phenotype Summary" color={ACCENT5}>
              {breakdown.phenotype_summary && Object.entries(breakdown.phenotype_summary).map(([k, v]) => (
                <div key={k} className="d-flex justify-content-between small border-bottom py-1">
                  <span className="text-capitalize">{k.replace('_', ' ')}</span>
                  <span className="fw-bold">{v.n} / {N_COHORT} ({v.pct}%)</span>
                </div>
              ))}
            </Section>
          </div>

          <div className="col-12">
            <Section title="Notable JBTS28 Variants — MKS1 Hypomorphic Alleles" color={ACCENT10}>
              <div className="row g-2">
                {breakdown.notable_variants?.map((v, i) => (
                  <div key={i} className="col-md-6">
                    <div className="card h-100 shadow-sm p-2" style={{ borderLeft: `3px solid ${ACCENT10}` }}>
                      <div className="fw-bold" style={{ color: ACCENT10 }}>{v.name} <span className="text-muted font-monospace small">({v.cdna})</span></div>
                      <div className="small text-muted mb-1">{v.domain}</div>
                      <div className="small mb-1"><strong>Population:</strong> {v.population}</div>
                      <div className="small mb-1"><strong>Severity:</strong> <span style={{ color: ACCENT }}>{v.severity}</span></div>
                      <div className="small">{v.mechanism}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* ── TAB 2: B9 COMPLEX PEARLS ── */}
      {tab === 2 && (
        <div>
          <Section title="MKS1 B9-Complex Clinical Pearls — JBTS28" color={ACCENT}>
            <div className="row g-3">
              {/* Pearl 1 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT6 }}>1. MKS TIER GENETIC COUNSELLING — PRE-PREGNANCY MANDATORY</div>
                  <p className="small mb-1">MKS1 biallelic null → Meckel-Gruber Syndrome Type 1 (perinatal lethal). In JBTS28 families where one parent is a known carrier of a MKS1 NULL allele (frameshift, stop, splice abolishing exon inclusion) and the other parent carries a JBTS28 HYPOMORPHIC allele, 25% of offspring face the MKS1 lethal outcome and 25% face JBTS28.</p>
                  <p className="small mb-1"><strong>Action:</strong> Full partner MKS1 genotyping (WES or MKS1 gene panel) before any pregnancy attempt in a family with a known JBTS28 proband or known MKS1 hypomorphic carrier. IVS14 Finnish founder screen (c.1408-7_1408-3delATTTT) must be included in Finnish/Northern European partner panel alongside full MKS1 sequencing.</p>
                  <p className="small mb-0"><strong>Comparison:</strong> This counselling mandate applies to ALL MKS-tier JBTS genes: JBTS2/TMEM216 (MKS2), JBTS6/TMEM67 (MKS3), JBTS7/RPGRIP1L (MKS5), JBTS9/CC2D2A (MKS6), JBTS19/B9D1 (MKS9), JBTS28/MKS1 (MKS1), JBTS34/B9D2 (MKS10). Non-MKS-tier JBTS types (JBTS25/CEP104, JBTS26/KIAA0556, JBTS27/ARMC9) do NOT require MKS lethal counselling — a critical distinction for clinic communication.</p>
                </div>
              </div>

              {/* Pearl 2 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT7}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT7 }}>2. HEPATIC CHF — SEVERITY DOES NOT CORRELATE WITH NEUROLOGICAL SEVERITY</div>
                  <p className="small mb-1">Hepatic CHF in JBTS28 (~30%) can progress to portal hypertension independently of cerebellar/neurological severity. A JBTS28 patient with a mild MTS and preserved cognitive function may still develop clinically significant portal hypertension by age 20–30. The hepatic penetrance is driven by the MKS1 CC2-MKS3/TMEM67 biliary docking module (Arg296Gln is the key allele: ~40% CHF in Arg296Gln homozygotes) — biliary cilia gate function is distinct from cerebellar granule cell Hedgehog transduction.</p>
                  <p className="small mb-1"><strong>Protocol:</strong> LFTs + hepatic US at diagnosis, annually. If hepatic fibrosis confirmed: (1) UDCA (ursodeoxycholic acid) 10–15 mg/kg/day; (2) portal hypertension screen (endoscopy for varices) from age 12; (3) hepatic transplant consideration if ESLD (end-stage liver disease) — hepatic transplant curative, no hepatic recurrence in transplanted organ.</p>
                  <p className="small mb-0"><strong>Contrast JBTS6/TMEM67-COACH:</strong> TMEM67/JBTS6 (COACH syndrome) has CHF ~60% + coloboma ~40%. JBTS28 has CHF ~30% and NO coloboma. Distinguish on panel: coloboma on ophthalmology assessment immediately separates JBTS6 from JBTS28.</p>
                </div>
              </div>

              {/* Pearl 3 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT2 }}>3. B9-COMPLEX PANEL INTERPRETATION — ALL THREE GENES MANDATORY</div>
                  <p className="small mb-1">MKS1 (JBTS28), B9D1 (JBTS19/MKS9), and B9D2 (JBTS34/MKS10) are the three subunits of the B9-complex TZ inner-leaflet gate. All three have MKS lethal null tiers. Their EM phenotypes on TEM differ by subunit:</p>
                  <ul className="small mb-1">
                    <li><strong>MKS1/JBTS28:</strong> Y-link inner-leaflet detachment (MKS1 = scaffold that anchors ALL subunits); Y-link spacing wider; outer necklace intact</li>
                    <li><strong>B9D1/JBTS19:</strong> Y-link membrane anchor absent (B9-domain contacts inner leaflet); Y-link legs absent at membrane contact</li>
                    <li><strong>B9D2/JBTS34:</strong> Y-link bridge destabilised (B9D2 β-strand exchange between B9D1); inter-subunit bridge absent</li>
                  </ul>
                  <p className="small mb-0">NPHP4 docking impaired in JBTS28 (CC1 module) → higher renal NPHP penetrance (~35%) vs B9D2/JBTS34. MKS3/TMEM67 docking impaired in JBTS28 (CC2 module) → higher hepatic CHF (~30%) vs B9D1/JBTS19 (~15%) and B9D2/JBTS34 (~18%). Functional B9-complex biochemistry (co-IP: MKS1-B9D1, MKS1-B9D2) is the most reliable VUS interpretation tool for all three genes.</p>
                </div>
              </div>

              {/* Pearl 4 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT10}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT10 }}>4. FINNISH FOUNDER — IVS14 DELETION PRE-SCREEN BEFORE WES</div>
                  <p className="small mb-1">c.1408-7_1408-3delATTTT is an intronic 5-nt deletion at the branch-point of MKS1 intron 14 (IVS14), causing exon 15 skipping (47 nt skipped → 15.7 aa in-frame deletion Δaa 452–481 in TZ amphipathic helix array). The deletion is NOT always detectable by standard WES (intronic variant: WES coverage of intronic regions varies; deep intronic; only −3 to −7 upstream of splice acceptor).</p>
                  <p className="small mb-1"><strong>Pre-screen protocol:</strong> In any Finnish or Northern European patient with MTS (suspect JBTS), screen c.1408-7_1408-3del by Sanger sequencing of MKS1 intron 14 BEFORE sending WES — if homozygous Finnish founder detected, diagnosis is JBTS28 (moderate phenotype) without waiting for WES. If heterozygous IVS14 detected, WES or MKS1 gene panel needed to identify trans allele (second MKS1 allele). If second allele is null → MKS1 tier risk for future pregnancies.</p>
                  <p className="small mb-0"><strong>Carrier frequency:</strong> ~1/50–1/80 in Finland. For context: B9D1/JBTS19 has no Finnish founder; B9D2/JBTS34 no Finnish founder. MKS1 IVS14 is THE Finnish ciliopathy founder — include in any Finnish ciliopathy carrier panel.</p>
                </div>
              </div>

              {/* Pearl 5 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT4 }}>5. RENAL SURVEILLANCE — B9-NPHP4 AXIS ELEVATED RISK</div>
                  <p className="small mb-1">JBTS28 renal penetrance (~35%) is higher than the JBTS average (~18–25%) due to the MKS1-NPHP4 axis. NPHP4 (Nephronophthisis Type 4) docks to MKS1 CC1 at the TZ inner leaflet. MKS1 hypomorphic LOF partially reduces NPHP4 TZ localisation → compound B9-gate + NPHP-module defect → tubular cilia gate partially dysfunctional → NPHP-like tubulointerstitial nephritis → progressive ESRD (median ~18–28 yr when renal affected).</p>
                  <p className="small mb-1"><strong>Protocol:</strong> Annual renal US + creatinine/eGFR from diagnosis. If renal US shows increased echogenicity or cysts: confirm NPHP pattern (small/normal kidneys + corticomedullary cysts on MRI); nephrology referral early. ESRD: renal transplant curative (cell-autonomous MKS1 defect; transplanted kidney has WT MKS1).</p>
                  <p className="small mb-0"><strong>Contrast:</strong> Pure NPHP4 mutations (NPHP4 gene LOF) cause nephronophthisis without MTS or MKS tier — very different from JBTS28 which has both MTS and elevated NPHP-like renal risk from the same B9-NPHP4 axis disruption. Do not conflate NPHP4-gene nephronophthisis with JBTS28 renal phenotype.</p>
                </div>
              </div>

              {/* Pearl 6 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT3 }}>6. TZ Y-LINK LEAKINESS — CILIA PRESENT, GATE FUNCTIONALLY IMPAIRED</div>
                  <p className="small mb-1">Unlike ARMC9/JBTS27 (cilia absent in biallelic null) or CEP83/JBTS22 (cilia absent via DA foundation loss), MKS1 hypomorphic → cilia FORM (TZ structure partially intact) but TZ diffusion barrier LEAKY. Primary fibroblast cilia are present on immunofluorescence (ARL13B+) but show: (1) wider Y-link inner-leaflet spacing on TEM (MKS1 scaffold partially detached); (2) partial SMO mis-trafficking (SMO partially enters cilia — not constitutively present, not constitutively excluded); (3) reduced beat frequency in nasal brushing.</p>
                  <p className="small mb-1"><strong>Diagnostic IF panel for JBTS28:</strong> ARL13B (cilia present); NPHP4 (reduced TZ localisation in MKS1 null fibroblasts — key readout); MKS3/TMEM67 (partially reduced TZ); SMO (partially mis-trafficked). Contrast: CP110 IF normal in JBTS28 (cilia initiate normally; CP110 cleared — unlike JBTS27/ARMC9 where CP110 persists).</p>
                  <p className="small mb-0"><strong>Functional assay:</strong> Co-immunoprecipitation of MKS1 with B9D1, B9D2, NPHP4 — reduced interaction in patient fibroblasts provides functional evidence for VUS classification (ACMG PS3 criterion).</p>
                </div>
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && definitions && (
        <div>
          <Section title="Gene & Disease Summary" color={ACCENT}>
            <div className="row g-2 mb-3">
              <div className="col-md-6">
                <table className="table table-sm small mb-0">
                  <tbody>
                    <tr><td className="fw-bold">Gene</td><td>MKS1 — Meckel Syndrome Type 1 Protein</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*{definitions.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM JBTS28</td><td>#{definitions.omim_jbts28}</td></tr>
                    <tr><td className="fw-bold">OMIM MKS1</td><td>#{definitions.omim_mks1}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                    <tr><td className="fw-bold">Protein Size</td><td>{definitions.protein_size}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    <tr><td className="fw-bold">MKS Tier</td><td><span className="badge" style={{ background: ACCENT6 }}>Yes — MKS1 (null/null lethal)</span></td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="card p-2 h-100" style={{ borderLeft: `3px solid ${ACCENT2}` }}>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>Mechanism Class</div>
                  <div className="small mb-2">{definitions.mechanism_class}</div>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT2 }}>Cilia Phenotype</div>
                  <div className="small mb-2">{definitions.cilia_phenotype}</div>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT7 }}>MTS Mechanism</div>
                  <div className="small">{definitions.mts_mechanism}</div>
                </div>
              </div>
            </div>
          </Section>

          {/* Allelic diseases */}
          {definitions.allelic_diseases?.length > 0 && (
            <Section title="Allelic Diseases (MKS Tier)" color={ACCENT6}>
              {definitions.allelic_diseases.map((d, i) => (
                <div key={i} className="card p-2 mb-2" style={{ borderLeft: `3px solid ${ACCENT6}` }}>
                  <div className="fw-bold" style={{ color: ACCENT6 }}>{d.name}</div>
                  <div className="small"><strong>Alleles:</strong> {d.alleles}</div>
                  <div className="small"><strong>Phenotype:</strong> {d.phenotype}</div>
                  <div className="small"><strong>Risk:</strong> {d.tier}</div>
                </div>
              ))}
            </Section>
          )}

          {/* Mechanism detail */}
          <Section title="Molecular Mechanism Detail" color={ACCENT3}>
            <div className="small p-2 rounded" style={{ background: ACCENT3 + '11', border: `1px solid ${ACCENT3}33` }}>
              {definitions.mechanism_detail}
            </div>
          </Section>

          {/* Surveillance */}
          <Section title="Surveillance Protocol" color={ACCENT4}>
            <div className="row g-2">
              {definitions.surveillance_protocol && Object.entries(definitions.surveillance_protocol).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card p-2 h-100" style={{ borderLeft: `3px solid ${ACCENT4}` }}>
                    <div className="fw-bold small text-uppercase mb-1" style={{ color: ACCENT4 }}>{k.replace('_', ' ')}</div>
                    <div className="small">{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Key DDx */}
          <Section title="Key Differential Diagnoses" color={ACCENT5}>
            {definitions.key_ddx?.map((d, i) => (
              <div key={i} className="small border-bottom py-2">
                <span style={{ color: ACCENT5 }}>&#x25b8;</span> {d}
              </div>
            ))}
          </Section>

          {/* Founder variants */}
          {definitions.founder_variants?.length > 0 && (
            <Section title="Population Founder / Cluster Variants" color={ACCENT10}>
              <div className="row g-2">
                {definitions.founder_variants.map((v, i) => (
                  <div key={i} className="col-md-6">
                    <div className="card p-2 h-100" style={{ borderLeft: `3px solid ${ACCENT10}` }}>
                      <div className="fw-bold small" style={{ color: ACCENT10 }}>{v.variant}</div>
                      <div className="small"><strong>Population:</strong> {v.population}</div>
                      <div className="small"><strong>Frequency:</strong> {v.frequency}</div>
                      <div className="small"><strong>Domain:</strong> {v.domain}</div>
                      <div className="small"><strong>Severity:</strong> {v.severity}</div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* Definitions glossary */}
          <Section title="Definitions Glossary" color={ACCENT2}>
            {definitions.definitions?.map((d, i) => (
              <div key={i} className="mb-2 small">
                <span className="fw-bold" style={{ color: ACCENT2 }}>{d.term}:</span> {d.definition}
              </div>
            ))}
          </Section>

          {/* Nav back */}
          <div className="mt-3">
            <Link href="/jbts27" className="btn btn-sm me-2" style={{ background: ACCENT + '22', color: ACCENT, border: `1px solid ${ACCENT}` }}>
              &#x2190; JBTS27 ARMC9
            </Link>
            <Link href="/bbs1" className="btn btn-sm" style={{ background: ACCENT2 + '22', color: ACCENT2, border: `1px solid ${ACCENT2}` }}>
              BBS1 &#x2192;
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
