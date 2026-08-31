'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'B9D2 β-Strand Bridge Pearls', 'Definitions'];

// JBTS34 colour scheme — B9D2 / β-Strand Bridge / B9-Complex / MKS TIER / 19q13.2
// Deep violet for B9D2 β-exchange structural role; teal for B9-complex; crimson for MKS tier
const ACCENT   = '#6a1b9a';   // deep violet — B9D2 β-strand bridge structural role
const ACCENT2  = '#00695c';   // dark teal — B9-complex TZ gate
const ACCENT3  = '#004d40';   // darker teal — TZ inner leaflet
const ACCENT4  = '#0277bd';   // sky blue — renal NPHP-like (B9D1 destabilisation indirect)
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#b71c1c';   // dark crimson — MKS10 perinatal lethal / MKS tier
const ACCENT7  = '#e65100';   // deep orange — hepatic CHF (lower ~18%; TMEM231 bridge)
const ACCENT8  = '#4a148c';   // deep purple — retinal rod-cone / connecting cilia
const ACCENT9  = '#1b5e20';   // forest green — cerebellar / neurological
const ACCENT10 = '#1a237e';   // deep indigo — B9D1 barrel cap / β-strand exchange

const SEED = 473;
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

export default function JBTS34Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts34/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts34/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts34/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>B9D2 — Joubert Syndrome Type 34 (JBTS34)</h4>
            <div className="small text-muted">
              B9-Complex β-Strand Bridge · B9D1 Barrel Cap · MKS TIER (MKS10 Allelic) · No Population Founder · 19q13.2 · OMIM *611951 / #614749
            </div>
            <div className="mt-1">
              <span className="badge me-1" style={{ background: ACCENT6 }}>&#x26a0;&#xfe0f; MKS TIER</span>
              <span className="badge me-1" style={{ background: ACCENT10 }}>β-Strand Bridge / B9D1 Cap</span>
              <span className="badge me-1" style={{ background: ACCENT7 }}>Hepatic CHF ~18%</span>
              <span className="badge me-1" style={{ background: ACCENT4 }}>Renal ~30%</span>
              <span className="badge me-1 bg-secondary">No Population Founder</span>
              <span className="badge" style={{ background: ACCENT2 }}>B9-Complex Subunit</span>
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
            <KPI label="Cohort (JBTS34)" value={kpis.total_patients ?? N_COHORT} color={ACCENT} />
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
            <strong style={{ color: ACCENT6 }}>&#x26a0;&#xfe0f; MKS TIER — B9D2 BIALLELIC NULL → MECKEL-GRUBER SYNDROME TYPE 10 (PERINATAL LETHAL):</strong>{' '}
            {alerts.mks_tier_allelic}
          </Alert>

          <Alert color={ACCENT10}>
            <strong style={{ color: ACCENT10 }}>B9-COMPLEX β-STRAND BRIDGE (B9D2 vs B9D1/JBTS19 vs MKS1/JBTS28):</strong>{' '}
            {alerts.b9_complex_bridge}
          </Alert>

          <Alert color={ACCENT7}>
            <strong style={{ color: ACCENT7 }}>LOWER HEPATIC CHF (~18%) AND RENAL (~30%) — NO DIRECT NPHP4/MKS3 MODULE DOCKING:</strong>{' '}
            {alerts.lower_hepatic_renal}
          </Alert>

          <Alert color={ACCENT5}>
            <strong style={{ color: ACCENT5 }}>NO CONFIRMED POPULATION FOUNDER — FULL B9D2 SEQUENCING REQUIRED:</strong>{' '}
            {alerts.no_founder}
          </Alert>

          {/* Key facts */}
          <Section title="Key Clinical Facts — JBTS34 (B9D2)" color={ACCENT}>
            <ul className="list-unstyled mb-0">
              {facts.map((f, i) => (
                <li key={i} className="mb-1 small"><span style={{ color: ACCENT }}>&#x25b8;</span> {f}</li>
              ))}
            </ul>
          </Section>

          {/* Patient table */}
          <Section title={`40-Patient JBTS34 Educational Cohort (Seed ${SEED})`} color={ACCENT5}>
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
            <Section title="Notable JBTS34 Variants — B9D2 Hypomorphic Alleles" color={ACCENT10}>
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

      {/* ── TAB 2: B9D2 β-STRAND BRIDGE PEARLS ── */}
      {tab === 2 && (
        <div>
          <Section title="B9D2 β-Strand Bridge Clinical Pearls — JBTS34" color={ACCENT}>
            <div className="row g-3">
              {/* Pearl 1 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT6 }}>1. MKS TIER GENETIC COUNSELLING — B9D2 NULL PARTNER RISK</div>
                  <p className="small mb-1">B9D2 biallelic null → Meckel-Gruber Syndrome Type 10 (MKS10, perinatal lethal). In JBTS34 families where one parent carries a B9D2 NULL allele and the other parent carries a JBTS34 HYPOMORPHIC allele, 25% of offspring face MKS10 lethal outcome and 25% face JBTS34.</p>
                  <p className="small mb-1"><strong>Action:</strong> Full partner B9D2 genotyping (WES or targeted B9D2 panel) before any pregnancy attempt in a JBTS34 family. B9D1 and MKS1 co-genotyping recommended because B9D1 (JBTS19/MKS9, 17p11.2) and MKS1 (JBTS28/MKS1, 17q22) are obligate B9-complex partners — digenic B9D2 + B9D1 compound heterozygosity has been reported.</p>
                  <p className="small mb-0"><strong>Key difference from JBTS28/MKS1:</strong> No Finnish founder pre-screen available for JBTS34/B9D2 — full B9D2 sequencing required (WES + intronic splice site coverage). Contrast MKS1/JBTS28 where Finnish IVS14 founder pre-screen (Sanger) can precede WES in Finnish patients.</p>
                </div>
              </div>

              {/* Pearl 2 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT10}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT10 }}>2. β-STRAND EXCHANGE EM HALLMARK — BRIDGE ABSENT, NOT ANCHOR ABSENT</div>
                  <p className="small mb-1">The EM distinction between B9D2/JBTS34 and B9D1/JBTS19 is critical for mechanistic diagnosis and is only resolvable on high-resolution TEM cryosections:</p>
                  <ul className="small mb-1">
                    <li><strong>B9D2/JBTS34:</strong> Y-link inter-subunit β-strand bridge absent — the cross-link between B9D1 and B9D2 within the inner-leaflet Y-link arm is missing; the membrane anchor legs (B9D1) are present but partially disordered</li>
                    <li><strong>B9D1/JBTS19:</strong> Y-link membrane anchor legs absent — B9D1 B9-domain contacts with inner leaflet are fully lost; the bridge (B9D2) is present but unanchored</li>
                    <li><strong>MKS1/JBTS28:</strong> Y-link inner-leaflet spacing wider — MKS1 scaffold detachment; both B9D1 and B9D2 present but mispositioned</li>
                  </ul>
                  <p className="small mb-0">Clinical availability: nasal brushing cilia TEM (non-invasive) can reveal Y-link architecture in experienced centres. Functional co-IP (B9D2–B9D1, B9D2–MKS1) in patient fibroblasts provides complementary biochemical evidence for VUS classification (ACMG PS3 criterion).</p>
                </div>
              </div>

              {/* Pearl 3 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT7}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT7 }}>3. LOWER HEPATIC CHF THAN JBTS28 — DISTINGUISH ON CLINIC REVIEW</div>
                  <p className="small mb-1">JBTS34 hepatic CHF penetrance (~18%) is substantially lower than MKS1/JBTS28 (~30%) because B9D2 does NOT directly contact MKS3/TMEM67 (that is the MKS1 CC2 function). B9D2 contributes to biliary cilia gate only indirectly via TMEM231 bridge uncoupling.</p>
                  <p className="small mb-1"><strong>Clinical decision tree:</strong> In a MTS patient with MKS-tier alleles and hepatic CHF: CHF &gt;25% probability → favour MKS1/JBTS28 (or TMEM67/JBTS6-COACH if coloboma present). CHF ~15–20% → favour B9D2/JBTS34 or B9D1/JBTS19. Hepatic CHF in B9D2/JBTS34 requires the same annual LFT + hepatic US protocol as JBTS28 — presence at lower penetrance does not mean absent.</p>
                  <p className="small mb-0"><strong>Protocol:</strong> LFTs + hepatic US at diagnosis and annually. If CHF confirmed: UDCA 10–15 mg/kg/day; portal hypertension screen (endoscopy) from age 12; hepatic transplant if ESLD. No coloboma in JBTS34 (distinguishes from JBTS6/TMEM67-COACH).</p>
                </div>
              </div>

              {/* Pearl 4 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT2 }}>4. B9-COMPLEX PANEL INTERPRETATION — ALL THREE B9 GENES MANDATORY</div>
                  <p className="small mb-1">B9D2 (JBTS34/MKS10), B9D1 (JBTS19/MKS9), and MKS1 (JBTS28) are the three subunits of the B9-complex TZ inner-leaflet gate. WES ciliopathy panels MUST include all three separately. VUS interpretation: functional co-IP between all three B9 subunits is the gold standard.</p>
                  <p className="small mb-1">Digenic scenario: B9D2 heterozygous VUS + B9D1 heterozygous VUS (one pathogenic allele in each gene) can produce a B9-complex assembly defect — digenic ciliopathy. This is rare but documented in B9-complex literature. Full trio WES (proband + both parents) is necessary to correctly phase compound het from digenic configurations.</p>
                  <p className="small mb-0"><strong>IF panel for JBTS34 fibroblasts:</strong> B9D2 (confirm expression level), B9D1 (confirm TZ localisation — partially impaired in B9D2 null), NPHP4 (mildly reduced, indirect), TMEM231 (reduced TZ localisation via TMEM231-B9D2 bridge uncoupling). ARL13B: cilia present in JBTS34 hypomorphic (shortened, not absent).</p>
                </div>
              </div>

              {/* Pearl 5 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT4 }}>5. RENAL SURVEILLANCE — INDIRECT B9D1 DESTABILISATION</div>
                  <p className="small mb-1">JBTS34 renal penetrance (~30%) is lower than MKS1/JBTS28 (~35%) but higher than the JBTS average. The mechanism is indirect: B9D2 LOF → B9D1 B9-barrel partially uncapped → B9D1 membrane insertion partially impaired → TZ gate partially open → NPHP-like tubular dysfunction. B9D2 does not directly contact NPHP4 (unlike MKS1 via CC1).</p>
                  <p className="small mb-1"><strong>Protocol:</strong> Annual renal US + creatinine/eGFR from diagnosis. ESRD when renal affected (median ~20–28 yr); renal transplant curative (cell-autonomous B9D2 defect; transplanted kidney has WT B9D2). No recurrence post-transplant.</p>
                  <p className="small mb-0"><strong>Contrast B9D1/JBTS19 vs B9D2/JBTS34 renal:</strong> B9D1/JBTS19 renal ~35% (direct B9D1 anchor loss → more complete NPHP axis disruption). B9D2/JBTS34 renal ~30% (indirect via B9D1 destabilisation → partial NPHP axis disruption). Subtle clinical gradient distinguishable only on WES (both MKS tier, similar phenotype on clinical review).</p>
                </div>
              </div>

              {/* Pearl 6 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT3}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT3 }}>6. NO POPULATION FOUNDER — ETHNICITY-AGNOSTIC PANEL DESIGN</div>
                  <p className="small mb-1">B9D2/JBTS34 has no confirmed high-frequency population founder (contrast: MKS1/JBTS28 Finnish IVS14 founder, carrier ~1/50 Finland; CEP290/JBTS5 Ashkenazi Jewish p.Gln1584Pro, carrier ~1/92 AJ). JBTS34 is ascertained across MENA (Arg54Gln cluster), South Asian (Leu89Pro), North African (Asp143His), and European (Arg185Gln + c.289+1G>A) populations without any single ethnicity disproportionately affected.</p>
                  <p className="small mb-1"><strong>Implication for panel design:</strong> Do NOT design a B9D2 targeted pre-screen (no single founder variant to test first). ALL B9D2/JBTS34 diagnoses require full B9D2 gene sequencing (WES or B9D2-targeted NGS panel) covering all exons + splice sites. Deep intronic variants: B9D2 has 9 exons; standard WES covers most splice sites but intronic branch-point regions may need Sanger confirmation for c.289+1-type splice variants.</p>
                  <p className="small mb-0"><strong>Variant interpretation:</strong> B9D2 missense VUSs: functional co-IP with B9D1 and MKS1 in patient fibroblasts (or HEK293T overexpression) is the most reliable VUS classification tool. ACMG PS3 applies if co-IP shows reduced interaction; PM3 if trans-allele is clearly pathogenic.</p>
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
                    <tr><td className="fw-bold">Gene</td><td>B9D2 — B9-Domain-Containing Protein 2 (MKSR2/C19orf52)</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>*{definitions.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM JBTS34</td><td>#{definitions.omim_jbts34}</td></tr>
                    <tr><td className="fw-bold">OMIM MKS10</td><td>#{definitions.omim_mks10}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                    <tr><td className="fw-bold">Protein Size</td><td>{definitions.protein_size}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    <tr><td className="fw-bold">MKS Tier</td><td><span className="badge" style={{ background: ACCENT6 }}>Yes — MKS10 (null/null lethal)</span></td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="card p-2 h-100" style={{ borderLeft: `3px solid ${ACCENT10}` }}>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT10 }}>Mechanism Class</div>
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

          {/* Founder / cluster variants */}
          {definitions.founder_variants?.length > 0 && (
            <Section title="Population Cluster Variants (No High-Frequency Founder)" color={ACCENT10}>
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

          {/* Nav back/forward */}
          <div className="mt-3">
            <Link href="/jbts29" className="btn btn-sm me-2" style={{ background: ACCENT + '22', color: ACCENT, border: `1px solid ${ACCENT}` }}>
              &#x2190; JBTS29 TOGARAM1
            </Link>
            <Link href="/jbts28" className="btn btn-sm" style={{ background: ACCENT2 + '22', color: ACCENT2, border: `1px solid ${ACCENT2}` }}>
              JBTS28 MKS1 &#x2192;
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
