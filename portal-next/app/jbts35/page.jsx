'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'KIAA0753 CP110-Removal Pearls', 'Definitions'];

// JBTS35 colour scheme — KIAA0753 / OFD1-interacting / CP110 removal / BB distal appendage / 17p13.1
// Steel blue for BB distal appendage; teal for OFD1 axis; amber for CP110 retention; purple for TTBK2 kinase
const ACCENT   = '#1565c0';   // deep steel blue — KIAA0753 BB distal appendage scaffold
const ACCENT2  = '#00695c';   // dark teal — OFD1-KIAA0753 axis
const ACCENT3  = '#004d40';   // darker teal — cilia initiation
const ACCENT4  = '#0277bd';   // sky blue — renal NPHP-like (tubular primary cilia)
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#e65100';   // deep amber — CP110 retention / cilia initiation block
const ACCENT7  = '#827717';   // olive — hepatic (very low ~5%; minimal biliary involvement)
const ACCENT8  = '#4a148c';   // deep purple — TTBK2 kinase mis-docking
const ACCENT9  = '#1b5e20';   // forest green — cerebellar / neurological
const ACCENT10 = '#880e4f';   // deep magenta — OFD1 protein partner (distinct from OFD1 gene X-linked)

const SEED     = 487;
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

export default function JBTS35Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts35/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts35/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts35/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="alert alert-danger m-4">Error: {error}</div>;

  const kpis     = overview?.kpis     || {};
  const alerts   = overview?.alerts   || {};
  const facts    = overview?.key_facts || [];
  const patients = overview?.patients  || [];

  return (
    <div className="container-fluid py-3 px-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}22, ${ACCENT2}22)`, border: `1px solid ${ACCENT}55` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: 28 }}>&#x1f9ec;</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>KIAA0753 / OFIP — Joubert Syndrome Type 35 (JBTS35)</h4>
            <div className="small text-muted">
              OFD1-Interacting Protein · CP110 Removal Coordinator · BB Distal Appendage Scaffold · TTBK2 Docking · No MKS Tier · 17p13.1 · OMIM *617518 / #619712
            </div>
            <div className="mt-1">
              <span className="badge me-1 bg-success">&#x2714; No MKS Tier</span>
              <span className="badge me-1" style={{ background: ACCENT6 }}>CP110 Retained (cilia absent)</span>
              <span className="badge me-1" style={{ background: ACCENT8 }}>TTBK2 Mis-docked</span>
              <span className="badge me-1" style={{ background: ACCENT10 }}>OFD1 Protein Partner</span>
              <span className="badge me-1" style={{ background: ACCENT4 }}>Renal ~15%</span>
              <span className="badge" style={{ background: ACCENT2 }}>Cilia Initiation Defect</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Cohort (JBTS35)" value={kpis.total_patients ?? N_COHORT} color={ACCENT} />
            <KPI label="MTS %" value={`${kpis.mts_pct ?? 100}%`} color={ACCENT2} />
            <KPI label="Cerebellar Ataxia" value={`${kpis.ataxia_pct ?? '—'}%`} color={ACCENT9} />
            <KPI label="Hypotonia" value={`${kpis.hypotonia_pct ?? '—'}%`} color={ACCENT3} />
            <KPI label="OMA" value={`${kpis.oma_pct ?? '—'}%`} color={ACCENT} />
            <KPI label="Breathing" value={`${kpis.breathing_pct ?? '—'}%`} color={ACCENT2} />
            <KPI label="Retinal" value={`${kpis.retinal_pct ?? '—'}%`} color={ACCENT8} />
            <KPI label="Renal NPHP" value={`${kpis.renal_pct ?? '—'}%`} color={ACCENT4} />
            <KPI label="Hepatic" value={`${kpis.hepatic_pct ?? '—'}%`} color={ACCENT7} />
            <KPI label="Polydactyly" value={`${kpis.poly_pct ?? '—'}%`} color={ACCENT5} />
            <KPI label="ID" value={`${kpis.id_pct ?? '—'}%`} color={ACCENT5} />
            <KPI label="ESRD" value={`${kpis.esrd_pct ?? '—'}%`} color={ACCENT6} />
          </div>

          {/* Alerts */}
          <Alert color={ACCENT2}>
            <strong style={{ color: ACCENT2 }}>&#x2714; NO MKS TIER — ALL JBTS35 LIVEBORN:</strong>{' '}
            {alerts.no_mks_tier}
          </Alert>

          <Alert color={ACCENT6}>
            <strong style={{ color: ACCENT6 }}>&#x26a0;&#xfe0f; CP110 REMOVAL BLOCK — CILIA ABSENT, NOT MERELY SHORT:</strong>{' '}
            {alerts.cp110_removal_upstream}
          </Alert>

          <Alert color={ACCENT10}>
            <strong style={{ color: ACCENT10 }}>OFD1 PROTEIN PARTNER ≠ OFD1 GENE (X-LINKED OFD SYNDROME):</strong>{' '}
            {alerts.ofd1_axis}
          </Alert>

          <Alert color={ACCENT8}>
            <strong style={{ color: ACCENT8 }}>TTBK2 KINASE SCAFFOLD — DDx vs JBTS11/TTBK2 (KINASE ABSENT):</strong>{' '}
            {alerts.ttbk2_kinase_dependency}
          </Alert>

          {/* Key facts */}
          <Section title="Key Clinical Facts — JBTS35 (KIAA0753 / OFIP)" color={ACCENT}>
            <ul className="list-unstyled mb-0">
              {facts.map((f, i) => (
                <li key={i} className="mb-1 small"><span style={{ color: ACCENT }}>&#x25b8;</span> {f}</li>
              ))}
            </ul>
          </Section>

          {/* Patient table */}
          <Section title={`40-Patient JBTS35 Educational Cohort (Seed ${SEED})`} color={ACCENT5}>
            <div style={{ overflowX: 'auto' }}>
              <table className="table table-sm table-hover small mb-0">
                <thead style={{ background: ACCENT5 + '22' }}>
                  <tr>
                    <th>ID</th><th>Age</th><th>Sex</th><th>Ethnicity</th>
                    <th>Variant</th><th>MTS</th><th>Ataxia</th><th>Renal</th><th>Retinal</th><th>Poly</th>
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
                      <td>{p.mts     ? <span style={{ color: ACCENT2 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.ataxia  ? <span style={{ color: ACCENT9 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.renal   ? <span style={{ color: ACCENT4 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.retinal ? <span style={{ color: ACCENT8 }}>&#x2713;</span> : '—'}</td>
                      <td>{p.poly    ? <span style={{ color: ACCENT5 }}>&#x2713;</span> : '—'}</td>
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
            <Section title="Notable JBTS35 Variants — KIAA0753/OFIP Hypomorphic Alleles" color={ACCENT8}>
              <div className="row g-2">
                {breakdown.notable_variants?.map((v, i) => (
                  <div key={i} className="col-md-6">
                    <div className="card h-100 shadow-sm p-2" style={{ borderLeft: `3px solid ${ACCENT8}` }}>
                      <div className="fw-bold" style={{ color: ACCENT8 }}>{v.name} <span className="text-muted font-monospace small">({v.cdna})</span></div>
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

      {/* ── TAB 2: KIAA0753 CP110-REMOVAL PEARLS ── */}
      {tab === 2 && (
        <div>
          <Section title="KIAA0753 / OFIP Clinical Pearls — JBTS35 CP110-Removal Scaffold" color={ACCENT}>
            <div className="row g-3">

              {/* Pearl 1 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT6}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT6 }}>1. CILIA ABSENT (CP110 RETAINED) — SUPER-RESOLUTION DIAGNOSTIC</div>
                  <p className="small mb-1">In JBTS35 patient fibroblasts, cilia are absent (not merely short): CP110 is retained at the mother-centriole distal appendage tip, blocking cilia membrane nucleation. This is a defining mechanistic signature distinguishable by super-resolution microscopy (STED or STORM).</p>
                  <p className="small mb-1"><strong>Imaging protocol:</strong> Co-stain with CP110 (retained at BB tip) + ARL13B (absent/stub) + acetylated tubulin (axoneme absent). Compare: CPLANE1/JBTS33 → ARL13B present, short/misoriented axoneme; JBTS35 → ARL13B absent (no axoneme). CEP164 (TTBK2 recruiter) is present at distal appendage in JBTS35 — confirms defect is downstream of CEP164.</p>
                  <p className="small mb-0"><strong>Clinical implication:</strong> Nasal brushing or rectal biopsy cilia IF/TEM from JBTS35 patients: cilia absent on TEM (vs. structurally abnormal but present in IFT/TZ-gate subtypes). This is the most striking cilia phenotype among JBTS subtypes that are not MKS tier.</p>
                </div>
              </div>

              {/* Pearl 2 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT8}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT8 }}>2. TTBK2 SCAFFOLD vs JBTS11/TTBK2 KINASE — CRITICAL DDx</div>
                  <p className="small mb-1">KIAA0753/JBTS35 and TTBK2/JBTS11 both result in CP110 retention and absent cilia, but via mechanistically distinct defects:</p>
                  <ul className="small mb-1">
                    <li><strong>JBTS35 (KIAA0753):</strong> TTBK2 kinase present and active, but mis-docked at the distal appendage → CEP164 Ser1285 phosphorylation reduced → MPP9 not degraded → CP110 retained</li>
                    <li><strong>JBTS11 (TTBK2):</strong> TTBK2 kinase itself absent → no CEP164 Ser1285 phosphorylation → MP9 retained → CP110 retained</li>
                  </ul>
                  <p className="small mb-0"><strong>Biochemical distinction:</strong> In JBTS35 fibroblasts: TTBK2 is detectable at the distal appendage (by IF, reduced but present); pSer1285-CEP164 is reduced (~40–55% WT). In JBTS11 fibroblasts: TTBK2 absent from distal appendage; pSer1285-CEP164 undetectable. WES: KIAA0753 + TTBK2 + CEP164 on targeted ciliopathy panel — all three must be included.</p>
                </div>
              </div>

              {/* Pearl 3 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT10}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT10 }}>3. OFD1 PROTEIN PARTNER — NOT OFD1 GENE (X-LINKED OFD SYNDROME)</div>
                  <p className="small mb-1">KIAA0753 directly interacts with OFD1 PROTEIN (the oral-facial-digital type 1 protein, a distal-appendage scaffold component). However, KIAA0753 is an AUTOSOMAL RECESSIVE gene on chromosome 17p13.1. The OFD1 GENE (Xp22.2) causes X-linked OFD syndrome (OMIM #311200), a completely different condition.</p>
                  <p className="small mb-1"><strong>Reporting risk:</strong> A WES report that lists "OFD1 pathway involvement" or incorrectly labels KIAA0753 variants as "OFD1" would cause serious clinical mismanagement — OFD syndrome has X-linked inheritance (affected males, carrier females), oral-facial-digital features, and different surveillance requirements.</p>
                  <p className="small mb-0"><strong>Male patients:</strong> Co-sequence OFD1 gene (Xp22.2) only if unexplained oral/facial/digital features are present beyond typical JBTS35. The KIAA0753-OFD1 protein interaction is a biological connection, not a genetic or clinical overlap with OFD syndrome.</p>
                </div>
              </div>

              {/* Pearl 4 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT2 }}>4. NO MKS TIER — COUNSELLING DISTINCT FROM B9-COMPLEX AND TZ-GATE SUBTYPES</div>
                  <p className="small mb-1">KIAA0753/JBTS35 does not disrupt the TZ diffusion barrier (B9-complex, TMEM proteins, nephrocystin module). No biallelic null KIAA0753 patient has presented with Meckel-Gruber Syndrome perinatal lethality. All JBTS35 patients are liveborn.</p>
                  <p className="small mb-1"><strong>Counselling critical difference vs JBTS5/CEP290, JBTS28/MKS1, JBTS34/B9D2:</strong> In MKS-tier subtypes, a parent carrying a NULL allele combined with a partner carrying a HYPOMORPHIC allele creates a 25% MKS risk per pregnancy. This risk does NOT apply in JBTS35. Recurrence risk in JBTS35: 25% JBTS35 (survivable), 50% carrier, 25% normal.</p>
                  <p className="small mb-0"><strong>Biallelic null KIAA0753:</strong> Biallelic null may be embryonic lethal (no live-born null/null reported), but this is not MKS syndrome — it lacks the encephalocele, cystic dysplastic kidneys, and polydactyly triad of MKS. Distinguished from MKS by absence of encephalocele and cystic renal dysplasia on prenatal US.</p>
                </div>
              </div>

              {/* Pearl 5 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT4 }}>5. VERY LOW RENAL/HEPATIC — NOT A TZ GATE OR BILIARY GENE</div>
                  <p className="small mb-1">JBTS35 has among the lowest renal (~15%) and hepatic (~5%) penetrance of all JBTS subtypes, because KIAA0753 acts at cilia INITIATION (upstream), not at the TZ diffusion barrier (which governs NPHP-axis and biliary cilia gate function).</p>
                  <p className="small mb-1"><strong>Renal in JBTS35:</strong> NPHP-like tubular disease (renal concentrating defect, mild proteinuria) when affected; rate ~15%. Annual creatinine/eGFR + renal US from diagnosis. ESRD uncommon (only ~6% by adulthood). Renal transplant curative if needed (cell-autonomous KIAA0753 defect).</p>
                  <p className="small mb-0"><strong>Hepatic in JBTS35:</strong> Hepatic CHF extremely rare (~5%); KIAA0753 is not a biliary cilia gate gene (no TMEM67, no NPHP4 contact). LFTs at diagnosis; if abnormal, hepatic US + UDCA consideration. Do NOT apply COACH syndrome protocol (no coloboma; no hepatic-dominant JBTS35 described). Very different from JBTS34/B9D2 (~18% hepatic).</p>
                </div>
              </div>

              {/* Pearl 6 */}
              <div className="col-md-6">
                <div className="card h-100 shadow-sm p-3" style={{ borderLeft: `4px solid ${ACCENT9}` }}>
                  <div className="fw-bold mb-2" style={{ color: ACCENT9 }}>6. WES PANEL DESIGN — KIAA0753 COVERAGE REQUIREMENTS</div>
                  <p className="small mb-1">KIAA0753 encodes a 1453-aa protein from 28 exons on chr17p13.1. Standard WES covers coding exons well but KIAA0753 splice site variants (e.g. c.1256+1G>A, intron 9) require confirmation of deep intronic coverage. The gene symbol KIAA0753 (or OFIP) should appear explicitly on the panel report — not OFD1, not OFIP-only if KIAA0753 is the canonical RefSeq ID.</p>
                  <p className="small mb-1"><strong>Functional assays for VUS:</strong>
                    (1) Co-IP in patient fibroblasts: KIAA0753-OFD1, KIAA0753-TTBK2, KIAA0753-CEP97 interactions (all should be reduced in pathogenic alleles).
                    (2) CP110 IF: retained at BB tip in pathogenic homozygotes (vs WT where CP110 is absent after serum starvation cilia induction).
                    (3) Cilia frequency: reduced (ARL13B-positive cilia absent or stunted stubs).
                  </p>
                  <p className="small mb-0"><strong>Panel co-sequencing recommended:</strong> KIAA0753 + TTBK2 (JBTS11) + CEP164 (TTBK2 recruiter, distal appendage) + OFD1 gene (X-linked DDx for males). CPLANE1/JBTS33 is NOT a DDx (CPLANE acts at BB docking geometry, not CP110 removal; cilia present in CPLANE1, absent in KIAA0753).</p>
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
                    <tr><td className="fw-bold">Gene</td><td>{definitions.gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Gene</td><td>{definitions.omim_gene}</td></tr>
                    <tr><td className="fw-bold">OMIM Disease</td><td>{definitions.omim_disease}</td></tr>
                    <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                    <tr><td className="fw-bold">Protein Size</td><td>{definitions.protein_size}</td></tr>
                    <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    <tr><td className="fw-bold">MKS Tier</td><td><span className="badge bg-success">No — all liveborn</span></td></tr>
                    <tr><td className="fw-bold">SRTD Allelic</td><td>{definitions.srtd_allelic ? 'Yes' : 'No'}</td></tr>
                    <tr><td className="fw-bold">Frequency (JBTS)</td><td>{definitions.frequency_jbts}</td></tr>
                    <tr><td className="fw-bold">Worldwide Prevalence</td><td>{definitions.worldwide_prevalence}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="col-md-6">
                <div className="card p-2 h-100" style={{ borderLeft: `3px solid ${ACCENT6}` }}>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT6 }}>Molecular Mechanism</div>
                  <div className="small">{definitions.mechanism}</div>
                </div>
              </div>
            </div>
          </Section>

          {/* DDx pearls */}
          <Section title="Differential Diagnosis Pearls" color={ACCENT5}>
            {definitions.ddx_pearls?.map((d, i) => (
              <div key={i} className="small border-bottom py-2">
                <span style={{ color: ACCENT5 }}>&#x25b8;</span> {d}
              </div>
            ))}
          </Section>

          {/* Key biomarkers */}
          <Section title="Key Biomarkers — IF / TEM Signatures" color={ACCENT6}>
            <div className="row g-2">
              {definitions.key_biomarkers && Object.entries(definitions.key_biomarkers).map(([k, v]) => (
                <div key={k} className="col-md-6">
                  <div className="card p-2 h-100" style={{ borderLeft: `3px solid ${ACCENT6}` }}>
                    <div className="fw-bold small text-uppercase" style={{ color: ACCENT6 }}>{k.replace(/_/g, ' ')}</div>
                    <div className="small">{v}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* Glossary */}
          <Section title="Definitions Glossary" color={ACCENT2}>
            {definitions.glossary && Object.entries(definitions.glossary).map(([term, def]) => (
              <div key={term} className="mb-2 small">
                <span className="fw-bold" style={{ color: ACCENT2 }}>{term}:</span> {def}
              </div>
            ))}
          </Section>

          {/* Nav */}
          <div className="mt-3">
            <Link href="/jbts34" className="btn btn-sm me-2" style={{ background: ACCENT + '22', color: ACCENT, border: `1px solid ${ACCENT}` }}>
              &#x2190; JBTS34 B9D2
            </Link>
            <Link href="/joubert" className="btn btn-sm" style={{ background: ACCENT2 + '22', color: ACCENT2, border: `1px solid ${ACCENT2}` }}>
              Joubert Overview &#x2192;
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
