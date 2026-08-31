'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'OFD4 & TCTN Pearls', 'Definitions'];

// JBTS18 colour scheme — TCTN3 / Tectonic complex / OFD4 allelic / No MKS tier
// Teal/green tones — distinct from TCTN1 (purple) and TCTN2 (indigo)
const ACCENT   = '#00695c';   // dark teal — tectonic complex
const ACCENT2  = '#00796b';   // teal — TCTN3
const ACCENT3  = '#1565c0';   // blue — neurological
const ACCENT4  = '#37474f';   // slate — renal
const ACCENT5  = '#e65100';   // orange — polydactyly / OFD4
const ACCENT6  = '#4a148c';   // deep purple — domain matrix
const ACCENT7  = '#b71c1c';   // red — retinal
const ACCENT8  = '#1b5e20';   // dark green — hepatic
const ACCENT9  = '#004d40';   // very dark teal — NPHP
const ACCENT10 = '#880e4f';   // deep pink — OFD4 features

const SEED = 443;
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

export default function JBTS18Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts18/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts18/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts18/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="container py-5"><div className="alert alert-danger">API error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT10} 100%)`, color: '#fff' }}>
        <h4 className="mb-1 fw-bold">&#x1f9ec; TCTN3 — Joubert Syndrome Type 18 (JBTS18)</h4>
        <div className="small opacity-90">
          Tectonic-3 · 10q24.1 · ~1,377 aa · Tectonic Complex / TZ Lipid Gate · OFD4 Allelic (AR, OMIM #258860) · No MKS Tier · AR · OMIM Gene *613847 · Disease #614815
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS18 patients (seed {SEED}) · all biallelic genotypes → live birth (no MKS lethal tier) · OFD4 allelic overlap ~18% · TCTN1-TCTN2-TCTN3 tectonic complex
        </div>
      </div>

      {/* No MKS tier banner */}
      <Alert color={ACCENT}>
        <strong>&#x2714; No MKS Lethal Tier — TCTN3-Specific Rule:</strong>{' '}
        {overview?.no_mks_pearl?.slice(0, 220)}… TCTN3 structural role is partially compensated by TCTN1/TCTN2 dimerisation during embryogenesis.
        All biallelic genotypes → JBTS18 live birth. Standard 25% AR recurrence applies; no MKS tier calculation needed.
      </Alert>

      {/* OFD4 banner */}
      <Alert color={ACCENT10}>
        <strong>&#x1f9f0; OFD4 Allelic Overlap — AR (Not X-linked):</strong>{' '}
        {overview?.ofd4_pearl?.slice(0, 200)}… TCTN3/OFD4 is AR (25% recurrence); critical DDx from OFD1 (X-linked, male lethal in utero).
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPIs */}
          <div className="row mb-3">
            {overview?.kpis?.map((k, i) => <KPI key={i} {...k} />)}
          </div>

          {/* TCTN3 Function Pearl */}
          <Alert color={ACCENT2}>
            <strong>&#x1f9ea; TCTN3 / Tectonic Complex Mechanism:</strong>{' '}
            {overview?.tctn3_function_pearl}
          </Alert>

          {/* Gene Summary */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT + '22' }}>Gene &amp; Protein Summary</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><th style={{ width: 180 }}>Gene</th><td>{overview?.gene} (Tectonic-3) — 10q24.1</td></tr>
                  <tr><th>OMIM (Gene)</th><td>*{overview?.omim_gene}</td></tr>
                  <tr><th>OMIM (Disease)</th><td>#{overview?.omim_disease} — JBTS18</td></tr>
                  <tr><th>OMIM (OFD4)</th><td>#258860 — Oro-Facial-Digital Syndrome Type 4 (allelic)</td></tr>
                  <tr><th>Protein</th><td>{overview?.protein}</td></tr>
                  <tr><th>Inheritance</th><td>{overview?.inheritance}</td></tr>
                  <tr><th>Prevalence</th><td>{overview?.prevalence}</td></tr>
                  <tr><th>First Description</th><td>{overview?.first_description}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Phenotype bar chart */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3 + '22' }}>Phenotype Frequency — JBTS18 Cohort (N={N_COHORT})</div>
            <div className="card-body">
              {[
                { label: 'Molar Tooth Sign (pathognomonic)', pct: overview?.phenotype_summary?.mts_pct,      color: '#00695c' },
                { label: 'Cerebellar Ataxia',                pct: overview?.phenotype_summary?.ataxia_pct,   color: ACCENT3 },
                { label: 'Neonatal Hypotonia',               pct: overview?.phenotype_summary?.hypotonia_pct,color: ACCENT4 },
                { label: 'Oculomotor Apraxia',               pct: overview?.phenotype_summary?.oma_pct,      color: ACCENT2 },
                { label: 'Breathing Dysregulation',          pct: overview?.phenotype_summary?.breathing_pct,color: '#880e4f' },
                { label: 'Intellectual Disability',          pct: overview?.phenotype_summary?.id_pct,       color: '#5d4037' },
                { label: 'OFD4 Features (AR overlap)',       pct: overview?.phenotype_summary?.ofd4_pct,     color: ACCENT10 },
                { label: 'Polydactyly (post-axial)',         pct: overview?.phenotype_summary?.poly_pct,     color: ACCENT5 },
                { label: 'Retinal Dystrophy (rod-cone)',     pct: overview?.phenotype_summary?.retinal_pct,  color: ACCENT7 },
                { label: 'Renal NPHP-like TIN',             pct: overview?.phenotype_summary?.renal_pct,    color: ACCENT9 },
                { label: 'Hepatic CHF',                      pct: overview?.phenotype_summary?.hepatic_pct,  color: ACCENT8 },
                { label: 'Corpus Callosum Anomaly',          pct: overview?.phenotype_summary?.cc_pct,       color: ACCENT6 },
              ].map((row, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{row.label}</span><span className="fw-bold">{row.pct}%</span>
                  </div>
                  <div className="progress" style={{ height: 12 }}>
                    <div className="progress-bar" style={{ width: `${row.pct}%`, background: row.color }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Allele class distribution */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2 + '22' }}>Allele Class Distribution (cohort)</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead><tr><th>Allele Class</th><th>Count</th><th>%</th></tr></thead>
                <tbody>
                  {overview?.allele_class_distribution?.map((row, i) => (
                    <tr key={i}><td>{row.allele_class}</td><td>{row.count}</td><td>{row.pct}%</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: DIAGNOSTIC BREAKDOWN ── */}
      {tab === 1 && (
        <div>
          {/* Allele tiers */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT + '22' }}>Allele Class → Clinical Tier — JBTS18</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead>
                  <tr><th>Allele Class</th><th>Clinical Tier</th><th>Outcome</th><th>Example</th><th>Counselling</th></tr>
                </thead>
                <tbody>
                  {breakdown?.allele_tiers?.map((row, i) => (
                    <tr key={i}>
                      <td><strong>{row.allele_class}</strong></td>
                      <td>{row.clinical_tier}</td>
                      <td>{row.outcome}</td>
                      <td className="text-muted small">{row.example}</td>
                      <td className="text-muted small">{row.counselling}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Ethnicity distribution */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3 + '22' }}>Ethnicity Distribution (cohort)</div>
            <div className="card-body">
              {breakdown?.ethnicity_distribution?.map((row, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span>{row.ethnicity}</span><span className="fw-bold">{row.count} ({row.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar" style={{ width: `${row.pct}%`, background: ACCENT2 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Key variants / genotype table */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT7 + '22' }}>Key Pathogenic Variants — TCTN3 / JBTS18</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead>
                    <tr><th>Variant</th><th>cDNA</th><th>Domain</th><th>Population</th><th>Severity</th><th>Mechanism</th></tr>
                  </thead>
                  <tbody>
                    {breakdown?.genotype_table?.map((v, i) => (
                      <tr key={i}>
                        <td><strong>{v.name}</strong></td>
                        <td className="small text-muted">{v.cdna}</td>
                        <td className="small">{v.domain}</td>
                        <td className="small">{v.population}</td>
                        <td>
                          <span className="badge" style={{
                            background: v.severity.includes('Null') ? ACCENT7 : v.severity.includes('Mild') || v.severity.includes('OFD4') ? ACCENT5 : ACCENT2,
                            color: '#fff'
                          }}>{v.severity}</span>
                        </td>
                        <td className="small">{v.mechanism}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Domain-phenotype matrix */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6 + '22' }}>TCTN3 Domain → Phenotype Matrix</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead>
                  <tr><th>Domain</th><th>Key Variants</th><th>Function Lost</th><th>Severity</th><th>Retinal</th><th>Renal</th></tr>
                </thead>
                <tbody>
                  {breakdown?.domain_phenotype_matrix?.map((row, i) => (
                    <tr key={i}>
                      <td><strong>{row.domain}</strong></td>
                      <td className="small">{row.key_variants}</td>
                      <td className="small">{row.function_lost}</td>
                      <td className="small">{row.severity}</td>
                      <td>{row.retinal_risk}</td>
                      <td>{row.renal_risk}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Variant distribution */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2 + '22' }}>Variant / Allele Class Distribution</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead><tr><th>Allele Class</th><th>Count</th><th>%</th></tr></thead>
                <tbody>
                  {breakdown?.variant_distribution?.map((row, i) => (
                    <tr key={i}><td>{row.allele_class}</td><td>{row.count}</td><td>{row.pct}%</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Patient table (first 20) */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT + '22' }}>Patient Cohort Sample (first 20 / 40)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0" style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>ID</th><th>Sex</th><th>Ethnicity</th><th>Allele</th><th>Age Dx</th>
                      <th>MTS</th><th>Ataxia</th><th>OMA</th><th>OFD4</th><th>Poly</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>ID</th><th>Breathing</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown?.patient_table?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td><td>{p.sex}</td><td>{p.ethnicity}</td><td>{p.allele}</td><td>{p.age_dx_yr}yr</td>
                        <td><span className="text-success">&#x2714;</span></td>
                        <td>{p.ataxia === 'Yes' ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.oma === 'Yes' ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.ofd4 && p.ofd4.includes('Yes') ? <span style={{ color: ACCENT10 }}>Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.poly.includes('Yes') ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.retinal.includes('Yes') ? <span className="text-danger">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.renal.includes('Yes') ? <span className="text-info">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.hepatic.includes('Yes') ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.id_ === 'Yes' ? <span className="text-secondary">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.breathing === 'Yes' ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: OFD4 & TCTN PEARLS ── */}
      {tab === 2 && (
        <div>
          <Alert color={ACCENT2}>
            <strong>&#x1f9ea; Mechanism Pearl — TCTN3 Tectonic Complex:</strong>{' '}
            {overview?.tctn3_function_pearl}
          </Alert>

          <Alert color={ACCENT10}>
            <strong>&#x1f9f0; OFD4 Allelic Overlap — AR (Not X-linked):</strong>{' '}
            {overview?.ofd4_pearl}
          </Alert>

          <Alert color={ACCENT}>
            <strong>&#x2714; No MKS Tier Explained:</strong>{' '}
            {overview?.no_mks_pearl}
          </Alert>

          {/* Clinical pearls from get_definitions().clinical_pearls */}
          {definitions?.clinical_pearls?.map((pearl, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header fw-bold" style={{ background: (i % 2 === 0 ? ACCENT2 : ACCENT10) + '22' }}>
                &#x1f4a1; {pearl.title}
              </div>
              <div className="card-body">
                <p className="small mb-0">{pearl.detail}</p>
              </div>
            </div>
          ))}

          {/* TCTN1 / TCTN2 / TCTN3 comparison */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6 + '22' }}>
              TCTN1 / TCTN2 / TCTN3 — Same Complex, Different Disease Tiers
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead>
                  <tr><th>Gene</th><th>Locus</th><th>Disease</th><th>MKS Tier?</th><th>OFD Allelic?</th><th>Retinal</th><th>Renal</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong style={{ color: '#7b1fa2' }}>TCTN1</strong></td>
                    <td>12q24.11</td>
                    <td>JBTS11</td>
                    <td><span className="badge bg-success">No MKS</span></td>
                    <td><span className="text-muted">No</span></td>
                    <td>~30%</td>
                    <td>~20%</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: '#283593' }}>TCTN2</strong></td>
                    <td>12q24.31</td>
                    <td>JBTS13 / MKS8</td>
                    <td><span className="badge bg-danger">MKS8 — Null/Null Lethal</span></td>
                    <td><span className="text-muted">No</span></td>
                    <td>~45%</td>
                    <td>~32%</td>
                  </tr>
                  <tr>
                    <td><strong style={{ color: ACCENT }}>TCTN3</strong></td>
                    <td>10q24.1</td>
                    <td>JBTS18 / OFD4</td>
                    <td><span className="badge bg-success">No MKS</span></td>
                    <td><span className="badge" style={{ background: ACCENT10, color: '#fff' }}>OFD4 (AR) ~18%</span></td>
                    <td>~25%</td>
                    <td>~20%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Literature */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6 + '22' }}>Key Literature</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {definitions?.literature_highlights?.map((ref, i) => (
                  <li key={i} className="mb-1">{ref}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && (
        <div>
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT + '22' }}>Gene &amp; Disease Identifiers</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><th style={{ width: 220 }}>Gene Full Name</th><td>{definitions?.gene_full_name}</td></tr>
                  <tr><th>OMIM Gene</th><td>*{definitions?.omim_gene}</td></tr>
                  <tr><th>OMIM JBTS18</th><td>#{definitions?.omim_jbts18}</td></tr>
                  <tr><th>OMIM OFD4 (allelic)</th><td>#{definitions?.omim_ofd4} — Oro-Facial-Digital Syndrome Type 4 (AR)</td></tr>
                  <tr><th>Chromosome</th><td>{definitions?.chromosome}</td></tr>
                  <tr><th>Protein Size</th><td>{definitions?.protein_size}</td></tr>
                  <tr><th>Inheritance</th><td>{definitions?.inheritance}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <Alert color={ACCENT}>
            <strong>No MKS Tier Rule:</strong> {definitions?.no_mks_tier_rule}
          </Alert>

          {/* Glossary */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2 + '22' }}>Glossary — JBTS18 / TCTN3 / OFD4</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <thead><tr><th style={{ width: 200 }}>Term</th><th>Definition</th></tr></thead>
                <tbody>
                  {definitions?.glossary?.map((g, i) => (
                    <tr key={i}>
                      <th>{g.term}</th>
                      <td className="small">{g.definition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3 + '22' }}>Phenotype Frequencies — JBTS18</div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0">
                <tbody>
                  {definitions?.phenotype_frequencies && Object.entries(definitions.phenotype_frequencies).map(([k, v], i) => (
                    <tr key={i}>
                      <th style={{ width: 260 }}>{k.replace(/_/g, ' ')}</th>
                      <td>{v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Domain matrix */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6 + '22' }}>TCTN3 Domain Matrix (3 Domains)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead>
                    <tr><th>Domain</th><th>Location</th><th>Function</th><th>Variant Examples</th></tr>
                  </thead>
                  <tbody>
                    {definitions?.domain_matrix?.map((row, i) => (
                      <tr key={i}>
                        <td><strong>{row.domain}</strong></td>
                        <td className="small">{row.location}</td>
                        <td className="small">{row.function}</td>
                        <td className="small text-muted">{row.variant_examples}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Literature */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT4 + '22' }}>Literature Highlights</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {definitions?.literature_highlights?.map((h, i) => (
                  <li key={i} className="mb-1">{h}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Back nav */}
      <div className="mt-4">
        <Link href="/" className="btn btn-outline-secondary btn-sm">&#x2190; Back to Dashboard</Link>
      </div>
    </div>
  );
}
