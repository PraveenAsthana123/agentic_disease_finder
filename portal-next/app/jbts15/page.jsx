'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CEP41 Glutamylation Pearls', 'Definitions'];

// JBTS15 colour scheme — CEP41 / tubulin glutamylation / axoneme modifier / No MKS tier
const ACCENT  = '#1b5e20';   // deep green — No MKS tier / tubulin code
const ACCENT2 = '#2e7d32';   // medium green — glutamylation / IFT-B
const ACCENT3 = '#1565c0';   // strong blue — neurological / cerebellar
const ACCENT4 = '#00695c';   // dark teal — renal mild
const ACCENT5 = '#e65100';   // burnt orange — polydactyly (~10%)
const ACCENT6 = '#37474f';   // dark slate — domain matrix / axoneme
const ACCENT7 = '#b71c1c';   // deep red — retinal rod-cone (~20%)
const ACCENT8 = '#33691e';   // dark olive — hepatic mild CHF (~8%)
const ACCENT9 = '#004d40';   // very dark teal — renal NPHP-like (~15%)
const ACCENT10 = '#4a148c';  // deep purple — tubulin glutamylation pathway

const SEED = 437;
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

export default function JBTS15Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts15/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts15/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts15/definitions`).then(r => r.json()),
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
        <h4 className="mb-1 fw-bold">&#x1f9ec; CEP41 — Joubert Syndrome Type 15 (JBTS15)</h4>
        <div className="small opacity-90">
          Centrosomal Protein 41 · 7q32.2 · 232 aa · Axoneme TTLL6 Scaffold · Tubulin Glutamylation Regulator · No MKS Tier · AR · OMIM Gene #612112 · Disease #614464
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS15 patients (seed {SEED}) · all biallelic genotypes → live birth (no MKS lethal tier)
        </div>
      </div>

      {/* No MKS tier banner */}
      <Alert color={ACCENT}>
        <strong>&#x2714; No MKS Lethal Tier — CEP41-Specific Rule:</strong>{' '}
        {overview?.no_mks_pearl?.slice(0, 220)}… CEP41 acts via post-translational axoneme modification, not TZ scaffold assembly.
        All biallelic genotypes → JBTS15 live birth. Standard 25% AR recurrence applies; no MKS tier calculation needed.
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

          {/* CEP41 Function Pearl */}
          <Alert color={ACCENT2}>
            <strong>&#x1f9ea; CEP41 / TTLL6 Axoneme Glutamylation Mechanism:</strong>{' '}
            {overview?.cep41_function_pearl}
          </Alert>

          {/* TTLL6 Glutamylation Pearl */}
          <Alert color={ACCENT10}>
            <strong>&#x1f9ec; Tubulin Glutamylation Code — IFT-B Link:</strong>{' '}
            {overview?.ttll6_glutamylation_pearl}
          </Alert>

          {/* Gene Summary */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT + '22' }}>Gene &amp; Protein Summary</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><th style={{ width: 180 }}>Gene</th><td>{overview?.gene} (CEP41) — 7q32.2</td></tr>
                  <tr><th>OMIM (Gene)</th><td>#{overview?.omim_gene}</td></tr>
                  <tr><th>OMIM (Disease)</th><td>#{overview?.omim_disease} — JBTS15</td></tr>
                  <tr><th>Protein</th><td>{overview?.protein}</td></tr>
                  <tr><th>Inheritance</th><td>{overview?.inheritance}</td></tr>
                  <tr><th>Prevalence</th><td>{overview?.prevalence}</td></tr>
                  <tr><th>First Description</th><td>{overview?.first_description}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Phenotype bar chart (visual) */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3 + '22' }}>Phenotype Frequency — JBTS15 Cohort (N={N_COHORT})</div>
            <div className="card-body">
              {[
                { label: 'Molar Tooth Sign (pathognomonic)', pct: overview?.phenotype_summary?.mts_pct,      color: '#1a237e' },
                { label: 'Cerebellar Ataxia',                pct: overview?.phenotype_summary?.ataxia_pct,   color: ACCENT3 },
                { label: 'Neonatal Hypotonia',               pct: overview?.phenotype_summary?.hypotonia_pct,color: ACCENT6 },
                { label: 'Oculomotor Apraxia',               pct: overview?.phenotype_summary?.oma_pct,      color: ACCENT10 },
                { label: 'Breathing Dysregulation',          pct: overview?.phenotype_summary?.breathing_pct,color: '#880e4f' },
                { label: 'Intellectual Disability',          pct: overview?.phenotype_summary?.id_pct,       color: '#5d4037' },
                { label: 'Retinal Dystrophy (rod-cone)',     pct: overview?.phenotype_summary?.retinal_pct,  color: ACCENT7 },
                { label: 'Renal NPHP-like TIN',             pct: overview?.phenotype_summary?.renal_pct,    color: ACCENT9 },
                { label: 'Hepatic Mild CHF',                 pct: overview?.phenotype_summary?.hepatic_pct,  color: ACCENT8 },
                { label: 'Polydactyly (post-axial)',         pct: overview?.phenotype_summary?.poly_pct,     color: ACCENT5 },
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
            <div className="card-header fw-bold" style={{ background: ACCENT10 + '22' }}>Allele Class Distribution (cohort)</div>
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
            <div className="card-header fw-bold" style={{ background: ACCENT + '22' }}>Allele Class → Clinical Tier — JBTS15</div>
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
                    <div className="progress-bar" style={{ width: `${row.pct}%`, background: ACCENT3 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Key variants */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT7 + '22' }}>Key Pathogenic Variants — CEP41 / JBTS15</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead>
                    <tr><th>Variant</th><th>Domain</th><th>Effect</th><th>Population</th><th>Severity</th><th>Retinal</th><th>Renal</th></tr>
                  </thead>
                  <tbody>
                    {breakdown?.key_variants?.map((v, i) => (
                      <tr key={i}>
                        <td><strong>{v.variant}</strong></td>
                        <td className="small">{v.domain}</td>
                        <td className="small">{v.effect}</td>
                        <td className="small">{v.population}</td>
                        <td><span className="badge" style={{ background: v.severity.includes('Null') ? ACCENT7 : v.severity.includes('Mild') ? ACCENT : ACCENT3, color: '#fff' }}>{v.severity}</span></td>
                        <td>{v.retinal_risk}</td>
                        <td>{v.renal_risk}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Domain-phenotype matrix */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6 + '22' }}>CEP41 Domain → Phenotype Matrix</div>
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

          {/* Pathway steps */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2 + '22' }}>CEP41 → Hedgehog Pathway — Step-by-Step</div>
            <div className="card-body">
              {breakdown?.pathway_steps?.map((step, i) => (
                <div key={i} className="d-flex mb-3">
                  <div className="me-3">
                    <div className="rounded-circle d-flex align-items-center justify-content-center fw-bold text-white"
                      style={{ width: 32, height: 32, background: ACCENT2, fontSize: 14 }}>{step.step}</div>
                  </div>
                  <div>
                    <div className="small text-muted mb-1"><strong>Normal:</strong> {step.event}</div>
                    <div className="small" style={{ color: ACCENT7 }}><strong>CEP41 LOF:</strong> {step.effect_when_lost}</div>
                  </div>
                </div>
              ))}
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
                      <th>MTS</th><th>Ataxia</th><th>OMA</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>Poly</th><th>ID</th><th>Breathing</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown?.patient_table?.map((p, i) => (
                      <tr key={i}>
                        <td>{p.id}</td><td>{p.sex}</td><td>{p.ethnicity}</td><td>{p.allele}</td><td>{p.age_dx_yr}yr</td>
                        <td><span className="text-success">&#x2714;</span></td>
                        <td>{p.ataxia === 'Yes' ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.oma === 'Yes' ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.retinal.includes('Yes') ? <span className="text-danger">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.renal.includes('Yes') ? <span className="text-info">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.hepatic.includes('Yes') ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.poly.includes('Yes') ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.id_ === 'Yes' ? <span className="text-secondary">Y</span> : <span className="text-muted">N</span>}</td>
                        <td>{p.breathing === 'Yes' ? <span className="text-warning">Y</span> : <span className="text-muted">N</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Management */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT4 + '22' }}>Management Protocol — JBTS15 / CEP41</div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead>
                  <tr><th>Intervention</th><th>Timing</th><th>Rationale</th><th>Level</th></tr>
                </thead>
                <tbody>
                  {breakdown?.management?.map((m, i) => (
                    <tr key={i}>
                      <td><strong>{m.intervention}</strong></td>
                      <td className="small">{m.timing}</td>
                      <td className="small">{m.rationale}</td>
                      <td><span className="badge bg-secondary">{m.level}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: CEP41 GLUTAMYLATION PEARLS ── */}
      {tab === 2 && (
        <div>
          <Alert color={ACCENT2}>
            <strong>&#x1f9ea; Mechanism Pearl — CEP41 &amp; TTLL6:</strong>{' '}
            {overview?.cep41_function_pearl}
          </Alert>

          <Alert color={ACCENT10}>
            <strong>&#x1f9ec; Tubulin Glutamylation Code &amp; IFT-B Processivity:</strong>{' '}
            {overview?.ttll6_glutamylation_pearl}
          </Alert>

          <Alert color={ACCENT}>
            <strong>&#x2714; No MKS Tier Explained:</strong>{' '}
            {overview?.no_mks_pearl}
          </Alert>

          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2 + '22' }}>
              JBTS15 vs Other No-MKS-Tier JBTS Types
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0">
                <thead>
                  <tr><th>Comparison</th><th>Key Distinction</th></tr>
                </thead>
                <tbody>
                  {definitions && Object.entries(definitions.key_clinical_distinctions || {}).map(([key, val], i) => (
                    <tr key={i}>
                      <td><strong style={{ color: ACCENT2 }}>{key.replace(/_/g, ' ')}</strong></td>
                      <td className="small">{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Unique functional biomarker */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT10 + '22' }}>
              Unique Functional Biomarker — GT335 Polyglutamylation IF
            </div>
            <div className="card-body">
              <p className="small mb-2">
                CEP41 is the only JBTS gene acting via tubulin glutamylation. Patient-derived fibroblast
                cilia can be stained with the GT335 antibody (anti-polyglutamylated tubulin) to detect
                reduced axonemal glutamylation. This provides a functional readout distinct from structural
                TZ scaffold assays (e.g. CEP290, NPHP1 localisation).
              </p>
              <p className="small mb-0">
                Clinical utility: when WES/WGS returns a CEP41 VUS (variant of uncertain significance),
                GT335 IF in patient fibroblast cilia showing ≥30% reduction in signal vs controls
                provides functional evidence for pathogenicity — supporting variant reclassification
                to likely pathogenic.
              </p>
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
                  <tr><th>OMIM Gene</th><td>#{definitions?.omim_gene}</td></tr>
                  <tr><th>OMIM JBTS15</th><td>#{definitions?.omim_jbts15}</td></tr>
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

          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT3 + '22' }}>Phenotype Frequencies — JBTS15</div>
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

          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT4 + '22' }}>Management Highlights</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {definitions?.management_highlights?.map((h, i) => (
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
