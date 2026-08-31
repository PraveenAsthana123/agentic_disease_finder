'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'TMEM237 TZ Pearls', 'Definitions'];

// JBTS14 colour scheme — TMEM237 / NPHP-module bridge / No MKS tier / TZ transition fibre
const ACCENT  = '#1a237e';   // deep indigo — TZ scaffold / MTS
const ACCENT2 = '#1565c0';   // strong blue — neurological / cerebellar
const ACCENT3 = '#1b5e20';   // deep green — No MKS tier / curative endpoint
const ACCENT4 = '#00695c';   // dark teal — NPHP curative / renal
const ACCENT5 = '#e65100';   // burnt orange — polydactyly (rare ~6%)
const ACCENT6 = '#37474f';   // dark slate — domain matrix / TZ complex
const ACCENT7 = '#b71c1c';   // deep red — retinal rod-cone (~27%)
const ACCENT8 = '#33691e';   // dark olive — hepatic mild CHF (~8%)
const ACCENT9 = '#00695c';   // dark teal — renal NPHP-like (~28%)
const ACCENT10 = '#4527a0';  // deep purple — NPHP-module cross-link

const SEED = 435;
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

export default function JBTS14Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts14/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts14/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts14/definitions`).then(r => r.json()),
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
        <h4 className="mb-1 fw-bold">&#x1f9ec; TMEM237 — Joubert Syndrome Type 14 (JBTS14)</h4>
        <div className="small opacity-90">
          Transmembrane Protein 237 (ALS2CR4) · 2q33.1 · 541 aa · TZ NPHP-Module Bridge · No MKS Tier · AR · OMIM Gene #614423 · Disease #614424
        </div>
        <div className="small opacity-80 mt-1">
          Cohort: {N_COHORT} JBTS14 patients (seed {SEED}) · all biallelic genotypes → live birth (no MKS lethal tier)
        </div>
      </div>

      {/* No MKS tier banner */}
      <Alert color={ACCENT3}>
        <strong>&#x2705; No MKS Lethal Tier — TMEM237-Specific Rule:</strong> Unlike TCTN2 (MKS8), CC2D2A (MKS6), RPGRIP1L (MKS5), TMEM67 (MKS3), or CEP290 (MKS4), <em>all</em> biallelic TMEM237 genotypes produce JBTS14 live birth. Standard 25% JBTS14 recurrence counselling applies to all families — no MKS-specific prenatal urgency calculation needed.
      </Alert>

      {/* NPHP-bridge warning */}
      <Alert color={ACCENT10}>
        <strong>&#x26a0; NPHP1-Bridge Interaction — Renal Surveillance:</strong> TMEM237 directly contacts NPHP1 (Y-link scaffold). This NPHP-module dependency creates a ~28% tubulointerstitial nephritis risk even for missense alleles. Annual NPHP renal protocol (creatinine + urine osmolality) is mandatory from diagnosis — do not defer renal surveillance until proteinuria appears.
      </Alert>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* Overview tab */}
      {tab === 0 && overview && (
        <div>
          <div className="row g-2 mb-3">
            {(overview.kpis || []).map((k, i) => <KPI key={i} {...k} />)}
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>TMEM237 TZ Function — NPHP-Module Bridge</div>
                <div className="card-body small">{overview.tmem237_function_pearl}</div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>No MKS Tier — Clinical Pearl</div>
                <div className="card-body small">{overview.no_mks_pearl}</div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT10, color: '#fff' }}>NPHP-Module Bridge — Renal Surveillance Pearl</div>
            <div className="card-body small">{overview.nphp_bridge_pearl}</div>
          </div>

          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Gene Summary</div>
                <div className="card-body small">{overview.gene_summary}</div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Allele Class Distribution (N={N_COHORT})</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Allele Class</th><th>n</th><th>%</th></tr></thead>
                    <tbody>
                      {(overview.allele_class_distribution || []).map((r, i) => (
                        <tr key={i}><td className="small">{r.allele_class}</td><td>{r.count}</td><td>{r.pct}%</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Phenotype Summary — JBTS14 Cohort (N={N_COHORT})</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(overview.phenotype_summary || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-3">
                    <div className="border rounded p-2 text-center">
                      <div className="fw-bold">{v}%</div>
                      <div className="text-muted small">{k.replace(/_pct$/, '').replace(/_/g, ' ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Clinical Reference</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>{overview.gene}</td><td className="fw-bold">Disease</td><td className="small">{overview.disease}</td></tr>
                  <tr><td className="fw-bold">OMIM Gene</td><td>#{overview.omim_gene}</td><td className="fw-bold">OMIM JBTS14</td><td>#{overview.omim_disease}</td></tr>
                  <tr><td className="fw-bold">Chr</td><td>{overview.chromosome}</td><td className="fw-bold">MKS Tier</td><td className="text-success fw-bold">None (all live birth)</td></tr>
                  <tr><td className="fw-bold">Protein</td><td colSpan={3} className="small">{overview.protein}</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td colSpan={3} className="small">{overview.inheritance}</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td colSpan={3}>{overview.prevalence}</td></tr>
                  <tr><td className="fw-bold">First desc.</td><td colSpan={3} className="small">{overview.first_description}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Breakdown tab */}
      {tab === 1 && breakdown && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>Allele Class Distribution</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Allele Class</th><th>n</th><th>%</th></tr></thead>
                    <tbody>
                      {(breakdown.allele_distribution || []).map((r, i) => (
                        <tr key={i}><td className="small">{r.allele_class}</td><td>{r.count}</td><td>{r.pct}%</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Ethnicity Distribution</div>
                <div className="card-body">
                  <table className="table table-sm table-bordered mb-0">
                    <thead><tr><th>Ethnicity</th><th>n</th><th>%</th></tr></thead>
                    <tbody>
                      {(breakdown.ethnicity_distribution || []).map((r, i) => (
                        <tr key={i}><td className="small">{r.ethnicity}</td><td>{r.count}</td><td>{r.pct}%</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Allele tier table */}
          {breakdown.allele_tiers && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>Allele Class → Clinical Tier (TMEM237 — No MKS Rule)</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Allele Class</th><th>Clinical Tier</th><th>Outcome</th><th>Example</th><th>Counselling</th></tr></thead>
                    <tbody>
                      {breakdown.allele_tiers.map((r, i) => (
                        <tr key={i}>
                          <td className="small fw-bold">{r.allele_class}</td>
                          <td className="small">{r.clinical_tier}</td>
                          <td className="small">{r.outcome}</td>
                          <td className="small font-monospace">{r.example}</td>
                          <td className="small">{r.counselling}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Key variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Key TMEM237 Pathogenic Variants</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-dark"><tr><th>Variant</th><th>Domain</th><th>Effect</th><th>Population</th><th>Severity</th><th>Retinal Risk</th><th>Renal Risk</th></tr></thead>
                  <tbody>
                    {(breakdown.key_variants || []).map((v, i) => (
                      <tr key={i}>
                        <td className="small font-monospace fw-bold">{v.variant}</td>
                        <td className="small">{v.domain}</td>
                        <td className="small">{v.effect}</td>
                        <td className="small">{v.population}</td>
                        <td className="small">{v.severity}</td>
                        <td className="small">{v.retinal_risk}</td>
                        <td className="small">{v.renal_risk}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Domain matrix */}
          {breakdown.domain_phenotype_matrix && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Domain–Phenotype Matrix</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Domain</th><th>Key Variants</th><th>Function Lost</th><th>Severity</th><th>Retinal Risk</th><th>Renal Risk</th></tr></thead>
                    <tbody>
                      {breakdown.domain_phenotype_matrix.map((r, i) => (
                        <tr key={i}>
                          <td className="small fw-bold">{r.domain}</td>
                          <td className="small font-monospace">{r.key_variants}</td>
                          <td className="small">{r.function_lost}</td>
                          <td className="small">{r.severity}</td>
                          <td className="small">{r.retinal_risk}</td>
                          <td className="small">{r.renal_risk}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Patient table */}
          {breakdown.patient_table && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Patient Cohort Snapshot (first 20 of {N_COHORT})</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered table-hover mb-0">
                    <thead className="table-dark">
                      <tr><th>ID</th><th>Sex</th><th>Ethnicity</th><th>Allele</th><th>Age Dx</th><th>MTS</th><th>Ataxia</th><th>OMA</th><th>Retinal</th><th>Renal</th><th>Hepatic</th><th>Poly</th><th>ID</th><th>Breathing</th></tr>
                    </thead>
                    <tbody>
                      {breakdown.patient_table.map((p, i) => (
                        <tr key={i}>
                          <td className="small font-monospace">{p.id}</td>
                          <td className="small">{p.sex}</td>
                          <td className="small">{p.ethnicity}</td>
                          <td className="small">{p.allele}</td>
                          <td className="small">{p.age_dx_yr}y</td>
                          <td className="small text-success fw-bold">{p.mts}</td>
                          <td className="small">{p.ataxia}</td>
                          <td className="small">{p.oma}</td>
                          <td className="small" style={{ color: p.retinal !== 'No' ? ACCENT7 : 'inherit' }}>{p.retinal}</td>
                          <td className="small" style={{ color: p.renal !== 'No' ? ACCENT9 : 'inherit' }}>{p.renal}</td>
                          <td className="small" style={{ color: p.hepatic !== 'No' ? ACCENT8 : 'inherit' }}>{p.hepatic}</td>
                          <td className="small">{p.poly}</td>
                          <td className="small">{p.id_}</td>
                          <td className="small">{p.breathing}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Pathway steps */}
          {breakdown.pathway_steps && (
            <div className="card shadow-sm mb-3">
              <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>TMEM237 TZ Pathway — Loss-of-Function Cascade</div>
              <div className="card-body p-0">
                <table className="table table-sm table-bordered mb-0">
                  <thead className="table-dark"><tr><th>Step</th><th>Normal Event</th><th>Effect When TMEM237 Lost</th></tr></thead>
                  <tbody>
                    {breakdown.pathway_steps.map((s, i) => (
                      <tr key={i}>
                        <td className="small fw-bold text-center">{s.step}</td>
                        <td className="small">{s.event}</td>
                        <td className="small text-danger">{s.effect_when_lost}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Management */}
          {breakdown.management && (
            <div className="card shadow-sm">
              <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>Management & Surveillance Protocol</div>
              <div className="card-body p-0">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Intervention</th><th>Timing</th><th>Rationale</th><th>Evidence Level</th></tr></thead>
                    <tbody>
                      {breakdown.management.map((m, i) => (
                        <tr key={i}>
                          <td className="small fw-bold">{m.intervention}</td>
                          <td className="small">{m.timing}</td>
                          <td className="small">{m.rationale}</td>
                          <td className="small">{m.level}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TMEM237 TZ Pearls tab */}
      {tab === 2 && overview && breakdown && (
        <div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>TMEM237 TZ Biology — NPHP-Module Bridge</div>
                <div className="card-body small">{overview.tmem237_function_pearl}</div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>No MKS Tier — Why TMEM237 Differs</div>
                <div className="card-body small">{overview.no_mks_pearl}</div>
              </div>
            </div>
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-bold" style={{ background: ACCENT10, color: '#fff' }}>NPHP1-Bridge Interaction — Renal Surveillance Rationale</div>
                <div className="card-body small">{overview.nphp_bridge_pearl}</div>
              </div>
            </div>
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>JBTS14 vs Key DDx — Side-by-Side Comparisons</div>
                <div className="card-body p-0">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark"><tr><th>Feature</th><th>JBTS14 (TMEM237)</th><th>Comparator</th><th>Significance</th></tr></thead>
                    <tbody>
                      <tr>
                        <td className="small fw-bold">MKS lethal tier</td>
                        <td className="small text-success fw-bold">None — all live birth</td>
                        <td className="small">TCTN2 → MKS8 (perinatal lethal)</td>
                        <td className="small">Counselling substantially simpler for JBTS14 families</td>
                      </tr>
                      <tr>
                        <td className="small fw-bold">Renal risk</td>
                        <td className="small">~28% NPHP-like TIN</td>
                        <td className="small">KIF7 (JBTS12) ~12%; CC2D2A (JBTS9) ~38%</td>
                        <td className="small">NPHP1-bridge interaction drives renal risk</td>
                      </tr>
                      <tr>
                        <td className="small fw-bold">Hepatic risk</td>
                        <td className="small">~8% mild CHF</td>
                        <td className="small">TMEM67 (JBTS6) ~30%; TCTN2 (JBTS13) ~22%</td>
                        <td className="small">Mild; monitor but lower risk than COACH-spectrum</td>
                      </tr>
                      <tr>
                        <td className="small fw-bold">Polydactyly</td>
                        <td className="small">~6% (rare)</td>
                        <td className="small">KIF7 (JBTS12) 35–45%; CC2D2A (JBTS9) ~20%</td>
                        <td className="small">Polydactyly points away from JBTS14</td>
                      </tr>
                      <tr>
                        <td className="small fw-bold">NPHP protocol</td>
                        <td className="small text-danger fw-bold">Mandatory — NPHP1 bridge</td>
                        <td className="small">Many JBTS types use general renal screen only</td>
                        <td className="small">Annual creatinine + urine osm from diagnosis; do not defer</td>
                      </tr>
                      <tr>
                        <td className="small fw-bold">Corpus callosum</td>
                        <td className="small">Not expected</td>
                        <td className="small">KIF7 (JBTS12) 20–25% CC anomaly</td>
                        <td className="small">CC anomaly on MRI points away from JBTS14</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Definitions tab */}
      {tab === 3 && definitions && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>Gene & Disease Reference</div>
                <div className="card-body">
                  <table className="table table-sm mb-0">
                    <tbody>
                      <tr><td className="fw-bold">Full Name</td><td>{definitions.gene_full_name}</td></tr>
                      <tr><td className="fw-bold">OMIM Gene</td><td>#{definitions.omim_gene}</td></tr>
                      <tr><td className="fw-bold">OMIM JBTS14</td><td>#{definitions.omim_jbts14}</td></tr>
                      <tr><td className="fw-bold">Chromosome</td><td>{definitions.chromosome}</td></tr>
                      <tr><td className="fw-bold">Protein</td><td>{definitions.protein_size}</td></tr>
                      <tr><td className="fw-bold">Inheritance</td><td>{definitions.inheritance}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>No MKS Tier — Rule Statement</div>
                <div className="card-body small">{definitions.no_mks_tier_rule}</div>
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Phenotype Frequencies — JBTS14</div>
            <div className="card-body">
              <div className="row g-2">
                {Object.entries(definitions.phenotype_frequencies || {}).map(([k, v]) => (
                  <div key={k} className="col-6 col-md-4 col-lg-3">
                    <div className="border rounded p-2">
                      <div className="small fw-bold" style={{ color: k.includes('retinal') ? ACCENT7 : k.includes('renal') ? ACCENT9 : k.includes('hepatic') ? ACCENT8 : ACCENT }}>{v}</div>
                      <div className="text-muted" style={{ fontSize: '0.72rem' }}>{k.replace(/_/g, ' ')}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>Clinical Distinctions — Key DDx Points</div>
            <div className="card-body">
              <table className="table table-sm mb-0">
                <tbody>
                  {Object.entries(definitions.key_clinical_distinctions || {}).map(([k, v]) => (
                    <tr key={k}><td className="small fw-bold">{k.replace(/_/g, ' ')}</td><td className="small">{v}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>Management Highlights</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {(definitions.management_highlights || []).map((h, i) => <li key={i}>{h}</li>)}
              </ul>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: ACCENT6, color: '#fff' }}>Key Literature</div>
            <div className="card-body">
              <ul className="mb-0 small">
                {(definitions.literature_highlights || []).map((l, i) => <li key={i}>{l}</li>)}
              </ul>
            </div>
          </div>
        </div>
      )}

      <div className="mt-3 text-center">
        <Link href="/" className="btn btn-outline-secondary btn-sm me-2">&#8592; Home</Link>
        <Link href="/joubert" className="btn btn-outline-primary btn-sm me-2">Joubert Overview</Link>
        <Link href="/jbts13" className="btn btn-outline-secondary btn-sm me-2">&#8592; JBTS13 (TCTN2)</Link>
      </div>
    </div>
  );
}
